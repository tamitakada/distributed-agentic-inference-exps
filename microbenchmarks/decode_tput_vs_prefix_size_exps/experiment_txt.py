#!/usr/bin/env python3
"""
Online (Poisson-arrival) long-prompt interference experiment.

Per (trial, mode, prefix length L):
  0. Wait for server idle, reset prefix cache.
  1. Send R' = chat-templated head + first L corpus tokens, max_tokens=1, ALONE. Wait for it to finish.
  2. Measurement window starts the instant R' returns: send R (same prefix + summarize instruction) and,
     at the same moment, start Poisson filler arrivals (rate --lam, seeded: identical schedule and
     identical ShareGPT prompt order for every L / mode / trial).
  3. Window ends when R's output is fully received. In-flight fillers are cancelled.
  4. Matched control (unless --no-control): replay the SAME arrival schedule with no R for the same
     duration, so filler-only throughput/latency is directly comparable.
R's output: "fixed:<n>" = exactly n tokens (ignore_eos); "natural" = until EOS (guard: --natural-max).
Throughput and batch size come from vLLM /metrics (polled every 0.25 s). Per-request latencies from
streamed timestamps. Energy/power from NVML. Run inside the same container as the server.
"""
import argparse, asyncio, bisect, hashlib, json, os, random, re, resource, threading, time
import aiohttp, pynvml
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--corpus", default=None,
               help="plain text file; default: download Complete Works of Shakespeare from Project Gutenberg")
p.add_argument("--filler-data", default=None,
               help="ShareGPT json; default: download ShareGPT_V3_unfiltered_cleaned_split.json from Hugging Face")
p.add_argument("--lam", type=float, required=True, help="filler arrival rate, requests/s (Poisson)")
p.add_argument("--seed", type=int, default=0, help="arrival-process seed (same for every L/mode/trial)")
p.add_argument("--modes", nargs="+", default=["fixed:512", "natural"],
               help="R output modes: fixed:<n> and/or natural")
p.add_argument("--natural-max", type=int, default=4096, help="max_tokens guard for natural mode")
p.add_argument("--url", default="http://localhost:8000")
p.add_argument("--pool-size", type=int, default=20000, help="filler prompts; arrival i uses pool[i %% size]")
p.add_argument("--filler-seed", type=int, default=0, help="seed for ShareGPT pool order")
p.add_argument("--filler-min-len", type=int, default=32)
p.add_argument("--filler-max-len", type=int, default=1024)
p.add_argument("--filler-max-out", type=int, default=8192, help="runaway guard on filler output (0 = context limit)")
p.add_argument("--min-len", type=int, default=1024)
p.add_argument("--max-prompt", type=int, default=0, help="largest R prefix (tokens) to sweep; 0 = server limit")
p.add_argument("--budget-min", type=float, default=0,
               help="wall-clock budget (min) from sweep start; skips sizes predicted to overrun (0 = off)")
p.add_argument("--no-snap", action="store_true", help="cut corpus at exactly L tokens (may split a sentence)")
p.add_argument("--trials", type=int, default=3)
p.add_argument("--no-control", action="store_true", help="skip the matched fillers-only control run")
p.add_argument("--outdir", default="results_online")
args = p.parse_args()
os.makedirs(args.outdir, exist_ok=True)
CFG = {}
PREFIX = {}   # L -> token ids of (template head + corpus prefix), built in main

# open-loop arrivals can pile up connections if lam exceeds capacity -> raise fd limit
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
try: resource.setrlimit(resource.RLIMIT_NOFILE, (hard if hard != resource.RLIM_INFINITY else 65536, hard))
except Exception: pass
if resource.getrlimit(resource.RLIMIT_NOFILE)[0] < 10000:
    print(f"WARNING: fd limit is {resource.getrlimit(resource.RLIMIT_NOFILE)[0]}; run `ulimit -n 65536`", flush=True)

# ---------- power / energy (GPU module via NVML) ----------
pynvml.nvmlInit()
H = pynvml.nvmlDeviceGetHandleByIndex(0)
energy_mj = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(H)
stop_power = threading.Event()
def sample_power():
    with open(f"{args.outdir}/power.csv", "w") as f:
        f.write("t,power_w,energy_mj\n")
        while not stop_power.is_set():
            f.write(f"{time.time()},{pynvml.nvmlDeviceGetPowerUsage(H)/1000},{energy_mj()}\n")
            f.flush(); stop_power.wait(0.1)
threading.Thread(target=sample_power, daemon=True).start()

# ---------- vLLM /metrics ----------
MET = []   # per scrape: t, gen, prompt (cumulative tokens), running, waiting, kv, preempt
WANT = {"vllm:generation_tokens_total": "gen", "vllm:prompt_tokens_total": "prompt",
        "vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
        "vllm:kv_cache_usage_perc": "kv", "vllm:gpu_cache_usage_perc": "kv",
        "vllm:num_preemptions_total": "preempt"}
COLS = ["gen", "prompt", "running", "waiting", "kv", "preempt"]
LINE = re.compile(r'^(vllm:\w+)(?:\{[^}]*\})?\s+(\S+)\s*$')

async def poll_metrics(sess):
    with open(f"{args.outdir}/metrics.csv", "w") as f:
        f.write("t," + ",".join(COLS) + "\n")
        while True:
            try:
                async with sess.get(f"{args.url}/metrics") as r:
                    txt = await r.text()
                row = dict.fromkeys(COLS, 0.0); row["t"] = time.time()
                for line in txt.splitlines():
                    m = LINE.match(line)
                    if m and m.group(1) in WANT: row[WANT[m.group(1)]] += float(m.group(2))
                MET.append(row)
                f.write(f"{row['t']}," + ",".join(str(row[c]) for c in COLS) + "\n"); f.flush()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(0.25)

def interp(field, t):                      # linear interpolation of a sampled counter
    ts = [m["t"] for m in MET]
    i = bisect.bisect_left(ts, t)
    if i == 0: return MET[0][field]
    if i == len(MET): return MET[-1][field]
    a, b = MET[i - 1], MET[i]
    return a[field] + (t - a["t"]) / max(b["t"] - a["t"], 1e-9) * (b[field] - a[field])
rate = lambda f, a, b: (interp(f, b) - interp(f, a)) / max(b - a, 1e-9)
def gauge(field, a, b, fn):
    v = [m[field] for m in MET if a <= m["t"] <= b]
    return fn(v) if v else interp(field, (a + b) / 2)
mean = lambda v: sum(v) / len(v)
def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None

# ---------- requests ----------
async def stream(sess, ids, max_tokens, rec, ts=None, text=None, ignore_eos=False, on_first=None):
    """Streaming completion. Mutates `rec` live (t_submit, t_first, n_out, finish, usage, completed, t_end)
    so partial info survives cancellation. One SSE chunk ~ one token."""
    body = dict(model=args.model, prompt=ids, max_tokens=max_tokens, temperature=0,
                ignore_eos=ignore_eos, stream=True, stream_options={"include_usage": True})
    rec.setdefault("t_submit", time.time()); rec["n_out"] = 0; rec["completed"] = False
    try:
        async with sess.post(f"{args.url}/v1/completions", json=body) as r:
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {await r.text()}")
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                d = json.loads(line[5:])
                if d.get("usage"): rec["usage"] = d["usage"]
                ch = d.get("choices") or []
                if not ch: continue
                if ch[0].get("finish_reason"): rec["finish"] = ch[0]["finish_reason"]
                if ch[0]["text"] or not ch[0].get("finish_reason"):
                    now = time.time()
                    if rec["n_out"] == 0:
                        rec["t_first"] = now
                        if on_first: on_first()
                    rec["n_out"] += 1
                    if ts is not None: ts.append(now)
                    if text is not None: text.append(ch[0]["text"])
        rec["completed"] = True
    finally:
        rec["t_end"] = time.time()

async def filler(sess, ids, rec):
    cap = CFG["max_len"] - len(ids) - 1
    if args.filler_max_out: cap = min(cap, args.filler_max_out)
    try:
        await stream(sess, ids, cap, rec)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        rec["error"] = repr(e)

async def arrivals(sess, pool, recs, stop, t0):
    """Open-loop Poisson arrivals at rate --lam, scheduled on absolute times from t0 (seeded)."""
    rng = random.Random(args.seed)
    t, i, tasks = t0, 0, []
    while not stop.is_set():
        t += rng.expovariate(args.lam)
        delay = t - time.time()
        if delay > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay); break
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.sleep(0)
            if stop.is_set(): break
        ids = pool[i % len(pool)]
        rec = dict(idx=i, t_sched=t, t_submit=time.time(), n_prompt=len(ids))
        recs.append(rec)
        tasks.append(asyncio.create_task(filler(sess, ids, rec)))
        i += 1
    return tasks

async def finish_arrivals(atask, stop):
    stop.set()
    tasks = await atask
    for t in tasks: t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

async def wait_idle(timeout=900):
    await asyncio.sleep(2)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if MET and MET[-1]["running"] + MET[-1]["waiting"] == 0: return
        await asyncio.sleep(1)
    print("warning: server still busy after timeout", flush=True)

async def reset_prefix_cache(sess):
    try:
        async with sess.post(f"{args.url}/reset_prefix_cache") as r: return r.status == 200
    except Exception:
        return False

async def run_fillers_only(sess, pool, duration):
    """Matched control: same arrival schedule/prompts as the R trial, no R."""
    await wait_idle(); await reset_prefix_cache(sess)
    stop, recs = asyncio.Event(), []
    t0 = time.time()
    atask = asyncio.create_task(arrivals(sess, pool, recs, stop, t0))
    await asyncio.sleep(duration)
    t1 = time.time()
    await finish_arrivals(atask, stop)
    return t0, t1, recs

# ---------- window statistics ----------
def wstats(a, b, recs, r_tokens=None):
    """Throughput/batch/latency over [a,b]. r_tokens = R's tokens generated inside the window (None: no R)."""
    d = max(b - a, 1e-9)
    agg = rate("gen", a, b)
    done = [r for r in recs if r.get("completed") and a <= r["t_end"] <= b]
    ttft = [r["t_first"] - r["t_submit"] for r in done if "t_first" in r]
    e2e = [r["t_end"] - r["t_submit"] for r in done]
    tpot = [1000 * (r["t_end"] - r["t_first"]) / (r["n_out"] - 1) for r in done if r.get("n_out", 0) > 1]
    return dict(
        window_s=d, agg_tok_s=agg, filler_tok_s=agg - (r_tokens or 0) / d, prefill_tok_s=rate("prompt", a, b),
        arrived=sum(1 for r in recs if a <= r["t_submit"] <= b), completed=len(done),
        inflight_end=sum(1 for r in recs if r["t_submit"] <= b and not (r.get("completed") and r["t_end"] <= b)),
        ttft_p50=pct(ttft, .5), ttft_p95=pct(ttft, .95), ttft_p99=pct(ttft, .99),
        tpot_ms_p50=pct(tpot, .5), tpot_ms_p95=pct(tpot, .95),
        e2e_p50=pct(e2e, .5), e2e_p95=pct(e2e, .95),
        running_mean=gauge("running", a, b, mean), running_min=gauge("running", a, b, min),
        waiting_mean=gauge("waiting", a, b, mean), waiting_max=gauge("waiting", a, b, max),
        kv_mean=gauge("kv", a, b, mean), kv_max=gauge("kv", a, b, max),
        preemptions=interp("preempt", b) - interp("preempt", a))
pref = lambda pre, d: {pre + k: v for k, v in d.items()}

# ---------- one trial ----------
async def run_trial(sess, L, mode, trial, head_ids, corpus_ids, tail_ids, pool):
    kind, _, n = mode.partition(":")
    fixed = kind == "fixed"
    max_tok = int(n) if fixed else args.natural_max
    await wait_idle()
    reset_ok = await reset_prefix_cache(sess)
    prefix = PREFIX[L]

    # 1. R': alone on an idle server (nothing else can evict its blocks)
    rp = {}
    await stream(sess, prefix, 1, rp)

    # 2. measurement window: R and Poisson arrivals start together once R' has returned
    stop, frecs, rrec, rts, rtext, ef = asyncio.Event(), [], {}, [], [], {}
    t_start, e0 = time.time(), energy_mj()
    rtask = asyncio.create_task(stream(sess, prefix + tail_ids, max_tok, rrec, ts=rts, text=rtext,
                                       ignore_eos=fixed, on_first=lambda: ef.setdefault("e", energy_mj())))
    atask = asyncio.create_task(arrivals(sess, pool, frecs, stop, t_start))
    try:
        await rtask                                   # 3. window ends when R's output is fully received
        t1, e1 = time.time(), energy_mj()
    finally:
        await finish_arrivals(atask, stop)

    # 4. matched control with the identical arrival schedule
    ctl = None
    if not args.no_control:
        ctl = await run_fillers_only(sess, pool, t1 - t_start)

    t_first, t_last = rts[0], rts[-1]
    usage = rrec.get("usage", {})
    pt = usage.get("prompt_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    uncached = (pt - (cached or 0)) if pt else None
    comp = usage.get("completion_tokens", len(rts))
    rdur = max(t_last - t_first, 1e-9)
    full, dec = (t_start, t1), (t_first, t_last)
    rec = dict(trial=trial, mode=mode, prefix_len=L, lam=args.lam, seed=args.seed, prefix_cache_reset=reset_ok,
               prefix_tokens=len(prefix) - len(head_ids), pool_hash=CFG.get("pool_hash"),
               warmup_s=rp["t_end"] - rp["t_submit"], prompt_tokens=pt, cached_tokens=cached,
               uncached_tokens=uncached, prefix_hit_ok=(uncached is not None and uncached <= len(tail_ids) + 64),
               out_tokens=comp, R_finish=rrec.get("finish"), R_ttft_s=t_first - t_start,
               R_latency_s=t1 - t_start, R_decode_tok_s=(len(rts) - 1) / rdur,
               R_tpot_ms=1000 * rdur / max(len(rts) - 1, 1), pool_wrapped=len(frecs) > len(pool),
               energy_J_window=(e1 - e0) / 1000, avg_power_W_window=(e1 - e0) / 1000 / (t1 - t_start),
               energy_J_decode=(e1 - ef["e"]) / 1000, avg_power_W_decode=(e1 - ef["e"]) / 1000 / rdur,
               t_start=t_start, t_end=t1, output="".join(rtext))
    rec.update(pref("full_", wstats(*full, frecs, r_tokens=comp)))      # whole R lifetime (incl. its TTFT)
    rec.update(pref("dec_", wstats(*dec, frecs, r_tokens=len(rts) - 1)))  # R's decode phase only
    if ctl:
        c0, c1, crecs = ctl
        cfull = (c0, c1)
        cdec = (c0 + (t_first - t_start), c0 + (t_last - t_start))     # same relative offsets as R's decode
        rec.update(pref("ctl_full_", wstats(*cfull, crecs)), **pref("ctl_dec_", wstats(*cdec, crecs)))
        for w in ("full", "dec"):
            base = rec[f"ctl_{w}_filler_tok_s"]
            rec[f"{w}_filler_slowdown"] = rec[f"{w}_filler_tok_s"] / base if base else None
    with open(f"{args.outdir}/trial_{mode.replace(':', '')}_{L}_t{trial}.json", "w") as f:
        json.dump(dict(R=rts, filler_reqs=frecs, control_reqs=(ctl[2] if ctl else None),
                       windows=dict(full=full, dec=dec, ctl=(ctl[:2] if ctl else None)),
                       metrics=[m for m in MET if rp["t_submit"] <= m["t"] <= (ctl[1] if ctl else t1)]), f)
    return rec

# ---------- prompts ----------
def resolve_filler_data():
    if args.filler_data: return args.filler_data
    from huggingface_hub import hf_hub_download          # cached under HF_HOME
    return hf_hub_download(repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
                           filename="ShareGPT_V3_unfiltered_cleaned_split.json", repo_type="dataset")

def resolve_corpus():
    if args.corpus: return args.corpus
    path = f"{args.outdir}/shakespeare.txt"
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve("https://www.gutenberg.org/cache/epub/100/pg100.txt", path)
    return path

def load_fillers(tok):
    """Seeded, ordered pool of ShareGPT first user turns (chat-templated). Identical in every run."""
    data = json.load(open(resolve_filler_data()))
    random.Random(args.filler_seed).shuffle(data)
    out, texts = [], []
    for conv in data:
        turns = conv.get("conversations") or []
        if not turns or turns[0].get("from") != "human": continue
        msg = turns[0]["value"].strip()
        if not msg or len(msg) > 8000: continue
        text = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False)
        if args.filler_min_len <= len(ids) <= args.filler_max_len:
            out.append(ids); texts.append(msg)
        if len(out) == args.pool_size: break
    assert out, "no usable fillers"
    with open(f"{args.outdir}/fillers.json", "w") as f:
        json.dump([dict(n_tokens=len(i), prompt=t) for i, t in zip(out, texts)], f)
    return out

def chat_parts(tok):
    """head (before the document) / tail (instruction + assistant turn): R' = head+doc, R = head+doc+tail."""
    s = tok.apply_chat_template(
        [{"role": "user", "content": "@@DOC@@\n\nSummarize the text above in a few paragraphs."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    head, tail = s.split("@@DOC@@")
    return tok.encode(head, add_special_tokens=False), tok.encode(tail, add_special_tokens=False)

def snap_to_sentence(tok, ids, L, window=256):
    """Largest j <= L (within `window` tokens) such that ids[:j] ends at a sentence end (. ! ?).
    Falls back to exactly L if none found. The corpus is hard-wrapped, so line ends are NOT sentence ends."""
    for j in range(min(L, len(ids)), max(L - window, 1), -1):
        if tok.decode(ids[j - 1:j]).rstrip(" \r\n\t").endswith((".", "!", "?")):
            return j
    return min(L, len(ids))

async def healthy(sess):
    try:
        async with sess.get(f"{args.url}/health") as r: return r.status == 200
    except Exception:
        return False

SHOW = ["trial", "mode", "prefix_len", "prompt_tokens", "uncached_tokens", "prefix_hit_ok", "out_tokens",
        "R_finish", "R_ttft_s", "R_latency_s", "R_tpot_ms", "full_arrived", "full_inflight_end",
        "full_filler_tok_s", "ctl_full_filler_tok_s", "full_filler_slowdown", "dec_filler_slowdown",
        "dec_running_mean", "ctl_dec_running_mean", "dec_waiting_mean", "dec_preemptions", "pool_wrapped"]

async def main():
    T0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    corpus_ids = tok.encode(open(resolve_corpus(), encoding="utf-8", errors="ignore").read(), add_special_tokens=False)
    head_ids, tail_ids = chat_parts(tok)
    pool = load_fillers(tok)
    print(f"corpus tokens: {len(corpus_ids)}; pool: {len(pool)} prompts; lam={args.lam}/s; modes={args.modes}",
          flush=True)
    max_out = max(int(m.partition(":")[2]) if m.startswith("fixed") else args.natural_max for m in args.modes)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),
                                     connector=aiohttp.TCPConnector(limit=0)) as sess:
        async with sess.get(f"{args.url}/v1/models") as r:
            CFG["max_len"] = (await r.json())["data"][0]["max_model_len"]
        cap = min(len(corpus_ids), CFG["max_len"] - max_out - len(head_ids) - len(tail_ids) - 8)
        if args.max_prompt: cap = min(cap, args.max_prompt)
        lens, L = [], args.min_len
        while L <= cap: lens.append(L); L *= 2
        if lens[-1] != cap: lens.append(cap)
        print("prefix lengths:", lens, flush=True)
        for L in lens:
            j = L if args.no_snap else snap_to_sentence(tok, corpus_ids, L)
            PREFIX[L] = head_ids + corpus_ids[:j]
        CFG["pool_hash"] = hashlib.sha1(json.dumps(pool).encode()).hexdigest()[:12]
        print("corpus tokens per size after snapping:", {L: len(v) - len(head_ids) for L, v in PREFIX.items()},
              "| filler pool hash:", CFG["pool_hash"], flush=True)
        poller = asyncio.create_task(poll_metrics(sess))
        while not MET: await asyncio.sleep(0.25)
        failed_at, dead = None, False
        with open(f"{args.outdir}/runs.jsonl", "a") as out:
            for trial in range(args.trials):
                for mode in args.modes:
                    prev_wall = None
                    for L in lens:
                        if failed_at is not None and L >= failed_at: continue
                        el = time.time() - T0
                        if args.budget_min and (el > 60 * args.budget_min or
                                                el + 2.2 * (prev_wall or 0) > 60 * args.budget_min):
                            print(f"budget: skipping L>={L} for trial {trial} mode={mode} "
                                  f"(elapsed {el / 60:.1f} min)", flush=True)
                            break
                        t_run = time.time()
                        print(f"--- trial {trial} mode={mode} L={L}", flush=True)
                        try:
                            rec = await run_trial(sess, L, mode, trial, head_ids, corpus_ids, tail_ids, pool)
                        except Exception as e:      # memory bound / server failure
                            print(f"failed at L={L}: {e}", flush=True)
                            out.write(json.dumps(dict(trial=trial, mode=mode, prefix_len=L, error=str(e))) + "\n")
                            out.flush()
                            if not await healthy(sess): dead = True; break
                            failed_at = L
                            continue
                        out.write(json.dumps(rec) + "\n"); out.flush()
                        prev_wall = time.time() - t_run
                        print({k: (round(rec[k], 3) if isinstance(rec[k], float) else rec[k])
                               for k in SHOW if k in rec}, flush=True)
                        if not rec["prefix_hit_ok"]:
                            print("WARNING: R missed the prefix cache; discard this trial", flush=True)
                        if rec["pool_wrapped"]:
                            print("WARNING: arrivals wrapped the filler pool; raise --pool-size", flush=True)
                    if dead: break
                if dead:
                    print("server unhealthy; stopping", flush=True); break
        poller.cancel()
    stop_power.set()

asyncio.run(main())
