#!/usr/bin/env python3
"""
Online (Poisson-arrival) long-prompt interference experiment.

Per (trial, mode, prefix length L):
  0. Wait for server idle, reset prefix cache.
  1. Send R' = chat-templated head + Django source files (appended whole, in a fixed order, target file first),
     max_tokens=1, ALONE. Wait for it to finish. Larger sizes = the same files plus more files appended.
  2. Measurement window starts the instant R' returns: send R (same prefix + test-suite-generation instruction) and,
     at the same moment, start Poisson filler arrivals (rate --lam, seeded: identical schedule and
     identical LiveCodeBench prompt order for every L / mode / trial).
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
p.add_argument("--filler-data", required=True,
               help="LiveCodeBench problems json produced by prep_livecodebench.py")
p.add_argument("--repo-dir", default=None,
               help="directory containing (or above) the django/ package; default: download a pinned Django sdist from PyPI")
p.add_argument("--django-version", default="6.1.3", help="Django version to download when --repo-dir is not given")
p.add_argument("--first-files", nargs="+", default=["django/utils/crypto.py", "django/utils/dateparse.py"],
               help="files appended first, in this order; the first one is the test-generation target")
p.add_argument("--lam", type=float, required=True, help="filler arrival rate, requests/s (Poisson)")
p.add_argument("--seed", type=int, default=0, help="arrival-process seed (same for every L/mode/trial)")
p.add_argument("--modes", nargs="+", default=["fixed:512", "natural"],
               help="R output modes: fixed:<n> and/or natural")
p.add_argument("--natural-max", type=int, default=4096, help="max_tokens guard for natural mode")
p.add_argument("--url", default="http://localhost:8000")
p.add_argument("--pool-size", type=int, default=2000, help="max filler prompts; arrival i uses pool[i %% size]")
p.add_argument("--filler-seed", type=int, default=0, help="seed for LiveCodeBench pool order")
p.add_argument("--filler-min-len", type=int, default=32)
p.add_argument("--filler-max-len", type=int, default=2048)
p.add_argument("--filler-max-out", type=int, default=8192, help="runaway guard on filler output (0 = context limit)")
p.add_argument("--min-len", type=int, default=1024, help="first token budget; budgets double from here (snapped down to whole files)")
p.add_argument("--max-prompt", type=int, default=0, help="largest R prefix (tokens) to sweep; 0 = server limit")
p.add_argument("--budget-min", type=float, default=0,
               help="wall-clock budget (min) from sweep start; skips sizes predicted to overrun (0 = off)")
p.add_argument("--trials", type=int, default=3)
p.add_argument("--no-control", action="store_true", help="skip the matched fillers-only control run")
p.add_argument("--outdir", default="results_online")
args = p.parse_args()
os.makedirs(args.outdir, exist_ok=True)
CFG = {}
PREFIX = {}   # L (repo tokens) -> token ids of (template head + files), built in main
META = {}     # L -> dict(n_files, last_file)

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
async def run_trial(sess, L, mode, trial, head_ids, tail_ids, pool):
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
               prefix_tokens=len(prefix) - len(head_ids), n_files=META[L]["n_files"], last_file=META[L]["last_file"],
               pool_hash=CFG.get("pool_hash"), repo_hash=CFG.get("repo_hash"),
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
EXCLUDE_DIRS = {"tests", "test", "testing", "migrations", "locale", "docs", "examples",
                "benchmarks", "__pycache__", "_vendor", "vendored"}

def find_pkg_root(base):
    for root, _, _ in os.walk(base):
        if os.path.isfile(os.path.join(root, "django", "utils", "crypto.py")): return root
    raise SystemExit(f"no django/ package found under {base}")

def resolve_repo():
    """Directory containing the django/ package: --repo-dir, or a pinned Django sdist downloaded from PyPI."""
    if args.repo_dir: return find_pkg_root(args.repo_dir)
    import io, tarfile, urllib.request
    dest = f"{args.outdir}/django-{args.django_version}"
    if not os.path.isdir(dest):
        meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/django/{args.django_version}/json", timeout=60))
        url = next(u["url"] for u in meta["urls"] if u["packagetype"] == "sdist")
        with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url, timeout=300).read())) as t:
            t.extractall(dest)
    return find_pkg_root(dest)

def ordered_files(root):
    """Fixed append order: --first-files, then the rest of the first file's directory (alphabetical),
    then every other non-test source file under django/ (alphabetical by path)."""
    allf = []
    for d, dirs, fs in os.walk(os.path.join(root, "django")):
        dirs[:] = [x for x in dirs if x not in EXCLUDE_DIRS]
        for f in fs:
            if f.endswith(".py") and not f.startswith("test_") and f != "conftest.py":
                allf.append(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/"))
    allf.sort()
    first = list(args.first_files)
    for f in first: assert f in allf, f"{f} not found among the repo's source files"
    same_dir = [f for f in allf if os.path.dirname(f) == os.path.dirname(first[0]) and f not in first]
    rest = [f for f in allf if f not in first and f not in same_dir]
    return first + same_dir + rest

def build_prefix(tok, root, cap):
    """Tokenize whole files in the fixed order until the next file would push the total past `cap` tokens.
    Each file is wrapped as <file path="..."> ... </file>. Returns (files, per-file token ids, cumulative counts)."""
    files, chunks, cum, total = [], [], [], 0
    for rel in ordered_files(root):
        text = open(os.path.join(root, rel), encoding="utf-8", errors="ignore").read()
        ids = tok.encode(f'<file path="{rel}">\n{text}\n</file>\n\n', add_special_tokens=False)
        if files and total + len(ids) > cap: break
        files.append(rel); chunks.append(ids); total += len(ids); cum.append(total)
    return files, chunks, cum

def choose_sizes(cum, cap):
    """Doubling token budgets from --min-len up to cap, each snapped DOWN to a whole-file boundary
    (always at least the first file); duplicates dropped. Every size is a prefix of the next one."""
    budgets, B = [], args.min_len
    while B <= cap: budgets.append(B); B *= 2
    budgets.append(cap)
    sizes = []                                   # (n_files, repo tokens)
    for B in budgets:
        k = max([i + 1 for i, c in enumerate(cum) if c <= B], default=1)
        if not sizes or k > sizes[-1][0]: sizes.append((k, cum[k - 1]))
    return sizes

def lcb_prompt(p):
    """LiveCodeBench-style code-generation prompt for one problem."""
    sc = (p.get("starter_code") or "").strip()
    if sc:
        fmt = ("You will use the following starter code to write the solution to the problem and enclose your "
               "code within delimiters.\n```python\n" + sc + "\n```")
    else:
        fmt = ("Read the inputs from stdin, solve the problem and write the answer to stdout (do not directly "
               "test on the sample inputs). Enclose your code within delimiters as follows.\n"
               "```python\n# YOUR CODE HERE\n```")
    return ("You will be given a question (problem specification) and will generate a correct Python program "
            "that matches the specification and passes all tests.\n\nQuestion: "
            + p["question_content"].strip() + "\n\n" + fmt)

def load_fillers(tok):
    """Seeded, ordered pool of LiveCodeBench prompts (chat-templated). Identical in every run."""
    data = json.load(open(args.filler_data))
    random.Random(args.filler_seed).shuffle(data)
    out, texts, dropped = [], [], 0
    for p in data:
        if not (p.get("question_content") or "").strip(): continue
        msg = lcb_prompt(p)
        text = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False)
        if args.filler_min_len <= len(ids) <= args.filler_max_len:
            out.append(ids); texts.append(msg)
            if len(out) == args.pool_size: break
        else:
            dropped += 1
    assert out, "no usable fillers"
    print(f"fillers: kept {len(out)}, dropped {dropped} by the length filter "
          f"[{args.filler_min_len}, {args.filler_max_len}]; mean {sum(map(len, out)) / len(out):.0f} tokens", flush=True)
    with open(f"{args.outdir}/fillers.json", "w") as f:
        json.dump([dict(n_tokens=len(i), prompt=t) for i, t in zip(out, texts)], f)
    return out

def chat_parts(tok):
    """head (before the repo files) / tail (instruction + assistant turn): R' = head+files, R = head+files+tail."""
    instr = (f"Write a comprehensive unittest test suite for the module `{args.first_files[0]}`, "
             "which is the first file shown above. Output the complete test file.")
    CFG["instruction"] = instr
    s = tok.apply_chat_template([{"role": "user", "content": "@@DOC@@\n\n" + instr}],
                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
    head, tail = s.split("@@DOC@@")
    return tok.encode(head, add_special_tokens=False), tok.encode(tail, add_special_tokens=False)

async def healthy(sess):
    try:
        async with sess.get(f"{args.url}/health") as r: return r.status == 200
    except Exception:
        return False

SHOW = ["trial", "mode", "prefix_len", "n_files", "prompt_tokens", "uncached_tokens", "prefix_hit_ok", "out_tokens",
        "R_finish", "R_ttft_s", "R_latency_s", "R_tpot_ms", "full_arrived", "full_inflight_end",
        "full_filler_tok_s", "ctl_full_filler_tok_s", "full_filler_slowdown", "dec_filler_slowdown",
        "dec_running_mean", "ctl_dec_running_mean", "dec_waiting_mean", "dec_preemptions", "pool_wrapped"]

async def main():
    T0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    head_ids, tail_ids = chat_parts(tok)
    pool = load_fillers(tok)
    root = resolve_repo()
    print(f"repo root: {root}; task: {CFG['instruction']}\nlam={args.lam}/s; modes={args.modes}", flush=True)
    max_out = max(int(m.partition(":")[2]) if m.startswith("fixed") else args.natural_max for m in args.modes)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),
                                     connector=aiohttp.TCPConnector(limit=0)) as sess:
        async with sess.get(f"{args.url}/v1/models") as r:
            CFG["max_len"] = (await r.json())["data"][0]["max_model_len"]
        cap = CFG["max_len"] - max_out - len(head_ids) - len(tail_ids) - 8
        if args.max_prompt: cap = min(cap, args.max_prompt)
        files, chunks, cum = build_prefix(tok, root, cap)
        sizes = choose_sizes(cum, cap)
        lens = [t for _, t in sizes]
        for k, t in sizes:
            PREFIX[t] = head_ids + [x for ch in chunks[:k] for x in ch]
            META[t] = dict(n_files=k, last_file=files[k - 1])
        CFG["pool_hash"] = hashlib.sha1(json.dumps(pool).encode()).hexdigest()[:12]
        CFG["repo_hash"] = hashlib.sha1(json.dumps(PREFIX[lens[-1]]).encode()).hexdigest()[:12]
        with open(f"{args.outdir}/prefix_manifest.json", "w") as f:
            json.dump(dict(instruction=CFG["instruction"], django_version=args.django_version, cap=cap,
                           sizes=[dict(repo_tokens=t, n_files=k, last_file=files[k - 1]) for k, t in sizes],
                           files=[dict(path=p_, cum_tokens=c) for p_, c in zip(files, cum)]), f, indent=1)
        print("sizes (repo tokens, n_files, last file appended):", flush=True)
        for k, t in sizes: print(f"   {t:>8}  {k:>4}  {files[k - 1]}", flush=True)
        print("filler pool hash:", CFG["pool_hash"], "| prefix hash:", CFG["repo_hash"], flush=True)
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
                            rec = await run_trial(sess, L, mode, trial, head_ids, tail_ids, pool)
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
