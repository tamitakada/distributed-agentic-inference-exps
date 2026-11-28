import json, base64, zlib, pickle, random, re, subprocess, sys, time, os, signal, itertools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import requests, pynvml
from huggingface_hub import hf_hub_download

MODEL = "Qwen/Qwen3.8-27B"
SIF = os.path.expanduser("~/vllm.sif")
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/hf_cache"))
PORT = 8000
URL = f"http://localhost:{PORT}"

# Two independent knobs:
#   BUDGETS        -> token budget STATED IN THE PROMPT (soft; the model may ignore it)
#   THINK_BUDGETS  -> thinking_token_budget sent to vLLM (enforced: thinking is force-closed at this many tokens)
# None in either list = "not specified" (no prompt line / no thinking_token_budget in the request).
BUDGETS = [10, 50, 100, 500, 1000]
THINK_BUDGETS = [256, 1024, 4096]
PROMPT_OPTS = BUDGETS + [None]
THINK_OPTS = THINK_BUDGETS + [None]
GRID = True    # True: every (prompt budget x thinking budget) pair; False: pair the two lists by position (zip)
if GRID:
    CONDITIONS = list(itertools.product(PROMPT_OPTS, THINK_OPTS))
else:
    assert len(PROMPT_OPTS) == len(THINK_OPTS), "GRID=False needs lists of equal length"
    CONDITIONS = list(zip(PROMPT_OPTS, THINK_OPTS))

MAX_LEN = 32768                 # server context window
HARD_CAP = 15000                # safety limit on TOTAL generation (thinking + answer), same for every run
PER_DIFF = 3
TRIALS = 3
TEMP = 0.7
SEED = 14
TIMEOUT = 5

label = lambda b: "none" if b is None else str(b)
cond_idx = {(label(pb), label(tb)): i for i, (pb, tb) in enumerate(CONDITIONS)}

# ---------- server (vLLM inside apptainer) ----------
def healthy():
    try: return requests.get(f"{URL}/health", timeout=2).status_code == 200
    except Exception: return False

server = None
if not healthy():
    cmd = ["apptainer", "exec", "--nv",
           "--bind", f"{HF_HOME}:{HF_HOME}", "--env", f"HF_HOME={HF_HOME}",
           SIF, "vllm", "serve", MODEL,
           "--port", str(PORT), "--max-model-len", str(MAX_LEN),
           "--max-num-seqs", "1",               # one request at a time
           "--no-enable-prefix-caching",        # repeated prompts must not reuse cached prefill
           "--gpu-memory-utilization", "0.95",
           "--reasoning-parser", "qwen3"]
    server = subprocess.Popen(cmd, stdout=open("server.log", "w"), stderr=subprocess.STDOUT,
                              preexec_fn=os.setsid)
    t_start = time.time()
    while not healthy():
        if server.poll() is not None:
            sys.exit("vLLM server exited early; see server.log")
        if time.time() - t_start > 1800:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            sys.exit("server did not become healthy in 30 min; see server.log")
        time.sleep(5)
print("server ready", flush=True)

def generate(prompt, seed, think_budget=None, max_tokens=HARD_CAP):
    """One chat request. Returns the answer and the reasoning separately (parser splits them)."""
    body = dict(model=MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=TEMP, max_tokens=max_tokens, seed=seed,
                repetition_penalty=1.05, top_k=20, top_p=0.8)
    if think_budget is not None:
        body["thinking_token_budget"] = think_budget     # omitted entirely when None
    r = requests.post(f"{URL}/v1/chat/completions", json=body, timeout=3600)
    if r.status_code != 200:
        raise RuntimeError(f"server error {r.status_code}: {r.text[:500]}")
    j = r.json()
    ch = j["choices"][0]
    msg = ch["message"]
    return dict(answer=msg.get("content") or "",         # None if thinking never finished
                reasoning=msg.get("reasoning") or msg.get("reasoning_content") or "",
                finish=ch["finish_reason"], usage=j["usage"])

def count_tokens(text):
    """Token count via the vLLM server's /tokenize endpoint (no tokenizer needed on the host)."""
    if not text: return 0
    r = requests.post(f"{URL}/tokenize",
                      json=dict(model=MODEL, prompt=text, add_special_tokens=False), timeout=60)
    r.raise_for_status()
    j = r.json()
    return j.get("count", len(j["tokens"]))

def split_counts(g):
    """(reasoning_tokens, answer_tokens, source). Prefers the server's own reasoning-token count if it
    reports one; otherwise tokenizes each text. In the tokenize case the two counts can sum to a few
    tokens less than completion_tokens (the </think> tag and boundary whitespace belong to neither)."""
    n_out = g["usage"]["completion_tokens"]
    rt = (g["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens")
    if rt:
        return rt, n_out - rt, "usage"
    return count_tokens(g["reasoning"]), count_tokens(g["answer"]), "tokenize"

try:
    # ---------- sample ----------
    files = [hf_hub_download("livecodebench/code_generation_lite", f, repo_type="dataset")
             for f in ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"]]
    ds = [json.loads(line) for f in files for line in open(f)]
    rng = random.Random(SEED)
    probs = []
    for d in ["easy", "medium", "hard"]:
        rows = [r for r in ds if r["difficulty"] == d]
        probs += rng.sample(rows, PER_DIFF)

    def decode(s):
        try: return json.loads(s)
        except Exception: return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode()))))

    def make_prompt(p, budget):
        if p["starter_code"]:
            fmt = f"Complete this starter code:\n```python\n{p['starter_code']}\n```"
        else:
            fmt = "Read input from stdin and write the answer to stdout."
        prefix = "" if budget is None else f"You MUST respond in {budget} tokens or fewer. Directly give the answer follow the format WITHOUT ANY explanation.\n\n"
        return (f"{prefix}{p['question_content']}\n\n{fmt}\n"
                f"Give the final solution in one ```python code block.")

    # ---------- generate + measure ----------
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    generate(make_prompt(probs[0], None), SEED, max_tokens=32)  # warmup

    # sanity check: is thinking_token_budget actually enforced, and is reasoning returned separately?
    chk = generate(make_prompt(probs[0], None), SEED, think_budget=20, max_tokens=600)
    chk_rt, chk_at, _ = split_counts(chk)
    print(f"[check] thinking_token_budget=20 -> reasoning_tok={chk_rt} answer_tok={chk_at}", flush=True)
    if not chk["reasoning"]:
        print("WARNING: no reasoning field returned; the reasoning parser may not match this model", flush=True)
    elif chk_rt > 100:
        print("WARNING: thinking_token_budget does not appear to be enforced", flush=True)

    total = len(probs) * len(CONDITIONS) * TRIALS
    print(f"{len(probs)} problems x {len(CONDITIONS)} conditions x {TRIALS} trials = {total} runs", flush=True)
    results = []
    for p in probs:
        for b, tb in CONDITIONS:
            prompt = make_prompt(p, b)
            for t in range(TRIALS):
                e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu); t0 = time.perf_counter()
                g = generate(prompt, SEED + t, think_budget=tb)
                e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu); t1 = time.perf_counter()
                joules, secs = (e1 - e0) / 1000, t1 - t0
                # token split happens AFTER the timed window so it never affects energy/time
                rt, at, src = split_counts(g)
                n_out = g["usage"]["completion_tokens"]
                results.append(dict(
                    run_id=f"{p['question_id']}_b{label(b)}_tb{label(tb)}_t{t}", trial=t,
                    id=p["question_id"], title=p.get("question_title"), platform=p.get("platform"),
                    diff=p["difficulty"],
                    budget=b, budget_label=label(b),                          # budget stated in the prompt
                    think_budget=tb, think_budget_label=label(tb),            # thinking_token_budget sent to vLLM
                    control=(b is None and tb is None),
                    max_tokens=HARD_CAP, prompt=prompt, prompt_tokens=g["usage"]["prompt_tokens"],
                    reasoning=g["reasoning"], answer=g["answer"],
                    n_tok=n_out, reasoning_tok=rt, answer_tok=at, tok_source=src,
                    over_budget=(n_out > b) if b is not None else None,                   # total vs prompt budget
                    over_budget_answer=(at > b) if b is not None else None,               # answer only vs prompt budget
                    over_think_budget=(rt > tb) if tb is not None else None,              # reasoning vs enforced budget
                    truncated=g["finish"] == "length",           # hit the hard cap
                    energy_J=joules, time_s=secs, avg_W=joules / secs))
                r = results[-1]
                print(f"[{len(results)}/{total}] {r['run_id']} tok={r['n_tok']} "
                      f"(think={r['reasoning_tok']} ans={r['answer_tok']}) "
                      f"cap_hit={r['truncated']} t={r['time_s']:.0f}s E={r['energy_J']:.0f}J", flush=True)
                json.dump(results, open("results_partial.json", "w"), indent=1)
finally:
    if server is not None:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)   # stop vLLM once generation is done

# ---------- score (after all generation) ----------
meta = {p["question_id"]: (p, json.loads(p["metadata"]) if p["metadata"] else {}) for p in probs}
tests = {p["question_id"]: decode(p["public_test_cases"]) + decode(p["private_test_cases"]) for p in probs}

def extract(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m[-1] if m else ""

def run_test(code, t, fn):
    if t["testtype"] == "functional":
        src = ("from typing import *\nimport sys, json\n" + code +
               f"\nargs=[json.loads(l) for l in sys.stdin.read().split('\\n') if l.strip()]\n"
               f"print(json.dumps(Solution().{fn}(*args)))")
    else:
        src = code
    try:
        r = subprocess.run([sys.executable, "-c", src], input=t["input"],
                           capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return False
    if r.returncode: return False
    if t["testtype"] == "functional":
        try: return json.loads(r.stdout) == json.loads(t["output"])
        except Exception: return False
    return r.stdout.split() == t["output"].split()

def score(key):
    pid, answer, truncated = key
    code = extract(answer)               # answer only; the reasoning trace is never searched for code
    if truncated or not code: return False
    fn = meta[pid][1].get("func_name")
    return all(run_test(code, t, fn) for t in tests[pid])

keys = sorted({(r["id"], r["answer"], r["truncated"]) for r in results})
with ThreadPoolExecutor(16) as ex:
    verdict = dict(zip(keys, ex.map(score, keys)))
for r in results:
    r["correct"] = verdict[(r["id"], r["answer"], r["truncated"])]

for r in results:
    print(f"{r['run_id']:38} {r['diff']:6} budget={r['budget_label']:>5} think_budget={r['think_budget_label']:>5} "
          f"prompt_tok={r['prompt_tokens']:5} tok={r['n_tok']:5} think={r['reasoning_tok']:5} ans={r['answer_tok']:5} "
          f"cap_hit={r['truncated']!s:5} correct={r['correct']!s:5} "
          f"E={r['energy_J']:.1f}J t={r['time_s']:.1f}s P={r['avg_W']:.0f}W")
json.dump(results, open("results.json", "w"), indent=1)

groups = defaultdict(list)
for r in results:
    groups[(r["diff"], r["budget_label"], r["think_budget_label"])].append(r)
order = lambda kv: (["easy", "medium", "hard"].index(kv[0][0]), cond_idx[(kv[0][1], kv[0][2])])
print("\ndiff    budget  think_b    n   acc    tokens    think   answer   energy_J   time_s")
for (d, bl, tl), rs in sorted(groups.items(), key=order):
    m = lambda k: sum(float(x[k]) for x in rs) / len(rs)
    print(f"{d:7} {bl:>6} {tl:>8} {len(rs):4} {m('correct'):5.2f} {m('n_tok'):9.0f} {m('reasoning_tok'):8.0f} "
          f"{m('answer_tok'):8.0f} {m('energy_J'):10.1f} {m('time_s'):8.1f}")
