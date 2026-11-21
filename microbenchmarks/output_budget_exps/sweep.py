import json, base64, zlib, pickle, random, re, subprocess, sys, time, os, signal
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import requests, pynvml
from huggingface_hub import hf_hub_download

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
SIF = os.path.expanduser("~/vllm.sif")
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/hf_cache"))
PORT = 8000
URL = f"http://localhost:{PORT}"

BUDGETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
CONDITIONS = BUDGETS + [None]                # None = control (no budget in the prompt)
MAX_LEN = 32768                              # server context window
HARD_CAP = 32768                             # safety limit on generation, same for every condition
PER_DIFF = 3
TRIALS = 3
TEMP = 0.0
SEED = 14
TIMEOUT = 6

label = lambda b: "none" if b is None else str(b)

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
           "--gpu-memory-utilization", "0.9"]
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

def generate(prompt, max_tokens, seed):
    r = requests.post(f"{URL}/v1/chat/completions", json=dict(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=TEMP, max_tokens=max_tokens, seed=seed), timeout=3600)
    if r.status_code != 200:
        raise RuntimeError(f"server error {r.status_code}: {r.text[:500]}")
    j = r.json()
    ch = j["choices"][0]
    return (ch["message"]["content"], ch["finish_reason"],
            j["usage"]["prompt_tokens"], j["usage"]["completion_tokens"])

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
        prefix = "" if budget is None else f"Use at most {budget} tokens.\n\n"
        return (f"{prefix}{p['question_content']}\n\n{fmt}\n"
                f"Give the final solution in one ```python code block.")

    # ---------- generate + measure ----------
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    generate(make_prompt(probs[0], BUDGETS[0]), 32, SEED)  # warmup

    total = len(probs) * len(CONDITIONS) * TRIALS
    results = []
    for p in probs:
        for b in CONDITIONS:
            prompt = make_prompt(p, b)
            for t in range(TRIALS):
                e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu); t0 = time.perf_counter()
                text, finish, n_prompt, n_out = generate(prompt, HARD_CAP, SEED + t)
                e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu); t1 = time.perf_counter()
                joules, secs = (e1 - e0) / 1000, t1 - t0
                results.append(dict(
                    run_id=f"{p['question_id']}_b{label(b)}_t{t}", trial=t,
                    id=p["question_id"], title=p.get("question_title"), platform=p.get("platform"),
                    diff=p["difficulty"], budget=b, budget_label=label(b), control=b is None,
                    max_tokens=HARD_CAP, prompt=prompt, prompt_tokens=n_prompt,
                    text=text, n_tok=n_out,
                    over_budget=(n_out > b) if b is not None else None,
                    truncated=finish == "length",           # hit the hard cap
                    energy_J=joules, time_s=secs, avg_W=joules / secs))
                r = results[-1]
                print(f"[{len(results)}/{total}] {r['run_id']} tok={r['n_tok']} "
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
    pid, text, truncated = key
    code = extract(text)
    if truncated or not code: return False
    fn = meta[pid][1].get("func_name")
    return all(run_test(code, t, fn) for t in tests[pid])

keys = sorted({(r["id"], r["text"], r["truncated"]) for r in results})
with ThreadPoolExecutor(16) as ex:
    verdict = dict(zip(keys, ex.map(score, keys)))
for r in results:
    r["correct"] = verdict[(r["id"], r["text"], r["truncated"])]

for r in results:
    print(f"{r['run_id']:30} {r['diff']:6} budget={r['budget_label']:>5} prompt_tok={r['prompt_tokens']:5} "
          f"tok={r['n_tok']:5} over={r['over_budget']!s:5} cap_hit={r['truncated']!s:5} "
          f"correct={r['correct']!s:5} E={r['energy_J']:.1f}J t={r['time_s']:.1f}s P={r['avg_W']:.0f}W")
json.dump(results, open("results.json", "w"), indent=1)

groups = defaultdict(list)
for r in results:
    groups[(r["diff"], r["budget_label"])].append(r)
order = lambda kv: (["easy", "medium", "hard"].index(kv[0][0]),
                    CONDITIONS.index(None if kv[0][1] == "none" else int(kv[0][1])))
print("\ndiff    budget    n   acc    tokens   energy_J   time_s")
for (d, bl), rs in sorted(groups.items(), key=order):
    m = lambda k: sum(float(x[k]) for x in rs) / len(rs)
    print(f"{d:7} {bl:>6} {len(rs):4} {m('correct'):5.2f} {m('n_tok'):8.0f} {m('energy_J'):10.1f} {m('time_s'):8.1f}")
