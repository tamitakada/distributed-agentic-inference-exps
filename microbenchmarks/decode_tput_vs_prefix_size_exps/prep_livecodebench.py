#!/usr/bin/env python3
"""
Extract LiveCodeBench problem statements into a small JSON file for experiment_online.py (--filler-data).

The dataset is ~2 GB because of the hidden test cases, which we do not need, so run this once on a machine
with internet and disk, then copy the small JSON to the cluster:

    python prep_livecodebench.py --out livecodebench_prompts.json

Untested here: I could not reach Hugging Face from my sandbox. The 'lighteval/code_generation_lite' repo
lists parquet configs (release_v1, release_v2, release_latest, ...). If loading fails with a dataset-script
error, check your `datasets` version, or try --repo livecodebench/code_generation_lite with an older
`datasets` (<4) release.
"""
import argparse, json
from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--repo", default="lighteval/code_generation_lite")
p.add_argument("--config", default="release_v2", help="e.g. release_v2 (511 problems), release_latest")
p.add_argument("--out", default="livecodebench_prompts.json")
a = p.parse_args()

ds = load_dataset(a.repo, a.config, split="test")
keep = [c for c in ("question_title", "question_content", "starter_code", "platform", "difficulty")
        if c in ds.column_names]
ds = ds.select_columns(keep)          # drop the huge private_test_cases column before iterating
rows = [{k: r[k] for k in keep} for r in ds]
json.dump(rows, open(a.out, "w"))
print(f"{len(rows)} problems -> {a.out}")
