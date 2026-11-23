To run:
```
# terminal 1
MODEL=Qwen/Qwen3.8-27B SIF=vllm.sif bash serve.sh

# terminal 2
python experiment.py --model Qwen/Qwen3.8-27B --filler-data livecodebench_prompts.json --repo-dir <code src e.g. django_src> --lam <Poisson lambda e.g. 10> --trials 3 --modes fixed:<fixed out tok count> natural
```

`experiment.py` adds complete full files from the source repo to the prefix, growing the number of context files each time. It issues the same instruction to design a comprehensive test suite for ONLY the very first context file for all runs.
`experiment_txt.py` grows the prefix by taking a larger excerpt from the provided corpus. The task is always to summarize the provided text.
