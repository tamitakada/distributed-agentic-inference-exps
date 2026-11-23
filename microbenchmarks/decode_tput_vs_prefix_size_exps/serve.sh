#!/bin/bash
# Usage: MODEL=<hf id or local path> SIF=vllm.sif MAX_LEN=1048576 ./serve.sh
# Run `ulimit -n 65536` (or the hard limit) in this shell first: the client opens one connection per filler.
set -e
SIF=${SIF:-vllm.sif}
MODEL=${MODEL:?set MODEL to the HF id or local path}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

apptainer exec --nv --bind "$HF_HOME:$HF_HOME" --env HF_HOME="$HF_HOME" \
  --env VLLM_SERVER_DEV_MODE=1 "$SIF" \
  vllm serve "$MODEL" \
    --port 8000 \
    --gpu-memory-utilization 0.92 --max-num-seqs 620 \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    ${EXTRA_ARGS}
