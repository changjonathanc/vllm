#!/bin/bash
#
# Minimal server launcher for reproducing the old routed-experts hybrid-KV
# buffer bug with GPT-OSS. Pair this with
# examples/online_serving/gpt_oss_routed_experts_stress.py.
#
# Example:
#   MODEL=openai/gpt-oss-20b TP=4 MAX_MODEL_LEN=32768 \
#   GPU_MEMORY_UTILIZATION=0.8 \
#   bash examples/online_serving/gpt_oss_routed_experts_repro_server.sh
#
# Then, in another shell:
#   python3 examples/online_serving/gpt_oss_routed_experts_stress.py \
#     --base-url http://localhost:8000/v1 \
#     --model openai/gpt-oss-20b

set -euo pipefail

MODEL="${MODEL:-openai/gpt-oss-20b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"

echo "Starting routed-experts repro server:"
echo "  model=${MODEL}"
echo "  host=${HOST}"
echo "  port=${PORT}"
echo "  tensor_parallel_size=${TP}"
echo "  max_model_len=${MAX_MODEL_LEN}"
echo "  max_num_seqs=${MAX_NUM_SEQS}"
echo "  max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
echo "  gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
echo ""
echo "Companion stress command:"
printf '  python3 examples/online_serving/gpt_oss_routed_experts_stress.py --base-url http://localhost:%s/v1 --model %s\n' "${PORT}" "${MODEL}"
echo ""

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --enable-return-routed-experts \
    "$@"
