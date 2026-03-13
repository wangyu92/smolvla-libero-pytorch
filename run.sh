#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-all}"
N_EPISODES="${2:-10}"
NUM_WORKERS="${3:-0}"   # 0 = auto-detect GPU count

echo "=== SmolVLA LIBERO Evaluation ==="
echo "Task:       $TASK"
echo "Episodes:   $N_EPISODES"
echo "Workers:    $NUM_WORKERS (0=auto)"
echo

docker compose build
docker compose run --rm eval \
    --task "$TASK" \
    --n-episodes "$N_EPISODES" \
    --num-workers "$NUM_WORKERS"
