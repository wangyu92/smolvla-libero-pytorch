#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-libero_10}"
N_EPISODES="${2:-10}"
BATCH_SIZE="${3:-1}"

echo "=== SmolVLA LIBERO Evaluation ==="
echo "Task:       $TASK"
echo "Episodes:   $N_EPISODES"
echo "Batch size: $BATCH_SIZE"
echo

docker compose build
docker compose run --rm eval \
    --task "$TASK" \
    --n-episodes "$N_EPISODES" \
    --batch-size "$BATCH_SIZE"
