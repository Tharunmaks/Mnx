#!/usr/bin/env bash
# Pretrain the 7.11B MNX 2.0 Max Coder ("max" preset) with FSDP2.
#
# Run the same command on every node, e.g. 8 nodes x 8 H100:
#   NNODES=8 NODE_RANK=<0..7> MASTER_ADDR=<node-0 host> ./scripts/launch_7b.sh
#
# Defaults: ~4M tokens per optimizer step at 4096 context, 142B tokens total
# (~20 tokens/parameter). Check the cost first:
#   python -m mnx.budget --config max --gpus $((NNODES * GPUS_PER_NODE)) --seq-len 4096
set -euo pipefail

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

DATA=${DATA:-/data/mnx-shards}
OUT=${OUT:-runs/mnx-2.0-max}
SEQ_LEN=${SEQ_LEN:-4096}
MICRO_BATCH=${MICRO_BATCH:-2}
TOKENS_PER_STEP=${TOKENS_PER_STEP:-4194304}
TOTAL_TOKENS=${TOTAL_TOKENS:-142000000000}

WORLD=$((NNODES * GPUS_PER_NODE))
GRAD_ACCUM=$(( (TOKENS_PER_STEP + WORLD * MICRO_BATCH * SEQ_LEN - 1) / (WORLD * MICRO_BATCH * SEQ_LEN) ))
STEPS=$(( TOTAL_TOKENS / (WORLD * MICRO_BATCH * SEQ_LEN * GRAD_ACCUM) ))
echo "world=$WORLD grad_accum=$GRAD_ACCUM steps=$STEPS"

torchrun \
    --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$GPUS_PER_NODE" \
    --rdzv_backend c10d --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
    -m mnx.pretrain \
    --config max --data "$DATA" --out "$OUT" \
    --seq-len "$SEQ_LEN" --micro-batch "$MICRO_BATCH" --grad-accum "$GRAD_ACCUM" \
    --steps "$STEPS" --lr 3e-4 --min-lr 3e-5 --warmup 2000 --weight-decay 0.1 \
    --eval-interval 500 --save-interval 1000 \
    --grad-checkpointing --compile "$@"
