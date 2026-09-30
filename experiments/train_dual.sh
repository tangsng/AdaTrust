#!/usr/bin/env bash
# train_dual.sh - dual-H100 parallel seed/config training (GPU utilization A)
# Usage: bash train_dual.sh "<spec1>" "<spec2>" ...
#   spec = "tag:iters:seed[:extra args]"  e.g. "main_s0:500:0" "ablA1_s1:500:1:--no_sw_cost"
# Each spec runs on alternating GPUs (even->GPU0, odd->GPU1), two at a time.
set -u
CODE=/home/user1/projects/code
OUTBASE=/home/user1/runs/train
i=0
pids=()
for spec in "$@"; do
  IFS=':' read -r tag iters seed extra <<< "$spec"
  gpu=$((i % 2))
  dir=$OUTBASE/$tag
  mkdir -p "$dir"
  echo "launch $tag on GPU$gpu (iters=$iters seed=$seed extra=$extra)"
  setsid env CUDA_VISIBLE_DEVICES=$gpu python3 -u "$CODE/experiments/train.py" \
    --iters "$iters" --seed "$seed" --out "$dir/ppo.pt" $extra \
    > "$dir/train.log" 2>&1 < /dev/null &
  pids+=($!)
  i=$((i+1))
  # throttle: at most 2 concurrent (one per GPU)
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 10; done
done
wait
echo "ALL TRAINING DONE $(date '+%F %T')"
for spec in "$@"; do
  IFS=':' read -r tag iters seed extra <<< "$spec"
  echo "--- $tag ---"
  tail -2 "$OUTBASE/$tag/train.log" 2>/dev/null
done
