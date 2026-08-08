#!/bin/bash
cd /Users/alexchuang/Documents/flashkv0516/turbo-fieldfare-github-official
BIN=.build/release/TurboFieldfareCLI
M=/Users/alexchuang/Documents/flashkv0516/models/gemma4.gturbo
P="Write a short technical explanation of how a B-tree index speeds up database lookups."
run() {
  echo "=== $1 ==="
  env TURBO_FIELDFARE_EXPERT_SLOTS=16 TURBO_FIELDFARE_EXPERT_STATS=1 TURBO_FIELDFARE_GPU_TIMING=1 \
    $BIN --model "$M" --prompt "$P" --max-new 128 $2 2>&1 \
    | grep -E "^\[stop|^\[expert-io|^\[stage|^\[gpu\]|^\[sync"
}
for r in r1 r2; do
  run "$r sha-full" ""
  run "$r trusted"  "--trust-receipt"
done
echo SHA-AB-DONE
