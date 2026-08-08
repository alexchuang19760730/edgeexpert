#!/bin/bash
# B1 final A/B: fused (TURBO_FIELDFARE_FUSE_SHARED=1) vs split (default).
# Strict interleaved, 64 slots, fixed prompt, 6 rounds, alternating start,
# median per mode, plus CB-count verification of the path taken.
set -u
BIN_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$BIN_DIR/TurboFieldfareCLI-b1"
PROMPT="Write a short technical explanation of how a B-tree index speeds up database lookups."

run_one() { # model mode -> "tok/s|ttft|decode|cbs"
  local model="$1" mode="$2"
  local ef=""
  [ "$mode" = "fused" ] && ef="TURBO_FIELDFARE_FUSE_SHARED=1"
  local out
  out=$(env $ef TURBO_FIELDFARE_EXPERT_STATS=1 TURBO_FIELDFARE_GPU_TIMING=1 \
        TURBO_FIELDFARE_EXPERT_SLOTS=64 TURBO_FIELDFARE_EXPERT_PREFETCH=1 \
        TURBO_FIELDFARE_EXPERT_LOOKAHEAD=1 \
        "$BIN" --model "$model" --prompt "$PROMPT" --max-new 128 --trust-receipt 2>&1)
  local t d c
  t=$(echo "$out" | grep -oE 'tok/s=[0-9.]+' | tail -1 | cut -d= -f2)
  d=$(echo "$out" | grep -oE 'ttft=[0-9.]+s' | tail -1 | cut -d= -f2 | tr -d s)
  c=$(echo "$out" | grep -oE 'cbs=[0-9]+' | tail -1 | cut -d= -f2)
  echo "${t:-NA}|${d:-NA}|${c:-NA}"
}

median() { # numbers on stdin, one per line -> median
  sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'
}

for model in "$@"; do
  echo "=== $(basename "$model") : 6 rounds x {split,fused} (alternating start) ==="
  sp=(); fu=()
  for r in 1 2 3 4 5 6; do
    if (( r % 2 == 1 )); then order="split fused"; else order="fused split"; fi
    for mode in $order; do
      res=$(run_one "$model" "$mode")
      ts=$(echo "$res" | cut -d'|' -f1)
      echo "r$r/$mode: tok/s=$ts (ttft=$(echo "$res" | cut -d'|' -f2)s cbs=$(echo "$res" | cut -d'|' -f3))"
      if [ "$mode" = "split" ]; then sp+=("$ts"); else fu+=("$ts"); fi
    done
  done
  echo "--- medians ---"
  echo "split: $(printf '%s\n' "${sp[@]}" | median) tok/s"
  echo "fused: $(printf '%s\n' "${fu[@]}" | median) tok/s"
done
