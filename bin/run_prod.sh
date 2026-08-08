#!/bin/bash
# Production launch for the on-device MoE decoder.
#
# Bundles every setting proven by A/B during the 2026-08-07 optimization
# session (see docs / TURBOFIELDFARE_MMAP_EXPERT_IO_PLAN_2026-08-07.md §8-§11):
#
#   - Expert cache slots 64 (was 16)        -> cache hit 65% -> 95%
#   - Hot pool (pinned high-frequency exp.) -> decode +25% (code), TTFT -0.5s
#   - --trust-receipt (skip SHA-256 rehash) -> TTFT -3.9s
#   - Adaptive MTP gate (code default ON)   -> MTP never slower than base
#   - Early shared commit (code default ON) -> GPU never starves cb1->shared
#
# Knobs (env):
#   HOT_POOL_PROFILE_SIZE   32|48|64  (default 64; 64 = 97.1% expert
#                                  coverage + sync preload, decode +32%
#                                  on r4 128-tok / +18% r3; 32 has best
#                                  TTFT)
#   TURBO_FIELDFARE_MODEL          model dir (default models/gemma4-r3.gturbo)
#   MTP_MODEL                      draft model dir (DEFAULT EMPTY = MTP off since
#                                  2026-08-08; MTP is net-negative at r2/r3/r4 once base
#                                  decode got fast. Set to models/gemma-4-mtp-head to
#                                  enable speculative decoding explicitly)
#   TURBO_FIELDFARE_EXPERT_READ_WORKERS
#                                   decode miss-read depth (default 8; prefill
#                                   capped at 2 in code: +11% decode, no TTFT tax)
#   TURBO_FIELDFARE_WAKE_POLL_US      spin-wait on MTLCommandBuffer.status before the
#                                   blocking wait (default 5000us; 0 = off). Load-noise
#                                   immunity + earlier CPU chain start: +35% median
#                                   decode under load, neutral in quiet windows, ~+28%
#                                   CPU on the decode thread. Byte-identical output.
#   MODEL_BITS                     2|3  routed-expert bit width (default 3 = r3).
#                                  2 reads ~44% fewer bytes: TTFT -37%, decode
#                                  +68% (2026-08-07 A/B) at a small quality cost
#                                  (weight-space RMSE ~0.013-0.019).
#
# Usage:
#   ./bin/run_prod.sh --messages-file /tmp/msg_code.json --max-new 128
# All extra args are passed through to TurboFieldfareCLI unchanged.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/.build/release/TurboFieldfareCLI"

MODEL_BITS="${MODEL_BITS:-3}"
case "$MODEL_BITS" in
  2|3) ;;
  *) echo "error: MODEL_BITS must be 2 or 3 (got $MODEL_BITS)" >&2; exit 2 ;;
esac
# Resolve the model dir from the bit width; an explicit TURBO_FIELDFARE_MODEL
# still wins (full path, e.g. a fine-tuned or custom repack).
if [ -n "${TURBO_FIELDFARE_MODEL:-}" ]; then
  MODEL="$TURBO_FIELDFARE_MODEL"
else
  MODEL="/Users/alexchuang/Documents/flashkv0516/models/gemma4-r${MODEL_BITS}.gturbo"
fi
PROFILE_SIZE="${HOT_POOL_PROFILE_SIZE:-64}"
# Draft model for speculative decoding (MTP). DEFAULT OFF since 2026-08-08:
# interleaved A/B (256 tok, r2/r3/r4, wake-poll config) showed MTP is NET
# NEGATIVE at every bit width once base decode got fast (wake polling + w8 +
# tensor-ops + B4): r2 17.1 vs 23.5 (-37%), r3 ~17 vs 19 (-12%), r4 15.3 vs
# 19.7 (-29%). FIXED 2026-08-08 (MTPAdaptive.swift): the gate now uses the MIN
# of the warmBaseline d=0 samples (the mean was 2x slow because in-loop d=0
# steps degrade after a cold verify batch), skips the cold first calibratingMTP
# sample, and disables IMMEDIATELY on a large losing margin (rate < 0.75 x
# baseline). Verified: r4 gate now disables 5x/run with 82% of steps at d=0
# (was 0-1x / ~0% before); in-loop d=0 is at parity with RawCompletion decode
# under the same load. MTP is still NET NEGATIVE on r2/r3/r4, so the default
# stays OFF; the gate is now correct if you opt in. Enable MTP explicitly per-
# run with: MTP_MODEL=/path/to/draft-model ./bin/run_prod.sh
# NOTE: plain ${VAR-default} (not :-) — UNSET or EMPTY MTP_MODEL means
# "disable MTP"; only an explicitly SET non-empty value enables it.
MTP_MODEL="${MTP_MODEL-}"

# The script owns --model and --mtp-model (via env); reject caller-passed
# ones so the two can never disagree.
for a in "$@"; do
  case "$a" in
    --model|--mtp-model|--model=*|--mtp-model=*)
      echo "error: pass the model via TURBO_FIELDFARE_MODEL/MTP_MODEL, not $a" >&2
      exit 2 ;;
  esac
done

case "$PROFILE_SIZE" in
  32|48|64) ;;
  *) echo "error: HOT_POOL_PROFILE_SIZE must be 32, 48 or 64 (got $PROFILE_SIZE)" >&2; exit 2 ;;
esac

# Profile lives next to the model (model-specific expert IDs).
PROFILE_DIR="$(dirname "$MODEL")/profiles"
PROFILE="${PROFILE_DIR}/top${PROFILE_SIZE}_code.json"
if [ ! -f "$PROFILE" ]; then
  echo "error: hot-pool profile not found: $PROFILE" >&2
  echo "  generate one first: ./bin/make_hotpool_profile.sh <prompt.json> $PROFILE_SIZE $PROFILE" >&2
  exit 2
fi

if [ "$PROFILE_SIZE" = "64" ]; then
  # pool64 keeps 32 evictable LRU slots -> needs 96 total (r4 expert = 3.2MB
  # means ~9GB virtual; commit stays lazy until touched, RSS ~2.5GB).
  export TURBO_FIELDFARE_EXPERT_SLOTS=96
else
  export TURBO_FIELDFARE_EXPERT_SLOTS=64
fi
export TURBO_FIELDFARE_HOT_POOL=1
export TURBO_FIELDFARE_HOT_POOL_EXPERTS="$PROFILE_SIZE"
export TURBO_FIELDFARE_HOT_POOL_PROFILE="$PROFILE"
# Preload mode: async for 32/48 (data loads in background, TTFT -0.35s vs
# sync, decode hit rate still ~95% once the pool fills); pool64 must use
# sync — async's 6.1GB background reads fight decode misses for SSD and
# collapse decode by ~50% (measured 2026-08-07). Override with
# TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync|async.
if [ "$PROFILE_SIZE" = "64" ]; then
  export TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
else
  export TURBO_FIELDFARE_HOT_POOL_PRELOAD=async
fi

# B3 single-pass MPP tensor-ops decode attention (512/16/2 full-attn).
# INERT ON THIS SDK (2026-08-08): the MPP kernel never compiles — it is
# guarded by `#if defined(__HAVE_TENSOR__)` and the local CLT SDK has no
# MetalPerformancePrimitives.h, so psoDecodeTensorOps is nil and every
# decode layer falls through to the tiled split path regardless. Verified:
# =0 vs =1 produce byte-identical output AND identical GPU time. Default 0.
# Callers may set TURBO_FIELDFARE_ATTN_TENSOROPS=1 on an SDK with MPP.
export TURBO_FIELDFARE_ATTN_TENSOROPS="${TURBO_FIELDFARE_ATTN_TENSOROPS:-0}"
# Expert miss-read depth (decode): 8 parallel preads per layer vs default 2.
# Clean-boot 6-round interleaved A/B (2026-08-08): decode +11.1% (15.43 vs
# 13.89 tok/s); TTFT unchanged because the prefill read path is capped at 2
# workers in code (split-worker change) — raising this knob no longer taxes
# first-token latency. Pool64 keeps decode hits ~99%, so the win is the rare
# miss being read in one I/O round instead of up to four.
export TURBO_FIELDFARE_EXPERT_READ_WORKERS=8
# Spin-wait on MTLCommandBuffer.status instead of parking in
# waitUntilCompleted: the decode thread has no other work during the cb1
# wait, so polling (sched_yield between checks) lets the CPU chain start
# ~100us earlier per layer AND immunizes the wake against load noise
# (parked-thread wakeups balloon under load; a spinning thread does not).
# Measured 2026-08-08: byte-identical output; median +35% decode under
# load (15.7 vs 11.6 tok/s, 4-round interleaved), neutral in quiet
# windows (17.1 vs 17.2), cost ~28% CPU on the decode thread. 5000us
# covers the average cb1 wait AND miss-expert read waits. Opt-out with 0.
export TURBO_FIELDFARE_WAKE_POLL_US="${TURBO_FIELDFARE_WAKE_POLL_US:-5000}"

MTP_ARGS=()
if [ -n "$MTP_MODEL" ]; then
  if [ ! -f "$MTP_MODEL/config.json" ] || [ ! -f "$MTP_MODEL/model.safetensors" ]; then
    echo "error: MTP draft model not found: $MTP_MODEL (missing config.json or model.safetensors; set MTP_MODEL to empty to disable MTP)" >&2
    exit 2
  fi
  MTP_ARGS=(--mtp-model "$MTP_MODEL")
fi

# ${arr[@]+...} idiom: bash 3.2 (macOS) treats "${MTP_ARGS[@]}" on an
# empty array as unbound under set -u; the guard expands only when non-empty.
exec "$BIN" --model "$MODEL" --trust-receipt ${MTP_ARGS[@]+"${MTP_ARGS[@]}"} "$@"
