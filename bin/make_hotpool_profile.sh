#!/bin/bash
# Generate a hot-pool profile for the production model from a representative
# workload prompt. The profile pins the top-N most-requested experts per layer
# so decode never re-reads them from disk.
#
# Usage:
#   ./bin/make_hotpool_profile.sh <messages-file> [N] [out.json]
#     N       experts pinned per layer (default 48)
#     out     default: <model-dir>/profiles/top<N>_code.json
#
# After generating, run with: ./bin/run_prod.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/.build/release/TurboFieldfareCLI"
MODEL="${TURBO_FIELDFARE_MODEL:-/Users/alexchuang/Documents/flashkv0516/models/gemma4-r3.gturbo}"

MSG="${1:?usage: make_hotpool_profile.sh <messages-file> [N] [out.json]}"
N="${2:-48}"
OUT="${3:-"$(dirname "$MODEL")/profiles/top${N}_code.json"}"

mkdir -p "$(dirname "$OUT")"
TRACE="$(mktemp -t hotpool_trace.XXXXXX.csv)"
trap 'rm -f "$TRACE"' EXIT

echo "== collecting expert access trace ($MSG) =="
TURBO_FIELDFARE_EXPERT_TRACE="$TRACE" TURBO_FIELDFARE_EXPERT_SLOTS=64 \
  "$BIN" --model "$MODEL" --messages-file "$MSG" \
  --max-new 128 --temperature 0 --trust-receipt >/dev/null
if [ ! -s "$TRACE" ]; then
  echo "error: trace is empty - the CLI produced no plan records" >&2
  exit 1
fi
echo "== trace rows: $(wc -l < "$TRACE" | tr -d ' ') =="

python3 "${ROOT}/Scripts/gen_hotpool_profile.py" "$TRACE" "$N" "$OUT"
echo "== profile ready: $OUT =="
echo "run with: HOT_POOL_PROFILE_SIZE=$N ./bin/run_prod.sh"
