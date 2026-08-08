#!/bin/zsh
# ============================================================================
# TurboFieldfare CLI 内核切换 / A-B 对比测试工具
#
# 两个已编译二进制（含今天 2026-08-07 的 GEMV tgx + GQA-full attention 优化）：
#   bin/TurboFieldfareCLI-newkernel   → 新内核（GQA-full + encodeTgx）md5 e3a5b2b9
#   bin/TurboFieldfareCLI-baseline    → 旧内核（generic + encode）       md5 4706c6ba
#
# 用法：
#   ./bin/bench_ab.sh new    --model <dir> --messages-file <f> --max-new 128   # 只跑新内核
#   ./bin/bench_ab.sh base   --model <dir> --prompt "..."                      # 只跑旧内核
#   ./bin/bench_ab.sh ab     --model <dir> --messages-file <f> --max-new 128   # 交错 A/B ×3（防页缓存漂移）
#
# 除 --model 外的 CLI 参数原样透传；--quiet 会隐藏 tok/s 统计，对比时不要加。
# 专家缓存槽数可用环境变量覆盖：TURBO_FIELDFARE_EXPERT_SLOTS=32 ./bin/bench_ab.sh ...
# ============================================================================
set -u
BIN_DIR="$(cd "$(dirname "$0")" && pwd)"
NEW="$BIN_DIR/TurboFieldfareCLI-newkernel"
BASE="$BIN_DIR/TurboFieldfareCLI-baseline"
MODEL="${TURBO_FIELDFARE_BENCH_MODEL:-/Users/alexchuang/Documents/flashkv0516/models/gemma4.gturbo}"

mode="${1:-ab}"; shift || true

run_one() {
    local bin="$1" tag="$2"
    # 插入 --model（若用户没给），其余参数透传
    local args=("$@")
    local has_model=0
    for a in "${args[@]}"; do [[ "$a" == "--model" ]] && has_model=1; done
    if (( has_model == 0 )); then
        "$bin" --model "$MODEL" "${args[@]:2}" 2>&1 | grep -oE 'ttft=[0-9.]+s.*tok/s=[0-9.]+' | tail -1
    else
        "$bin" "${args[@]:2}" 2>&1 | grep -oE 'ttft=[0-9.]+s.*tok/s=[0-9.]+' | tail -1
    fi
}

case "$mode" in
  new)
    echo "[newkernel] $(basename $NEW)"
    run_one "$NEW" B "$@"
    ;;
  base|baseline)
    echo "[baseline] $(basename $BASE)"
    run_one "$BASE" A "$@"
    ;;
  ab|a-b|abab)
    echo "=== 交错 A/B ×3: A=baseline(旧) B=newkernel(新) ==="
    echo "round,tag,tok/s"
    for round in 1 2 3; do
        r=$(run_one "$BASE" A "$@"); echo "$round,A,$r"
        r=$(run_one "$NEW"  B "$@"); echo "$round,B,$r"
        r=$(run_one "$BASE" A "$@"); echo "$round,A,$r"
        r=$(run_one "$NEW"  B "$@"); echo "$round,B,$r"
    done
    ;;
  *)
    echo "用法: $0 {new|base|ab} [--model <dir>] [CLI 参数...]"
    exit 1
    ;;
esac
