#!/usr/bin/env python3
"""Analyze the TURBO_FIELDFARE_GPU_TIMELINE_CSV dump.

Usage: python3 analyze_gpu_timeline.py /tmp/gpu_timeline.csv
Answers: (1) idle-gap distribution — thousands of sub-ms slivers vs few long
stalls; (2) which transition owns the idle (correct attribution: rows[i].gap
is idle BEFORE span i, i.e. AFTER span i-1); (3) whether gaps drift with
context length.
"""
import sys, re, statistics
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gpu_timeline.csv"
rows = []
for line in open(path).read().splitlines()[1:]:
    m = re.match(r"(\d+),([0-9.]+),([0-9.]+),([0-9.]+),(\w+),gap=([0-9.]+)ms", line)
    if m:
        rows.append({"st": float(m.group(2)), "dur": float(m.group(4)),
                     "label": m.group(5), "gap": float(m.group(6))})

if not rows:
    print("no rows"); sys.exit(1)
n = len(rows)
busy = sum(r["dur"] for r in rows) / 1000
gap_total = sum(r["gap"] for r in rows) / 1000
print(f"spans={n} window={rows[-1]['st']+rows[-1]['dur']-rows[0]['st']:.2f}s "
      f"gpuBusy={busy:.2f}s gpuIdle={gap_total:.2f}s")

print("\n=== gap distribution ===")
for name, lo, hi in [("<0.2ms", 0, 0.2), ("0.2-1ms", 0.2, 1), ("1-5ms", 1, 5),
                     ("5-20ms", 5, 20), ("20-100ms", 20, 100), (">100ms", 100, 1e9)]:
    sel = [r for r in rows if lo <= r["gap"] < hi]
    if sel:
        print(f"  {name:10s}: {len(sel):5d}  {sum(r['gap'] for r in sel)/1000:6.2f}s")

print("\n=== idle after each label (corrected: gap i is idle after span i-1) ===")
by_lab = {}
for i in range(1, len(rows)):
    by_lab.setdefault(rows[i-1]["label"], []).append(rows[i]["gap"])
for lab in ["cb1", "sharedFFN", "phase1Hit", "routed", "head"]:
    gs = by_lab.get(lab, [])
    if gs:
        print(f"  after {lab:9s}: n={len(gs):5d} mean={statistics.mean(gs):.2f}ms "
              f"med={statistics.median(gs):.2f}ms total={sum(gs)/1000:.2f}s")

print("\n=== stalls (>5ms) by transition ===")
stalls = [(i, r) for i, r in enumerate(rows) if r["gap"] > 5]
ctx = Counter((rows[i-1]["label"], rows[i]["label"]) for i, r in stalls)
for (a, b), c in ctx.most_common():
    tot = sum(r["gap"] for i, r in stalls if rows[i-1]["label"] == a and rows[i]["label"] == b) / 1000
    print(f"  {a} -> {b}: {c}x  ({tot:.2f}s)")

print("\n=== 1-5ms gaps by transition (the workhorse slivers) ===")
med = [(i, r) for i, r in enumerate(rows) if 1 <= r["gap"] < 5]
ctx2 = Counter((rows[i-1]["label"], rows[i]["label"]) for i, r in med)
for (a, b), c in ctx2.most_common():
    tot = sum(r["gap"] for i, r in med if rows[i-1]["label"] == a and rows[i]["label"] == b) / 1000
    print(f"  {a} -> {b}: {c}x  ({tot:.2f}s)")

print("\n=== sharedFFN->routed gap drift (context-length effect) ===")
sidx = [i for i, r in enumerate(rows) if r["label"] == "sharedFFN"]
gaps = [rows[i+1]["gap"] for i in sidx]
q = lambda arr, f: arr[int(len(arr) * f)]
sg = sorted(gaps)
print(f"  n={len(gaps)} p10={q(sg,0.1):.2f}ms p50={q(sg,0.5):.2f}ms p90={q(sg,0.9):.2f}ms max={max(gaps):.1f}ms")
print(f"  first 3 tokens med={statistics.median(gaps[:90]):.2f}ms  last 3 tokens med={statistics.median(gaps[-90:]):.2f}ms")
