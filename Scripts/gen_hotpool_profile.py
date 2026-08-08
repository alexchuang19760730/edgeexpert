#!/usr/bin/env python3
"""Generate hot-pool profile (per-layer top-N expert IDs) from a trace CSV.

The profile is a fixed-length list indexed by LAYER NUMBER (parsed from
"layer_05.bin"), so a missing layer cannot shift subsequent entries. Layers
absent from the trace get an empty list (the streamer pins nothing for them —
safe, not misaligned).

Usage: gen_hotpool_profile.py <trace.csv> <N> <out.json>
"""
import sys, collections, json, re

path, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = []
with open(path) as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 5:
            continue
        m = re.match(r"layer_(\d+)\.bin", parts[0])
        if not m:
            continue
        layer = int(m.group(1))
        experts = [int(x) for x in parts[4].split()]
        rows.append((layer, experts))

if not rows:
    print(f"error: empty trace {path} — the trace run produced no plan records", file=sys.stderr)
    sys.exit(1)

by_layer = collections.defaultdict(collections.Counter)
for layer, experts in rows:
    by_layer[layer].update(experts)

num_layers = max(by_layer) + 1
total_req = sum(len(e) for _, e in rows)
profile = [[] for _ in range(num_layers)]
cov = 0
for layer in range(num_layers):
    counter = by_layer.get(layer, collections.Counter())
    top = [e for e, _ in counter.most_common(n)]
    profile[layer] = top
    cov += sum(c for e, c in counter.items() if e in set(top))

print(f"layers={num_layers} (max observed {max(by_layer)}) top-{n}/layer "
      f"coverage={cov/total_req*100:.1f}% of request volume")
for i, p in enumerate(profile):
    assert len(set(p)) == len(p), f"duplicate expert in layer {i}"
json.dump(profile, open(out, "w"))
print(f"wrote {out}")
