#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def greedy_order(expert_counts, pair_counts, experts_per_layer):
    remaining = set(range(experts_per_layer))
    order = []
    counts = Counter(expert_counts)
    pair_map = defaultdict(int, pair_counts)

    def pair_weight(a, b):
        if a > b:
            a, b = b, a
        return pair_map[(a, b)]

    if not remaining:
        return order

    first = max(remaining, key=lambda e: (counts[e], -e))
    order.append(first)
    remaining.remove(first)

    while remaining:
        best_expert = None
        best_side = "right"
        best_score = None
        left = order[0]
        right = order[-1]
        for expert in remaining:
            left_score = (pair_weight(expert, left), counts[expert], -expert)
            right_score = (pair_weight(expert, right), counts[expert], -expert)
            candidate_score, side = (left_score, "left") if left_score > right_score else (right_score, "right")
            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_expert = expert
                best_side = side
        if best_side == "left":
            order.insert(0, best_expert)
        else:
            order.append(best_expert)
        remaining.remove(best_expert)
    return order


def main():
    parser = argparse.ArgumentParser(description="Aggregate prefill expert trace JSONL into co-access summary and layout order.")
    parser.add_argument("--input", required=True, help="Path to prefill expert trace JSONL")
    parser.add_argument("--output", required=True, help="Path to output layout-order JSON")
    parser.add_argument("--experts-per-layer", type=int, default=128, help="Experts per layer for permutation output")
    parser.add_argument("--top-pairs", type=int, default=10, help="Top co-access pairs to keep in summary")
    args = parser.parse_args()

    per_layer_counts = defaultdict(Counter)
    per_layer_pairs = defaultdict(Counter)
    request_count = 0

    with open(args.input, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            request_count += 1
            record = json.loads(line)
            for layer in record.get("layers", []):
                layer_index = int(layer["layer"])
                for item in layer.get("expert_counts", []):
                    per_layer_counts[layer_index][int(item["expert"])] += int(item["count"])
                for item in layer.get("coaccess_pairs", []):
                    key = (int(item["first"]), int(item["second"]))
                    per_layer_pairs[layer_index][key] += int(item["count"])

    layers = []
    for layer_index in sorted(set(per_layer_counts) | set(per_layer_pairs)):
        counts = per_layer_counts[layer_index]
        pairs = per_layer_pairs[layer_index]
        order = greedy_order(counts, pairs, args.experts_per_layer)
        top_pairs = [
            {"first": first, "second": second, "count": count}
            for (first, second), count in pairs.most_common(args.top_pairs)
        ]
        layers.append({
            "layer": layer_index,
            "order": order,
            "top_pairs": top_pairs,
            "top_experts": [
                {"expert": expert, "count": count}
                for expert, count in counts.most_common(args.top_pairs)
            ],
        })

    output = {
        "strategy": "coaccess-greedy-v1",
        "request_count": request_count,
        "experts_per_layer": args.experts_per_layer,
        "layers": layers,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
