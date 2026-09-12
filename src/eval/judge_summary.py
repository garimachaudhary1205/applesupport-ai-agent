"""Print mean judge scores per system (the reply-quality table in REPORT.md)."""
import collections
import json

rows = [json.loads(l) for l in open("runs/judge_scores.jsonl") if "overall" in l]
by_sys = collections.defaultdict(list)
for r in rows:
    by_sys[r["system"]].append(r)
dims = ["grounding", "actionability", "tone", "safety", "overall"]
print(f"{'system':12s}" + "".join(f"{d:>14s}" for d in dims) + f"{'n':>5s}")
for s in ("nn", "historical", "agent"):
    items = by_sys.get(s, [])
    if not items:
        continue
    means = [sum(i[d] for i in items) / len(items) for d in dims]
    print(f"{s:12s}" + "".join(f"{m:>14.2f}" for m in means) + f"{len(items):>5d}")
