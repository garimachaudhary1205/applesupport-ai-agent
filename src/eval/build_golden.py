"""Sample golden-set candidates from the processed pairs.

Sampling design (see REPORT.md):
- 150 tweets drawn from 7 keyword strata proportional to their share of the
  stream, with a floor of 12 for rare-but-critical strata (account/billing,
  hardware) so every intent has enough support to measure.
- 50 tweets drawn uniformly at random (no keyword filter) as an honesty
  check against stratification bias.
- ~15% of the total drawn from mid-thread messages (with context) so the
  "troubleshooting already failed -> escalate" policy is exercised.
- Fixed seed; candidate file records the stratum each example came from.
Labels are added by hand afterwards (see docs/labeling_guide.md).
"""
import json
import random
import re

random.seed(42)

STRATA = {
    "update": r"updat|upgrad|ios ?11",
    "battery_perf": r"batter|charg|drain|overheat|hot\b|slow|lag|freez|froze",
    "crash_feature": r"crash|glitch|keyboard|autocorrect|alarm|imessage|facetime|bluetooth|wi-?fi",
    "hardware": r"screen|broke|cracked|repair|speaker|button|headphone|won'?t turn on|genius",
    "account_billing": r"icloud|apple id|password|refund|charge[sd] me|gift card|subscript|payment|locked",
    "howto": r"how do i|how to|where (?:is|are|can)|can i\b|is there a way",
    "complaint": r"fix (?:this|it|your)|trash|garbage|worst|hate|ridiculous|unacceptable",
}
FLOORS = {"account_billing": 12, "hardware": 12, "howto": 12}
N_STRATIFIED = 150
N_RANDOM = 50
FRACTION_MIDTHREAD = 0.15

pairs = [json.loads(l) for l in open("data/processed/pairs.jsonl")]
roots = [p for p in pairs if p["is_root"]]
mids = [p for p in pairs if not p["is_root"] and p["context"]]


def bucket(p):
    t = p["customer_text"].lower()
    for name, rx in STRATA.items():
        if re.search(rx, t):
            return name
    return None


by_stratum = {name: [] for name in STRATA}
for p in roots:
    b = bucket(p)
    if b:
        by_stratum[b].append(p)

total = sum(len(v) for v in by_stratum.values())
counts = {k: max(FLOORS.get(k, 0), round(N_STRATIFIED * len(v) / total)) for k, v in by_stratum.items()}
# trim proportionally if floors pushed us over
while sum(counts.values()) > N_STRATIFIED:
    biggest = max(counts, key=lambda k: counts[k] - FLOORS.get(k, 0))
    counts[biggest] -= 1

chosen, seen = [], set()
for name, n in counts.items():
    for p in random.sample(by_stratum[name], n):
        if p["pair_id"] not in seen:
            seen.add(p["pair_id"])
            chosen.append({**p, "stratum": name})

n_mid = round((N_STRATIFIED + N_RANDOM) * FRACTION_MIDTHREAD)
for p in random.sample(mids, n_mid):
    if p["pair_id"] not in seen:
        seen.add(p["pair_id"])
        chosen.append({**p, "stratum": "midthread"})

pool = [p for p in roots if p["pair_id"] not in seen]
for p in random.sample(pool, N_RANDOM):
    seen.add(p["pair_id"])
    chosen.append({**p, "stratum": "random"})

random.shuffle(chosen)
with open("data/golden/candidates.jsonl", "w") as f:
    for p in chosen:
        f.write(json.dumps(p) + "\n")
print(f"wrote {len(chosen)} candidates")
from collections import Counter

print(Counter(p["stratum"] for p in chosen))
