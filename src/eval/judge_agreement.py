"""Judge-vs-human agreement on the `overall` reply score.

data/golden/human_reply_scores.csv holds the author's blind rubric scores
(assigned before running the judge) for a mixed sample of agent, baseline,
and historical replies. Reports Spearman rho, exact and within-1 agreement,
and quadratic-weighted kappa.
"""
import csv
import json

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

human = {}
with open("data/golden/human_reply_scores.csv") as f:
    for row in csv.DictReader(f):
        human[(int(row["pair_id"]), row["system"])] = int(row["human_overall"])

judge = {}
for line in open("runs/judge_scores.jsonl"):
    r = json.loads(line)
    if "overall" in r:
        judge[(r["pair_id"], r["system"])] = r["overall"]

keys = sorted(set(human) & set(judge))
h = [human[k] for k in keys]
j = [judge[k] for k in keys]
n = len(keys)
exact = sum(a == b for a, b in zip(h, j)) / n
within1 = sum(abs(a - b) <= 1 for a, b in zip(h, j)) / n
rho, p = spearmanr(h, j)
kappa = cohen_kappa_score(h, j, weights="quadratic")

print(f"n={n} paired scores")
print(f"spearman rho      = {rho:.3f} (p={p:.4f})")
print(f"exact agreement   = {exact:.2%}")
print(f"within-1 agreement= {within1:.2%}")
print(f"quadratic kappa   = {kappa:.3f}")
json.dump(
    {"n": n, "spearman": round(rho, 3), "exact": round(exact, 3),
     "within1": round(within1, 3), "quadratic_kappa": round(kappa, 3)},
    open("runs/judge_agreement.json", "w"), indent=1,
)
