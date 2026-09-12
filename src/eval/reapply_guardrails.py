"""Re-apply the deterministic guardrail layer to saved agent outputs.

The guardrails run after the LLM call, so a guardrail bugfix does not
require re-running the model: reconstruct the model's own action (an
override always appends a '[guardrail override: ...]' marker to reason),
then apply the current patterns from taxonomy.py.
"""
import json
import re
import sys

sys.path.insert(0, "src")
from taxonomy import CONFIDENCE_FLOOR, GUARDRAIL_PATTERNS

MARKER = re.compile(r" \[guardrail override: [^\]]*\]$")

golden = {json.loads(l)["pair_id"]: json.loads(l) for l in open("data/golden/golden.jsonl")}
out = []
n_changed = 0
for line in open("runs/agent_outputs.jsonl"):
    r = json.loads(line)
    if "error" in r:
        out.append(r)
        continue
    reason = MARKER.sub("", r["reason"])
    base_action = "auto_handle" if MARKER.search(r["reason"]) else r["action"]
    lowered = golden[r["pair_id"]]["customer_text"].lower()
    guardrails = [n for n, p in GUARDRAIL_PATTERNS.items() if re.search(p, lowered)]
    if r["confidence"] < CONFIDENCE_FLOOR:
        guardrails.append("low_confidence")
    action = base_action
    if guardrails and action != "escalate":
        action = "escalate"
        reason += f" [guardrail override: {', '.join(guardrails)}]"
    if (action, guardrails) != (r["action"], r["guardrails"]):
        n_changed += 1
    out.append({**r, "action": action, "reason": reason, "guardrails": guardrails})

with open("runs/agent_outputs.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"reprocessed {len(out)}, changed {n_changed}")
