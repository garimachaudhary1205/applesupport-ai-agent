"""Compute intent + escalation metrics for the agent and both baselines."""
import json
import sys
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)


def esc_metrics(gold, pred):
    p, r, f, _ = precision_recall_fscore_support(
        gold, pred, labels=["escalate"], zero_division=0
    )
    return {"precision": round(p[0], 3), "recall": round(r[0], 3), "f1": round(f[0], 3)}


def main():
    golden = [json.loads(l) for l in open("data/golden/golden.jsonl")]
    agent_out = {json.loads(l)["pair_id"]: json.loads(l) for l in open("runs/agent_outputs.jsonl")}
    baselines = json.load(open("runs/baseline_outputs.json"))

    ok = [g for g in golden if g["pair_id"] in agent_out and "error" not in agent_out[g["pair_id"]]]
    n_err = len(golden) - len(ok)
    idx = {g["pair_id"]: i for i, g in enumerate(golden)}

    gold_intent = [g["gold_intent"] for g in ok]
    gold_action = [g["gold_action"] for g in ok]

    systems = {
        "agent": {
            "intent": [agent_out[g["pair_id"]]["intent"] for g in ok],
            "action": [agent_out[g["pair_id"]]["action"] for g in ok],
        },
        "trivial": {
            "intent": [baselines["trivial"][idx[g["pair_id"]]]["intent"] for g in ok],
            "action": [baselines["trivial"][idx[g["pair_id"]]]["action"] for g in ok],
        },
        "simple": {
            "intent": [baselines["simple"][idx[g["pair_id"]]]["intent"] for g in ok],
            "action": [baselines["simple"][idx[g["pair_id"]]]["action"] for g in ok],
        },
    }

    metrics = {"n_scored": len(ok), "n_agent_errors": n_err}
    for name, preds in systems.items():
        metrics[name] = {
            "intent_accuracy": round(accuracy_score(gold_intent, preds["intent"]), 3),
            "intent_macro_f1": round(
                f1_score(gold_intent, preds["intent"], average="macro", zero_division=0), 3
            ),
            "escalation": esc_metrics(gold_action, preds["action"]),
        }

    # detail for failure analysis
    confusions = Counter(
        (g, p) for g, p in zip(gold_intent, systems["agent"]["intent"]) if g != p
    )
    metrics["agent_top_confusions"] = [
        {"gold": g, "pred": p, "count": c} for (g, p), c in confusions.most_common(8)
    ]
    metrics["agent_per_class"] = classification_report(
        gold_intent, systems["agent"]["intent"], zero_division=0, output_dict=True
    )

    with open("runs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)

    print(f"scored {len(ok)}/{len(golden)} (agent errors: {n_err})\n")
    print(f"{'system':10s} {'intent acc':>10s} {'macro F1':>9s} {'esc P':>7s} {'esc R':>7s} {'esc F1':>7s}")
    for name in systems:
        m = metrics[name]
        e = m["escalation"]
        print(
            f"{name:10s} {m['intent_accuracy']:>10.3f} {m['intent_macro_f1']:>9.3f} "
            f"{e['precision']:>7.3f} {e['recall']:>7.3f} {e['f1']:>7.3f}"
        )
    print("\ntop agent confusions (gold -> pred):")
    for c in metrics["agent_top_confusions"]:
        print(f"  {c['gold']} -> {c['pred']}: {c['count']}")


if __name__ == "__main__":
    main()
