"""Baselines the agent must beat.

Trivial: majority-class intent, never-escalate, one template reply.
Simple:  TF-IDF + logistic regression for intent (5-fold CV on the golden
         set); nearest-neighbour for replies and escalation (copy the reply
         Apple gave to the most similar historical message; escalate iff
         that reply moved the customer to DM).
"""
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from retrieval import Retriever, normalize

TEMPLATE_REPLY = (
    "We're sorry to hear you're having trouble. Please update to the latest "
    "iOS in Settings > General > Software Update, and reach out in DM if the "
    "issue continues."
)


def majority_baseline(golden):
    intents = [g["gold_intent"] for g in golden]
    majority = max(set(intents), key=intents.count)
    return [
        {"intent": majority, "action": "auto_handle", "reply": TEMPLATE_REPLY}
        for _ in golden
    ]


def tfidf_logreg_intent(golden, seed=42):
    """5-fold cross-validated predictions so every example is scored
    out-of-fold — the simple baseline never sees its own label."""
    texts = [g["customer_text"] for g in golden]
    labels = [g["gold_intent"] for g in golden]
    pipe = make_pipeline(
        TfidfVectorizer(preprocessor=normalize, stop_words="english", ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(cross_val_predict(pipe, texts, labels, cv=cv))


def nearest_neighbor(golden, retriever: Retriever):
    """Copy the historical reply to the most similar message; escalate iff
    Apple historically took that conversation to DM."""
    out = []
    for g in golden:
        nn = retriever.top_k(g["customer_text"], k=1)[0]
        out.append(
            {
                "reply": nn["reply_text"],
                "action": "escalate" if nn["reply_asks_dm"] else "auto_handle",
                "nn_similarity": nn["similarity"],
            }
        )
    return out


def run_all(golden_path="data/golden/golden.jsonl", out_path="runs/baseline_outputs.json"):
    golden = [json.loads(l) for l in open(golden_path)]
    retriever = Retriever(exclude_ids={g["pair_id"] for g in golden})
    simple_intents = tfidf_logreg_intent(golden)
    nn = nearest_neighbor(golden, retriever)
    results = {
        "trivial": majority_baseline(golden),
        "simple": [
            {"intent": si, **n} for si, n in zip(simple_intents, nn)
        ],
    }
    import os

    os.makedirs("runs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    run_all()
