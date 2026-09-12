"""TF-IDF retrieval over historical (customer message -> Apple reply) pairs.

Grounds the agent's drafts in how AppleSupport actually resolved similar
issues. Index is built over root-of-thread customer messages only, so a new
incoming tweet is compared against like-for-like first contacts.
"""
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BOILERPLATE = re.compile(r"@\w+|<link>")


def normalize(text: str) -> str:
    return BOILERPLATE.sub(" ", text).lower()


class Retriever:
    def __init__(self, pairs_path="data/processed/pairs.jsonl", exclude_ids=None):
        exclude_ids = exclude_ids or set()
        self.pairs = []
        with open(pairs_path) as f:
            for line in f:
                p = json.loads(line)
                if p["is_root"] and p["pair_id"] not in exclude_ids:
                    self.pairs.append(p)
        self.vectorizer = TfidfVectorizer(
            preprocessor=normalize, stop_words="english", ngram_range=(1, 2), min_df=2
        )
        self.matrix = self.vectorizer.fit_transform(p["customer_text"] for p in self.pairs)

    def top_k(self, query: str, k: int = 6):
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [
            {**self.pairs[i], "similarity": round(float(sims[i]), 3)} for i in order
        ]
