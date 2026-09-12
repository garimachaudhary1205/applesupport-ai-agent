"""Extract AppleSupport (customer message -> brand reply) pairs from twcs.csv.

Reads the raw Kaggle dump in chunks, keeps only rows involving AppleSupport,
reconstructs reply chains, and writes one JSON line per (customer tweet that
AppleSupport replied to directly). Each record carries the upstream thread
context and the brand reply (with same-thread continuation tweets merged).

Usage:
    python src/prepare_data.py [--twcs PATH] [--out data/processed/pairs.jsonl]
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd

BRAND = "AppleSupport"
URL_RE = re.compile(r"https?://\S+")
ANON_HANDLE_RE = re.compile(r"@(\d+)")
DM_RE = re.compile(r"\b(dm|direct message)\b", re.IGNORECASE)


def default_twcs_path() -> str:
    """Resolve the raw csv via kagglehub cache, downloading if absent."""
    import kagglehub

    root = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
    return str(Path(root) / "twcs" / "twcs.csv")


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = URL_RE.sub("<link>", text)
    text = ANON_HANDLE_RE.sub("@customer", text)
    return re.sub(r"\s+", " ", text).strip()


def load_brand_rows(twcs_path: str) -> pd.DataFrame:
    keep = []
    for chunk in pd.read_csv(twcs_path, chunksize=500_000):
        mask = (chunk["author_id"] == BRAND) | chunk["text"].str.contains(
            BRAND, case=False, na=False
        )
        keep.append(chunk[mask])
    df = pd.concat(keep, ignore_index=True)
    df["tweet_id"] = df["tweet_id"].astype("int64")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twcs", default=None, help="path to twcs.csv (default: kagglehub cache)")
    ap.add_argument("--out", default="data/processed/pairs.jsonl")
    args = ap.parse_args()

    twcs = args.twcs or default_twcs_path()
    print(f"reading {twcs}", file=sys.stderr)
    df = load_brand_rows(twcs)
    print(f"{len(df)} rows involve {BRAND}", file=sys.stderr)

    by_id = df.set_index("tweet_id")
    is_brand = by_id["author_id"] == BRAND

    def parent_id(tid):
        v = by_id.at[tid, "in_response_to_tweet_id"]
        if pd.isna(v):
            return None
        v = int(v)
        return v if v in by_id.index else None

    def walk_up(tid, max_hops=6):
        """Context above a tweet, oldest first, excluding the tweet itself."""
        chain = []
        cur = parent_id(tid)
        while cur is not None and len(chain) < max_hops:
            chain.append(cur)
            cur = parent_id(cur)
        return list(reversed(chain))

    def merge_continuations(reply_id, max_extra=2):
        """Apple splits long answers into a self-reply chain; merge them."""
        texts = [by_id.at[reply_id, "text"]]
        cur = reply_id
        for _ in range(max_extra):
            resp = by_id.at[cur, "response_tweet_id"]
            if pd.isna(resp):
                break
            nxt = None
            for cand in str(resp).split(","):
                cand = int(cand)
                if cand in by_id.index and is_brand.get(cand, False):
                    nxt = cand
                    break
            if nxt is None:
                break
            texts.append(by_id.at[nxt, "text"])
            cur = nxt
        return " ".join(texts)

    brand_replies = df[(df["author_id"] == BRAND) & (~df["in_response_to_tweet_id"].isna())]
    seen_customer = set()
    n_written = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in brand_replies.itertuples():
            cust_id = int(row.in_response_to_tweet_id)
            if cust_id not in by_id.index or cust_id in seen_customer:
                continue
            if is_brand.get(cust_id, False) or not by_id.at[cust_id, "inbound"]:
                continue
            seen_customer.add(cust_id)
            ctx_ids = walk_up(cust_id)
            context = [
                {
                    "role": "brand" if is_brand.get(t, False) else "customer",
                    "text": clean_text(by_id.at[t, "text"]),
                }
                for t in ctx_ids
            ]
            reply_text = clean_text(merge_continuations(int(row.tweet_id)))
            rec = {
                "pair_id": cust_id,
                "customer_text": clean_text(by_id.at[cust_id, "text"]),
                "reply_text": reply_text,
                "context": context,
                "is_root": len(ctx_ids) == 0,
                "reply_asks_dm": bool(DM_RE.search(reply_text)),
                "created_at": by_id.at[cust_id, "created_at"],
            }
            f.write(json.dumps(rec) + "\n")
            n_written += 1
    print(f"wrote {n_written} pairs -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
