"""Run the agent over the golden set and save raw outputs.

Retrieval excludes all golden pair_ids so the agent can never look up the
exact tweet (and Apple's real answer) it is being evaluated on.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "src")
from agent import SupportAgent
from llm import get_backend
from retrieval import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/golden/golden.jsonl")
    ap.add_argument("--out", default="runs/agent_outputs.jsonl")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="run on first N only (smoke test)")
    args = ap.parse_args()

    golden = [json.loads(l) for l in open(args.golden)]
    if args.limit:
        golden = golden[: args.limit]
    retriever = Retriever(exclude_ids={g["pair_id"] for g in golden})
    agent = SupportAgent(get_backend(), retriever, model=args.model)

    done = {}
    if os.path.exists(args.out):  # resume support
        for line in open(args.out):
            r = json.loads(line)
            done[r["pair_id"]] = r
        print(f"resuming: {len(done)} already done", file=sys.stderr)

    todo = [g for g in golden if g["pair_id"] not in done]

    def work(g):
        try:
            res = agent.handle(g["customer_text"], context=g["context"] or None)
            return {"pair_id": g["pair_id"], **res}
        except Exception as e:
            return {"pair_id": g["pair_id"], "error": str(e)}

    with open(args.out, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, g) for g in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            f.write(json.dumps(fut.result()) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"{i}/{len(todo)}", file=sys.stderr)
    print(f"done -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
