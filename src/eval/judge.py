"""LLM-as-judge for reply quality.

Scores each reply 1-5 on four dimensions, blind to which system produced it.
The judge model (Sonnet) is deliberately a different, stronger model than
the agent (Haiku) to reduce self-preference bias. Systems judged:
- agent      : the agent's drafted reply
- nn         : nearest-neighbour baseline (copied historical reply)
- historical : what AppleSupport actually replied (calibration anchor)
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "src")
from llm import get_backend

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "grounding": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "actionability": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "tone": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "safety": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "overall": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "rationale": {"type": "string"},
    },
    "required": ["grounding", "actionability", "tone", "safety", "overall", "rationale"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """You are grading a single public-Twitter support reply from AppleSupport
(November 2017). Score each dimension 1-5 using this rubric:

grounding — Is the reply consistent with how Apple historically handled
similar issues (examples provided)? 5 = uses the same concrete steps/paths;
3 = plausible but generic; 1 = contradicts historical practice or invents
policies, URLs, prices, or promises.

actionability — Does the reply move the customer forward? 5 = concrete next
step (exact setting path, specific question, clear handoff); 3 = generic
advice; 1 = says nothing useful.

tone — 5 = warm, professional, matches AppleSupport's voice, fits a tweet;
3 = acceptable but stiff or slightly off; 1 = rude, dismissive, or wildly
off-brand.

safety — 5 = makes no unverifiable promises, requests no sensitive info in
public, appropriately hands off account/repair matters; 3 = minor
overreach; 1 = asks for credentials publicly, promises refunds/repairs, or
gives risky advice.

overall — holistic quality as a support reply (not an average).

Judge only the reply text. Do not reward length. A short reply that asks
the one right question can be a 5."""


def judge_one(backend, model, message, reply, exemplars):
    ex = "\n".join(f"- customer: {e['c']}\n  apple: {e['a']}" for e in exemplars)
    user = (
        f"Historical Apple replies to similar messages:\n{ex}\n\n"
        f"Customer message:\n{message}\n\nReply to grade:\n{reply}"
    )
    return backend.complete(JUDGE_SYSTEM, user, schema=JUDGE_SCHEMA, model=model, max_tokens=500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/judge_scores.jsonl")
    args = ap.parse_args()

    from retrieval import Retriever

    golden = [json.loads(l) for l in open("data/golden/golden.jsonl")]
    agent_out = {json.loads(l)["pair_id"]: json.loads(l) for l in open("runs/agent_outputs.jsonl")}
    baselines = json.load(open("runs/baseline_outputs.json"))
    idx = {g["pair_id"]: i for i, g in enumerate(golden)}
    retriever = Retriever(exclude_ids={g["pair_id"] for g in golden})

    if args.limit:
        golden = golden[: args.limit]

    jobs = []
    for g in golden:
        pid = g["pair_id"]
        exemplars = [
            {"c": e["customer_text"][:200], "a": e["reply_text"][:250]}
            for e in retriever.top_k(g["customer_text"], k=4)
        ]
        candidates = {
            "historical": g["reply_text"],
            "nn": baselines["simple"][idx[pid]]["reply"],
        }
        if pid in agent_out and "error" not in agent_out[pid]:
            candidates["agent"] = agent_out[pid]["reply"]
        for system, reply in candidates.items():
            jobs.append((pid, system, g["customer_text"], reply, exemplars))

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["pair_id"], r["system"]))
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    print(f"{len(jobs)} judgements to run", file=sys.stderr)

    backend = get_backend()

    def work(job):
        pid, system, msg, reply, ex = job
        try:
            scores = judge_one(backend, args.model, msg, reply, ex)
            return {"pair_id": pid, "system": system, **scores}
        except Exception as e:
            return {"pair_id": pid, "system": system, "error": str(e)}

    with open(args.out, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            f.write(json.dumps(fut.result()) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"{i}/{len(jobs)}", file=sys.stderr)


if __name__ == "__main__":
    main()
