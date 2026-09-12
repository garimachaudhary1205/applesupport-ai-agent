# AppleSupport AI Agent — Hiver take-home

An AI support agent for the **AppleSupport** brand from the [Customer Support
on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. For each incoming customer tweet it:

1. **Classifies** the message into one of 8 intents defined from the data
   (`src/taxonomy.py`),
2. **Drafts a reply** grounded in how AppleSupport historically resolved
   similar issues (TF-IDF retrieval over ~51k resolved first contacts),
3. **Decides auto-handle vs escalate** with a stated reason, plus
   deterministic guardrails (safety/security/legal regexes, confidence floor).

The full report — problem framing, results vs baselines, failure analysis,
and what's misleading about the headline numbers — is in **[REPORT.md](REPORT.md)**.
The decision log is in **[DECISIONS.md](DECISIONS.md)**.

## Reproduce the headline results (< 15 min)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# LLM access — either:
export ANTHROPIC_API_KEY=sk-ant-...      # preferred (uses the SDK)
# ...or, if you have Claude Code logged in, the pipeline falls back to
# `claude -p` automatically (set LLM_BACKEND=cli to force it).
```

All processed data, the golden set, and the raw run outputs are **committed
to the repo**, so each step below can be run independently — you can verify
the headline metrics in under a minute (step 5) and re-run any earlier stage
to regenerate its inputs.

```bash
# 1. (optional, ~3 min) rebuild the processed dataset from the raw Kaggle dump
#    downloads via kagglehub (no Kaggle account needed for public datasets)
python src/prepare_data.py

# 2. (optional) re-sample golden candidates (labels are hand-assigned, committed)
python src/eval/build_golden.py

# 3. baselines (no LLM, ~1 min)
PYTHONPATH=src python src/baselines.py

# 4. run the agent over the 230-example golden set (~20 min with --workers 12;
#    or verify the committed run instantly by skipping to step 5)
python src/eval/run_agent.py --workers 12   # resumable; --limit 10 to smoke-test

# 5. metrics: intent accuracy/F1 + escalation P/R vs both baselines
python src/eval/evaluate.py

# 6. LLM-as-judge over agent / nearest-neighbour / historical replies
python src/eval/judge.py --limit 120    # ~10 min
python src/eval/judge_summary.py        # reply-quality table
python src/eval/judge_agreement.py      # judge vs human agreement
```

Try the agent interactively:

```bash
PYTHONPATH=src python src/agent.py "my battery is draining so fast since ios 11, help!"
```

## Repo map

```
src/prepare_data.py       raw twcs.csv -> data/processed/pairs.jsonl (78k pairs)
src/taxonomy.py           intent definitions + escalation policy + guardrails
src/retrieval.py          TF-IDF retrieval over historical resolutions
src/agent.py              the agent (single LLM call + guardrails) + CLI demo
src/llm.py                LLM backends: Anthropic SDK or `claude -p`
src/baselines.py          majority-class + TF-IDF/logreg + nearest-neighbour
src/eval/build_golden.py  stratified sampler for the golden set
src/eval/run_agent.py     batch runner (parallel, resumable)
src/eval/evaluate.py      intent + escalation metrics
src/eval/judge.py         LLM judge (rubric, blind, judged by a stronger model)
src/eval/judge_summary.py    mean judge scores per system (report table)
src/eval/judge_agreement.py  judge vs human agreement stats
src/eval/reapply_guardrails.py  re-run the deterministic guardrail layer on saved outputs
data/golden/golden.jsonl  230 hand-labelled examples (intent + action)
docs/labeling_guide.md    labeling rules and process
runs/                     committed outputs of the headline run
```

## Models

- Agent: `claude-haiku-4-5` (cheap enough to run on every inbound tweet).
- Judge: `claude-sonnet-5` — deliberately a different, stronger model than
  the agent to reduce self-preference bias.

## Borrowed / cited

- Dataset: thoughtvector/customer-support-on-twitter (Kaggle).
- sklearn TF-IDF + LogisticRegression for baselines/retrieval; anthropic SDK.
- LLM-as-judge rubric structure loosely follows common practice (e.g.
  MT-Bench-style 1–5 single-answer grading); implementation is original.
