# Decision log — non-obvious decisions and why

1. **Brand: AppleSupport, not AmazonHelp.** AmazonHelp is bigger but its
   replies are near-uniform redirection boilerplate; Apple's replies contain
   real troubleshooting content, which makes grounded drafting measurable.

2. **Apple's 52.6% "move to DM" rate became the escalation signal.** Apple
   moving a conversation to DM is a historical proxy for "needs
   account-specific/human handling" — used for the nearest-neighbour
   escalation baseline and to sanity-check the policy, not as gold truth.

3. **Pairs, not threads, as the unit.** Each (customer tweet → direct Apple
   reply) pair is one example, with upstream thread context attached. This
   matches the production shape ("a message arrives, respond") and gives a
   reference reply for every input.

4. **Intent = symptom, not cause.** "Battery drains since iOS 11" is
   `performance_battery`, not an update issue — support flows branch on the
   symptom. Consequence: `update_install_issue` (install trouble only) is
   genuinely rare (2/230), and the "everything broke after the update"
   cluster distributes across symptom intents.

5. **8 intents, including explicit `complaint_feedback` and
   `other_unclear`.** ~13% of the stream is venting or unintelligible;
   without those buckets the model is forced to hallucinate a technical
   intent for them.

6. **Escalation is policy + guardrails, not just the LLM.** The LLM decides
   first; regex guardrails (safety, account security, legal, explicit human
   request) and a confidence floor can only *force* escalation, never
   de-escalate. False escalations are cheap; a bot mishandling a smoking
   battery is not.

7. **Stratified golden sampling with a floor for rare strata + a 50-example
   pure-random slice.** Proportional sampling alone gives ~3 account/billing
   examples (unmeasurable); pure stratification hides the true distribution.
   The random slice keeps an unbiased view; strata are recorded per example.

8. **~15% of golden examples are mid-thread** (with conversation context) so
   the "standard troubleshooting already failed → escalate" rule is actually
   exercised, not just written down.

9. **Retrieval index excludes all golden pair_ids.** Otherwise the agent
   retrieves the evaluation tweet itself along with Apple's actual answer —
   leakage that inflates every metric.

10. **Simple intent baseline scored with 5-fold cross-validation** on the
    golden set, so it never sees its own test label — a fair comparison even
    with only 230 labelled examples.

11. **Judge is a different, stronger model (Sonnet) than the agent (Haiku),
    blind to which system wrote the reply**, and also scores Apple's real
    historical replies as a calibration anchor. If the judge scored the
    anchor absurdly, we'd distrust the judge before trusting the agent.

12. **Human labels for judge agreement were assigned before running the
    judge** on a mixed, shuffled sample of agent/baseline/historical
    replies — the only way a solo author can approximate a blind comparison.

13. **LLM client supports two backends** (Anthropic SDK / `claude -p` CLI)
    behind one interface, because the dev machine had a Claude Code login
    but no API key. The CLI path parses JSON with retry; the SDK path uses
    server-enforced structured outputs.

14. **Committed processed data and run outputs to the repo.** Graders can
    verify the headline numbers in under a minute without a 500MB download
    or an API key; every stage can still be regenerated from scratch.

15. **Agent = one LLM call, not a chain.** Classify + draft + decide share
    one prompt and one retrieval pass. At Twitter-support volume, a 3-call
    chain triples cost/latency for marginal gains a 230-example eval could
    not detect anyway.

16. **Guardrail bugfix applied post-hoc without re-running the model.** The
    eval caught guardrail regexes matching substrings ("is**sue**" →
    legal_threat). Because guardrails are a deterministic layer *after* the
    LLM call, `src/eval/reapply_guardrails.py` reconstructs the model's own
    decision and re-applies the fixed patterns — saving 230 LLM calls and
    keeping the LLM outputs byte-identical to the original run.

17. **Non-English → escalate rather than reply in-language.** The historical
    corpus (and thus grounding) is English; replying in unverified Spanish
    with confident troubleshooting steps is a worse failure mode than a
    human handoff.
