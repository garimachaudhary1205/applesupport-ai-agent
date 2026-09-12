# Report: AppleSupport AI agent

## 1. Problem framing: what "good" means for this brand

The dataset window is November 2017: the iOS 11 rollout, the infamous
"I → A⍰" autocorrect bug, and a flood of battery/freezing complaints. The
@AppleSupport queue is dominated by **high-volume, low-variance first
contacts**: the same ~10 problems asked thousands of ways, which Apple
answers with a small set of concrete troubleshooting scripts (exact
Settings paths, "update to the latest iOS", "join us in DM").

For this brand, a good agent therefore means:

- **Correct triage.** Route each tweet to the right support flow (the
  8 intents in `src/taxonomy.py`) and, more importantly, make the right
  **auto vs escalate** call. Apple historically moved 52.6% of public
  conversations to DM; account, repair, and diagnostic matters do not
  belong in public 280-character replies, and neither does a bot.
- **Grounded first responses.** A draft reply is good if it does what Apple
  actually did for similar issues, same concrete steps, same voice, and
  never invents policies, prices, or promises. Novelty is a liability here.
- **Cheap and safe at volume.** ~500 inbound tweets/day for one brand means
  the agent runs on a small model (Haiku), with deterministic guardrails
  that can only *force* escalation (safety, account security, legal threats,
  explicit human requests, low classifier confidence) and never suppress it.

**What I chose not to build:** multi-turn dialogue management (the agent
sees thread context but each run drafts one reply); reply *sending* and DM
handoff plumbing; fine-tuned classifiers (an LLM + retrieval beats them at
this scale of labelled data); embedding/vector retrieval (TF-IDF is
transparent, fast, and strong on short noisy text; an embedding upgrade is
in "next week"); non-English support (escalates instead); and any use of
Banking77 (its intents are banking-specific and would not transfer).

## 2. System in one paragraph

`prepare_data.py` reconstructs 78,423 (customer tweet → direct Apple reply)
pairs with upstream thread context from the raw dump. At inference, the
agent retrieves the 6 most similar historical first contacts (TF-IDF,
bigrams, golden examples excluded from the index) and makes **one** Haiku
call that returns `{intent, confidence, reply, action, reason}` as
schema-validated JSON, grounded in those exemplars. Regex guardrails and a
confidence floor can then override `action` to `escalate`.

## 3. Golden set

230 examples, hand-labelled by the author (`data/golden/golden.jsonl`,
process and rules in `docs/labeling_guide.md`):

- **Sampling.** 150 drawn from 7 keyword strata proportional to stream
  share with floors of 12 on rare-but-critical strata (account/billing,
  hardware, how-to); 50 drawn uniformly at random as a stratification-bias
  check; 30 mid-thread messages (with context) so the "troubleshooting
  already failed → escalate" rule is actually exercised. Fixed seed;
  stratum recorded per example.
- **Labelling.** Single labeller (the author) in one pass over full thread
  context, then a per-intent consistency pass. Ambiguous cases carry a
  `note`. Label distribution: feature/app malfunction 85, performance/
  battery 64, how-to 24, other/unclear 21, account/billing 16, complaint 9,
  hardware 9, update-install 2. Actions: 174 auto / 56 escalate (24%).

## 4. Results

All systems scored on the same 230 examples (230 scored;
0 agent errors). The simple intent baseline is 5-fold
cross-validated so it never sees its own test label.

| System | Intent acc | Intent macro-F1 | Esc. P | Esc. R | Esc. F1 |
|---|---|---|---|---|---|
| Trivial (majority / never-escalate) | 0.370 | 0.067 | 0.000 | 0.000 | 0.000 |
| Simple (TF-IDF+logreg / NN-asks-DM) | 0.483 | 0.298 | 0.233 | 0.500 | 0.318 |
| **Agent (Haiku + retrieval + guardrails)** | **0.826** | **0.702** | 0.632 | **0.768** | **0.694** |

**Reply quality (LLM judge, 1-5).** A Sonnet judge, a different, stronger
model than the agent, blind to which system wrote each reply, scored the
agent's draft, the nearest-neighbour baseline (copy the most similar
historical reply), and Apple's **actual historical reply** as a calibration
anchor, on 120 golden examples:

| System | Grounding | Actionability | Tone | Safety | Overall |
|---|---|---|---|---|---|
| NN baseline | 3.83 | 3.23 | 4.34 | 4.62 | 3.36 |
| Historical Apple reply | 3.98 | 3.86 | 4.51 | 4.80 | 3.83 |
| **Agent** | 4.12 | 4.31 | 4.74 | 4.83 | **4.17** |

**Judge validity.** The judge's `overall` scores were compared against the
author's blind rubric scores (assigned before running the judge) on
36 mixed replies: Spearman ρ = 0.458, within-1 agreement
88.9%, quadratic-weighted κ = 0.556.

## 5. Failure analysis: top 5 failure modes

**1. Guardrail regexes over-fired on substrings (found by this eval, then
fixed).** The `legal_threat` pattern `sue\b` matched inside "is**sue**" and
`fire` matched "are you trying to get me **fired**", so ordinary tweets like
*"I just upgraded to iOS 11.2 and got a wifi issue"* were force-escalated as
legal threats. 7/230 outputs were wrongly overridden; fixing the word
boundaries (a deterministic post-LLM layer, so no re-run needed) raised
escalation precision 0.589 → 0.632. Hypothesis confirmed: keyword guardrails
on informal text are themselves a failure surface and must be inside the
eval, not assumed safe.

**2. "Standard fix already failed" escalations are missed.** 8 of the 13
missed escalations are messages where the historical fix was already tried:
*"what's up with the letter I️ and I've updated my iOS to 11.1 and it's
still jacked up?"* → auto-handled with force-close advice; *"it autocorrects
to I.T… I've changed I.T in shortcuts literally 5 times"* → auto-handled
asking for the device model. Hypothesis: the policy bullet requires temporal
reasoning about *implied* prior attempts, and the retrieved exemplars (where
Apple replied publicly anyway) pull the model toward auto-handling. Fix
candidates: a dedicated "has the customer already tried X?" field in the
schema, or few-shot exemplars of exactly this pattern.

**3. `other_unclear` is the biggest intent-confusion source** (12 of 40
errors). The model reads *through* the noise and assigns a semantic class
where the taxonomy wants a routing class: a thanks/closure like *"Power
button plus volume down did the trick. Rebooted."* becomes
`performance_battery`; a German tweet about Bluetooth becomes
`feature_app_malfunction` (and gets an English reply instead of a language
escalation). Hypothesis: the class is defined by what support should *do*
(ask/route), which conflicts with the model's instinct to classify content;
the intent description needs "even if you can recognise the topic" language.

**4. Vague-but-benign messages get escalated to DM instead of asked in
public.** Roughly half the 25 false escalations are rants or vague reports
(*"#iOS11 an absolute disaster, total disgrace. Phone totally unworkable"*)
where the gold action is a public clarifying question but the model reasons
"requires DM clarification", mimicking Apple's own DM-heavy style from the
exemplars. This is the *safe* direction of error, but at volume it defeats
the point of automation. Hypothesis: the exemplars teach "Apple's next step
is DM" as the default resolution; the policy's "may ask ONE clarifying
question" needs to be framed as the preferred first move.

**5. Fault-attribution errors at the hardware/software and
hardware/account boundaries.** *"My iPhone7 is all static during calls and
audio comes through the speakers with headphones plugged in"* (gold:
hardware) → `feature_app_malfunction`; *"work MacBook password locked by
accident"* (gold: account/billing) → `hardware_damage`. The model anchors on
device nouns rather than the fault mechanism or the tool needed to fix it
(repair bench vs account console). A related cluster: question-phrased
symptoms (*"may I run a diagnostic for my battery? It's dying quickly"*)
land on the symptom class where I labelled the ask (`how_to_question`) -
some of these are genuinely arguable labels, which is itself a finding
(taxonomy boundaries rather than model capability cap measurable accuracy).

## 6. What is misleading about my headline number?

1. **The golden labels are single-annotator.** I wrote the taxonomy, the
   labelling guide, the agent prompt, *and* the labels. Some agent
   "accuracy" is really taxonomy-alignment: an independent annotator would
   disagree with me on the ~15% of examples I flagged as borderline, so a
   realistic ceiling on label quality bounds true accuracy well below the
   measured number. Escalation labels are worse: several encode *my* policy
   judgment ("3rd replacement device → escalate"), not observable fact.
2. **Intent accuracy is propped up by two easy majority classes.** 65% of
   the golden set is feature-malfunction + performance/battery. Macro-F1 is
   the honest number, and it is dragged by classes with tiny support
   (update-install has n=2; one error moves its F1 massively).
3. **The agent "beating" Apple's real agents (4.17 vs 3.83) is a judge
   artifact, not a product claim.** The judge rewards complete, polished,
   self-contained replies; Apple's real replies are often terse
   context-dependent thread continuations ("We've received your DM, we'll
   continue there") that score poorly out of context but were the *right*
   move in the actual conversation. Treat the historical row as a
   calibration anchor showing the judge's bias, not as evidence the agent
   outperforms humans.
4. **The judge and the agent share a model family.** Sonnet judging Haiku
   reduces but does not eliminate family self-preference; both may share
   blind spots (e.g., both consider "have you tried updating?" adequate).
   The judge-human agreement is computed against *my* scores, and I am not
   independent of the system design. And the judge sees the same retrieved
   exemplars the agent used, so "grounding" partly measures "agrees with
   the retriever".
5. **Escalation recall is measured against a 24% escalate base rate** on a
   set where I over-sampled escalate-y strata. On the true stream the base
   rate is lower and rarer patterns (legal threats, safety) appear in
   numbers too small to measure; the guardrails for exactly those cases
   are essentially untested by this eval.
6. **Judged "reply quality" ≠ resolved customers.** The only ground truth
   this dataset could give (did the customer come back angry?) is not in
   the loop. A reply can score 5/5 and still be wrong for a specific device
   (the judge can't verify that Settings path existed on iOS 11).
7. **Temporal leakage.** The retrieval index spans the whole window, so the
   agent can ground November answers in December replies. Deployed
   for real, it would only have the past. (Mitigated but not eliminated by
   excluding the golden tweets themselves from the index.)

## 7. With one more week

1. **Second annotator + adjudication** on all 230 examples; report
   inter-annotator agreement and re-baseline every metric against
   adjudicated labels. This is the highest-leverage item; it hardens the
   yardstick everything else is measured with.
2. **Escalation-focused eval**: mine the dataset for threads that *did*
   blow up (multi-turn anger, churn statements) and build a dedicated
   escalate/don't test set with a realistic base rate; tune the confidence
   floor and guardrails on precision-recall curves instead of defaults.
3. **Embedding retrieval + time-aware index** (only retrieve replies from
   before the tweet's timestamp), A/B'd against TF-IDF through the same
   judge harness.
4. **Multi-turn evaluation**: replay full historical threads, letting the
   agent respond at each customer turn, and measure resolution-shaped
   proxies (did the thread end? did Apple's real agent do what our agent
   did next?).
5. **Prompt/model ablations**: Haiku vs Sonnet agent, k=0 (no retrieval) vs
   k=6, guardrails on/off, each cell through the same eval to attribute
   the headline number to its components.
