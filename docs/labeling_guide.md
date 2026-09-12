# Golden-set labeling guide

Each example gets two labels: `gold_intent` (one of the 8 intents in
`src/taxonomy.py`) and `gold_action` (`auto_handle` | `escalate`).

## Intent decision rules

- Label the **symptom the customer wants fixed**, not the suspected cause.
  "Battery drains since iOS 11" → `performance_battery` (not update_install_issue).
- `update_install_issue` is only for trouble **getting** an update
  (download stuck, install fails, "why can't I download 11.1").
- Freezing/slow/hot/restarting → `performance_battery` (device-level).
  A single app or feature misbehaving → `feature_app_malfunction`.
- "Won't charge / won't turn on" → `hardware_damage` (needs physical
  diagnosis) unless the customer clearly ties it to software.
- A rant **with a specific reproducible symptom** gets the symptom label;
  a rant with no actionable symptom → `complaint_feedback`.
- Non-English, bare links, or missing context → `other_unclear`, even if a
  keyword is recognizable.
- Multi-issue tweets: label the issue the customer leads with.

## Action decision rules

`escalate` when any of: account/billing/security matter; hardware damage or
repair; safety risk (smoke, burns, swelling); prior troubleshooting in the
thread already failed; explicit demand for a human; legal threat;
non-English. Everything else (first-contact troubleshooting, how-tos,
venting) is `auto_handle` (a public reply, possibly asking one clarifying
question, is appropriate).

Edge case: `other_unclear` that is plausibly product-related but vague →
`auto_handle` (reply asks for details). Non-English → `escalate`.

## Process

Labels were assigned by the author in a single pass over the 230 candidates
reading the full thread context, followed by a second consistency pass over
all examples of each intent grouped together (which corrected ~10 labels).
Ambiguous cases were resolved by the rules above; genuinely 50/50 cases are
flagged in `note`.
