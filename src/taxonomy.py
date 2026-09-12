"""Intent taxonomy and escalation policy for the AppleSupport agent.

Derived from reading samples of the Nov-2017 @AppleSupport inbound stream
(iOS 11 era). Definitions are written for a labeler first, model second -
the same text is used in the labeling guide and in the agent prompt.
"""

INTENTS = {
    "update_install_issue": (
        "Trouble downloading or installing an iOS/macOS/watchOS update, or an "
        "update that is stuck/failing. NOT for problems that merely started "
        "after an update (classify those by the symptom)."
    ),
    "performance_battery": (
        "Battery draining fast, device running hot, slow, laggy, freezing, "
        "random restarts, or boot loops, device/OS-level performance and "
        "power problems, whatever the suspected cause."
    ),
    "feature_app_malfunction": (
        "A specific app or feature misbehaving: keyboard/autocorrect bugs "
        "(e.g. the 'I -> A[?]' bug), alarms not firing, Safari crashing, "
        "iMessage/FaceTime issues, missing settings, Bluetooth/Wi-Fi bugs."
    ),
    "hardware_damage": (
        "Physical or hardware faults: cracked screen, broken buttons/speaker, "
        "won't charge or power on, stuck in headphone mode, water damage, "
        "repair or Genius Bar questions."
    ),
    "account_billing": (
        "Apple ID / iCloud sign-in, locked accounts, passwords, purchases, "
        "refunds, gift cards, subscriptions, App Store or iTunes charges."
    ),
    "how_to_question": (
        "The product works; the customer wants to know how to do something, "
        "where a feature lives, or whether something is possible."
    ),
    "complaint_feedback": (
        "Venting, sarcasm, opinions, or feature requests with no specific "
        "troubleshootable ask ('iOS 11 is trash, fix it')."
    ),
    "other_unclear": (
        "Non-English, unintelligible, missing context (bare link/image), or "
        "not about an Apple product/service."
    ),
}

ACTIONS = ("auto_handle", "escalate")

ESCALATION_POLICY = """\
Escalate to a human when ANY of the following holds:
- account_billing issues (account access, payments, refunds; these need identity
  verification and account tools the bot must not touch)
- hardware_damage (repair decisions, warranty, physical inspection)
- safety risk (device smoking/burning, injury, data loss claims)
- the thread shows Apple's standard troubleshooting was already tried and
  failed, or the customer is on a repeat contact for the same issue
- explicit demand for a human, legal threat, or extreme anger where a
  templated reply would inflame
- the message is non-English (route to language support)
Otherwise auto-handle: first-contact troubleshooting for update/performance/
feature issues, how-to answers, and empathetic acknowledgements of
complaints. Auto-handled replies may ask ONE clarifying question when the
message is vague but plausibly product-related."""

# Deterministic guardrails applied AFTER the model's decision. Regexes are
# matched on the lowercased customer message; any hit forces escalation.
GUARDRAIL_PATTERNS = {
    "safety_risk": r"\bfire\b|\bsmok(?:e|ing)\b|\bfumes?\b|\bburn(?:ed|ing|t)?\b|explod|swell|swollen|electric(?:al)? shock",
    "account_security": r"hack(?:ed|ing)?|stolen|theft|fraud|scam|unauthori[sz]ed|locked out|disabled account",
    "legal_threat": r"\blawyers?\b|\blawsuits?\b|\bsue\b|\bsuing\b|\battorneys?\b|\blegal action\b",
    "explicit_human_request": r"real person|real human|speak to (?:a )?(?:human|person|someone)|talk to (?:a )?(?:human|person)",
}

CONFIDENCE_FLOOR = 0.55  # below this, force escalate regardless of model action
