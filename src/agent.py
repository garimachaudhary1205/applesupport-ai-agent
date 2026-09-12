"""The support agent: classify -> retrieve -> draft -> escalation decision.

One LLM call returns intent + confidence + drafted reply + action + reason,
grounded in retrieved historical resolutions. Deterministic guardrails then
override the action for safety/policy-critical patterns and low confidence.
"""
import json
import re

from taxonomy import (
    ACTIONS,
    CONFIDENCE_FLOOR,
    ESCALATION_POLICY,
    GUARDRAIL_PATTERNS,
    INTENTS,
)

AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "confidence": {"type": "number"},
        "reply": {"type": "string"},
        "action": {"type": "string", "enum": list(ACTIONS)},
        "reason": {"type": "string"},
    },
    "required": ["intent", "confidence", "reply", "action", "reason"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = f"""You are the AI support agent for AppleSupport on Twitter (November 2017 era —
iOS 11 rollout). For each incoming customer tweet you must:

1. Classify it into exactly one intent:
{chr(10).join(f'- {name}: {desc}' for name, desc in INTENTS.items())}

2. Draft a reply in AppleSupport's public-Twitter voice:
- warm, professional, no slang, no emoji, under 280 characters
- grounded in how Apple historically resolved similar issues (examples
  provided per message); prefer their concrete steps (exact Settings paths,
  "update to the latest iOS", "force close the app") over generic advice
- never invent Apple policies, URLs, prices, or promises
- if escalating, the reply should acknowledge the issue and hand off (e.g.
  invite to DM), not attempt account/repair actions in public

3. Decide the action per this policy:
{ESCALATION_POLICY}

4. Give confidence in [0,1] for the intent classification and a one-sentence
reason for the action."""


class SupportAgent:
    def __init__(self, backend, retriever, model="claude-haiku-4-5", k=6):
        self.backend = backend
        self.retriever = retriever
        self.model = model
        self.k = k

    def build_user_prompt(self, message, context=None):
        exemplars = self.retriever.top_k(message, self.k)
        ex_lines = []
        for i, e in enumerate(exemplars, 1):
            ex_lines.append(
                f"[{i}] (sim {e['similarity']}) customer: {e['customer_text']}\n"
                f"    apple: {e['reply_text']}"
            )
        parts = ["Historical resolutions of similar messages:", "\n".join(ex_lines)]
        if context:
            ctx = "\n".join(f"{t['role']}: {t['text']}" for t in context)
            parts.append(f"Earlier messages in this thread:\n{ctx}")
        parts.append(f"Incoming customer message:\n{message}")
        return "\n\n".join(parts), exemplars

    def handle(self, message, context=None):
        user_prompt, exemplars = self.build_user_prompt(message, context)
        result = self.backend.complete(
            SYSTEM_PROMPT, user_prompt, schema=AGENT_SCHEMA, model=self.model
        )
        result["guardrails"] = []
        lowered = message.lower()
        for name, pattern in GUARDRAIL_PATTERNS.items():
            if re.search(pattern, lowered):
                result["guardrails"].append(name)
        if result["confidence"] < CONFIDENCE_FLOOR:
            result["guardrails"].append("low_confidence")
        if result["guardrails"] and result["action"] != "escalate":
            result["action"] = "escalate"
            result["reason"] += f" [guardrail override: {', '.join(result['guardrails'])}]"
        result["exemplar_ids"] = [e["pair_id"] for e in exemplars]
        return result


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, "src")
    from llm import get_backend
    from retrieval import Retriever

    ap = argparse.ArgumentParser(description="Run the agent on one message")
    ap.add_argument("message")
    ap.add_argument("--model", default="claude-haiku-4-5")
    args = ap.parse_args()

    agent = SupportAgent(get_backend(), Retriever(), model=args.model)
    print(json.dumps(agent.handle(args.message), indent=2))
