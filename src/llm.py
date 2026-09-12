"""LLM client with two interchangeable backends.

- "api": official Anthropic SDK. Used when ANTHROPIC_API_KEY (or an `ant auth
  login` profile) is available. JSON output is enforced server-side via
  structured outputs (output_config.format).
- "cli": shells out to `claude -p` (Claude Code headless mode). Used for
  development on a machine with a Claude Code login but no API key. JSON is
  requested in the prompt and parsed with one retry.

Both expose: complete(system, user, schema=None, model=...) -> str | dict
Backend selection: LLM_BACKEND env var, else auto (api if creds resolve,
else cli).
"""
import json
import os
import re
import subprocess
import time

DEFAULT_MODEL = "claude-haiku-4-5"


def _extract_json(text: str):
    """Parse the first JSON object found in text (handles ```json fences)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"unbalanced JSON in: {text[:200]!r}")


def _validate(obj: dict, schema: dict) -> dict:
    """Minimal check: required keys present, enums respected."""
    for key in schema.get("required", []):
        if key not in obj:
            raise ValueError(f"missing key {key!r}")
    for key, spec in schema.get("properties", {}).items():
        if key in obj and "enum" in spec and obj[key] not in spec["enum"]:
            raise ValueError(f"{key}={obj[key]!r} not in {spec['enum']}")
    return obj


class ApiBackend:
    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()

    def complete(self, system, user, schema=None, model=DEFAULT_MODEL, max_tokens=1024):
        kwargs = {}
        if schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        resp = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
        text = next(b.text for b in resp.content if b.type == "text")
        return _validate(json.loads(text), schema) if schema is not None else text


class CliBackend:
    def complete(self, system, user, schema=None, model=DEFAULT_MODEL, max_tokens=1024):
        prompt = f"{system}\n\n---\n\n{user}"
        if schema is not None:
            prompt += (
                "\n\nRespond with ONLY a single JSON object matching this JSON "
                "schema (no prose, no markdown fences):\n" + json.dumps(schema)
            )
        last_err = None
        for attempt in range(3):
            try:
                out = subprocess.run(
                    ["claude", "-p", "--model", model],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if out.returncode != 0:
                    raise RuntimeError(f"claude CLI failed: {out.stderr[:300]}")
                text = out.stdout.strip()
                if schema is None:
                    return text
                return _validate(_extract_json(text), schema)
            except Exception as e:  # parse errors, timeouts, transient CLI failures
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"CLI backend failed after retries: {last_err}")


def get_backend():
    choice = os.environ.get("LLM_BACKEND", "auto")
    if choice in ("api", "auto"):
        try:
            backend = ApiBackend()
            # constructor only checks that credentials resolve, not that they work
            if choice == "api" or os.environ.get("ANTHROPIC_API_KEY"):
                return backend
        except Exception:
            if choice == "api":
                raise
    if choice in ("cli", "auto"):
        return CliBackend()
    raise ValueError(f"unknown LLM_BACKEND={choice!r}")
