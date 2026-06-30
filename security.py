"""
FundaScope — security layer
===========================
Two defenses, mirroring the course's expense-agent exercise:
  1. screen_input()  — input-side: detects prompt-injection / advice-seeking
     queries before they reach the model.
  2. guardrail()     — output-side: neutralizes investment-advice language in
     the LLM's response and always attaches an educational disclaimer.

Kept LLM-free on purpose: deterministic regex checks are cheap, auditable, and
cannot themselves be prompt-injected.
"""

from __future__ import annotations

import re
from typing import Any

# ── Input-side: prompt-injection / advice-seeking patterns ──
_INJECTION_PATTERNS = re.compile(
    r"(?:ignore|disregard|forget|bypass|override)\b.{0,100}"
    r"(?:instruction|rule|prompt|above|previous|system)"
    r"|system prompt|you are now|act as|jailbreak|developer mode"
    r"|(?:should|recommend|advise|suggest).{0,30}(?:buy|sell|invest)"
    r"|auto[- ]?approve",
    re.IGNORECASE | re.DOTALL,
)

# ── Output-side: explicit investment-advice language ──
_ADVICE_PATTERNS = re.compile(
    r"\bstrong (?:buy|sell)\b"
    r"|\byou should (?:buy|sell|short|invest)\b"
    r"|\b(?:i|we) (?:recommend|suggest|advise)\b"
    r"|\bmy recommendation\b"
    r"|\bprice target of\b"
    r"|\bis a (?:buy|sell|good buy|great buy)\b"
    r"|\bworth buying\b|\bshould be bought\b"
    r"|\brecommend (?:buying|selling|holding)\b",
    re.IGNORECASE,
)

_DISCLAIMER = (
    " This is an automated, educational summary of reported figures, "
    "not investment advice."
)


def is_malicious_input(text: str) -> bool:
    """True if the query contains prompt-injection or advice-seeking language."""
    return bool(_INJECTION_PATTERNS.search(text or ""))


def guardrail(ctx: Any, node_input: Any = None) -> dict:
    """Output safety layer. Runs after the LLM interpretation node.

    1. Coerces the upstream interpretation to plain text (handles str/Content).
    2. If the text contains investment-advice language, it is withheld and
       replaced with a safe notice (defense-in-depth — the interpret prompt
       already forbids advice).
    3. Ensures the educational disclaimer is always present.
    """
    # 1. Normalize whatever the LLM node passed downstream into a string
    text = node_input
    if hasattr(text, "parts"):  # types.Content object
        text = "".join((p.text or "") for p in (text.parts or []))
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()

    # 2. Block investment-advice language
    advice_flagged = bool(_ADVICE_PATTERNS.search(text))
    if advice_flagged:
        text = (
            "The analysis was withheld because it contained language resembling "
            "investment advice. FundaScope reports factual summaries of a "
            "company's published fundamentals only."
        )

    # 3. Always attach the disclaimer
    if "not investment advice" not in text.lower():
        text = text + _DISCLAIMER

    ctx.actions.state_delta["advice_flagged"] = advice_flagged

    return {
        "output": {
            "status": "ok",
            "analysis": text,
            "advice_flagged": advice_flagged,
        }
    }