"""Agent dispatch layer.

`solve()` is the single entry point the harness calls per task:
classify (zero tokens) -> pick a category-specific prompt -> one Fireworks call.

Prompts are deliberately terse and cap `max_tokens` per category: the leaderboard
ranks passing submissions by fewest total tokens, so we spend the minimum needed
to clear the accuracy gate. FACTUAL doubles as the fallback for any misroute, so
its handler is the most general.
"""

from __future__ import annotations

from classifier import Category, classify
from llm import complete, model_for_tier

# Shared preamble: kept short (input tokens count too). Pushes direct answers
# with no restating of the question, which also trims output tokens.
_BASE = "Answer in English. Be correct and concise. No preamble; do not restate the question."

# Tier per category (resolved to concrete models in llm.py from ALLOWED_MODELS):
#   strong = biggest general reasoner   cheap = smallest/fastest
#   code   = code-specialised model (falls back to strong if none allowed)
# FACTUAL is `strong` because it is also the fallback for any misroute.
STRONG = "strong"
CHEAP = "cheap"
CODE = "code"

# (system_prompt, max_tokens, tier) per category.
_PROMPTS: dict[Category, tuple[str, int, str]] = {
    Category.FACTUAL: (
        f"{_BASE} Explain clearly and completely in one short paragraph; never "
        f"exceed 120 words.",
        300, STRONG,
    ),
    Category.MATH: (
        f"{_BASE} Work through the problem step by step, then state the final answer "
        f"on its own line as 'Answer: <value>'.",
        400, STRONG,
    ),
    Category.SENTIMENT: (
        f"{_BASE} Give the sentiment label (positive, negative, or neutral) followed "
        f"by one short sentence of justification.",
        120, CHEAP,
    ),
    Category.SUMMARIZATION: (
        f"{_BASE} Produce only the summary, obeying any length or format constraint "
        f"stated in the request.",
        220, CHEAP,
    ),
    Category.NER: (
        f"{_BASE} Extract the named entities and label each as person, organization, "
        f"location, or date. Output one 'label: value' per line.",
        260, CHEAP,
    ),
    Category.CODE_DEBUG: (
        f"{_BASE} Identify the bug in one sentence, then give the corrected code in a "
        f"single fenced block.",
        520, CODE,
    ),
    Category.LOGIC: (
        f"{_BASE} Show the deduction in a few brief numbered steps, checking every "
        f"constraint, then state the answer on its own line as 'Answer: <value>'.",
        420, STRONG,
    ),
    Category.CODE_GEN: (
        f"{_BASE} Return only the requested function/code in a single fenced block, "
        f"correct and self-contained, with no explanation.",
        520, CODE,
    ),
}


def _handle(prompt: str, category: Category) -> str:
    system, max_tokens, tier = _PROMPTS[category]
    primary = model_for_tier(tier)
    # Fall back to the cheap tier if the primary returns blank (e.g. a reasoning
    # model truncated before emitting an answer). Cheap is usually non-reasoning.
    fallback = model_for_tier(CHEAP)
    return complete(
        prompt, system=system, max_tokens=max_tokens,
        model=primary, fallback_model=fallback,
    )


def solve(prompt: str) -> str:
    """Classify a single prompt and produce an answer string."""
    category = classify(prompt)
    return _handle(prompt, category)
