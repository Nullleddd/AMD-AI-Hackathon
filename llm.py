"""Fireworks client wrapper.

Fireworks exposes an OpenAI-compatible API, so we use the `openai` SDK pointed
at FIREWORKS_BASE_URL. Everything is read from the environment at call time (the
harness injects the real values at eval); nothing is hardcoded or bundled.

Rules honoured here:
  - ALL calls go through FIREWORKS_BASE_URL (else tokens aren't recorded).
  - Model IDs come from ALLOWED_MODELS, never hardcoded.
  - Responses are English, deterministic (temperature 0), and length-capped to
    keep token usage — and therefore the token-efficiency rank — low.
"""

from __future__ import annotations

import os
import re
import threading
from functools import lru_cache

from openai import OpenAI


def _load_local_env(path: str = ".env") -> None:
    """Best-effort .env loader for LOCAL dev only (never bundled in the image).

    Uses setdefault so real environment values always win over the file.
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_local_env()


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ["FIREWORKS_BASE_URL"],
        timeout=25.0,  # per-request cap must stay under the 30s rule
        max_retries=2,
    )


@lru_cache(maxsize=1)
def allowed_models() -> tuple[str, ...]:
    raw = os.environ.get("ALLOWED_MODELS", "")
    models = tuple(m.strip() for m in raw.split(",") if m.strip())
    if not models:
        raise RuntimeError("ALLOWED_MODELS is empty; cannot select a model")
    return models


def pick_model() -> str:
    """Model to use. MODEL env overrides (handy locally); else first allowed."""
    override = os.environ.get("MODEL")
    if override:
        return override
    return allowed_models()[0]


# --- Model tiering ----------------------------------------------------------
# Model IDs are published launch day, so we can't hardcode them. Instead we read
# ALLOWED_MODELS and infer three tiers from the ID strings:
#   strong : biggest GENERAL reasoner  -> math, logic, factual
#   code   : a code-specialised model  -> code_debug, code_gen (else = strong)
#   cheap  : smallest by ACTIVE params -> sentiment, ner, summarisation
# All picks are overridable via MODEL / MODEL_<TIER> env vars.

TIERS = ("cheap", "strong", "code")

_MOE = re.compile(r"(\d+)\s*x\s*(\d+)\s*b\b")      # mixtral-8x7b -> 8*7
_ACTIVE = re.compile(r"\ba(\d+)b\b")               # gemma-...-a4b -> 4 active
_DENSE = re.compile(r"(\d+)\s*b\b")                # llama-...-8b -> 8
_CODE_HINT = re.compile(r"\bcode|coder|-code\b")
_QUANT_HINT = re.compile(r"nvfp4|fp4|fp8|int8|int4|awq|gptq|gguf")


def _total_params(model_id: str) -> int:
    """Total params (billions) parsed from the ID; unknown -> large frontier."""
    mid = model_id.lower()
    moe = _MOE.search(mid)
    if moe:
        return int(moe.group(1)) * int(moe.group(2))
    sizes = [int(m.group(1)) for m in _DENSE.finditer(mid)]
    return max(sizes) if sizes else 100


def _active_params(model_id: str) -> int:
    """Active params (billions) — MoE 'aNb' notation, else total. Drives speed/cost."""
    m = _ACTIVE.search(model_id.lower())
    if m:
        return int(m.group(1))
    return _total_params(model_id)


# Kept as the public capability proxy (used by tests / diagnostics).
_capability_score = _total_params


def _is_code_model(model_id: str) -> bool:
    return bool(_CODE_HINT.search(model_id.lower()))


def _is_quantized(model_id: str) -> bool:
    return bool(_QUANT_HINT.search(model_id.lower()))


@lru_cache(maxsize=1)
def _tiers() -> dict[str, str]:
    """Resolve {'cheap','strong','code'} from ALLOWED_MODELS via heuristics."""
    models = list(allowed_models())

    # strong: largest general (non-code) reasoner; prefer full precision on ties.
    general = [m for m in models if not _is_code_model(m)] or models
    strong = max(general, key=lambda m: (_total_params(m), not _is_quantized(m)))

    # code: a code-specialised model if any, else reuse strong.
    code_models = [m for m in models if _is_code_model(m)]
    code = max(code_models, key=_total_params) if code_models else strong

    # cheap: fewest active params (fast/credit-light); prefer quantized on ties.
    cheap = min(models, key=lambda m: (_active_params(m), not _is_quantized(m)))

    return {"cheap": cheap, "strong": strong, "code": code}


def model_for_tier(tier: str) -> str:
    """Model ID for a tier ('cheap' | 'strong' | 'code'), honouring overrides.

    Precedence: MODEL (forces one model everywhere) > MODEL_<TIER> > inferred.
    """
    forced = os.environ.get("MODEL")
    if forced:
        return forced
    per_tier = os.environ.get(f"MODEL_{tier.upper()}")
    if per_tier:
        return per_tier
    return _tiers()[tier]


def describe_tiers() -> str:
    """Human-readable tier->model mapping for startup logging."""
    return "  ".join(f"{t}={model_for_tier(t)}" for t in TIERS)


# Running token totals — the leaderboard ranks by total tokens, so track them.
# Thread-safe: main.py runs tasks concurrently.
_USAGE = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
_USAGE_LOCK = threading.Lock()

# Models that rejected `reasoning_effort` (e.g. non-reasoning instruct models);
# remembered so we don't pay a failed round-trip on every subsequent call.
_NO_EFFORT_PARAM: set[str] = set()

# Reasoning models (minimax-m3, gpt-oss, ...) burn hundreds of hidden tokens
# and can truncate to a BLANK answer. reasoning_effort='none' suppresses that:
# measured on minimax-m3 it turned a 420-token blank into a 4-token correct
# answer. Tokens are the scored metric, so 'none' is the default everywhere.
DEFAULT_REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "none")


def usage() -> dict[str, int]:
    """Cumulative token usage across all completions this run."""
    with _USAGE_LOCK:
        return dict(_USAGE)


def _chat(model: str, messages: list[dict], max_tokens: int, temperature: float,
          reasoning_effort: str | None):
    kwargs = {}
    if reasoning_effort and model not in _NO_EFFORT_PARAM:
        kwargs["reasoning_effort"] = reasoning_effort
    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
    except Exception as e:
        # Model doesn't support the reasoning knob -> retry once without it.
        if kwargs and "invalid_request_error" in str(e):
            _NO_EFFORT_PARAM.add(model)
            resp = _client().chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            raise
    u = getattr(resp, "usage", None)
    if u:
        with _USAGE_LOCK:
            _USAGE["prompt"] += u.prompt_tokens or 0
            _USAGE["completion"] += u.completion_tokens or 0
            _USAGE["total"] += u.total_tokens or 0
            _USAGE["calls"] += 1
    return resp.choices[0]


def complete(
    prompt: str,
    system: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    model: str | None = None,
    fallback_model: str | None = None,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
) -> str:
    """Single chat completion. Returns the message text (stripped).

    If the primary model returns blank content (e.g. a reasoning model spent its
    whole budget on hidden reasoning and got truncated), retry once on
    `fallback_model` — a blank answer scores zero, so any real text is better.
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    primary = model or pick_model()
    choice = _chat(primary, messages, max_tokens, temperature, reasoning_effort)
    content = (choice.message.content or "").strip()

    if not content and fallback_model and fallback_model != primary:
        choice = _chat(fallback_model, messages, max_tokens, temperature,
                       reasoning_effort)
        content = (choice.message.content or "").strip()

    return content
