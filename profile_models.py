"""Launch-day model profiler.

Run this once the official models are known to measure, per model:
  - token burn on a simple prompt (lower = better for the efficiency rank)
  - whether it is a reasoning model (emits hidden reasoning_content)
  - rough latency (must stay under the 30s/request limit)

Usage (models from ALLOWED_MODELS, or pass a comma list):
    python profile_models.py
    python profile_models.py "modelA,modelB"

Reads FIREWORKS_API_KEY / FIREWORKS_BASE_URL from env (or a local .env).
"""

from __future__ import annotations

import os
import sys
import time

import llm

# A deliberately tiny task: a good model answers in a handful of tokens. Models
# that emit hundreds of tokens here will tank the token-efficiency score.
PROBE_SYSTEM = "Answer in English. Be concise. No preamble."
PROBE_PROMPT = "What is 15% of 240? Give only the number."


def profile(model: str) -> dict:
    client = llm._client()
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROBE_SYSTEM},
                {"role": "user", "content": PROBE_PROMPT},
            ],
            max_tokens=400,
            temperature=0.0,
        )
    except Exception as e:
        return {"model": model, "error": f"{type(e).__name__}: {str(e)[:100]}"}
    dt = time.time() - t0

    choice = resp.choices[0]
    msg = choice.message
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    return {
        "model": model,
        "latency_s": round(dt, 1),
        "finish": choice.finish_reason,
        "prompt_toks": resp.usage.prompt_tokens,
        "completion_toks": resp.usage.completion_tokens,
        "reasoning": bool(reasoning),
        "content": (msg.content or "").strip()[:60],
    }


def main() -> int:
    if len(sys.argv) > 1:
        models = [m.strip() for m in sys.argv[1].split(",") if m.strip()]
    else:
        models = list(llm.allowed_models())

    print(f"Profiling {len(models)} model(s) via {os.environ.get('FIREWORKS_BASE_URL')}\n")
    rows = [profile(m) for m in models]

    for r in rows:
        if "error" in r:
            print(f"  {r['model']:44s} ERROR {r['error']}")
            continue
        flag = "  <-- REASONING (token-heavy)" if r["reasoning"] else ""
        print(
            f"  {r['model']:44s} completion={r['completion_toks']:<4d} "
            f"finish={r['finish']:<6s} {r['latency_s']:>4}s{flag}"
        )
        print(f"      -> {r['content']!r}")

    print("\nHint: prefer low-completion, non-reasoning models for the strong tier;")
    print("set MODEL_STRONG / MODEL_CODE / MODEL_CHEAP to lock in your picks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
