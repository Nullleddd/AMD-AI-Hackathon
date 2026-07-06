# AMD Developer Hackathon — Track 1: General-Purpose AI Agent

A token-efficient agent that reads tasks, classifies each into one of 8 capability
categories with **zero-token keyword heuristics**, then answers with a
category-specific handler. Routing spends no LLM tokens so the budget goes to answers
(scoring ranks passing submissions by fewest tokens).

## Layout

| File | Role |
|------|------|
| `main.py` | Entrypoint. Reads `/input/tasks.json` → solves tasks **concurrently** (`MAX_WORKERS`, default 8) → writes `/output/results.json` → exit 0. Logs tier mapping + token usage to stderr. |
| `classifier.py` | Single-pass regex/keyword router → one of 8 `Category` values (zero tokens). |
| `agent.py` | Dispatch layer: category → (system prompt, max_tokens, model tier) → one Fireworks call. |
| `llm.py` | Fireworks client (OpenAI SDK @ `FIREWORKS_BASE_URL`). Model tiering from `ALLOWED_MODELS`, `reasoning_effort="none"` by default, blank-answer fallback, token accounting, 25s timeout. |
| `profile_models.py` | Probe each allowed model: token burn, reasoning behaviour, latency. |
| `test_classifier.py` | Classifier stress test (tuned + held-out sets, per-category accuracy). |
| `sample_input/tasks.json` | 8 example tasks, one per category, for local runs. |
| `Dockerfile` | Submission image (`linux/amd64`). |

## Model tiers (inferred from `ALLOWED_MODELS` at runtime)

| Tier | Launch-day pick | Categories |
|------|-----------------|-----------|
| `strong` | `minimax-m3` | factual, math, logic |
| `code` | `kimi-k2p7-code` | code_debug, code_gen |
| `cheap` | `gemma-4-26b-a4b-it` (4B active MoE) | sentiment, ner, summarization |

Overrides: `MODEL` (everything) > `MODEL_STRONG` / `MODEL_CODE` / `MODEL_CHEAP` > inferred.

**Key finding:** `minimax-m3` is a reasoning model — without `reasoning_effort="none"` it
burns its whole token budget on hidden reasoning and returns a **blank** answer on hard
prompts. With `"none"` it answers correctly in a handful of tokens. `llm.py` sends `"none"`
by default and auto-retries without the param for models that reject it.

## Run locally (Windows / PowerShell)

Put your own key in `.env` (gitignored; see `llm.py` for the loader), then:

```powershell
python main.py            # full pipeline on sample_input/, prints token usage
python test_classifier.py # classifier regression
python profile_models.py  # probe allowed models for token burn / reasoning
```

Note: personal Fireworks keys can't reach the gemma models (404) — `glm-5p2` stands in
as the cheap tier locally. The harness list works fully at eval time.

Run the classifier stress test:

```powershell
python test_classifier.py
```

## Contract (from the participant guide)

- Input `/input/tasks.json`: `[ { "task_id": "t1", "prompt": "..." }, ... ]`
- Output `/output/results.json`: `[ { "task_id": "t1", "answer": "..." }, ... ]`
- Exit 0 on success; output must be valid JSON.
- Env injected by the harness at eval time — **read from the environment, do not hardcode or bundle `.env`**:
  `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL` (route ALL calls through it), `ALLOWED_MODELS` (comma-separated).

## Status / next steps

- [x] Harness skeleton, exit-code-0 verified end-to-end
- [x] Heuristic classifier + stress test (57/57 tuned, 19/19 held-out)
- [x] Fireworks handlers wired, per-category prompts + token caps
- [x] Model tiering (strong/code/cheap) from `ALLOWED_MODELS`
- [x] `reasoning_effort="none"` (fixes blank answers, huge token savings)
- [x] Parallel task execution + 25s request timeout
- [x] Verified vs real hackathon models: 8 tasks, ~1330 tokens, ~6s, all correct
- [ ] Build & push Docker image, submit
- [ ] On eval day: sanity-check tier log line in the harness output
