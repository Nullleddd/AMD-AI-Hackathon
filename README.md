# AMD Developer Hackathon — Track 1: General-Purpose AI Agent

A token-efficient agent that reads tasks, classifies each into one of 8 capability
categories with **zero-token keyword heuristics**, then answers with a
category-specific handler. Routing spends no LLM tokens so the budget goes to answers
(scoring ranks passing submissions by fewest tokens).

## Layout

| File | Role |
|------|------|
| `main.py` | Entrypoint. Reads `/input/tasks.json` → solves each → writes `/output/results.json` → exit 0. |
| `classifier.py` | Single-pass regex/keyword router → one of 8 `Category` values. |
| `agent.py` | Dispatch layer. `solve()` classifies then calls a per-category handler (**currently a dummy echo stub**). |
| `test_classifier.py` | Classifier stress test (tuned + held-out sets, per-category accuracy). |
| `sample_input/tasks.json` | 8 example tasks, one per category, for local runs. |
| `Dockerfile` | Submission image (`linux/amd64`, stdlib-only for now). |

## Run locally (Windows / PowerShell)

```powershell
$env:INPUT_PATH="sample_input/tasks.json"; $env:OUTPUT_PATH="sample_output/results.json"; python main.py
```

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
- [x] Heuristic classifier + stress test
- [ ] Replace `_echo` in `agent.py` with Fireworks-backed handlers (start with a strong general/factual handler — it's the fallback for any misroute)
- [ ] Add confidence signal so low-confidence prompts route to the general handler
