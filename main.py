"""Track 1 harness entrypoint.

Contract (from the participant guide):
  - Read tasks from   /input/tasks.json   on startup
  - Write results to  /output/results.json before exiting
  - Exit code 0 on success, non-zero on failure
  - results.json must be valid JSON

Input  : [ { "task_id": "t1", "prompt": "..." }, ... ]
Output : [ { "task_id": "t1", "answer": "..." }, ... ]

Paths default to the harness locations but can be overridden with INPUT_PATH /
OUTPUT_PATH env vars for local development on non-Linux machines.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

from agent import solve

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
# Concurrent Fireworks calls: the 10-min budget with up-to-30s requests makes
# sequential runs risky beyond ~20 tasks; parallelism removes that bottleneck.
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))


def load_tasks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError(f"Expected a JSON list of tasks, got {type(tasks).__name__}")
    return tasks


def write_results(path: str, results: list[dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _solve_one(task: dict, index: int) -> dict:
    task_id = task.get("task_id", f"idx_{index}")
    prompt = task.get("prompt", "")
    try:
        answer = solve(prompt)
    except Exception:  # never let one task abort the batch
        traceback.print_exc()
        answer = ""
    return {"task_id": task_id, "answer": answer}


def run(tasks: list[dict]) -> list[dict]:
    if len(tasks) <= 1:
        return [_solve_one(t, i) for i, t in enumerate(tasks)]
    workers = min(MAX_WORKERS, len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # pool.map preserves input order in its results
        return list(pool.map(_solve_one, tasks, range(len(tasks))))


def main() -> int:
    try:
        tasks = load_tasks(INPUT_PATH)
    except Exception as e:
        print(f"FATAL: could not read tasks from {INPUT_PATH}: {e}", file=sys.stderr)
        return 1

    print(f"Loaded {len(tasks)} task(s) from {INPUT_PATH}", file=sys.stderr)

    try:  # log tier->model mapping; never fatal (e.g. env not set locally)
        from llm import describe_tiers

        print(f"Model tiers: {describe_tiers()}", file=sys.stderr)
    except Exception as e:
        print(f"WARN: could not resolve model tiers: {e}", file=sys.stderr)

    results = run(tasks)

    try:
        write_results(OUTPUT_PATH, results)
    except Exception as e:
        print(f"FATAL: could not write results to {OUTPUT_PATH}: {e}", file=sys.stderr)
        return 1

    print(f"Wrote {len(results)} result(s) to {OUTPUT_PATH}", file=sys.stderr)

    try:  # report token usage — the scored metric
        from llm import usage

        u = usage()
        print(
            f"Tokens: total={u['total']} (prompt={u['prompt']} "
            f"completion={u['completion']}) over {u['calls']} call(s)",
            file=sys.stderr,
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
