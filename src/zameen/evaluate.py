"""Evaluation with resume: predictions are appended to disk as they finish."""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import score

CHUNK = 128   # flush to disk every CHUNK prompts


def run_eval(backend, splits: dict[str, list[dict]], path: str | Path, max_new_tokens: int,
             show: bool = True, log=print) -> list[dict]:
    path = Path(path)
    done: dict = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[(r["split"], r["id"])] = r
    todo = [(s, e) for s, exs in splits.items() for e in exs if (s, e["id"]) not in done]
    if done:
        log(f"Resuming {path.name}: {len(done)} done, {len(todo)} to go")
    with open(path, "a", encoding="utf-8") as f:
        for c in range(0, len(todo), CHUNK):
            chunk = todo[c:c + CHUNK]
            raws = backend.generate([e["prompt"] for _, e in chunk], max_new_tokens,
                                    desc=f"{path.stem} {c + len(chunk)}/{len(todo)}", show=show)
            for (s, e), raw in zip(chunk, raws):
                row = {"split": s, "id": e["id"], "poet": e["poet"], "src": e["src"], "tgt": e["tgt"],
                       "sher": e["sher"], "gold": e["answer"], "raw": raw, **score(e, raw)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[(s, e["id"])] = row
            f.flush()
    return [done[(s, e["id"])] for s, exs in splits.items() for e in exs if (s, e["id"]) in done]
