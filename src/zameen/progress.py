"""Stage timing and logging: every stage is timed, printed and appended to progress.jsonl."""

from __future__ import annotations

import json
import time
from pathlib import Path


def fmt_dur(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


class Progress:
    def __init__(self, path: str | Path | None = None, echo: bool = True):
        self.path, self.echo = (Path(path) if path else None), echo
        self.stages: list[dict] = []
        self._t: dict[str, float] = {}

    def log(self, msg: str) -> None:
        if self.echo:
            print(msg, flush=True)

    def start(self, name: str, note: str = "") -> None:
        self._t[name] = time.time()
        self.log(f"\n▶ {name}{(' · ' + note) if note else ''}  [{time.strftime('%H:%M:%S')}]")

    def done(self, name: str, **info) -> None:
        dt = time.time() - self._t.pop(name, time.time())
        rec = {"stage": name, "seconds": round(dt, 1), "at": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
        self.stages.append(rec)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        extra = " · ".join(f"{k}={v}" for k, v in info.items())
        self.log(f"✔ {name} in {fmt_dur(dt)}{(' · ' + extra) if extra else ''}")

    def minutes(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.stages:
            out[s["stage"]] = round(out.get(s["stage"], 0) + s["seconds"] / 60, 1)
        return out


class NullBar:
    """Stand-in for tqdm when it isn't installed."""

    def __init__(self, *a, **k):
        pass

    def update(self, n=1):
        pass

    def close(self):
        pass
