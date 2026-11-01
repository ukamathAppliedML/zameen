"""Loading the three-script corpus.

Two input formats are supported:

* **JSONL** (the format to export Rekhta's full corpus in), one poem per line::

    {"id": "mirza-ghalib/aa-ki-...", "poet": "mirza-ghalib", "form": "ghazal",
     "text": {"nastaliq": [...lines], "devanagari": [...], "roman": [...]}}

* **A Rekhta-style folder**: ``<poet>/{ur,hi,en}/<poem>``, one line per line of verse, as in the
  public sample at github.com/amir9ume/urdu_ghazals_rekhta (MIT; text from rekhta.org).

With no corpus given, the public sample is downloaded once into the cache directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import zipfile
from pathlib import Path

SAMPLE_REPO = "https://github.com/amir9ume/urdu_ghazals_rekhta.git"
FOLDERS = {"nastaliq": "ur", "devanagari": "hi", "roman": "en"}


def _read_lines(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_folder(root: str | Path) -> list[dict]:
    root = Path(root)
    poems = []
    for pdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (pdir / "en").is_dir():
            continue
        names: set[str] = set()
        for d in FOLDERS.values():
            if (pdir / d).is_dir():
                names |= {f.name for f in (pdir / d).iterdir() if not f.name.startswith(".")}
        for name in sorted(names):
            text = {s: _read_lines(pdir / d / name) for s, d in FOLDERS.items() if (pdir / d / name).is_file()}
            text = {s: v for s, v in text.items() if v}
            if text:
                poems.append({"id": f"{pdir.name}/{name}", "poet": pdir.name, "form": "ghazal", "text": text})
    return poems


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fetch_public_sample(cache_dir: str | Path) -> Path:
    """Clone the public sample (once) and unzip its dataset. Returns the dataset folder."""
    cache = Path(cache_dir).expanduser()
    dataset = cache / "rekhta_sample" / "dataset"
    if dataset.is_dir():
        return dataset
    repo = cache / "urdu_ghazals_rekhta"
    if not repo.is_dir():
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", SAMPLE_REPO, str(repo)], check=True,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    with zipfile.ZipFile(repo / "dataset" / "dataset.zip") as z:
        z.extractall(cache / "rekhta_sample")
    return dataset


def aligned(poem: dict) -> bool:
    """All scripts present have the same number of lines (at least one sher)."""
    lens = {len(v) for v in poem["text"].values()}
    return len(lens) == 1 and min(lens) >= 2


def load_corpus(source: str, cache_dir: str = "~/.cache/zameen") -> tuple[list[dict], int]:
    """Returns (usable ghazals, number of poems read). Unaligned poems are skipped."""
    if not source:
        poems_all = load_folder(fetch_public_sample(cache_dir))
    elif str(source).endswith(".jsonl"):
        poems_all = load_jsonl(source)
    else:
        poems_all = load_folder(source)
    poems = [p for p in poems_all if aligned(p) and p.get("form", "ghazal") == "ghazal"]
    return poems, len(poems_all)


def write_jsonl(poems: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in poems)
