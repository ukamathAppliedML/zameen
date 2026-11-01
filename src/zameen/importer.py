"""Turn an executed Colab notebook into a results folder with the standard ``summary.json``.

The Colab notebook (``notebooks/ghazal_transliteration_a100.ipynb``) writes its full outputs to
Google Drive. When only the executed notebook is at hand, this importer recovers the reported
numbers from its displayed tables: per-direction metrics before and after, word accuracy by
phenomenon, form checks, the validation curve, timing and training facts, plus the figures.
"""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

from .report import SCHEMA, write_run_reports

_SPLITS = ("test_seen", "test_unseen")


def _cells(nb: dict):
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            yield "".join(c.get("source", "")), c.get("outputs", [])


def _tables(outputs) -> list[list[list[str]]]:
    tabs = []
    for o in outputs:
        h = "".join(o.get("data", {}).get("text/html", ""))
        for t in re.findall(r"<table.*?</table>", h, re.DOTALL):
            rows = []
            for r in re.findall(r"<tr>(.*?)</tr>", t, re.DOTALL):
                cells = [html.unescape(re.sub("<.*?>", "", c)).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.DOTALL)]
                rows.append(cells)
            tabs.append(rows)
    return tabs


def _text(outputs) -> str:
    return "".join("".join(o.get("text", "")) for o in outputs if o.get("output_type") == "stream")


def _pngs(outputs) -> list[bytes]:
    return [base64.b64decode(o["data"]["image/png"]) for o in outputs if "image/png" in o.get("data", {})]


def _p(s: str) -> float | None:
    s = s.strip().rstrip("%")
    try:
        return round(float(s) / 100, 6)
    except ValueError:
        return None


def _data_rows(tab):
    """Rows whose first cell names a direction (skips header rows)."""
    return [r for r in tab if r and ("→" in r[0] or r[0] == "all directions")]


def import_notebook(path: str | Path, out_dir: str | Path, run_name: str | None = None) -> dict:
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    s: dict = {"schema": SCHEMA, "source": f"colab notebook ({Path(path).name})", "environment": {}, "preflight": {},
               "training": {}, "splits": {sp: {"directions": {}, "phenomena": {}, "form": {}} for sp in _SPLITS},
               "validation_curve": [], "timing_minutes": {}}
    figures: dict[str, bytes] = {}
    for src, outs in _cells(nb):
        text, tabs = _text(outs), _tables(outs)
        if (m := re.search(r"Model (\S+) · size (\w+): (\{.*\})", text)):
            s["model"], s["size"] = m.group(1), m.group(2)
            s["preset"] = json.loads(m.group(3).replace("'", '"').replace("None", "null"))
        if (m := re.search(r"GPU (.+?) \((\d+) GB\).*?loading (\S+) as (\w+)", text)):
            s["environment"].update(gpu=m.group(1), gpu_gb=int(m.group(2)), quantization=m.group(4))
            s["model_key"] = m.group(3)
        if (m := re.search(r"torch (\S+) · transformers (\S+) · peft (\S+)", text)):
            s["environment"].update(torch=m.group(1), transformers=m.group(2), peft=m.group(3))
        if (m := re.search(r"think_prefill=(\w+)", text)):
            s["preflight"]["think_prefill"] = m.group(1) == "True"
        if src.startswith('PROG.start("baseline eval"'):
            for sp, tab in zip(_SPLITS, tabs):
                for r in _data_rows(tab):
                    vals = dict(zip(["CER", "CER loose", "WER", "exact", "no answer", "wrong script"], map(_p, r[1:7])))
                    vals["n"] = float(r[7]) if len(r) > 7 else None
                    s["splits"][sp]["directions"].setdefault(r[0], {})["before"] = vals
        if src.startswith("def before_after"):
            for sp, tab in zip(_SPLITS, tabs):
                for r in _data_rows(tab):
                    rec = s["splits"][sp]["directions"].setdefault(r[0], {})
                    after = {}
                    for name, cell in zip(["CER", "CER loose", "WER", "exact"], r[1:5]):
                        b, a = (cell.split("→") + [""])[:2]
                        after[name] = _p(a)
                        rec.setdefault("before", {}).setdefault(name, _p(b))
                    rec["after"] = after
            figures.update(dict(zip(["before_after.png"], _pngs(outs))))
        if src.startswith("# Learning curve"):
            for tab in tabs:
                head = next((r for r in tab if "val_CER" in r), None)
                if not head:
                    continue
                cols = [c for c in head if c]
                for r in tab:
                    if r and r[0].isdigit():
                        rec = {"examples_seen": int(r[0])}
                        for c, v in zip(cols, r[1:]):
                            rec[c] = _p(v)
                        s["validation_curve"].append(rec)
            figures.update(dict(zip(["learning_curve.png"], _pngs(outs))))
        if src.startswith("for split in TEST:"):
            for sp, tab in zip(_SPLITS, tabs):
                script = None
                for r in tab:
                    if len(r) == 5 and r[0] in ("Devanagari", "Roman", "Nastaliq"):
                        script, feat, b, a = r[0], r[1], r[2], r[3]
                    elif len(r) == 4 and script and r[1].endswith("%"):
                        feat, b, a = r[0], r[1], r[2]
                    else:
                        continue
                    s["splits"][sp]["phenomena"].setdefault(script, {})[feat] = {"before": _p(b), "after": _p(a)}
        if src.startswith("rows = []"):
            for tab in tabs:
                sp = None
                for r in tab:
                    if r and r[0] in _SPLITS:
                        sp, check, b, a = r[0], r[1], r[2], r[3]
                    elif len(r) == 3 and sp:
                        check, b, a = r
                    else:
                        continue
                    key = "radif_kept" if "radif" in check else "qafiya_kept"
                    s["splits"][sp]["form"][key] = {"before": _p(b), "after": _p(a)}
        if (m := re.search(r"trainable params: ([\d,]+)", text)):
            s["training"]["trainable_params"] = int(m.group(1).replace(",", ""))
        if (m := re.search(r"(\d+) training sequences", text)):
            s["training"]["train_examples"] = int(m.group(1))
        if (m := re.search(r"best validation checkpoint: step (\d+)", text)):
            s["training"]["best_val_step"] = int(m.group(1))
        if src.startswith("summary = {"):
            for tab in tabs:
                for r in tab:
                    if len(r) == 4 and r[0].isdigit():
                        s["timing_minutes"][r[1]] = float(r[3])
    s["model"] = s.get("model", "unknown")
    s["run"] = run_name or f"{s.get('model_key', 'model')}-{s.get('size', 'run')}"
    out = Path(out_dir)
    write_run_reports(s, out)
    for name, png in figures.items():     # keep the notebook's own figures alongside the redrawn ones
        (out / f"colab_{name}").write_bytes(png)
    return s
