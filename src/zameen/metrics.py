"""Scoring transliterations.

* CER / WER (strict): character and word error rate against Rekhta's text; every nukta, macron
  and vowel sign counts.
* CER loose: the same with diacritics, nukta and macrons removed. Strict minus loose is the share
  of error that is orthographic detail only.
* exact: share of shers transliterated perfectly (strict).
* phenomena: word accuracy on words that contain a given feature in the target script.
* form kept: for a matla, the transliteration still has the right radif and qafiya (rule-based).

Answers are extracted robustly: reasoning blocks, labels and quotes are stripped, and lines that
merely repeat the source are dropped. A model that only echoes the source is scored on the echo,
which counts as a wrong-script answer rather than no answer.
"""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd

from .tasks import DIR_LABEL
from .text import LABEL, detect_script, edit_distance, loose, qafiya_words, radif, strict, strip_think

_LABEL = re.compile(r"^\s*(?:\*\*)?(transliteration|answer|here is|output|ترجمہ|लिप्यंतरण)[^:：]*[:：]\s*", re.IGNORECASE)


def extract(raw: str, e: dict) -> str | None:
    """The model's transliteration from its raw output (None if it never finished reasoning)."""
    ans, unfinished = strip_think(raw)
    if unfinished:
        return None
    lines, echoed = [], []
    for line in ans.splitlines():
        line = _LABEL.sub("", line.strip()).strip(' "“”«»*`').strip()
        if not line:
            continue
        if detect_script(line) == e["src"] != e["tgt"] and any(
                SequenceMatcher(None, loose(line, e["src"]), loose(x, e["src"])).ratio() > 0.85 for x in e["source"]):
            echoed.append(line)
            continue
        lines.append(line)
    want = [l for l in lines if detect_script(l) == e["tgt"]]
    lines = want or lines or echoed
    if len(lines) == 1 and " / " in lines[0]:
        lines = lines[0].split(" / ")
    return "\n".join(lines[:2])


PHENOMENA = {
    "devanagari": {
        "nukta (ज़ ख़ क़ फ़ ग़)": lambda w: "़" in w,
        "aspirates (ख घ थ ध फ भ …)": lambda w: any(c in "खघछझठढथधफभ" for c in w),
        "nasalization (ँ ं)": lambda w: "ँ" in w or "ं" in w,
        "conjuncts (्)": lambda w: "्" in w,
        "izafat (-ए- / -ओ-)": lambda w: bool(re.search("-(ए|ओ)-", w)),
    },
    "roman": {
        "long vowels (ā ī ū)": lambda w: any(c in w for c in "āīū"),
        "nasalization (ñ)": lambda w: "ñ" in w,
        "aspirates (kh gh th dh ph bh …)": lambda w: bool(re.search(r"(?<!ḳ)[kgcjtdpbTD]h", w)),
        "retroflex (T D R Ḍ Ṛ)": lambda w: any(c in w for c in "TDRḌṚṬ"),
        "Perso-Arabic sounds (ḳh ġ q z f)": lambda w: bool(re.search("ḳh|ġ|q|z|f", w)),
        "izafat (-e- / -o-)": lambda w: bool(re.search("-(e|o)-", w)),
    },
    "nastaliq": {
        "aspirates (ھ)": lambda w: "ھ" in w,
        "noon ghunna (ں)": lambda w: "ں" in w,
        "hamza / izafat (ء ئ ۂ ؤ)": lambda w: any(c in w for c in "ءئۂؤ"),
        "spelling choice (ث ذ ص ض ط ظ ع ح)": lambda w: any(c in w for c in "ثذصضطظعح"),
        "bari ye (ے)": lambda w: "ے" in w,
    },
}


def word_hits(gold: str, pred: str) -> tuple[list[str], list[bool]]:
    """Which gold words appear correctly (strict) in the prediction, by word alignment."""
    gw, pw = strict(gold).split(), strict(pred).split()
    ok = [False] * len(gw)
    for tag, i1, i2, _j1, _j2 in SequenceMatcher(None, gw, pw, autojunk=False).get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                ok[i] = True
    return gw, ok


def score(e: dict, raw: str) -> dict:
    pred = extract(raw, e)
    gold = e["answer"]
    r: dict = {"no_answer": pred is None or not pred.strip()}
    pred = pred or ""
    gs, ps = strict(gold), strict(pred)
    gl, pl = loose(gold, e["tgt"]), loose(pred, e["tgt"])
    r["pred"] = pred
    r["chars"], r["char_err"] = len(gs), min(edit_distance(gs, ps), len(gs))
    r["chars_loose"], r["char_err_loose"] = len(gl), min(edit_distance(gl, pl), len(gl))
    r["words"], r["word_err"] = len(gs.split()), min(edit_distance(gs.split(), ps.split()), len(gs.split()))
    r["exact"] = gs == ps
    r["wrong_script"] = bool(pred.strip()) and detect_script(pred) != e["tgt"]
    gw, ok = word_hits(gold, pred)
    r["phen"] = {name: [sum(o for w, o in zip(gw, ok) if f(w)), sum(1 for w in gw if f(w))]
                 for name, f in PHENOMENA[e["tgt"]].items()}
    if "form" in e:
        l1, l2 = (pred.splitlines() + ["", ""])[:2]
        rr = radif(l1, l2, e["tgt"])
        r["radif_kept"] = bool(rr) and loose(rr, e["tgt"]) == loose(e["form"]["radif"], e["tgt"])
        if e["form"]["qafiya"]:
            q = qafiya_words(l1, l2, e["tgt"])
            r["qafiya_kept"] = bool(q) and [loose(x, e["tgt"]) for x in q] == [loose(x, e["tgt"]) for x in e["form"]["qafiya"]]
    return r


# ---------------------------------------------------------------------------- aggregation
METRIC_COLS = ["CER", "CER loose", "WER", "exact", "no answer", "wrong script", "n"]


def summarize(rows: list[dict], by=("split", "src", "tgt")) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    g = df.groupby(list(by))
    return pd.DataFrame({
        "CER": g.char_err.sum() / g.chars.sum(),
        "CER loose": g.char_err_loose.sum() / g.chars_loose.sum(),
        "WER": g.word_err.sum() / g.words.sum(),
        "exact": g.exact.mean(),
        "no answer": g.no_answer.mean(),
        "wrong script": g.wrong_script.mean(),
        "n": g.size(),
    })


def direction_table(rows: list[dict], split: str) -> pd.DataFrame:
    """Metrics per direction for one split, plus an 'all directions' row."""
    sub = [r for r in rows if r["split"] == split]
    t = summarize(sub).droplevel(0)
    t.index = [DIR_LABEL[i] for i in t.index]
    t.loc["all directions"] = summarize([dict(r, split="x", src="all", tgt="all") for r in sub]).iloc[0]
    return t


def phenomena_table(rows: list[dict], split: str) -> dict[tuple[str, str], float]:
    acc: dict = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["split"] == split:
            for name, (hit, tot) in r["phen"].items():
                acc[(LABEL[r["tgt"]], name)][0] += hit
                acc[(LABEL[r["tgt"]], name)][1] += tot
    return {k: h / t for k, (h, t) in acc.items() if t}


def form_rates(rows: list[dict], split: str) -> dict[str, float]:
    out = {}
    for k in ("radif_kept", "qafiya_kept"):
        v = [r[k] for r in rows if r["split"] == split and r.get(k) is not None]
        if v:
            out[k] = sum(v) / len(v)
    return out


def mark(gold: str, pred: str) -> str:
    """Predicted text with words that differ from Rekhta's in ⟦ ⟧ (⟦∅⟧ = word missing)."""
    glines, plines = strict(gold).splitlines(), strict(pred).splitlines()
    if len(plines) != len(glines):
        glines, plines = [" ".join(glines)], [" ".join(plines)]
    marked = []
    for gl, pl in zip(glines, plines):
        gw, pw, out = gl.split(), pl.split(), []
        for tag, _i1, _i2, j1, j2 in SequenceMatcher(None, gw, pw, autojunk=False).get_opcodes():
            seg = " ".join(pw[j1:j2])
            out.append(seg if tag == "equal" else (f"⟦{seg}⟧" if seg else "⟦∅⟧"))
        marked.append(" ".join(x for x in out if x))
    return " / ".join(marked)
