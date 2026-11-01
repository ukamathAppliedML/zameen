"""The task: transliterate one sher between two scripts.

Six directions. Training mixes the directions and the language of the instruction (English, Urdu,
Hindustani in Devanagari, Roman Urdu), so the model learns the scripts as one tradition.
Evaluation always uses English instructions so that only the scripts vary.
"""

from __future__ import annotations

from .text import LABEL, SCRIPTS, qafiya_words, radif, scripts_of, shers

SCRIPT_NAMES = {
    "en": {"nastaliq": "Urdu script (Nastaliq)", "devanagari": "Devanagari", "roman": "Roman script (Latin letters)"},
    "nastaliq": {"nastaliq": "اردو رسم الخط", "devanagari": "دیوناگری", "roman": "رومن رسم الخط"},
    "devanagari": {"nastaliq": "उर्दू लिपि", "devanagari": "देवनागरी", "roman": "रोमन लिपि"},
    "roman": {"nastaliq": "Urdu script", "devanagari": "Devanagari", "roman": "Roman"},
}
TEMPLATE = {
    "en": "Transliterate this sher into {script}. Keep every word as it is; write only the transliterated sher, in two lines.",
    "nastaliq": "اس شعر کو {script} میں لکھیے۔ ہر لفظ برقرار رکھیے؛ صرف شعر، دو مصرعوں میں، لکھیے۔",
    "devanagari": "इस शेर को {script} में लिखिए। हर लफ़्ज़ बरक़रार रखिए; सिर्फ़ शेर, दो मिसरों में, लिखिए।",
    "roman": "Is sher ko {script} mein likhiye. Har lafz barqaraar rakhiye; sirf sher, do misron mein, likhiye.",
}
DIRECTIONS = [(a, b) for a in SCRIPTS for b in SCRIPTS if a != b]
DIR_LABEL = {d: f"{LABEL[d[0]]} → {LABEL[d[1]]}" for d in DIRECTIONS}


def build_prompt(lines: list[str], tgt: str, lang: str = "en") -> str:
    return TEMPLATE[lang].format(script=SCRIPT_NAMES[lang][tgt]) + "\n\n" + "\n".join(lines)


def make_example(poem: dict, src: str, tgt: str, idx: int, lang: str = "en") -> dict | None:
    """One example: sher `idx` of `poem`, from script `src` to `tgt`, instruction in `lang`."""
    sc = scripts_of(poem)
    if src not in sc or tgt not in sc:
        return None
    a, b = shers(poem, src), shers(poem, tgt)
    if idx >= min(len(a), len(b)):
        return None
    e = {
        "id": f"{poem['id']}|s{idx}|{src}>{tgt}|{lang}", "poem_id": poem["id"], "poet": poem["poet"],
        "sher": idx, "src": src, "tgt": tgt, "lang": lang,
        "prompt": build_prompt(list(a[idx]), tgt, lang), "answer": f"{b[idx][0]}\n{b[idx][1]}",
        "source": list(a[idx]), "item_key": f"{poem['id']}|s{idx}",
    }
    if idx == 0:
        # Matla: record the rule-based form, but only if every script agrees there is a radif
        # (Urdu writes the suffix گے as a separate word where Hindi joins it).
        rads = {s: radif(*poem["text"][s][:2], s) for s in sc}
        if all(rads.values()):
            e["form"] = {"radif": rads[tgt], "qafiya": qafiya_words(*poem["text"][tgt][:2], tgt)}
    return e
