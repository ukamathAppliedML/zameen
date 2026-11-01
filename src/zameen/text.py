"""Script-aware text utilities and the rule-based radif / qafiya extractors.

Two normal forms are used throughout:

* ``strict``: Unicode NFC, whitespace collapsed, typographic quotes unified. Case is kept,
  because Rekhta's Roman marks retroflex consonants with capitals (DhūñDh, ujḌe).
* ``loose``: diacritics, nukta, macrons and punctuation removed. Used by the rule checks and by
  the loose error rate, so that "strict minus loose" isolates purely orthographic errors.
"""

from __future__ import annotations

import re
import unicodedata

SCRIPTS = ("nastaliq", "devanagari", "roman")
LABEL = {"nastaliq": "Nastaliq", "devanagari": "Devanagari", "roman": "Roman"}

_MARKS = re.compile("[ً-ٰٟँ़]")
_ROMAN_JOIN = re.compile(r"[.'’ʼ‘]")
_IZAFAT = re.compile(r"-(?:e|o|ए|ओ)-")


def tokens(text: str, script: str) -> list[str]:
    """Loose tokens: lower-cased, diacritics and punctuation removed."""
    s = unicodedata.normalize("NFC", str(text).lower())
    if script == "roman":
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
        s = _ROMAN_JOIN.sub("", s)
    s = _MARKS.sub("", s)
    s = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in s)
    return s.split()


def loose(text: str, script: str) -> str:
    return " ".join(tokens(text, script))


def strict(text: str) -> str:
    s = unicodedata.normalize("NFC", str(text)).replace("’", "'").replace("‘", "'")
    return "\n".join(" ".join(line.split()) for line in s.strip().splitlines() if line.strip())


def detect_script(text: str) -> str | None:
    """The dominant script of a string, or None if it has no letters."""
    counts = {
        "nastaliq": sum("؀" <= ch <= "ۿ" for ch in text),
        "devanagari": sum("ऀ" <= ch <= "ॿ" for ch in text),
        "roman": sum(ch.isalpha() and unicodedata.normalize("NFKD", ch)[0].isascii() for ch in text),
    }
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] else None


def radif(l1: str, l2: str, script: str) -> str:
    """The refrain: words shared at the end of both lines of a matla (compared loosely)."""
    w1, w2, shared = l1.split(), l2.split(), []
    for a, b in zip(reversed(w1), reversed(w2)):
        if not loose(a, script) or loose(a, script) != loose(b, script):
            break
        shared.insert(0, b)
    return " ".join(shared)


def qafiya_words(l1: str, l2: str, script: str) -> tuple[str, str] | None:
    """The rhyming words right before the radif, with izafat compounds reduced to their last part."""
    r = radif(l1, l2, script)
    if not r:
        return None
    n, w1, w2 = len(r.split()), l1.split(), l2.split()
    if len(w1) <= n or len(w2) <= n:
        return None
    q1, q2 = (_IZAFAT.split(w[len(w) - n - 1])[-1] for w in (w1, w2))
    if not loose(q1, script) or not loose(q2, script) or loose(q1, script) == loose(q2, script):
        return None
    return q1, q2


def strip_think(text: str) -> tuple[str, bool]:
    """Remove a reasoning block. Returns (answer, unfinished): unfinished if <think> never closed."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip(), False
    if "<think>" in text:
        return "", True
    return text.strip(), False


def shers(poem: dict, script: str) -> list[tuple[str, str]]:
    lines = poem["text"][script]
    return [(lines[i], lines[i + 1]) for i in range(0, len(lines) - 1, 2)]


def scripts_of(poem: dict) -> list[str]:
    return [s for s in SCRIPTS if poem["text"].get(s)]


def edit_distance(a, b) -> int:
    """Levenshtein distance over sequences (strings or word lists)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
