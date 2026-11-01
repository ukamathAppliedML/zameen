from zameen.metrics import extract, mark, score
from zameen.tasks import make_example


def ex(ghalib, src="nastaliq", tgt="devanagari"):
    return make_example(ghalib, src, tgt, 0)


def test_perfect_answer(ghalib):
    e = ex(ghalib)
    r = score(e, e["answer"])
    assert r["exact"] and r["char_err"] == 0 and r["radif_kept"] and r["qafiya_kept"] and not r["wrong_script"]


def test_nukta_only_error_is_strict_not_loose(ghalib):
    e = ex(ghalib)
    r = score(e, e["answer"].replace("़", ""))
    assert r["char_err"] > 0 and r["char_err_loose"] == 0 and not r["exact"]


def test_echo_is_wrong_script_not_no_answer(ghalib):
    e = ex(ghalib)
    r = score(e, "\n".join(e["source"]))
    assert not r["no_answer"] and r["wrong_script"] and r["char_err"] > 0


def test_reasoning_labels_quotes_and_echo_are_stripped(ghalib):
    e = ex(ghalib)
    raw = "<think>let me see</think>\n" + "\n".join(e["source"]) + "\n**Transliteration:** “" + e["answer"].replace("\n", "”\n“") + "”"
    assert extract(raw, e) == e["answer"]


def test_unfinished_reasoning_is_no_answer(ghalib):
    assert score(ex(ghalib), "<think>going on and on")["no_answer"]


def test_single_line_with_slash(ghalib):
    e = ex(ghalib)
    assert extract(e["answer"].replace("\n", " / "), e) == e["answer"]


def test_phenomena_counted(ghalib):
    e = ex(ghalib)
    r = score(e, e["answer"])
    hit, tot = r["phen"]["nukta (ज़ ख़ क़ फ़ ग़)"]
    assert tot > 0 and hit == tot


def test_mark():
    assert mark("a b c\nd e", "a x c\nd e") == "a ⟦x⟧ c / d e"
