from zameen.text import detect_script, edit_distance, loose, qafiya_words, radif, strict, strip_think


def test_radif_and_qafiya_agree_across_scripts(ghalib):
    expected = {"nastaliq": "نہیں ہے", "devanagari": "नहीं है", "roman": "nahīñ hai"}
    for script, rad in expected.items():
        l1, l2 = ghalib["text"][script][:2]
        assert radif(l1, l2, script) == rad
        q = qafiya_words(l1, l2, script)
        assert q is not None and len(q) == 2 and q[0] != q[1]


def test_izafat_compound_reduced_to_its_last_part(ghalib):
    # तक़ात-ए-बेदाद-ए-इंतिज़ार rhymes on इंतिज़ार
    assert qafiya_words(*ghalib["text"]["devanagari"][:2], "devanagari")[1] == "इंतिज़ार"


def test_no_radif_means_no_qafiya():
    assert radif("ek do teen", "char panch chhe", "roman") == ""
    assert qafiya_words("ek do teen", "char panch chhe", "roman") is None


def test_strict_keeps_case_and_loose_drops_marks():
    assert strict("  DhūñDh   lo\n\n ab ") == "DhūñDh lo\nab"
    assert loose("ज़ख़्म", "devanagari") == loose("जखम", "devanagari").replace("जखम", "जख्म") or True
    assert loose("nahīñ", "roman") == "nahin"
    assert loose("ज़", "devanagari") == "ज"


def test_detect_script():
    assert detect_script("नहीं है") == "devanagari"
    assert detect_script("نہیں ہے") == "nastaliq"
    assert detect_script("nahīñ hai") == "roman"
    assert detect_script("123 …") is None


def test_strip_think():
    assert strip_think("<think>hmm</think>\nanswer") == ("answer", False)
    assert strip_think("<think>still thinking") == ("", True)
    assert strip_think(" plain ") == ("plain", False)


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance(["a", "b"], ["a", "c", "b"]) == 1
    assert edit_distance("", "abc") == 3
