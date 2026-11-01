from zameen.splits import TEST_SPLITS, assign_splits, build_datasets
from zameen.tasks import DIRECTIONS, make_example


def test_six_directions():
    assert len(DIRECTIONS) == 6 and len(set(DIRECTIONS)) == 6


def test_example_contents(ghalib):
    e = make_example(ghalib, "nastaliq", "devanagari", 0, "roman")
    assert e["prompt"].startswith("Is sher ko Devanagari mein likhiye")
    assert e["prompt"].endswith("\n".join(ghalib["text"]["nastaliq"][:2]))
    assert e["answer"] == "\n".join(ghalib["text"]["devanagari"][:2])
    assert e["form"]["radif"] == "नहीं है"
    assert make_example(ghalib, "nastaliq", "devanagari", 999) is None


def test_matla_only_gets_form(ghalib):
    assert "form" not in make_example(ghalib, "roman", "nastaliq", 1)


def test_holdout_poets_only_in_unseen(poems, cfg):
    sp = assign_splits(poems, cfg)
    assert {p["poet"] for p in sp["test_unseen"]} == set(cfg.holdout_poets)
    assert not any(p["poet"] in cfg.holdout_poets for s in ("train", "val", "test_seen") for p in sp[s])


def test_splits_stable_when_poems_are_added(poems, cfg):
    before = {p["id"]: s for s, ps in assign_splits(poems[:12], cfg).items() for p in ps}
    after = {p["id"]: s for s, ps in assign_splits(poems, cfg).items() for p in ps}
    assert all(after[k] == v for k, v in before.items())


def test_datasets_disjoint_and_sized(poems, cfg):
    d = build_datasets(poems, cfg)
    train_poems = {e["poem_id"] for e in d.train}
    assert len(d.train) == cfg.preset["n_train"]
    for s in TEST_SPLITS:
        assert d.test[s] and not train_poems & {e["poem_id"] for e in d.test[s]}
        assert all(e["lang"] == "en" for e in d.test[s])
    assert {e["lang"] for e in d.train} <= set(cfg.lang_weights)


def test_datasets_deterministic(poems, cfg):
    a, b = build_datasets(poems, cfg), build_datasets(poems, cfg)
    assert [e["id"] for e in a.train] == [e["id"] for e in b.train]
