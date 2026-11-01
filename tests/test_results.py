"""The committed results are complete and the comparison builds from them."""

from pathlib import Path

import pytest

from zameen.report import compare, load_summary

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["qwen2.5-14b-small", "qwen2.5-14b-medium", "sarvam-30b-small", "sarvam-30b-medium"]


@pytest.mark.parametrize("run", RUNS)
def test_summary_complete(run):
    s = load_summary(ROOT / "results" / run)
    for split in ("test_seen", "test_unseen"):
        d = s["splits"][split]["directions"]
        assert len(d) == 7
        assert all(v["before"]["CER"] is not None and v["after"]["CER"] is not None for v in d.values())
        assert s["splits"][split]["form"]["radif_kept"]["after"] > 0.8
        assert len(s["splits"][split]["phenomena"]) == 3
    assert len(s["validation_curve"]) >= 9
    assert s["training"]["train_examples"] in (3000, 10000)


def test_headline_numbers():
    u = lambda r: load_summary(ROOT / "results" / r)["splits"]["test_unseen"]["directions"]["all directions"]["after"]
    assert round(u("qwen2.5-14b-medium")["CER"], 3) == 0.015
    assert round(u("sarvam-30b-medium")["CER"], 3) == 0.050


def test_compare(tmp_path):
    out = compare([load_summary(ROOT / "results" / r) for r in RUNS], tmp_path)
    text = out.read_text()
    assert all(r in text for r in RUNS) and (tmp_path / "comparison_curves.png").exists()
