"""End to end on CPU with the oracle backend."""

import json

import pytest

from zameen.pipeline import evaluate_adapter, export_datasets, prepare, run
from zameen.progress import Progress


def test_full_run_writes_reports_and_passes_checks(cfg):
    s = run(cfg, backend_kind="oracle", show=False)
    out = cfg.out_dir
    for f in ("summary.json", "results.md", "predictions_base.jsonl", "predictions_tuned.jsonl",
              "validation_curve.csv", "learning_curve.png", "before_after.png", "examples.md", "config.yaml"):
        assert (out / f).exists(), f
    assert all(c["ok"] for c in s["checks"]), s["checks"]
    d = s["splits"]["test_unseen"]["directions"]["all directions"]
    assert d["after"]["CER"] < d["before"]["CER"]
    assert json.loads((out / "summary.json").read_text())["run"] == "t"


def test_evaluation_resumes(cfg):
    run(cfg, backend_kind="oracle", show=False)
    pred = cfg.out_dir / "predictions_tuned.jsonl"
    n = len(pred.read_text().splitlines())
    lines = pred.read_text().splitlines()[: n // 2]
    pred.write_text("\n".join(lines) + "\n")                     # simulate an interrupted evaluation
    run(cfg, backend_kind="oracle", show=False)
    assert len(pred.read_text().splitlines()) == n


def test_evaluate_saved_adapter(cfg):
    run(cfg, backend_kind="oracle", show=False)
    s = evaluate_adapter(cfg, str(cfg.out_dir / "adapter"), backend_kind="oracle", tag="again", show=False)
    assert (cfg.out_dir / "predictions_again.jsonl").exists()
    assert s["splits"]["test_unseen"]["directions"]["all directions"]["after"]["CER"] < 0.1


def test_export(cfg, tmp_path):
    _, data = prepare(cfg, Progress(echo=False))
    files = export_datasets(data, tmp_path / "data")
    assert {f.name for f in files} == {"train.jsonl", "val_probe.jsonl", "test_seen.jsonl", "test_unseen.jsonl", "splits.json"}


def test_placeholder_adapter_gives_a_clear_error(cfg, tmp_path):
    from zameen.backends import load_backend
    empty = tmp_path / "adapter"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="placeholder"):
        load_backend("hf", cfg, adapter=str(empty))


def test_transliterate_cli_with_oracle(cfg, capsys):
    from zameen.cli import main
    run(cfg, backend_kind="oracle", show=False)
    main(["transliterate", "--size", "smoke", "--model", "qwen2.5-7b", "--backend", "oracle",
          "--adapter", str(cfg.out_dir / "adapter"), "--to", "devanagari", "--text", "aa ki / jaan"])
    assert capsys.readouterr().out.strip()
