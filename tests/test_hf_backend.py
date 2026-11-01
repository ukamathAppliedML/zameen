"""The Hugging Face backend and LoRA training, against stand-in torch / transformers / peft."""

import fake_hf
import pytest

from zameen.pipeline import run
from zameen.splits import build_datasets
from zameen.tasks import DIRECTIONS, make_example


@pytest.fixture
def fake(monkeypatch, poems, cfg):
    model = fake_hf.install(monkeypatch)
    data = build_datasets(poems, cfg)
    g = next(p for p in poems if p["poet"] == "mirza-ghalib")
    extra = [make_example(g, s, t, 0) for s, t in DIRECTIONS]
    for e in data.train + data.probe + [x for v in data.test.values() for x in v] + extra:
        fake_hf.GOLD[e["prompt"]] = e["answer"]
    return model


def test_generate_drops_token_type_ids(fake, cfg):
    from zameen.backends.hf import HFBackend
    b = HFBackend(cfg)
    assert b.generate(["What is the capital of France? Answer with one word."], 8, show=False) == ["Paris"]
    assert all("token_type_ids" not in kw for kw in fake_hf.STATE["generate_kwargs"])


def test_think_prefill_rendering(fake, cfg):
    from zameen.backends.hf import HFBackend
    b = HFBackend(cfg)
    b.think_prefill = True
    assert b.render("x").endswith("<think>\n\n</think>\n\n")


def test_sarvam_rejects_transformers_5(fake, monkeypatch):
    import sys

    from zameen.backends.hf import check_versions
    from zameen.config import RunConfig
    monkeypatch.setattr(sys.modules["transformers"], "__version__", "5.2.0")
    assert check_versions(RunConfig(model="sarvam-30b"))
    assert not check_versions(RunConfig(model="qwen2.5-14b"))


def test_full_run_with_lora_targets_and_best_checkpoint(fake, cfg):
    s = run(cfg, backend_kind="hf", show=False)
    targets = fake_hf.STATE["lora_targets"]
    assert "model.layers.0.attention.dense" in targets and "model.layers.0.mlp.gate_proj" in targets
    assert not any("experts" in t or "router" in t for t in targets)
    assert (cfg.out_dir / "adapter" / "adapter_config.json").exists()
    assert (cfg.out_dir / "training_done.json").exists() and not (cfg.out_dir / "trainer").exists()
    d = s["splits"]["test_unseen"]["directions"]["all directions"]
    assert d["after"]["CER"] < d["before"]["CER"]


def test_training_resumes_after_a_disconnect(fake, cfg):
    long = cfg.__class__.from_yaml(None, **{**cfg.to_dict(), "n_train": 320})   # 20 steps, checkpoint every 10
    fake_hf.STATE["kill_at"] = 15
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        run(long, backend_kind="hf", show=False)
    assert [p.name for p in (long.out_dir / "trainer").glob("checkpoint-*")] == ["checkpoint-10"]
    assert (long.out_dir / "best_adapter_state.pt").exists()
    fake_hf.STATE["kill_at"] = 0
    fake_hf.MODEL.skill = 0.0
    s = run(long, backend_kind="hf", show=False)
    assert fake_hf.STATE["resumed_from"] == 10
    steps = [r["step"] for r in s["validation_curve"]]
    assert steps == sorted(steps) and steps[0] == 0 and steps[-1] == 20 and len(steps) == len(set(steps))
    assert (long.out_dir / "training_done.json").exists() and not (long.out_dir / "trainer").exists()


def test_finished_training_is_reused(fake, cfg):
    run(cfg, backend_kind="hf", show=False)
    fake_hf.STATE["resumed_from"] = None
    fake_hf.MODEL.skill = 0.0
    s = run(cfg, backend_kind="hf", show=False)
    assert fake_hf.STATE["resumed_from"] is None          # Trainer.train never called
    assert s["training"]["reused"]
