from pathlib import Path

import pytest

from zameen.config import RunConfig

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_loads(path):
    cfg = RunConfig.from_yaml(path)
    assert cfg.spec["id"] and cfg.preset["n_train"]


def test_overrides_and_presets():
    cfg = RunConfig.from_yaml(None, model="sarvam-30b", size="medium", n_train=1234)
    assert cfg.preset == {"n_train": 1234, "eval_poems": 100, "epochs": 1, "probe_poems": 15}
    assert cfg.spec["chat_kwargs"] == {"enable_thinking": False}
    assert cfg.name == "sarvam-30b-medium"


def test_bad_values_rejected():
    with pytest.raises(ValueError):
        RunConfig.from_yaml(None, size="huge")
    with pytest.raises(ValueError):
        RunConfig.from_yaml(None, model="custom")
    with pytest.raises(ValueError):
        RunConfig.from_yaml(None, nonsense=1)


def test_roundtrip(tmp_path):
    cfg = RunConfig.from_yaml(None, model="qwen2.5-7b", lora_r=64)
    cfg.save(tmp_path / "c.yaml")
    assert RunConfig.from_yaml(tmp_path / "c.yaml") == cfg
