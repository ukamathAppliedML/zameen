from pathlib import Path

import pytest

from zameen.config import RunConfig
from zameen.corpus import load_jsonl

MINI = Path(__file__).resolve().parents[1] / "src" / "zameen" / "data" / "mini_corpus.jsonl"


@pytest.fixture
def poems():
    return load_jsonl(MINI)


@pytest.fixture
def ghalib(poems):
    return next(p for p in poems if p["poet"] == "mirza-ghalib")


@pytest.fixture
def cfg(tmp_path):
    return RunConfig.from_yaml(None, model="qwen2.5-7b", size="smoke", corpus=str(MINI), output_dir=str(tmp_path),
                               run_name="t", test_frac=0.3, val_frac=0.15)
