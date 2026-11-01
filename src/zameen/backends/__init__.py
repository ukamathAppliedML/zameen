"""Model backends.

A backend turns prompts into raw model outputs and can be fine-tuned. The pipeline only talks to
this interface, so the same code runs a 30B model on an A100 or the deterministic ``oracle``
backend on a laptop (used by the test suite and ``zameen smoke``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class TrainResult:
    curve: list[dict] = field(default_factory=list)       # validation points along training
    loss_log: list[dict] = field(default_factory=list)    # {"step", "loss", ...}
    best_step: int = 0
    train_examples: int = 0
    trainable_params: int | None = None
    resumed_from: int = 0
    reused: bool = False                                   # finished in an earlier session


ProbeFn = Callable[[], tuple[float, float, dict]]          # () -> (val CER, val exact, CER by source script)


class Backend:
    """Interface shared by all backends."""

    name = "backend"
    think_prefill: bool = False     # pre-fill an empty, closed think block (reasoning models)
    use_cache: bool = True          # KV cache during generation
    gen_batch: int = 16

    def render(self, prompt: str) -> str:
        raise NotImplementedError

    def generate(self, prompts: list[str], max_new_tokens: int, desc: str = "generating",
                 show: bool = True) -> list[str]:
        raise NotImplementedError

    def tokenizer_stats(self, lines_by_script: dict[str, list[str]]) -> dict | None:
        return None

    def fit(self, train: list[dict], probe: ProbeFn, cfg, out_dir, log) -> TrainResult:
        raise NotImplementedError

    def environment(self) -> dict:
        return {"backend": self.name}


def load_backend(kind: str, cfg, adapter: str | None = None, examples: list[dict] | None = None) -> Backend:
    if kind == "hf":
        from pathlib import Path
        if adapter and Path(adapter).is_dir() and not (Path(adapter) / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"{adapter} has no adapter_config.json: it is still a placeholder. Copy the adapter folder from "
                "Google Drive (MyDrive/ghazal_tuning/<run>/adapter) into it, or pass a Hugging Face repo id. "
                "See models/README.md.")
        from .hf import HFBackend
        return HFBackend(cfg, adapter=adapter)
    if kind == "oracle":
        from .oracle import OracleBackend
        return OracleBackend(examples or [], adapter=adapter)
    raise ValueError(f"unknown backend {kind!r} (hf or oracle)")
