"""Run configuration: one YAML file per experiment, overridable from the command line."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Size presets. n_train counts examples (one sher in one direction each); eval_poems is per test
# split; probe_poems is the number of validation poems scored during training.
PRESETS: dict[str, dict[str, Any]] = {
    "small": {"n_train": 3000, "eval_poems": 40, "epochs": 1, "probe_poems": 10},
    "medium": {"n_train": 10000, "eval_poems": 100, "epochs": 1, "probe_poems": 15},
    "large": {"n_train": 30000, "eval_poems": None, "epochs": 1, "probe_poems": 20},
    "smoke": {"n_train": 60, "eval_poems": 3, "epochs": 1, "probe_poems": 2},
}

# Known models. bf16_min_gb: below this much GPU memory the model loads in 4-bit instead.
# chat_kwargs go to apply_chat_template (Sarvam: ask it not to reason before answering).
MODELS: dict[str, dict[str, Any]] = {
    "sarvam-30b": {"id": "sarvamai/sarvam-30b", "remote_code": True, "bf16_min_gb": 75,
                   "gen_batch": 16, "train_batch": 2, "grad_accum": 8,
                   "chat_kwargs": {"enable_thinking": False}},
    "qwen2.5-7b": {"id": "Qwen/Qwen2.5-7B-Instruct", "remote_code": False, "bf16_min_gb": 30,
                   "gen_batch": 32, "train_batch": 8, "grad_accum": 2, "chat_kwargs": {}},
    "qwen2.5-14b": {"id": "Qwen/Qwen2.5-14B-Instruct", "remote_code": False, "bf16_min_gb": 60,
                    "gen_batch": 24, "train_batch": 4, "grad_accum": 4, "chat_kwargs": {}},
    "custom": {"id": "", "remote_code": True, "bf16_min_gb": 60,
               "gen_batch": 16, "train_batch": 4, "grad_accum": 4, "chat_kwargs": {}},
}


@dataclass
class RunConfig:
    # what to run
    model: str = "qwen2.5-14b"
    model_id: str | None = None          # Hugging Face id; required for model: custom
    size: str = "small"
    run_name: str | None = None          # default: "<model>-<size>"

    # overrides of the size preset (None = preset value)
    n_train: int | None = None
    eval_poems: int | None = None
    epochs: int | None = None
    probe_poems: int | None = None

    # training
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_dropout: float = 0.05
    warmup_frac: float = 0.03
    max_seq_len: int = 512
    probes_per_run: int = 8                 # validation points along the learning curve
    gen_batch: int | None = None
    train_batch: int | None = None
    grad_accum: int | None = None
    quantization: str = "auto"              # auto | none | 4bit

    # generation
    answer_tokens: int = 200
    think_mode: str = "auto"                # auto | prefill | allow
    think_tokens: int = 768

    # data
    corpus: str = ""                        # "" = public Rekhta sample; or a JSONL export / Rekhta-style folder
    seed: int = 12345
    holdout_poets: list[str] = field(default_factory=lambda: ["jigar-moradabadi", "waseem-barelvi"])
    test_frac: float = 0.15
    val_frac: float = 0.05
    lang_weights: dict[str, float] = field(
        default_factory=lambda: {"en": 0.4, "nastaliq": 0.2, "devanagari": 0.2, "roman": 0.2})

    # where things go
    output_dir: str = "runs"
    cache_dir: str = "~/.cache/zameen"

    # ------------------------------------------------------------------ derived
    @property
    def name(self) -> str:
        return self.run_name or f"{self.model}-{self.size}"

    @property
    def out_dir(self) -> Path:
        return Path(self.output_dir).expanduser() / self.name

    @property
    def preset(self) -> dict[str, Any]:
        if self.size not in PRESETS:
            raise ValueError(f"size must be one of {sorted(PRESETS)}, got {self.size!r}")
        p = dict(PRESETS[self.size])
        for k in ("n_train", "eval_poems", "epochs", "probe_poems"):
            if getattr(self, k) is not None:
                p[k] = getattr(self, k)
        return p

    @property
    def spec(self) -> dict[str, Any]:
        """Model spec with this config's overrides applied."""
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {sorted(MODELS)}, got {self.model!r}")
        s = dict(MODELS[self.model])
        if self.model_id:
            s["id"] = self.model_id
        if not s["id"]:
            raise ValueError("model: custom needs model_id (a Hugging Face model id)")
        for k in ("gen_batch", "train_batch", "grad_accum"):
            if getattr(self, k) is not None:
                s[k] = getattr(self, k)
        return s

    # ------------------------------------------------------------------ io
    @classmethod
    def from_yaml(cls, path: str | Path | None = None, **overrides: Any) -> RunConfig:
        data: dict[str, Any] = {}
        if path:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        _ = self.preset, self.spec
        if self.think_mode not in ("auto", "prefill", "allow"):
            raise ValueError("think_mode must be auto, prefill or allow")
        if self.quantization not in ("auto", "none", "4bit"):
            raise ValueError("quantization must be auto, none or 4bit")
        if abs(sum(self.lang_weights.values()) - 1) > 1e-6:
            raise ValueError("lang_weights must sum to 1")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)
