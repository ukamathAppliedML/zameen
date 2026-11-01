"""A deterministic stand-in for a model, for tests and for ``zameen smoke``.

It knows every gold answer and gets a share of them right that grows as it "trains"; the rest are
orthographic slips, dropped words or echoes of the source. That is enough to exercise every stage
of the pipeline (evaluation, resume, training curve, best-checkpoint selection, reports) on a
laptop in seconds. Its numbers mean nothing about any real model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import Backend, TrainResult

BASE_SKILL, TUNED_SKILL = 0.15, 0.8


class OracleBackend(Backend):
    name = "oracle"

    def __init__(self, examples: list[dict], adapter: str | None = None):
        self.by_prompt = {e["prompt"]: e for e in examples}
        self.skill = BASE_SKILL
        self.gen_batch = 32
        if adapter:
            self.skill = json.loads((Path(adapter) / "oracle_adapter.json").read_text())["skill"]

    def add_examples(self, examples: list[dict]) -> None:
        self.by_prompt.update({e["prompt"]: e for e in examples})

    def render(self, prompt: str) -> str:
        return prompt

    def _answer(self, prompt: str) -> str:
        if "capital of France" in prompt:
            return "Paris"
        e = self.by_prompt.get(prompt)
        if e is None:
            return prompt.splitlines()[-1]
        h = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 1000 / 1000
        skill = max(self.skill, 0.9) if (e["src"], e["tgt"]) == ("roman", "devanagari") else self.skill
        if h < skill:
            return e["answer"]
        if h < skill + 0.15:   # orthographic slip only
            return e["answer"].replace("़", "").replace("ā", "a").replace("ñ", "n").replace("ھ", "ہ")
        if h < skill + 0.25:   # drops the last word of each line
            return "\n".join(" ".join(l.split()[:-1]) for l in e["answer"].splitlines())
        return "\n".join(e["source"])   # echoes the source script

    def generate(self, prompts, max_new_tokens, desc="generating", show=True):
        return [self._answer(p) for p in prompts]

    def tokenizer_stats(self, lines_by_script):
        return {s: {"tokens/word": 1.0, "round-trip": 1.0} for s in lines_by_script}

    def fit(self, train, probe, cfg, out_dir, log) -> TrainResult:
        out_dir = Path(out_dir)
        eff_batch = 16
        total = max(1, -(-len(train) // eff_batch) * cfg.preset["epochs"])
        every = max(1, total // cfg.probes_per_run)
        res = TrainResult(train_examples=len(train), trainable_params=0)
        best = float("inf")
        for step in range(total + 1):
            self.skill = BASE_SKILL + (TUNED_SKILL - BASE_SKILL) * step / total
            if step % every == 0 or step == total:
                cer, exact, by_src = probe()
                res.curve.append({"step": step, "examples_seen": step * eff_batch, "val_CER": cer,
                                  "val_exact": exact, **{f"from {k}": v for k, v in by_src.items()}})
                if cer < best:
                    best, res.best_step = cer, step
                log(f"[val] step {step}/{total} · CER {cer:.1%} · exact {exact:.0%}")
            if step:
                res.loss_log.append({"step": step, "loss": 2.0 / step ** 0.3})
        adapter = out_dir / "adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "oracle_adapter.json").write_text(json.dumps({"skill": self.skill}))
        return res
