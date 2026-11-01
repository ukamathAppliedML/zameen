"""LoRA fine-tuning with a validation curve, best-checkpoint selection and resume.

* Loss on answer tokens only; prompts rendered exactly as at evaluation.
* LoRA on attention and dense / shared-expert MLP projections, never on routed experts or the
  router (adapting experts multiplies cost and can destabilize routing).
* Every ``total_steps / probes_per_run`` steps the validation probe is scored. The adapter with the
  lowest validation CER is kept and is what gets evaluated on test.
* Checkpoints go to ``<run>/trainer`` at each validation point. Re-running after a crash or a
  disconnect resumes from the last one; once training has finished, re-running reuses the saved
  adapter instead of training again.
"""

from __future__ import annotations

import glob
import inspect
import json
import math
import shutil
import time
from pathlib import Path

from .backends import TrainResult
from .progress import fmt_dur
from .text import LABEL

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj", "query_key_value", "qkv_proj", "dense", "out_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def lora_targets(model) -> list[str]:
    import torch
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and (
        (n.rsplit(".", 1)[-1] in ATTN and "expert" not in n) or
        (n.rsplit(".", 1)[-1] in MLP and ".experts." not in n))]


def featurize(backend, examples, max_len):
    tok, feats, dropped = backend.tok, [], 0
    for e in examples:
        p_ids = tok(backend.render(e["prompt"]), add_special_tokens=False)["input_ids"]
        a_ids = tok(e["answer"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(a_ids) > max_len:
            dropped += 1
            continue
        feats.append({"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids})
    return feats, dropped


def make_collate(pad_id):
    import torch

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda x, v: x + [v] * (n - len(x))
        return {"input_ids": torch.tensor([pad(b["input_ids"], pad_id) for b in batch]),
                "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
                "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch])}
    return collate


def train_lora(backend, train, probe, cfg, out_dir, log) -> TrainResult:
    import pandas as pd
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import Trainer, TrainerCallback, TrainingArguments

    out_dir = Path(out_dir)
    adapter_dir, done_path = out_dir / "adapter", out_dir / "training_done.json"
    ckpt_dir, best_path, curve_path = out_dir / "trainer", out_dir / "best_adapter_state.pt", out_dir / "validation_curve.csv"
    spec = cfg.spec

    feats, dropped = featurize(backend, train, cfg.max_seq_len)
    log(f"{len(feats)} training sequences ({dropped} over {cfg.max_seq_len} tokens dropped)")

    model = backend.model
    targets = lora_targets(model)
    log(f"LoRA on {len(targets)} modules, e.g. {targets[:3]}")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=2 * cfg.lora_r, lora_dropout=cfg.lora_dropout,
                                             target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    backend.model = model
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.print_trainable_parameters()

    eff_batch = spec["train_batch"] * spec["grad_accum"]
    total_steps = math.ceil(len(feats) / eff_batch) * cfg.preset["epochs"]
    every = max(10, total_steps // cfg.probes_per_run)
    log(f"{total_steps} optimizer steps · effective batch {eff_batch} · validation every {every} steps")
    res = TrainResult(train_examples=len(feats), trainable_params=trainable)

    # ---------------------------------------------------------------- already trained?
    if done_path.exists() and adapter_dir.exists():
        from peft.utils import load_peft_weights
        set_peft_model_state_dict(model, load_peft_weights(str(adapter_dir)))
        done = json.loads(done_path.read_text(encoding="utf-8"))
        res.loss_log, res.best_step, res.reused = done["log_history"], done["best_step"], True
        if curve_path.exists():
            res.curve = pd.read_csv(curve_path).to_dict("records")
        log(f"✅ Training already finished in an earlier session: loaded {adapter_dir}")
        model.eval()
        model.config.use_cache = True
        return res

    ckpts = sorted(glob.glob(str(ckpt_dir / "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
    resume = ckpts[-1] if ckpts else None
    resume_step = int(resume.rsplit("-", 1)[-1]) if resume else 0
    if resume:
        log(f"↻ Resuming from {resume} (step {resume_step}/{total_steps})")

    class ValidationCurve(TrainerCallback):
        def __init__(self):
            self.curve, self.best, self.best_state = [], float("inf"), None
            self.t0, self.start = time.time(), resume_step
            if resume_step and curve_path.exists():
                self.curve = [r for r in pd.read_csv(curve_path).to_dict("records") if r["step"] <= resume_step]
                if self.curve:
                    self.best = min(r["val_CER"] for r in self.curve)
                if best_path.exists():
                    self.best_state = torch.load(best_path)

        def _probe(self, step):
            model.eval()
            cer, exact, by_src = probe()
            model.train()
            self.curve.append({"step": step, "examples_seen": step * eff_batch, "val_CER": cer, "val_exact": exact,
                               **{f"from {LABEL[k]}": v for k, v in by_src.items()}})
            mark = ""
            if cer < self.best:
                self.best, mark = cer, "★ best"
                self.best_state = {k: v.detach().to("cpu", copy=True)
                                   for k, v in get_peft_model_state_dict(model).items()}
                torch.save(self.best_state, best_path)
            done = step - self.start
            eta = f" · ETA {fmt_dur((time.time() - self.t0) / done * (total_steps - step))}" if done > 0 else ""
            src = " ".join(f"{LABEL[k][:4]}={v:.1%}" for k, v in by_src.items())
            log(f"[val] step {step}/{total_steps} · seen {step * eff_batch:,} · CER {cer:.1%} · "
                f"exact {exact:.0%} · from {src} {mark}{eta}")
            pd.DataFrame(self.curve).to_csv(curve_path, index=False)

        def on_train_begin(self, args, state, control, **kw):
            if not self.start:
                self._probe(0)

        def on_step_end(self, args, state, control, **kw):
            if state.global_step % every == 0:
                self._probe(state.global_step)

        def on_train_end(self, args, state, control, **kw):
            if not self.curve or self.curve[-1]["step"] != state.global_step:
                self._probe(state.global_step)

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                log(f"   step {state.global_step}/{total_steps} · loss {logs['loss']:.3f} · "
                    f"lr {logs.get('learning_rate', 0):.1e}")

    targs = dict(output_dir=str(ckpt_dir), num_train_epochs=cfg.preset["epochs"],
                 per_device_train_batch_size=spec["train_batch"], gradient_accumulation_steps=spec["grad_accum"],
                 learning_rate=cfg.learning_rate, warmup_steps=max(1, round(cfg.warmup_frac * total_steps)),
                 lr_scheduler_type="cosine", logging_steps=max(1, every // 4), report_to="none",
                 save_strategy="steps", save_steps=every, save_total_limit=1,
                 bf16=True, remove_unused_columns=False, seed=cfg.seed, disable_tqdm=False)
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)   # names differ across versions
    skipped = [k for k in targs if k not in accepted]
    if skipped:
        log(f"TrainingArguments: this transformers version doesn't take {skipped} (skipped)")
    targs = {k: v for k, v in targs.items() if k in accepted}

    cb = ValidationCurve()
    trainer = Trainer(model=model, train_dataset=feats, data_collator=make_collate(backend.tok.pad_token_id),
                      callbacks=[cb], args=TrainingArguments(**targs))
    trainer.train(resume_from_checkpoint=resume)

    set_peft_model_state_dict(model, cb.best_state)               # keep the best validation checkpoint
    best = min(cb.curve, key=lambda r: r["val_CER"])
    log(f"Using the best validation checkpoint: step {best['step']} (val CER {best['val_CER']:.1%})")
    model.save_pretrained(str(adapter_dir))
    backend.tok.save_pretrained(str(adapter_dir))
    res.curve, res.loss_log, res.best_step, res.resumed_from = cb.curve, trainer.state.log_history, int(best["step"]), resume_step
    done_path.write_text(json.dumps({"best_step": res.best_step, "log_history": res.loss_log}, default=str), encoding="utf-8")
    shutil.rmtree(ckpt_dir, ignore_errors=True)                    # resume checkpoints no longer needed
    model.eval()
    model.config.use_cache = True
    return res
