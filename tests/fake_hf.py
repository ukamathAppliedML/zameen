"""Minimal stand-ins for torch, transformers and peft.

They let the GPU code paths (generation, think prefill, LoRA targeting, training with validation,
checkpoint resume) run in CI without a GPU or any downloads. They model the behaviors that broke
real runs: Sarvam's tokenizer emits token_type_ids that generate() rejects, and training can be
killed part-way and must resume.
"""

from __future__ import annotations

import contextlib
import json
import os
import pickle
import sys
import types

GOLD: dict[str, str] = {}          # user prompt -> gold answer (filled by the test)
STATE = {"generate_kwargs": [], "kill_at": 0, "resumed_from": None, "lora_targets": None}


class Linear:
    pass


class _Props:
    name, total_memory = "FAKE A100", 85e9


def make_torch():
    t = types.ModuleType("torch")
    t.__version__ = "fake"
    t.bfloat16 = "bf16"
    t.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_properties=lambda i: _Props(),
                                   get_device_name=lambda i: _Props.name)
    t.no_grad = contextlib.nullcontext
    t.nn = types.SimpleNamespace(Linear=Linear)
    t.tensor = lambda x: x
    t.save = lambda obj, p: pickle.dump(obj, open(p, "wb"))
    t.load = lambda p, **k: pickle.load(open(p, "rb"))
    return t


class Batch:
    def __init__(self, texts):
        self.texts, self.shape = texts, (len(texts), max(len(x) for x in texts))

    def __getitem__(self, k):
        return self


class Enc(dict):
    def to(self, device):
        return self


class FakeTok:
    eos_token, pad_token, pad_token_id, padding_side = "</s>", None, 0, "right"
    all_special_tokens = ["</s>", "<think>", "</think>", "<pad>"]

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "<user>" + msgs[0]["content"] + "<assistant>" + ("<think>" if kw.get("thinks") else "")

    def __call__(self, text, return_tensors=None, padding=False, add_special_tokens=False, **kw):
        if isinstance(text, list):   # like Sarvam: always returns token_type_ids
            return Enc(input_ids=Batch(text), attention_mask=1, token_type_ids=0)
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def batch_decode(self, out, skip_special_tokens=False):
        return out.answers

    def save_pretrained(self, p):
        os.makedirs(p, exist_ok=True)


class Out:
    def __init__(self, answers):
        self.answers = answers

    def __getitem__(self, k):
        return self


class FakeModel:
    device = "cuda"

    def __init__(self):
        self.config = types.SimpleNamespace(use_cache=True)
        self.skill = 0.0

    def named_modules(self):
        return [("model.layers.0.self_attn.q_proj", Linear()), ("model.layers.0.attention.dense", Linear()),
                ("model.layers.0.mlp.gate_proj", Linear()), ("model.layers.0.mlp.experts.3.up_proj", Linear()),
                ("model.layers.0.mlp.router", Linear())]

    def generate(self, input_ids=None, attention_mask=None, **kw):
        if "token_type_ids" in kw:
            raise ValueError("The following `model_kwargs` are not used by the model: ['token_type_ids']")
        STATE["generate_kwargs"].append(kw)
        outs = []
        for t in input_ids.texts:
            user = t.split("<user>", 1)[1].split("<assistant>", 1)[0]
            if "capital of France" in user:
                outs.append("Paris")
            elif user in GOLD:
                outs.append(GOLD[user] if (self.skill > 0.5 or "Devanagari" in user.splitlines()[0]) else user.splitlines()[-1])
            else:
                outs.append(user.splitlines()[-1])
        return Out(outs)

    # training API
    def gradient_checkpointing_enable(self, **kw): pass
    def enable_input_require_grads(self): pass
    def print_trainable_parameters(self): pass
    def parameters(self): return []
    def eval(self): return self
    def train(self): return self

    def save_pretrained(self, p):
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, "adapter_config.json"), "w") as f:
            json.dump({"skill": self.skill}, f)


MODEL = None


class W:
    """A stand-in weight tensor (module level so it pickles)."""

    def __init__(self, v):
        self.v = v

    def detach(self):
        return self

    def to(self, *a, **k):
        return W(self.v)


def make_transformers():
    tr = types.ModuleType("transformers")
    tr.__version__ = "4.57.6"
    tr.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTok())
    tr.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *a, **k: MODEL)
    tr.BitsAndBytesConfig = lambda **k: k

    class TrainingArguments:
        def __init__(self, output_dir, num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=1,
                     learning_rate=1e-4, warmup_steps=0, lr_scheduler_type="linear", logging_steps=10, report_to="none",
                     save_strategy="no", save_steps=500, save_total_limit=None, bf16=False,
                     remove_unused_columns=True, seed=0, disable_tqdm=False):
            self.__dict__.update({k: v for k, v in locals().items() if k != "self"})

    class TrainerCallback:
        pass

    class Trainer:
        def __init__(self, model, train_dataset, data_collator, callbacks, args):
            self.model, self.ds, self.collate, self.cbs, self.args = model, train_dataset, data_collator, callbacks, args
            self.state = types.SimpleNamespace(global_step=0, log_history=[])

        def train(self, resume_from_checkpoint=None):
            a = self.args
            eff = a.per_device_train_batch_size * a.gradient_accumulation_steps
            total = -(-len(self.ds) // eff) * a.num_train_epochs
            start = int(resume_from_checkpoint.rsplit("-", 1)[-1]) if resume_from_checkpoint else 0
            STATE["resumed_from"] = start
            b = self.collate(self.ds[:2])
            assert len(b["input_ids"][0]) == len(b["labels"][0]) == len(b["attention_mask"][0])
            for cb in self.cbs:
                cb.on_train_begin(a, self.state, None)
            for step in range(start + 1, total + 1):
                if STATE["kill_at"] and step == STATE["kill_at"]:
                    raise RuntimeError("simulated disconnect")
                self.model.skill = step / total
                self.state.global_step = step
                if step % a.logging_steps == 0:
                    logs = {"loss": 2.0 / step ** 0.3, "learning_rate": 1e-4, "step": step}
                    self.state.log_history.append(logs)
                    for cb in self.cbs:
                        cb.on_log(a, self.state, None, logs=logs)
                for cb in self.cbs:
                    cb.on_step_end(a, self.state, None)
                if a.save_strategy == "steps" and step % a.save_steps == 0:
                    import glob
                    import shutil
                    for old in glob.glob(os.path.join(a.output_dir, "checkpoint-*")):
                        shutil.rmtree(old)
                    os.makedirs(os.path.join(a.output_dir, f"checkpoint-{step}"))
            self.state.log_history.append({"train_loss": 0.9, "step": total})
            for cb in self.cbs:
                cb.on_train_end(a, self.state, None)

    tr.TrainingArguments, tr.TrainerCallback, tr.Trainer = TrainingArguments, TrainerCallback, Trainer
    return tr


def make_peft():
    pf = types.ModuleType("peft")
    pf.__version__ = "fake"
    pf.LoraConfig = lambda **k: k

    def get_peft_model(m, c):
        STATE["lora_targets"] = c["target_modules"]
        return m

    pf.get_peft_model = get_peft_model
    pf.get_peft_model_state_dict = lambda m: {"lora": W(m.skill)}
    pf.set_peft_model_state_dict = lambda m, sd: setattr(m, "skill", sd["lora"].v if isinstance(sd["lora"], W) else sd["lora"])
    pf.PeftModel = types.SimpleNamespace(from_pretrained=lambda m, p: (setattr(
        m, "skill", json.load(open(os.path.join(p, "adapter_config.json")))["skill"]), m)[1])
    utils = types.ModuleType("peft.utils")
    utils.load_peft_weights = lambda p: {"lora": json.load(open(os.path.join(p, "adapter_config.json")))["skill"]}
    pf.utils = utils
    return pf, utils


def install(monkeypatch):
    global MODEL
    MODEL = FakeModel()
    STATE.update(generate_kwargs=[], kill_at=0, resumed_from=None, lora_targets=None)
    pf, utils = make_peft()
    for name, mod in {"torch": make_torch(), "transformers": make_transformers(), "peft": pf, "peft.utils": utils}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return MODEL
