"""Hugging Face backend: loads a causal LM (optionally with a LoRA adapter) on a GPU.

Lessons from the Colab runs are built in:

* Sarvam-30B's custom model code needs transformers 4.x; under 5.x it generates garbage.
* Sarvam's tokenizer returns ``token_type_ids``, which ``generate()`` rejects: only input ids and
  the attention mask are passed on.
* Sarvam reasons before answering even when asked not to: an empty, closed ``<think>`` block can be
  pre-filled (decided in preflight).
* Colab's old torchao breaks recent peft; ``zameen doctor`` warns about it.
"""

from __future__ import annotations

import os

from . import Backend, TrainResult


def _require_gpu_stack():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as err:  # pragma: no cover - depends on the machine
        raise RuntimeError("The GPU backend needs the training extras: pip install -e '.[train]' "
                           "(or '.[sarvam]' for Sarvam-30B)") from err


def check_versions(cfg) -> list[str]:
    """Problems with the installed stack for this model (empty list = fine)."""
    import transformers
    problems = []
    major = int(transformers.__version__.split(".")[0])
    if cfg.model == "sarvam-30b" and major >= 5:
        problems.append(f"Sarvam-30B needs transformers 4.x (installed {transformers.__version__}); "
                        "install with: pip install -e '.[sarvam]'")
    try:
        from importlib.metadata import version
        ao = version("torchao")
        if tuple(int(x) for x in ao.split(".")[:2]) < (0, 16):
            problems.append(f"torchao {ao} conflicts with recent peft; run: pip uninstall -y torchao")
    except Exception:
        pass
    return problems


class HFBackend(Backend):
    name = "hf"

    def __init__(self, cfg, adapter: str | None = None):
        _require_gpu_stack()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU found. The hf backend needs one (A100 80 GB recommended).")
        for p in check_versions(cfg):
            raise RuntimeError(p)

        self.cfg, self.spec = cfg, cfg.spec
        self.gen_batch = self.spec["gen_batch"]
        props = torch.cuda.get_device_properties(0)
        self.gpu_name, self.gpu_gb = props.name, props.total_memory / 1e9
        q = cfg.quantization
        self.quant = q if q != "auto" else ("4bit" if self.gpu_gb < self.spec["bf16_min_gb"] else "none")

        self.tok = AutoTokenizer.from_pretrained(self.spec["id"], trust_remote_code=self.spec["remote_code"])
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        kw = {"device_map": "auto", "trust_remote_code": self.spec["remote_code"], "torch_dtype": torch.bfloat16}
        if self.quant == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["lm_head"])
        self.model = AutoModelForCausalLM.from_pretrained(self.spec["id"], **kw)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self._specials: list[str] | None = None

    # ------------------------------------------------------------------ generation
    def render(self, prompt: str) -> str:
        p = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, **self.spec["chat_kwargs"])
        if self.think_prefill:
            if p.rstrip().endswith("<think>"):
                p += "\n\n</think>\n\n"
            elif "</think>" not in p[-60:]:
                p += "<think>\n\n</think>\n\n"
        return p

    def _clean(self, text: str) -> str:
        if self._specials is None:   # keep <think> tags (the scorer needs them), drop pad/eos/etc.
            self._specials = [t for t in set(self.tok.all_special_tokens) if t and "think" not in t.lower()]
        for t in self._specials:
            text = text.replace(t, "")
        return text.strip()

    def generate(self, prompts, max_new_tokens, desc="generating", show=True):
        import torch
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            from ..progress import NullBar as tqdm

        texts = [self.render(p) for p in prompts]
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))   # longest first: OOM shows early
        outs = [""] * len(texts)
        bar = tqdm(total=len(texts), desc=desc, disable=not show, leave=False)
        with torch.no_grad():
            for s in range(0, len(order), self.gen_batch):
                idx = order[s:s + self.gen_batch]
                enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                               add_special_tokens=False, return_token_type_ids=False).to(self.model.device)
                enc = {k: enc[k] for k in ("input_ids", "attention_mask") if k in enc}
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          use_cache=self.use_cache, pad_token_id=self.tok.pad_token_id)
                new = out[:, enc["input_ids"].shape[1]:]
                for i, t in zip(idx, self.tok.batch_decode(new, skip_special_tokens=False)):
                    outs[i] = self._clean(t)
                bar.update(len(idx))
        bar.close()
        return outs

    def tokenizer_stats(self, lines_by_script):
        out = {}
        for script, lines in lines_by_script.items():
            n_tok = n_words = rt = 0
            for line in lines:
                ids = self.tok(line, add_special_tokens=False)["input_ids"]
                n_tok += len(ids)
                n_words += len(line.split())
                rt += self.tok.decode(ids, skip_special_tokens=True).strip() == line.strip()
            out[script] = {"tokens/word": round(n_tok / max(n_words, 1), 2), "round-trip": rt / max(len(lines), 1)}
        return out

    # ------------------------------------------------------------------ training
    def fit(self, train, probe, cfg, out_dir, log) -> TrainResult:
        from ..train import train_lora
        return train_lora(self, train, probe, cfg, out_dir, log)

    def environment(self):
        import torch
        import transformers
        try:
            import peft
            pv = peft.__version__
        except ImportError:  # pragma: no cover
            pv = None
        return {"backend": self.name, "gpu": self.gpu_name, "gpu_gb": round(self.gpu_gb), "quantization": self.quant,
                "torch": torch.__version__, "transformers": transformers.__version__, "peft": pv,
                "model_id": self.spec["id"]}
