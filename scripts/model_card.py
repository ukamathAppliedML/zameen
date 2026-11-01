"""Write a Hugging Face model card (models/<run>/README.md) from results/<run>/summary.json.

    python scripts/model_card.py results/qwen2.5-14b-medium --hf-repo udaykamath/zameen-qwen2.5-14b-medium

Re-run after replacing a placeholder with a real adapter or a Hub link.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DIRS = ["Nastaliq → Devanagari", "Nastaliq → Roman", "Devanagari → Nastaliq",
        "Devanagari → Roman", "Roman → Nastaliq", "Roman → Devanagari", "all directions"]


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def card(s: dict, hf_repo: str | None, drive_path: str) -> str:
    run, model = s["run"], s["model"]
    tr = s.get("training") or {}
    u = s["splits"]["test_unseen"]["directions"]
    k = s["splits"]["test_seen"]["directions"]
    hub = f"[`{hf_repo}`](https://huggingface.co/{hf_repo})" if hf_repo else "_not published yet (placeholder)_"
    load_from = hf_repo or f"models/{run}/adapter"
    rows = "\n".join(
        f"| {d} | {pct(k.get(d, {}).get('after', {}).get('CER'))} | {pct(u.get(d, {}).get('after', {}).get('CER'))} | "
        f"{pct(u.get(d, {}).get('after', {}).get('exact'))} |" for d in DIRS)
    fu, fk = s["splits"]["test_unseen"]["form"], s["splits"]["test_seen"]["form"]
    return f"""---
base_model: {model}
library_name: peft
license: apache-2.0
language: [ur, hi]
tags: [lora, transliteration, urdu, hindustani, ghazal, rekhta, nastaliq, devanagari]
---

# zameen · {run}

A LoRA adapter for **{model}** that transliterates Urdu poetry (ghazals) between **Nastaliq,
Devanagari and Roman**, in all six directions, following Rekhta's orthography.

| | |
|---|---|
| Base model | `{model}` |
| Training examples | {tr.get('train_examples', '–'):,} (one sher, one direction each) |
| Trainable parameters | {(tr.get('trainable_params') or 0) / 1e6:.1f} M (LoRA rank 16, alpha 32) |
| Best validation step | {tr.get('best_val_step', '–')} |
| Hugging Face | {hub} |
| Trained adapter | `{drive_path}` |

## Results

Characters wrong after tuning (CER, lower is better) and perfectly transliterated shers, on poems
never seen in training. "New poets" are Jigar Moradabadi and Waseem Barelvi, held out entirely.

| Direction | CER, familiar poets | CER, new poets | Exact shers, new poets |
|---|---|---|---|
{rows}

Form kept on matlas (rule-based check), familiar / new poets: radif {pct(fk.get('radif_kept', {}).get('after'))} /
{pct(fu.get('radif_kept', {}).get('after'))}, qafiya {pct(fk.get('qafiya_kept', {}).get('after'))} /
{pct(fu.get('qafiya_kept', {}).get('after'))}.

## Use

```bash
pip install -e '.[train]'        # in the zameen repository
zameen transliterate -c configs/{run}.yaml --adapter {load_from} --to devanagari \\
  --text "آ کہ مری جان کو قرار نہیں ہے / طاقت بیداد انتظار نہیں ہے"
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("{model}")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("{model}", device_map="auto"), "{load_from}")
```

Prompt format (as in training): `Transliterate this sher into Devanagari. Keep every word as it is;
write only the transliterated sher, in two lines.` followed by a blank line and the two lines.

## Data

Trained on the public Rekhta sample ([amir9ume/urdu_ghazals_rekhta](https://github.com/amir9ume/urdu_ghazals_rekhta),
MIT; text from [Rekhta](https://www.rekhta.org)): 1,302 ghazals by 30 poets, line-aligned in three scripts.

## Limitations

Ghazals only; Rekhta's house orthography is the target, so other valid spellings count as errors;
part of the remaining error is inconsistency in the gold text itself.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--hf-repo")
    ap.add_argument("--out-root", default="models")
    a = ap.parse_args()
    s = json.loads((Path(a.results_dir) / "summary.json").read_text(encoding="utf-8"))
    out = Path(a.out_root) / s["run"]
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    (out / "adapter" / ".gitkeep").touch()
    on_drive = "drive" in json.dumps(s.get("environment", {})) or "sarvam" in s["run"]
    where = (f"MyDrive/ghazal_tuning/{s['run']}/adapter" if on_drive else
             f"Colab disk (/content/ghazal_tuning/{s['run']}/adapter), not Drive; re-create with zameen run")
    (out / "README.md").write_text(card(s, a.hf_repo, where), encoding="utf-8")
    print("wrote", out / "README.md")


if __name__ == "__main__":
    main()
