---
base_model: Qwen/Qwen2.5-14B-Instruct
library_name: peft
license: apache-2.0
language: [ur, hi]
tags: [lora, transliteration, urdu, hindustani, ghazal, rekhta, nastaliq, devanagari]
---

# zameen · qwen2.5-14b-medium

A LoRA adapter for **Qwen/Qwen2.5-14B-Instruct** that transliterates Urdu poetry (ghazals) between **Nastaliq,
Devanagari and Roman**, in all six directions, following Rekhta's orthography.

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-14B-Instruct` |
| Training examples | 10,000 (one sher, one direction each) |
| Trainable parameters | 68.8 M (LoRA rank 16, alpha 32) |
| Best validation step | 624 |
| Hugging Face | _not published yet (placeholder)_ |
| Trained adapter | `Colab disk (/content/ghazal_tuning/qwen2.5-14b-medium/adapter), not Drive; re-create with zameen run` |

## Results

Characters wrong after tuning (CER, lower is better) and perfectly transliterated shers, on poems
never seen in training. "New poets" are Jigar Moradabadi and Waseem Barelvi, held out entirely.

| Direction | CER, familiar poets | CER, new poets | Exact shers, new poets |
|---|---|---|---|
| Nastaliq → Devanagari | 3.6% | 2.6% | 33.0% |
| Nastaliq → Roman | 4.8% | 3.2% | 25.5% |
| Devanagari → Nastaliq | 1.2% | 1.0% | 63.5% |
| Devanagari → Roman | 1.1% | 0.8% | 63.0% |
| Roman → Nastaliq | 1.4% | 1.1% | 59.5% |
| Roman → Devanagari | 0.5% | 0.3% | 84.5% |
| all directions | 2.1% | 1.5% | 54.8% |

Form kept on matlas (rule-based check), familiar / new poets: radif 96.7% /
97.9%, qafiya 80.7% /
88.1%.

## Use

```bash
pip install -e '.[train]'        # in the zameen repository
zameen transliterate -c configs/qwen2.5-14b-medium.yaml --adapter models/qwen2.5-14b-medium/adapter --to devanagari \
  --text "آ کہ مری جان کو قرار نہیں ہے / طاقت بیداد انتظار نہیں ہے"
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct", device_map="auto"), "models/qwen2.5-14b-medium/adapter")
```

Prompt format (as in training): `Transliterate this sher into Devanagari. Keep every word as it is;
write only the transliterated sher, in two lines.` followed by a blank line and the two lines.

## Data

Trained on the public Rekhta sample ([amir9ume/urdu_ghazals_rekhta](https://github.com/amir9ume/urdu_ghazals_rekhta),
MIT; text from [Rekhta](https://www.rekhta.org)): 1,302 ghazals by 30 poets, line-aligned in three scripts.

## Limitations

Ghazals only; Rekhta's house orthography is the target, so other valid spellings count as errors;
part of the remaining error is inconsistency in the gold text itself.
