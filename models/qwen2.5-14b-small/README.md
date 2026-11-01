---
base_model: Qwen/Qwen2.5-14B-Instruct
library_name: peft
license: apache-2.0
language: [ur, hi]
tags: [lora, transliteration, urdu, hindustani, ghazal, rekhta, nastaliq, devanagari]
---

# zameen · qwen2.5-14b-small

A LoRA adapter for **Qwen/Qwen2.5-14B-Instruct** that transliterates Urdu poetry (ghazals) between **Nastaliq,
Devanagari and Roman**, in all six directions, following Rekhta's orthography.

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-14B-Instruct` |
| Training examples | 3,000 (one sher, one direction each) |
| Trainable parameters | 68.8 M (LoRA rank 16, alpha 32) |
| Best validation step | 188 |
| Hugging Face | _not published yet (placeholder)_ |
| Trained adapter | `Colab disk (/content/ghazal_tuning/qwen2.5-14b-small/adapter), not Drive; re-create with zameen run` |

## Results

Characters wrong after tuning (CER, lower is better) and perfectly transliterated shers, on poems
never seen in training. "New poets" are Jigar Moradabadi and Waseem Barelvi, held out entirely.

| Direction | CER, familiar poets | CER, new poets | Exact shers, new poets |
|---|---|---|---|
| Nastaliq → Devanagari | 7.2% | 4.7% | 15.0% |
| Nastaliq → Roman | 8.4% | 5.3% | 2.5% |
| Devanagari → Nastaliq | 2.4% | 2.2% | 35.0% |
| Devanagari → Roman | 3.1% | 1.9% | 35.0% |
| Roman → Nastaliq | 3.2% | 2.8% | 27.5% |
| Roman → Devanagari | 1.0% | 0.5% | 76.2% |
| all directions | 4.3% | 2.9% | 31.9% |

Form kept on matlas (rule-based check), familiar / new poets: radif 93.3% /
96.9%, qafiya 68.1% /
79.4%.

## Use

```bash
pip install -e '.[train]'        # in the zameen repository
zameen transliterate -c configs/qwen2.5-14b-small.yaml --adapter models/qwen2.5-14b-small/adapter --to devanagari \
  --text "آ کہ مری جان کو قرار نہیں ہے / طاقت بیداد انتظار نہیں ہے"
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct", device_map="auto"), "models/qwen2.5-14b-small/adapter")
```

Prompt format (as in training): `Transliterate this sher into Devanagari. Keep every word as it is;
write only the transliterated sher, in two lines.` followed by a blank line and the two lines.

## Data

Trained on the public Rekhta sample ([amir9ume/urdu_ghazals_rekhta](https://github.com/amir9ume/urdu_ghazals_rekhta),
MIT; text from [Rekhta](https://www.rekhta.org)): 1,302 ghazals by 30 poets, line-aligned in three scripts.

## Limitations

Ghazals only; Rekhta's house orthography is the target, so other valid spellings count as errors;
part of the remaining error is inconsistency in the gold text itself.
