---
base_model: sarvamai/sarvam-30b
library_name: peft
license: apache-2.0
language: [ur, hi]
tags: [lora, transliteration, urdu, hindustani, ghazal, rekhta, nastaliq, devanagari]
---

# zameen · sarvam-30b-small

A LoRA adapter for **sarvamai/sarvam-30b** that transliterates Urdu poetry (ghazals) between **Nastaliq,
Devanagari and Roman**, in all six directions, following Rekhta's orthography.

| | |
|---|---|
| Base model | `sarvamai/sarvam-30b` |
| Training examples | 3,000 (one sher, one direction each) |
| Trainable parameters | 10.1 M (LoRA rank 16, alpha 32) |
| Best validation step | 161 |
| Hugging Face | _not published yet (placeholder)_ |
| Trained adapter | `MyDrive/ghazal_tuning/sarvam-30b-small/adapter` |

## Results

Characters wrong after tuning (CER, lower is better) and perfectly transliterated shers, on poems
never seen in training. "New poets" are Jigar Moradabadi and Waseem Barelvi, held out entirely.

| Direction | CER, familiar poets | CER, new poets | Exact shers, new poets |
|---|---|---|---|
| Nastaliq → Devanagari | 13.5% | 8.8% | 11.2% |
| Nastaliq → Roman | 17.2% | 13.3% | 1.2% |
| Devanagari → Nastaliq | 6.5% | 4.6% | 23.8% |
| Devanagari → Roman | 8.9% | 7.7% | 1.2% |
| Roman → Nastaliq | 8.0% | 6.9% | 12.5% |
| Roman → Devanagari | 8.5% | 5.9% | 16.2% |
| all directions | 10.6% | 8.0% | 11.0% |

Form kept on matlas (rule-based check), familiar / new poets: radif 86.7% /
89.5%, qafiya 48.6% /
58.3%.

## Use

```bash
pip install -e '.[train]'        # in the zameen repository
zameen transliterate -c configs/sarvam-30b-small.yaml --adapter models/sarvam-30b-small/adapter --to devanagari \
  --text "آ کہ مری جان کو قرار نہیں ہے / طاقت بیداد انتظار نہیں ہے"
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("sarvamai/sarvam-30b")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-30b", device_map="auto"), "models/sarvam-30b-small/adapter")
```

Prompt format (as in training): `Transliterate this sher into Devanagari. Keep every word as it is;
write only the transliterated sher, in two lines.` followed by a blank line and the two lines.

## Data

Trained on the public Rekhta sample ([amir9ume/urdu_ghazals_rekhta](https://github.com/amir9ume/urdu_ghazals_rekhta),
MIT; text from [Rekhta](https://www.rekhta.org)): 1,302 ghazals by 30 poets, line-aligned in three scripts.

## Limitations

Ghazals only; Rekhta's house orthography is the target, so other valid spellings count as errors;
part of the remaining error is inconsistency in the gold text itself.
