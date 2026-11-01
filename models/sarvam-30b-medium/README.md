---
base_model: sarvamai/sarvam-30b
library_name: peft
license: apache-2.0
language: [ur, hi]
tags: [lora, transliteration, urdu, hindustani, ghazal, rekhta, nastaliq, devanagari]
---

# zameen · sarvam-30b-medium

A LoRA adapter for **sarvamai/sarvam-30b** that transliterates Urdu poetry (ghazals) between **Nastaliq,
Devanagari and Roman**, in all six directions, following Rekhta's orthography.

| | |
|---|---|
| Base model | `sarvamai/sarvam-30b` |
| Training examples | 10,000 (one sher, one direction each) |
| Trainable parameters | 10.1 M (LoRA rank 16, alpha 32) |
| Best validation step | 390 |
| Hugging Face | _not published yet (placeholder)_ |
| Trained adapter | `MyDrive/ghazal_tuning/sarvam-30b-medium/adapter` |

## Results

Characters wrong after tuning (CER, lower is better) and perfectly transliterated shers, on poems
never seen in training. "New poets" are Jigar Moradabadi and Waseem Barelvi, held out entirely.

| Direction | CER, familiar poets | CER, new poets | Exact shers, new poets |
|---|---|---|---|
| Nastaliq → Devanagari | 8.1% | 6.4% | 19.5% |
| Nastaliq → Roman | 10.1% | 8.8% | 5.0% |
| Devanagari → Nastaliq | 4.4% | 2.8% | 35.5% |
| Devanagari → Roman | 5.8% | 4.7% | 11.5% |
| Roman → Nastaliq | 5.0% | 3.4% | 30.0% |
| Roman → Devanagari | 4.6% | 3.3% | 29.0% |
| all directions | 6.5% | 5.0% | 21.8% |

Form kept on matlas (rule-based check), familiar / new poets: radif 89.6% /
93.0%, qafiya 56.3% /
68.2%.

## Use

```bash
pip install -e '.[train]'        # in the zameen repository
zameen transliterate -c configs/sarvam-30b-medium.yaml --adapter models/sarvam-30b-medium/adapter --to devanagari \
  --text "آ کہ مری جان کو قرار نہیں ہے / طاقت بیداد انتظار نہیں ہے"
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("sarvamai/sarvam-30b")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-30b", device_map="auto"), "models/sarvam-30b-medium/adapter")
```

Prompt format (as in training): `Transliterate this sher into Devanagari. Keep every word as it is;
write only the transliterated sher, in two lines.` followed by a blank line and the two lines.

## Data

Trained on the public Rekhta sample ([amir9ume/urdu_ghazals_rekhta](https://github.com/amir9ume/urdu_ghazals_rekhta),
MIT; text from [Rekhta](https://www.rekhta.org)): 1,302 ghazals by 30 poets, line-aligned in three scripts.

## Limitations

Ghazals only; Rekhta's house orthography is the target, so other valid spellings count as errors;
part of the remaining error is inconsistency in the gold text itself.
