# qwen2.5-14b-medium: Qwen/Qwen2.5-14B-Instruct

size **medium** · 10,000 training examples · 68.8 M trainable LoRA parameters · best validation checkpoint at step 624 · NVIDIA A100-SXM4-80GB, bf16

Source: colab notebook (f2147a43-ghazal_transliteration_a100__2___1_.ipynb).

## Familiar poets (test_seen)

| Direction | CER | CER loose | WER | Exact shers |
|---|---|---|---|---|
| Nastaliq → Devanagari | 32.7% → **3.6%** | 30.1% → **3.1%** | 62.6% → **11.1%** | 0.0% → **27.0%** |
| Nastaliq → Roman | 32.4% → **4.8%** | 18.0% → **3.4%** | 79.4% → **14.8%** | 0.0% → **17.5%** |
| Devanagari → Nastaliq | 26.5% → **1.2%** | 25.4% → **1.2%** | 46.1% → **3.8%** | 0.0% → **57.7%** |
| Devanagari → Roman | 21.1% → **1.1%** | 8.2% → **0.5%** | 60.1% → **4.2%** | 0.0% → **53.1%** |
| Roman → Nastaliq | 26.2% → **1.4%** | 25.5% → **1.3%** | 47.1% → **4.2%** | 0.5% → **57.5%** |
| Roman → Devanagari | 8.2% → **0.5%** | 6.2% → **0.5%** | 22.4% → **1.9%** | 5.1% → **76.0%** |
| **all directions** | 24.6% → **2.1%** | 18.6% → **1.7%** | 52.9% → **6.6%** | 0.9% → **48.1%** |

Form kept (rule-based check on matlas): radif 56.7% → **96.7%** · qafiya 33.0% → **80.7%**

<details><summary>Word accuracy by phenomenon</summary>

| Target script | Words with | Before | After |
|---|---|---|---|
| Devanagari | nukta (ज़ ख़ क़ फ़ ग़) | 8.2% | 82.2% |
| Devanagari | aspirates (ख घ थ ध फ भ …) | 44.9% | 90.8% |
| Devanagari | nasalization (ँ ं) | 52.1% | 91.2% |
| Devanagari | conjuncts (्) | 24.4% | 82.3% |
| Devanagari | izafat (-ए- / -ओ-) | 8.9% | 69.6% |
| Roman | long vowels (ā ī ū) | 4.7% | 88.1% |
| Roman | nasalization (ñ) | 1.0% | 88.8% |
| Roman | aspirates (kh gh th dh ph bh …) | 24.1% | 88.4% |
| Roman | retroflex (T D R Ḍ Ṛ) | 0.0% | 80.3% |
| Roman | Perso-Arabic sounds (ḳh ġ q z f) | 10.8% | 78.4% |
| Roman | izafat (-e- / -o-) | 3.5% | 62.4% |
| Nastaliq | aspirates (ھ) | 52.1% | 98.3% |
| Nastaliq | noon ghunna (ں) | 43.4% | 95.8% |
| Nastaliq | hamza / izafat (ء ئ ۂ ؤ) | 26.9% | 89.8% |
| Nastaliq | spelling choice (ث ذ ص ض ط ظ ع ح) | 21.8% | 84.2% |
| Nastaliq | bari ye (ے) | 66.1% | 97.2% |

</details>

## Poets never seen in training (test_unseen)

| Direction | CER | CER loose | WER | Exact shers |
|---|---|---|---|---|
| Nastaliq → Devanagari | 31.5% → **2.6%** | 29.2% → **2.1%** | 58.2% → **8.7%** | 0.0% → **33.0%** |
| Nastaliq → Roman | 29.9% → **3.2%** | 15.9% → **2.3%** | 78.1% → **10.6%** | 0.0% → **25.5%** |
| Devanagari → Nastaliq | 25.7% → **1.0%** | 24.9% → **1.0%** | 43.8% → **3.4%** | 1.0% → **63.5%** |
| Devanagari → Roman | 21.7% → **0.8%** | 8.8% → **0.4%** | 61.9% → **3.0%** | 0.0% → **63.0%** |
| Roman → Nastaliq | 23.7% → **1.1%** | 23.2% → **1.1%** | 42.1% → **4.0%** | 1.0% → **59.5%** |
| Roman → Devanagari | 7.8% → **0.3%** | 5.7% → **0.3%** | 21.9% → **1.2%** | 3.5% → **84.5%** |
| **all directions** | 23.5% → **1.5%** | 17.6% → **1.2%** | 50.9% → **5.1%** | 0.9% → **54.8%** |

Form kept (rule-based check on matlas): radif 55.6% → **97.9%** · qafiya 29.6% → **88.1%**

<details><summary>Word accuracy by phenomenon</summary>

| Target script | Words with | Before | After |
|---|---|---|---|
| Devanagari | nukta (ज़ ख़ क़ फ़ ग़) | 11.0% | 86.7% |
| Devanagari | aspirates (ख घ थ ध फ भ …) | 43.2% | 91.1% |
| Devanagari | nasalization (ँ ं) | 51.3% | 94.1% |
| Devanagari | conjuncts (्) | 29.7% | 84.1% |
| Devanagari | izafat (-ए- / -ओ-) | 5.6% | 70.4% |
| Roman | long vowels (ā ī ū) | 3.1% | 92.1% |
| Roman | nasalization (ñ) | 0.4% | 91.9% |
| Roman | aspirates (kh gh th dh ph bh …) | 25.8% | 90.7% |
| Roman | retroflex (T D R Ḍ Ṛ) | 0.7% | 82.8% |
| Roman | Perso-Arabic sounds (ḳh ġ q z f) | 12.5% | 83.0% |
| Roman | izafat (-e- / -o-) | 4.3% | 71.0% |
| Nastaliq | aspirates (ھ) | 53.1% | 98.8% |
| Nastaliq | noon ghunna (ں) | 54.0% | 96.5% |
| Nastaliq | hamza / izafat (ء ئ ۂ ؤ) | 32.2% | 82.2% |
| Nastaliq | spelling choice (ث ذ ص ض ط ظ ع ح) | 33.4% | 87.0% |
| Nastaliq | bari ye (ے) | 63.5% | 94.0% |

</details>

## Validation curve

| Examples seen | Validation CER | Validation exact |
|---|---|---|
| 0 | 23.5% | 0% |
| 1,248 | 5.1% | 20% |
| 2,496 | 3.3% | 30% |
| 3,744 | 2.7% | 41% |
| 4,992 | 2.6% | 39% |
| 6,240 | 1.8% | 51% |
| 7,488 | 1.8% | 52% |
| 8,736 | 1.8% | 53% |
| 9,984 | 1.8% | 53% |
| 10,000 | 1.8% | 53% |

## Time

| Stage | Minutes |
|---|---|
| corpus | 0.0 |
| datasets | 0.0 |
| load model | 0.2 |
| preflight | 0.3 |
| baseline eval | 20.7 |
| training | 43.2 |
| tuned eval | 23.4 |
| **total** | **88** |
