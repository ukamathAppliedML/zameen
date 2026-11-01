# qwen2.5-14b-small: Qwen/Qwen2.5-14B-Instruct

size **small** · 3,000 training examples · 68.8 M trainable LoRA parameters · best validation checkpoint at step 188 · NVIDIA A100-SXM4-80GB, bf16

Source: colab notebook (8d3f336b-ghazal_transliteration_a100__2_.ipynb).

## Familiar poets (test_seen)

| Direction | CER | CER loose | WER | Exact shers |
|---|---|---|---|---|
| Nastaliq → Devanagari | 32.1% → **7.2%** | 29.1% → **6.3%** | 62.3% → **18.0%** | 0.0% → **12.5%** |
| Nastaliq → Roman | 34.4% → **8.4%** | 20.1% → **5.9%** | 81.1% → **24.5%** | 0.0% → **3.8%** |
| Devanagari → Nastaliq | 24.7% → **2.4%** | 23.7% → **2.4%** | 44.9% → **7.9%** | 0.0% → **27.5%** |
| Devanagari → Roman | 21.2% → **3.1%** | 8.8% → **1.7%** | 59.7% → **10.8%** | 0.0% → **16.2%** |
| Roman → Nastaliq | 27.6% → **3.2%** | 27.0% → **3.2%** | 50.3% → **9.2%** | 0.0% → **26.2%** |
| Roman → Devanagari | 10.1% → **1.0%** | 7.9% → **0.8%** | 26.7% → **3.4%** | 2.5% → **57.5%** |
| **all directions** | 25.1% → **4.3%** | 19.1% → **3.4%** | 54.0% → **12.2%** | 0.4% → **24.0%** |

Form kept (rule-based check on matlas): radif 51.9% → **93.3%** · qafiya 34.3% → **68.1%**

<details><summary>Word accuracy by phenomenon</summary>

| Target script | Words with | Before | After |
|---|---|---|---|
| Devanagari | nukta (ज़ ख़ क़ फ़ ग़) | 7.2% | 70.7% |
| Devanagari | aspirates (ख घ थ ध फ भ …) | 39.0% | 83.7% |
| Devanagari | nasalization (ँ ं) | 47.9% | 85.6% |
| Devanagari | conjuncts (्) | 21.4% | 74.5% |
| Devanagari | izafat (-ए- / -ओ-) | 8.0% | 52.7% |
| Roman | long vowels (ā ī ū) | 3.3% | 76.1% |
| Roman | nasalization (ñ) | 0.8% | 76.2% |
| Roman | aspirates (kh gh th dh ph bh …) | 24.2% | 73.8% |
| Roman | retroflex (T D R Ḍ Ṛ) | 0.0% | 69.0% |
| Roman | Perso-Arabic sounds (ḳh ġ q z f) | 9.0% | 61.5% |
| Roman | izafat (-e- / -o-) | 3.6% | 43.8% |
| Nastaliq | aspirates (ھ) | 45.0% | 91.7% |
| Nastaliq | noon ghunna (ں) | 44.6% | 92.4% |
| Nastaliq | hamza / izafat (ء ئ ۂ ؤ) | 24.2% | 73.4% |
| Nastaliq | spelling choice (ث ذ ص ض ط ظ ع ح) | 22.4% | 72.4% |
| Nastaliq | bari ye (ے) | 64.3% | 93.6% |

</details>

## Poets never seen in training (test_unseen)

| Direction | CER | CER loose | WER | Exact shers |
|---|---|---|---|---|
| Nastaliq → Devanagari | 29.4% → **4.7%** | 27.0% → **4.0%** | 55.5% → **14.8%** | 0.0% → **15.0%** |
| Nastaliq → Roman | 29.6% → **5.3%** | 15.6% → **3.7%** | 77.1% → **17.4%** | 0.0% → **2.5%** |
| Devanagari → Nastaliq | 29.1% → **2.2%** | 28.2% → **2.2%** | 47.4% → **7.0%** | 0.0% → **35.0%** |
| Devanagari → Roman | 20.9% → **1.9%** | 8.1% → **0.8%** | 60.3% → **6.4%** | 0.0% → **35.0%** |
| Roman → Nastaliq | 23.0% → **2.8%** | 22.4% → **2.8%** | 41.9% → **8.5%** | 0.0% → **27.5%** |
| Roman → Devanagari | 7.8% → **0.5%** | 5.7% → **0.4%** | 21.8% → **1.9%** | 3.8% → **76.2%** |
| **all directions** | 23.3% → **2.9%** | 17.5% → **2.3%** | 50.6% → **9.3%** | 0.6% → **31.9%** |

Form kept (rule-based check on matlas): radif 60.5% → **96.9%** · qafiya 32.0% → **79.4%**

<details><summary>Word accuracy by phenomenon</summary>

| Target script | Words with | Before | After |
|---|---|---|---|
| Devanagari | nukta (ज़ ख़ क़ फ़ ग़) | 7.7% | 77.6% |
| Devanagari | aspirates (ख घ थ ध फ भ …) | 47.0% | 88.8% |
| Devanagari | nasalization (ँ ं) | 54.5% | 89.1% |
| Devanagari | conjuncts (्) | 26.7% | 80.1% |
| Devanagari | izafat (-ए- / -ओ-) | 6.2% | 59.4% |
| Roman | long vowels (ā ī ū) | 4.5% | 85.1% |
| Roman | nasalization (ñ) | 0.7% | 88.5% |
| Roman | aspirates (kh gh th dh ph bh …) | 26.6% | 83.3% |
| Roman | retroflex (T D R Ḍ Ṛ) | 1.9% | 70.4% |
| Roman | Perso-Arabic sounds (ḳh ġ q z f) | 12.1% | 74.2% |
| Roman | izafat (-e- / -o-) | 7.8% | 65.6% |
| Nastaliq | aspirates (ھ) | 44.1% | 93.2% |
| Nastaliq | noon ghunna (ں) | 56.5% | 93.1% |
| Nastaliq | hamza / izafat (ء ئ ۂ ؤ) | 32.3% | 76.2% |
| Nastaliq | spelling choice (ث ذ ص ض ط ظ ع ح) | 27.2% | 72.8% |
| Nastaliq | bari ye (ے) | 61.3% | 92.5% |

</details>

## Validation curve

| Examples seen | Validation CER | Validation exact |
|---|---|---|
| 0 | 27.2% | 0% |
| 368 | 9.0% | 0% |
| 736 | 6.3% | 7% |
| 1,104 | 5.0% | 12% |
| 1,472 | 4.0% | 22% |
| 1,840 | 3.4% | 32% |
| 2,208 | 3.1% | 32% |
| 2,576 | 3.1% | 35% |
| 2,944 | 3.0% | 38% |
| 3,008 | 2.9% | 38% |

## Time

| Stage | Minutes |
|---|---|
| corpus | 0.0 |
| datasets | 0.0 |
| load model | 1.7 |
| preflight | 0.4 |
| baseline eval | 8.1 |
| training | 17.0 |
| tuned eval | 9.4 |
| **total** | **37** |
