# sarvam-30b-small: sarvamai/sarvam-30b

size **small** · 3,000 training examples · 10.1 M trainable LoRA parameters · best validation checkpoint at step 161 · NVIDIA A100-SXM4-80GB, bf16

Source: colab notebook (f875c965-ghazal_transliteration_sarvam_a100__3_.ipynb).

## Familiar poets (test_seen)

| Direction | CER | CER loose | WER | Exact shers |
|---|---|---|---|---|
| Nastaliq → Devanagari | 37.4% → **13.5%** | 34.5% → **12.2%** | 53.4% → **25.0%** | 0.0% → **6.2%** |
| Nastaliq → Roman | 62.7% → **17.2%** | 53.5% → **12.1%** | 89.5% → **40.0%** | 0.0% → **0.0%** |
| Devanagari → Nastaliq | 74.3% → **6.5%** | 71.1% → **6.4%** | 83.0% → **15.6%** | 0.0% → **15.0%** |
| Devanagari → Roman | 46.4% → **8.9%** | 36.9% → **4.8%** | 76.2% → **26.0%** | 0.0% → **0.0%** |
| Roman → Nastaliq | 35.6% → **8.0%** | 29.5% → **7.9%** | 55.3% → **18.4%** | 0.0% → **11.2%** |
| Roman → Devanagari | 36.0% → **8.5%** | 34.8% → **7.9%** | 53.6% → **18.9%** | 0.0% → **2.5%** |
| **all directions** | 48.8% → **10.6%** | 43.4% → **8.6%** | 68.5% → **23.8%** | 0.0% → **5.8%** |

Form kept (rule-based check on matlas): radif 41.4% → **86.7%** · qafiya 18.1% → **48.6%**

<details><summary>Word accuracy by phenomenon</summary>

| Target script | Words with | Before | After |
|---|---|---|---|
| Devanagari | nukta (ज़ ख़ क़ फ़ ग़) | 20.1% | 50.3% |
| Devanagari | aspirates (ख घ थ ध फ भ …) | 39.0% | 65.7% |
| Devanagari | nasalization (ँ ं) | 36.8% | 69.4% |
| Devanagari | conjuncts (्) | 26.6% | 43.2% |
| Devanagari | izafat (-ए- / -ओ-) | 8.9% | 22.3% |
| Roman | long vowels (ā ī ū) | 3.4% | 55.3% |
| Roman | nasalization (ñ) | 0.8% | 60.3% |
| Roman | aspirates (kh gh th dh ph bh …) | 14.4% | 57.7% |
| Roman | retroflex (T D R Ḍ Ṛ) | 0.0% | 15.5% |
| Roman | Perso-Arabic sounds (ḳh ġ q z f) | 7.2% | 36.0% |
| Roman | izafat (-e- / -o-) | 3.6% | 18.8% |
| Nastaliq | aspirates (ھ) | 33.5% | 82.6% |
| Nastaliq | noon ghunna (ں) | 35.3% | 85.0% |
| Nastaliq | hamza / izafat (ء ئ ۂ ؤ) | 16.4% | 61.7% |
| Nastaliq | spelling choice (ث ذ ص ض ط ظ ع ح) | 12.5% | 54.6% |
| Nastaliq | bari ye (ے) | 38.2% | 89.5% |

</details>

## Poets never seen in training (test_unseen)

| Direction | CER | CER loose | WER | Exact shers |
|---|---|---|---|---|
| Nastaliq → Devanagari | 34.3% → **8.8%** | 32.4% → **8.0%** | 47.6% → **16.8%** | 0.0% → **11.2%** |
| Nastaliq → Roman | 61.2% → **13.3%** | 52.7% → **9.2%** | 87.7% → **33.6%** | 0.0% → **1.2%** |
| Devanagari → Nastaliq | 79.3% → **4.6%** | 77.4% → **4.6%** | 87.7% → **11.0%** | 1.2% → **23.8%** |
| Devanagari → Roman | 52.5% → **7.7%** | 44.1% → **4.4%** | 80.1% → **22.1%** | 0.0% → **1.2%** |
| Roman → Nastaliq | 38.8% → **6.9%** | 34.6% → **6.8%** | 54.8% → **14.8%** | 0.0% → **12.5%** |
| Roman → Devanagari | 46.3% → **5.9%** | 46.1% → **5.4%** | 61.6% → **13.0%** | 0.0% → **16.2%** |
| **all directions** | 52.1% → **8.0%** | 47.9% → **6.4%** | 69.9% → **18.5%** | 0.2% → **11.0%** |

Form kept (rule-based check on matlas): radif 29.4% → **89.5%** · qafiya 20.2% → **58.3%**

<details><summary>Word accuracy by phenomenon</summary>

| Target script | Words with | Before | After |
|---|---|---|---|
| Devanagari | nukta (ज़ ख़ क़ फ़ ग़) | 22.8% | 58.1% |
| Devanagari | aspirates (ख घ थ ध फ भ …) | 38.8% | 75.3% |
| Devanagari | nasalization (ँ ं) | 43.9% | 82.1% |
| Devanagari | conjuncts (्) | 22.2% | 59.1% |
| Devanagari | izafat (-ए- / -ओ-) | 6.2% | 25.0% |
| Roman | long vowels (ā ī ū) | 3.4% | 61.2% |
| Roman | nasalization (ñ) | 0.7% | 73.3% |
| Roman | aspirates (kh gh th dh ph bh …) | 15.1% | 61.5% |
| Roman | retroflex (T D R Ḍ Ṛ) | 0.0% | 13.0% |
| Roman | Perso-Arabic sounds (ḳh ġ q z f) | 7.9% | 46.2% |
| Roman | izafat (-e- / -o-) | 1.6% | 28.1% |
| Nastaliq | aspirates (ھ) | 29.7% | 84.2% |
| Nastaliq | noon ghunna (ں) | 29.0% | 85.1% |
| Nastaliq | hamza / izafat (ء ئ ۂ ؤ) | 13.1% | 66.2% |
| Nastaliq | spelling choice (ث ذ ص ض ط ظ ع ح) | 16.7% | 66.1% |
| Nastaliq | bari ye (ے) | 33.1% | 89.6% |

</details>

## Validation curve

| Examples seen | Validation CER | Validation exact |
|---|---|---|
| 0 | 46.4% | 0% |
| 368 | 16.3% | 3% |
| 736 | 11.6% | 3% |
| 1,104 | 9.7% | 7% |
| 1,472 | 10.0% | 7% |
| 1,840 | 9.2% | 10% |
| 2,208 | 9.3% | 7% |
| 2,576 | 8.9% | 12% |
| 2,944 | 9.1% | 10% |
| 3,008 | 9.0% | 10% |

## Time

| Stage | Minutes |
|---|---|
| corpus | 0.0 |
| datasets | 0.0 |
| load model | 12.1 |
| preflight | 1.3 |
| baseline eval | 29.2 |
| training | 99.3 |
| tuned eval | 8.6 |
| **total** | **150** |
