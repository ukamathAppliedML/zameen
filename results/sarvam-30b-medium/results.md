# sarvam-30b-medium: sarvamai/sarvam-30b

size **medium** · 10,000 training examples · 10.1 M trainable LoRA parameters · best validation checkpoint at step 390 · NVIDIA A100-SXM4-80GB, bf16

Source: colab notebook (6ac57826-ghazal_transliteration_sarvam_a100__3___1_.ipynb).

## Familiar poets (test_seen)

| Direction | CER | CER loose | WER | Exact shers |
|---|---|---|---|---|
| Nastaliq → Devanagari | 35.6% → **8.1%** | 33.6% → **7.1%** | 51.4% → **18.3%** | 0.0% → **12.2%** |
| Nastaliq → Roman | 61.4% → **10.1%** | 52.4% → **7.2%** | 87.9% → **25.7%** | 0.0% → **2.0%** |
| Devanagari → Nastaliq | 76.6% → **4.4%** | 73.9% → **4.3%** | 85.2% → **10.1%** | 1.0% → **23.5%** |
| Devanagari → Roman | 51.6% → **5.8%** | 43.1% → **3.6%** | 79.3% → **16.8%** | 0.0% → **6.6%** |
| Roman → Nastaliq | 41.0% → **5.0%** | 35.5% → **5.0%** | 59.1% → **11.9%** | 0.5% → **18.5%** |
| Roman → Devanagari | 40.1% → **4.6%** | 39.1% → **4.3%** | 56.9% → **11.1%** | 0.0% → **20.4%** |
| **all directions** | 51.1% → **6.5%** | 46.3% → **5.3%** | 70.0% → **15.6%** | 0.3% → **13.9%** |

Form kept (rule-based check on matlas): radif 37.0% → **89.6%** · qafiya 18.7% → **56.3%**

<details><summary>Word accuracy by phenomenon</summary>

| Target script | Words with | Before | After |
|---|---|---|---|
| Devanagari | nukta (ज़ ख़ क़ फ़ ग़) | 20.7% | 63.9% |
| Devanagari | aspirates (ख घ थ ध फ भ …) | 41.6% | 78.9% |
| Devanagari | nasalization (ँ ं) | 40.0% | 80.3% |
| Devanagari | conjuncts (्) | 25.0% | 60.3% |
| Devanagari | izafat (-ए- / -ओ-) | 7.1% | 39.3% |
| Roman | long vowels (ā ī ū) | 3.7% | 71.4% |
| Roman | nasalization (ñ) | 1.9% | 72.8% |
| Roman | aspirates (kh gh th dh ph bh …) | 14.1% | 75.0% |
| Roman | retroflex (T D R Ḍ Ṛ) | 0.0% | 38.1% |
| Roman | Perso-Arabic sounds (ḳh ġ q z f) | 7.9% | 57.5% |
| Roman | izafat (-e- / -o-) | 3.1% | 42.5% |
| Nastaliq | aspirates (ھ) | 31.0% | 89.8% |
| Nastaliq | noon ghunna (ں) | 27.1% | 89.1% |
| Nastaliq | hamza / izafat (ء ئ ۂ ؤ) | 20.4% | 72.4% |
| Nastaliq | spelling choice (ث ذ ص ض ط ظ ع ح) | 18.1% | 69.9% |
| Nastaliq | bari ye (ے) | 35.3% | 93.7% |

</details>

## Poets never seen in training (test_unseen)

| Direction | CER | CER loose | WER | Exact shers |
|---|---|---|---|---|
| Nastaliq → Devanagari | 35.7% → **6.4%** | 33.2% → **5.7%** | 49.7% → **14.4%** | 0.5% → **19.5%** |
| Nastaliq → Roman | 61.1% → **8.8%** | 52.7% → **6.4%** | 87.4% → **22.4%** | 0.0% → **5.0%** |
| Devanagari → Nastaliq | 75.4% → **2.8%** | 73.0% → **2.7%** | 83.7% → **7.0%** | 1.5% → **35.5%** |
| Devanagari → Roman | 53.8% → **4.7%** | 45.9% → **2.7%** | 81.4% → **13.8%** | 0.0% → **11.5%** |
| Roman → Nastaliq | 38.1% → **3.4%** | 34.6% → **3.3%** | 54.0% → **8.5%** | 1.0% → **30.0%** |
| Roman → Devanagari | 42.2% → **3.3%** | 41.6% → **3.1%** | 56.5% → **8.8%** | 0.0% → **29.0%** |
| **all directions** | 51.2% → **5.0%** | 46.9% → **4.1%** | 68.8% → **12.4%** | 0.5% → **21.8%** |

Form kept (rule-based check on matlas): radif 30.7% → **93.0%** · qafiya 18.9% → **68.2%**

<details><summary>Word accuracy by phenomenon</summary>

| Target script | Words with | Before | After |
|---|---|---|---|
| Devanagari | nukta (ज़ ख़ क़ फ़ ग़) | 22.6% | 70.0% |
| Devanagari | aspirates (ख घ थ ध फ भ …) | 38.0% | 78.6% |
| Devanagari | nasalization (ँ ं) | 43.2% | 84.3% |
| Devanagari | conjuncts (्) | 28.4% | 65.3% |
| Devanagari | izafat (-ए- / -ओ-) | 10.5% | 35.2% |
| Roman | long vowels (ā ī ū) | 3.9% | 75.8% |
| Roman | nasalization (ñ) | 1.1% | 76.7% |
| Roman | aspirates (kh gh th dh ph bh …) | 14.5% | 77.3% |
| Roman | retroflex (T D R Ḍ Ṛ) | 0.0% | 38.1% |
| Roman | Perso-Arabic sounds (ḳh ġ q z f) | 7.7% | 61.9% |
| Roman | izafat (-e- / -o-) | 1.9% | 41.4% |
| Nastaliq | aspirates (ھ) | 34.4% | 91.9% |
| Nastaliq | noon ghunna (ں) | 30.8% | 91.0% |
| Nastaliq | hamza / izafat (ء ئ ۂ ؤ) | 17.5% | 72.3% |
| Nastaliq | spelling choice (ث ذ ص ض ط ظ ع ح) | 21.3% | 79.1% |
| Nastaliq | bari ye (ے) | 36.3% | 92.6% |

</details>

## Validation curve

| Examples seen | Validation CER | Validation exact |
|---|---|---|
| 0 | 49.6% | 0% |
| 1,248 | 12.6% | 6% |
| 2,496 | 9.5% | 12% |
| 3,744 | 8.5% | 11% |
| 4,992 | 7.5% | 14% |
| 6,240 | 6.8% | 17% |
| 7,488 | 6.8% | 18% |
| 8,736 | 7.2% | 14% |
| 9,984 | 7.6% | 17% |
| 10,000 | 7.3% | 18% |

## Time

| Stage | Minutes |
|---|---|
| corpus | 0.1 |
| datasets | 0.0 |
| load model | 11.7 |
| preflight | 1.2 |
| baseline eval | 69.6 |
| training | 303.6 |
| tuned eval | 19.9 |
| **total** | **406** |
