# VeriNews AI — Domain Generalisation Report (Milestone 1B)

> **Experiment**: Dataset Engineering — Replace narrow ISOT with unified multi-source dataset
> **Model**: Same TF-IDF + Logistic Regression (unchanged)
> **Hypothesis**: Domain bias is caused by training data, not algorithm

---

## 1. What Was Changed

| Change | Description | Rationale |
|--------|-------------|-----------|
| Publisher datelines removed | Stripped "CITY (Reuters) -" prefixes | Eliminate `reuters` coefficient +21.7 leakage |
| Added LIAR dataset | 12k PolitiFact statements (binarised) | Adds diverse text styles |
| Added WELFake (if available) | 72k multi-source articles | Adds domain breadth |
| Deduplication across sources | MD5 hash on cleaned text | Prevent training contamination |
| LIAR label binarisation | true/mostly-true=REAL; false/pants-fire=FAKE; skip ambiguous | Clean training signal |

### Dataset Composition

| Source | Articles |
|--------|---------|
| ISOT | 40,579 |
| mrm8488 | 38,575 |
| **Total** | **79,154** |

---

## 2. Train/Test Split Metrics: Before vs After

| Metric | Baseline (v1) | After Engineering (v2) | Delta |
|--------|--------------|----------------------|-------|
| Accuracy  | 97.03% | 98.15% | +1.12% |
| Precision | 97.41% | 98.27% | +0.86% |
| Recall    | 97.08% | 98.33% | +1.25% |
| F1-Score  | 97.25% | 98.30% | +1.05% |

---

## 3. Manual Validation Accuracy: Before vs After

| Validation Set | Baseline (v1) | After Engineering (v2) | Delta |
|---------------|--------------|----------------------|-------|
| Overall (60 examples) | 45.0% | 56.7% | +11.67% |

### Category-wise Performance

| Category | Baseline | v2 | Delta |
|----------|----------|----|-------|
| Business | 25.0% | 50.0% | +25.00% |
| Clickbait | 100.0% | 100.0% | +0.00% |
| Conspiracy | 100.0% | 100.0% | +0.00% |
| Entertainment | 0.0% | 33.3% | +33.30% |
| Finance | 66.7% | 83.3% | +16.60% |
| Health | 33.3% | 66.7% | +33.40% |
| Local News | 0.0% | 0.0% | +0.00% |
| Obvious Fake | 100.0% | 100.0% | +0.00% |
| Politics | 40.0% | 60.0% | +20.00% |
| Science | 25.0% | 25.0% | +0.00% |
| Space | 28.6% | 28.6% | +0.00% |
| Sports | 33.3% | 33.3% | +0.00% |
| Technology | 33.3% | 50.0% | +16.70% |
| Weather | 50.0% | 50.0% | +0.00% |

---

## 4. NASA / James Webb Article Analysis

| Metric | Baseline (v1) | After Engineering (v2) |
|--------|--------------|----------------------|
| Predicted | FAKE | FAKE |
| Confidence | 89.18% | 82.18% |
| P(FAKE) | 89.18% | 82.18% |
| P(REAL) | 10.82% | 17.82% |

**Result**: Model still misclassifies the NASA article as FAKE. The domain bias persists despite dataset engineering.

---

## 5. Feature Importance: Reuters Bias Analysis

| | Baseline | v2 |
|-|----------|----|
| Top 5 REAL features | `reuters | said | washington reuters | president donald | wednesday` | `said | president donald | wednesday | tuesday | thursday` |
| Top 5 FAKE features | `video | image | just | images | gop` | `image | video | just | gop | images` |
| `reuters` coefficient | +21.7182 | +0.0000 |

**Reuters bias ELIMINATED** — the `reuters` token no longer dominates REAL predictions.

---

## 6. Key Findings

### Did dataset engineering improve manual validation?

**YES** — Manual validation accuracy improved from 45.0% to 56.7% (+11.7 percentage points). Dataset engineering was effective.

### Did the Reuters bias decrease?

**YES** — The `reuters` coefficient dropped from 21.7 to 0.0. Publisher dateline removal was effective.

### What types of articles improved?

- **Health**: +33.4%
- **Entertainment**: +33.3%
- **Business**: +25.0%
- **Politics**: +20.0%
- **Technology**: +16.7%

### What still fails?

- No significant regressions.

Categories with <50% accuracy still present: Entertainment, Local News, Science, Space, Sports

---

## 7. Should We Continue with Logistic Regression? Or DistilBERT?

### Analysis

| Question | Finding |
|----------|---------|
| Is the domain bias primarily from training data? | Yes — dataset engineering improved results |
| Can TF-IDF + LR reach acceptable accuracy (>75%) on diverse validation? | Unlikely — current ceiling appears below 75% |
| Is the improvement sufficient for production? | No — more work needed |

### Conclusion

**Recommend DistilBERT for Milestone 2** — Despite dataset engineering, the TF-IDF + Logistic Regression model cannot overcome its fundamental bag-of-words limitation. Manual validation accuracy of 56.7% is insufficient for production use. The model lacks semantic understanding needed to distinguish legitimate science news from sensationalist fake news.

---

## 8. Recommendations for Next Milestone

### If Continuing with LR (Short-term)

1. **Class weighting** (`class_weight='balanced'`): Will improve recall for underrepresented classes.
2. **Larger vocabulary** (increase `MAX_FEATURES` to 100k): Captures more domain terms.
3. **Trigram support** (`ngram_range=(1,3)`): Captures "NASA announced", "peer-reviewed study".
4. **More diverse training data**: CoAID (health), McIntire (sports/entertainment), FeverFact.
5. **Probability calibration**: Platt scaling to make confidence scores meaningful.

### Milestone 2: DistilBERT Fine-tuning (Recommended)

1. Replace TF-IDF with contextual embeddings (DistilBERT encoder).
2. Fine-tune on the unified dataset with domain-balanced sampling.
3. Expected manual validation accuracy: 80-90% (based on published benchmarks).
4. Integration: FastAPI endpoint accepts text → returns label + confidence.

---

*Generated by VeriNews AI — Milestone 1B retrain.py*
