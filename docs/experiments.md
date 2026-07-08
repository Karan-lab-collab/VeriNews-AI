# Experimentation Log: VeriNews AI Research Journey

This document logs the core experiments conducted to establish the baseline and identify the limits of non-semantic text classification for news trust assessment.

---

## Experiment 1: Baseline TF-IDF + Logistic Regression

### Objective
Establish a baseline classification model using standard NLP features and simple algorithms on a large benchmark dataset.

### Setup
* **Dataset**: ISOT dataset (~44k political articles: Fake vs True).
* **Features**: TF-IDF Vectorizer (max 50,000 features, unigrams and bigrams, English stop words).
* **Classifier**: Logistic Regression ($C=1.0$).
* **Evaluation**: 80/20 train/test split.

### Results
* **Train/Test Accuracy**: **97.59%**
* **Precision**: 98.00%
* **Recall**: 97.24%
* **F1-Score**: 97.62%

### Outcome
* **High Benchmark Performance**: The model successfully learned to separate classes within the test split of the benchmark dataset.
* **Status**: **Baseline Established.**

---

## Experiment 2: Independent Manual Validation & Error Analysis

### Objective
Validate whether the high benchmark accuracy translates to real-world out-of-domain articles, and identify any structural bias in the model.

### Setup
* **Validation Set**: A manually compiled dataset of 60 real-world articles representing 14 diverse categories (including Science, Space, Technology, Finance, Business, Politics, clickbait, and obvious fake news).
* **Case Study**: A factual scientific announcement from NASA regarding the James Webb Space Telescope.
* **Evaluation**: Metric calculation (accuracy, precision, recall) per category, feature importance coefficient diagnostic, and manual prediction verification.

### Results
* **Manual Validation Accuracy**: **45.0%** (a **52.59%** drop compared to test split).
* **Category Breakdown**: 
  * Clickbait, Conspiracy, Obvious Fake: 100% accurate.
  * Entertainment, Local News: 0% accurate.
  * Science, Space, Technology, Sports: ~25-33% accurate.
* **NASA Case Study**: **Classified as FAKE** with **89.18%** confidence.
* **Top REAL Coefficients**: `reuters` (+21.72), `said` (+12.25), `washington reuters` (+9.10).

### Outcome
* **Severe Domain Generalization Failure**: The model is highly overfit to the US political news style of the 2015-2018 ISOT benchmark.
* **Discovery of Publisher Leakage**: The model was not learning the content or characteristics of journalism. Instead, it was memorizing wire-service dateline prefixes (like `reuters`) as the definitive signal for genuine news.

---

## Experiment 3: Dataset Engineering & Bias Mitigation

### Objective
Determine if domain bias can be resolved solely by engineering the training dataset (without replacing the algorithm).

### Setup
* **Dataset Consolidation**: Merged ISOT with `mrm8488/fake-news` (~44.8k articles) to diversify domain coverage.
* **Label Correction**: Inverted the labels of the `mrm8488` dataset after discovering that its labels were flipped.
* **Deduplication**: Removed 5,611 cross-dataset exact duplicates using MD5 text hashing.
* **Bias Mitigation (Preprocessing)**: Added regex-based publisher dateline removal to strip prefix strings like `(Reuters) -`.
* **Bias Mitigation (Features)**: Blacklisted wire agencies (`reuters`, `nytimes`, `associated press`, etc.) by adding them to the TF-IDF stop words list.
* **Evaluation**: Retrained the same Logistic Regression pipeline on the unified 79,154-article dataset and re-ran all evaluations.

### Results
* **Train/Test Accuracy**: **98.15%**
* **Manual Validation Accuracy**: **56.7%** (an improvement of **+11.7** percentage points).
* **Reuters Coefficient**: **0.0000** (successfully removed from top features).
* **NASA Case Study**: Still predicted **FAKE** (with **82.18%** confidence).

### Outcome
* **Publisher Bias Successfully Removed**: Dateline stripping and stop words successfully eliminated publisher leakage features.
* **Limited Generalization Gain**: While manual validation accuracy increased to 56.7%, the model still failed to correctly categorize the NASA article and other out-of-domain factual articles.

---

## Final Conclusion

Dataset engineering helped confirm that data quality and preprocessing were critical factors in baseline performance (yielding a +11.7% improvement). However, it did not resolve the fundamental issue of semantic understanding. 

The baseline model relies entirely on a bag-of-words representation (TF-IDF), which fails to capture contextual relationships. Since factual science news and fabricated science rumors use similar vocabularies, TF-IDF cannot differentiate between them.

**TF-IDF + Logistic Regression has reached its practical ceiling.** This establishes a clear, evidence-based motivation to move to transformer models (such as DistilBERT) for Milestone 2.
