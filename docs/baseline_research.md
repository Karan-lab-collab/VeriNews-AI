# Baseline Research Report: TF-IDF + Logistic Regression Evaluation

## Objective
Evaluate whether a simple bag-of-words baseline model (TF-IDF vectorization + Logistic Regression) can serve as a reliable foundation for estimating the trustworthiness of news articles across diverse domains.

---

## Methodology
The research methodology is structured to systematically test the limits of non-semantic classification:
1. **Model Pipeline**: 
   * **Preprocessing**: Text cleaning, including lowercasing, HTML tag stripping, URL/punctuation removal, standalone number stripping, and whitespace normalization.
   * **Feature Extraction**: TF-IDF vectorization capturing unigrams and bigrams, limited to a vocabulary ceiling of 50,000 features.
   * **Classifier**: Scikit-Learn `LogisticRegression` (with default $C=1.0$ and LBFGS solver).
2. **Train/Test Evaluation**: Standard 80/20 split on the benchmark ISOT dataset to establish a performance benchmark.
3. **Independent Manual Validation**: Evaluating model performance on an out-of-domain manually curated validation set containing 60 real-world news articles representing 14 diverse categories (including Science, Space, Technology, Finance, Business, Politics, clickbait, and obvious fake news).
4. **Error Analysis**: Diagnostic feature importance extraction to understand class-specific token associations.
5. **Dataset Engineering**: Iterative improvements to training data (removing publisher datelines, adding domain-diverse datasets, deduplication, and label correction).
6. **Repeat Evaluation**: Re-running train/test split, manual validation, and feature analysis to quantify the impact of dataset engineering.

---

## Findings

### 1. Performance Overview
* **Train/Test Split Accuracy (ISOT benchmark)**: **97.59%**
* **Initial Manual Validation Accuracy (Out-of-Domain)**: **45.0%**

### 2. Major Observation
Despite achieving excellent benchmark accuracy (~97.6%), the baseline model severely failed to generalize to real-world scientific, international, and contextual news articles. 

### 3. Case Study: The NASA James Webb Article
* **Input Text**: *"NASA's James Webb Space Telescope has captured unprecedented infrared images of the Carina Nebula, revealing thousands of never-before-seen young stars in the stellar nursery located 7,600 light-years away."*
* **Baseline Prediction**: **FAKE** (with **89.18%** confidence).
* **Analysis**: The baseline model confidently classified a factual scientific press release from NASA as fake news because it lacked semantic understanding of science news patterns.

### 4. Root Cause: Publisher Leakage
Error analysis using Logistic Regression coefficients (feature importance) revealed that the model was exploiting dataset-specific features rather than learning genuine journalistic characteristics.

Specifically, wire-service attributions dominated the REAL predictions:
* Token `reuters` coefficient: **+21.7182**
* Token `said` coefficient: **+12.2546**
* Token `washington reuters` coefficient: **+9.1048**

Because genuine articles in the training set almost universally started with or contained publisher-specific datelines (e.g., *"WASHINGTON (Reuters) - "*), the model learned that the presence of the word "reuters" was the single most defining characteristic of truth. Any out-of-domain article lacking these specific markers was highly likely to be classified as FAKE.

---

## Dataset Engineering Experiment

To determine if the baseline's generalization failure was primarily due to training dataset issues, we engineered an improved training dataset:
* **Merged Datasets**: Blended the ISOT dataset with the `mrm8488/fake-news` dataset (McIntire-derived corpus).
* **Corrected Inverted Labels**: Addressed a critical issue in `mrm8488/fake-news` where genuine Reuters articles (labelled 0) and sensationalist fake articles (labelled 1) were flipped relative to our convention.
* **Removed Duplicates**: Eliminated 5,611 cross-dataset exact-match duplicate texts via MD5 hashes.
* **Removed Publisher Datelines**: Implemented regex stripping of wire-service prefixes (e.g. `(Reuters) -`, `(AP) -`) before processing.
* **Blacklisted Publisher Tokens**: Added wire agencies and byline signals (`reuters`, `nytimes`, `associated press`, etc.) directly to the TF-IDF stop words list.

### Retrained Model Results
* **Retrained Train/Test Accuracy**: **98.15%**
* **Retrained Manual Validation Accuracy**: **56.7%** (an improvement of **+11.7** percentage points).
* **Reuters Coefficient**: Reduced to **0.0000** (successfully eliminated publisher leakage).
* **NASA James Webb Article**: Still predicted **FAKE** (with **82.18%** confidence).

---

## Research Conclusion
Dataset engineering successfully eliminated domain bias related to publisher identity (dropping the `reuters` coefficient from +21.7 to 0) and raised manual validation performance by +11.7%. 

However, the model still failed on the NASA case study and other domain-diverse factual articles. Because the model relies on a bag-of-words representation, it cannot understand semantic context. It is unable to distinguish legitimate scientific announcements (which mention words like "scientists" or "announced") from clickbait or conspiracy theories that share similar low-level vocabulary.

This evidence demonstrates that we have reached the practical ceiling of TF-IDF and Logistic Regression, motivating the transition to contextual transformer models (such as DistilBERT) for Milestone 2.
