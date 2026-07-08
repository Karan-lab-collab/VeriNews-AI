# VeriNews AI Development Roadmap

VeriNews AI is positioned as a research-driven Machine Learning Engineering project rather than a simple classifier. Our roadmap is designed to build, validate, audit, and iteratively improve our models based on empirical evidence.

---

## Completed Milestones

### Milestone 0: Architecture & Scaffold
* **Objective**: Build a production-quality scaffold with a decoupled frontend and backend.
* **Why it exists**: Establishes the engineering foundation (React + FastAPI + modular directory structure) so future ML models can be cleanly integrated, served, and evaluated.

### Milestone 1A: Baseline Model
* **Objective**: Train a simple TF-IDF + Logistic Regression baseline model.
* **Why it exists**: Establishes a simple, reproducible baseline of reference metrics. We intentionally chose a fast, simple bag-of-words algorithm to establish our lower-bound performance limit.

### Milestone 1.5: Independent Validation
* **Objective**: Question the benchmark dataset's high accuracy using a manually curated validation set.
* **Why it exists**: In-domain test splits (which yielded ~97.6% accuracy) are often misleading. We built an independent manual validation dataset (60 rows, 14 categories) to evaluate real-world generalization. This led to the discovery of severe domain bias and publisher leakage (e.g., the model classifying news based on datelines like `(Reuters) -` rather than actual content).

### Milestone 1B: Dataset Engineering & Generalization
* **Objective**: Mitigate publisher leakage and evaluate whether dataset changes can fix the baseline model.
* **Why it exists**: Before replacing the model algorithm, we wanted to scientifically isolate the data's impact. We stripped datelines, blacklisted publisher tokens from TF-IDF, merged datasets, and corrected label errors. While manual validation improved from 45.0% to 56.7%, the model still failed on factual out-of-domain articles (such as the NASA case study), proving the baseline had hit its semantic bag-of-words ceiling.

---

## Upcoming Milestones

### Milestone 2: Contextual Classification (DistilBERT)
* **Objective**: Fine-tune a pre-trained transformer model (DistilBERT) on the unified multi-source dataset.
* **Why it exists**: Our experiments proved that non-semantic bag-of-words models cannot generalize. To distinguish genuine news from misinformation across diverse domains, we require a model that understands context, word order, and semantic relations. Fine-tuning DistilBERT is an evidence-based decision driven by the failure of TF-IDF on our manual validation sets.

### Milestone 3: API & UI Integration
* **Objective**: Expose the model through a REST API and build the core interactive interface.
* **Why it exists**: Once we have a robust classifier, we will implement the FastAPI inference endpoint, load the serialized model in the API server, and update the React frontend to allow users to submit text and visualize real-time classification results.

### Milestone 4: Explainable AI & Explainability
* **Objective**: Integrate explainability features (such as SHAP/LIME or attention weights visualization) in the UI.
* **Why it exists**: Trust is not just a label. To serve as a trust assessment platform, we must show users *why* the AI estimated a specific score, highlighting the key words and phrases that contributed to the model's confidence.

### Milestone 5: Trust Assessment Platform (Multi-Signal Scoring)
* **Objective**: Expand the backend to evaluate source credibility, bias analysis, and claim checking.
* **Why it exists**: False news detection is multi-faceted. Combining model classification with publisher credibility checks, structural metadata analysis, and bias evaluation yields a comprehensive, multi-signal Trust Score.
