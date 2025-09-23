## Reply Classifier — Reply Sentiment (Positive / Neutral / Negative)

- A lightweight production-ready reply classification service.  
- Trained baselines (TF-IDF + Logistic Regression, LightGBM) and a fine-tuned DistilBERT transformer.
- The project includes a Flask API (`/predict`) that returns label and confidence for incoming text.

---

### Project Overview

We built an end-to-end reply classification pipeline that labels incoming short replies as **positive**, **neutral**, or **negative**. The project includes:

- Data preprocessing and exploration
- Baseline models (TF-IDF + Logistic Regression, LightGBM)
- Transformer fine-tuning (DistilBERT)
- A Flask-based `/predict` API serving the best model
- Dockerfile and `requirements.txt` for easy local deployment

This project was created for the SvaraAI / AI_ML Engineer internship assignment.

---

### Roadmap & Deliverables

1. **Data inspection & cleaning** (tokenization, text normalization)  
2. **Baseline training**: TF-IDF → Logistic Regression, and LightGBM for comparison  
3. **Transformer fine-tune**: `distilbert-base-uncased` with Hugging Face Trainer  
4. **Evaluation**: accuracy, macro-F1, confusion matrices, error analysis  
5. **Deployment**: Flask API with `/predict` endpoint, Dockerfile, and instructions  
6. **Documentation**: README + Answers.md + requirements.txt

---

### Results Summary

| Model                     | Accuracy | Macro F1 |
|---------------------------|----------:|---------:|
| Logistic Regression (TF-IDF) | 0.9859  | 0.9859  |
| LightGBM (TF-IDF)         | 0.9812  | 0.9812  |
| DistilBERT (fine-tuned)   | **1.000** | **1.000** |

> All models perform exceptionally well. DistilBERT produced the highest performance on the held-out test set. See the `/notebooks` (or training logs) for full evaluation artifacts (confusion matrices, classification reports).

---

### Modeling & Evaluation (brief)

- Preprocessing: lowercasing, URL/mention replacement, simple punctuation cleaning, TF-IDF vectorization (n-grams 1–2).

- Baselines: Logistic Regression (interpretable, fast), LightGBM (tree-based).

- Transformer: DistilBERT fine-tuned using Hugging Face Trainer — achieved highest accuracy.

- Metrics: Accuracy and Macro F1 (macro F1 chosen because it treats all classes equally — important for multi-class balance).

  ---

### Dependencies

Key libraries used in this project:

- Transformers, torch — transformer training & inference

- Scikit-learn — baselines + metrics

- LightGBM — tree-based baseline

- Flask — lightweight API

- Datasets — helpers for HF fine-tuning (if used in training notebooks)
