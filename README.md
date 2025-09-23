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

All models perform exceptionally well. DistilBERT produced the highest performance on the held-out test set. See the `/notebooks` (or training logs) for full evaluation artifacts (confusion matrices, classification reports).

---

### How to run this API locally
To set up and run this project locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Anirudha-repo/Reply_Classification.git
   cd Reply_Classification

2. **Download the dataset (if not already included):**
 Ensure reply_classification_dataset.csv is placed in the root directory of the cloned repository.

3. **Create and activate a Python virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux: 
   source venv/bin/activate

4. **Install required libraries:**
   ```bash
   - pip install -r requirements.txt

5. **Run the Flask API:**
   ```bash
   python Development.py

6. **Test the API (e.g., with Postman):**
- Send a POST request to http://127.0.0.1:8000/predict
- Set the request body to raw JSON with content type application/json.

 ![!Example 2](https://github.com/Anirudha-repo/Reply_Classification/blob/main/screenshots/Screenshot%202025-09-23%20173837.png)
  
 ![Example 1](https://github.com/Anirudha-repo/Reply_Classification/blob/main/screenshots/Screenshot%202025-09-23%20173816.png)
 
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
