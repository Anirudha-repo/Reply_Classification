import os
import logging
from typing import Dict, Any
from flask import Flask, request, jsonify
from transformers import pipeline
import torch


# Logging & config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model directory from environment 
MODEL_DIR = os.getenv("MODEL_DIR", "models/distilbert_reply_classifier")

# Device configuration 
DEVICE = 0 if torch.cuda.is_available() else -1

logger.info(f"MODEL_DIR = {MODEL_DIR}  |  DEVICE = {DEVICE}")


# Loading model once at startup

try:
    logger.info("Loading pipeline (this may take a moment)...")
    nlp_pipeline = pipeline(
        task="text-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        device=DEVICE,
        return_all_scores=True,  # we want scores for all classes
        truncation=True
    )
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load model pipeline. Make sure MODEL_DIR exists and contains a saved HF model/tokenizer.")
    raise


# Flask app

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok", "model_dir": MODEL_DIR}), 200

@app.route("/predict", methods=["POST"])
def predict() -> Any:
    """
    Request JSON: {"text": "some text to classify"}
    Response JSON:
    {
      "label": "positive",
      "confidence": 0.9876,
      "all_scores": [
         {"label":"positive", "confidence": 0.9876},
         {"label":"neutral", "confidence": 0.0102},
         {"label":"negative", "confidence": 0.0022}
      ]
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON with {'text': '...'}"}), 400

    j = request.get_json()
    text = j.get("text") or j.get("input") or ""
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing or empty 'text' field in JSON body."}), 400

    try:
        outputs = nlp_pipeline(text)[0]  # list of dicts: [{label, score}, ...]
        # normalise results
        all_scores = [{"label": item["label"], "confidence": round(float(item["score"]), 6)} for item in outputs]
        best = max(outputs, key=lambda x: x["score"])
        response = {
            "label": best["label"],
            "confidence": round(float(best["score"]), 6),
            "all_scores": all_scores
        }
        return jsonify(response), 200
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


# Run (development)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    # For simple local testing
    app.run(host="0.0.0.0", port=port, debug=False)
