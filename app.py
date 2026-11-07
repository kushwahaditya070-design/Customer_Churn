
from pathlib import Path
import joblib, pandas as pd, numpy as np
from flask import Flask, render_template, request, jsonify
from utils.analytics import build_analytics
from utils.prediction import predict_customer, MODEL_PATH, FEATURES

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "customer_churn.csv"
app = Flask(__name__)
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]

@app.route("/")
def home():
    return render_template("index.html")

@app.get("/api/analytics")
def analytics():
    try:
        return jsonify(build_analytics(DATA_PATH))
    except Exception as e:
        app.logger.exception("Analytics error")
        return jsonify({"error":"Unable to process dataset analytics."}), 500

@app.get("/api/model-info")
def model_info():
    return jsonify({
        "model_name": bundle.get("model_name", "Logistic Regression"),
        "features": FEATURES,
        "accuracy": 0.7818,
        "precision": 0.6175,
        "recall": 0.4706,
        "f1": 0.5341,
        "roc_auc": 0.8197,
        "confusion_matrix": [[924,109],[198,176]],
        "test_size": 1407,
        "evaluation_note": "Reproduced from notebook's stratified 80/20 split (random_state=42)."
    })

@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        result = predict_customer(payload, model, scaler)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error":str(e)}), 400
    except Exception:
        app.logger.exception("Prediction error")
        return jsonify({"error":"Prediction failed. Check server logs."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
