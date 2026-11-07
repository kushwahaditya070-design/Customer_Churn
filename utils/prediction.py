
from pathlib import Path
import numpy as np, pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "telco_customer_churn_classification.pkl"
FEATURES = ["tenure","MonthlyCharges","TotalCharges","SeniorCitizen","gender_Male","Partner_Yes","Dependents_Yes","PaperlessBilling_Yes"]

def predict_customer(payload, model, scaler):
    required = FEATURES
    missing = [f for f in required if f not in payload or payload[f] in ("", None)]
    if missing:
        raise ValueError("Please complete all fields: " + ", ".join(missing))
    try:
        row = [float(payload[f]) for f in required]
    except (TypeError, ValueError):
        raise ValueError("Numeric fields must contain valid numbers.")
    if not 0 <= row[0] <= 72: raise ValueError("Tenure must be between 0 and 72 months.")
    if not 0 <= row[1] <= 200: raise ValueError("Monthly charges must be between 0 and 200.")
    if not 0 <= row[2] <= 10000: raise ValueError("Total charges must be between 0 and 10000.")
    for i in [3,4,5,6,7]:
        if row[i] not in (0,1): raise ValueError("Binary fields must be 0 or 1.")
    matrix = pd.DataFrame([row], columns=required)
    scaled = scaler.transform(matrix)
    prediction = int(model.predict(scaled)[0])
    probability = float(model.predict_proba(scaled)[0][1]) if hasattr(model,"predict_proba") else None
    return {"prediction":prediction, "label":"Likely to churn" if prediction else "Likely to stay",
            "churn_probability":round(probability,4) if probability is not None else None,
            "stay_probability":round(1-probability,4) if probability is not None else None,
            "risk_band":("High" if probability>=.65 else "Moderate" if probability>=.35 else "Low") if probability is not None else None,
            "model": "Logistic Regression"}
