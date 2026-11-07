# Churn Signal — Customer Churn Prediction

A Flask + vanilla JS application for dataset analytics and live inference using the **provided saved Logistic Regression model bundle**.

## What is wired in
- Original Telco CSV (`data/customer_churn.csv`) powers the analytics API and charts.
- Saved model bundle (`models/telco_customer_churn_classification.pkl`) is loaded with Joblib. It contains `model`, `scaler`, `features`, and `model_name`.
- The prediction form maps the exact 8 notebook features and passes them through the saved `StandardScaler` before model inference.
- Metrics and confusion matrix reproduce the notebook's stratified 80/20 split (`random_state=42`): accuracy 78.18%, churn precision 61.75%, recall 47.06%, F1 53.41%, ROC AUC 81.97%, matrix [[924,109],[198,176]].
- Dataset cleaning mirrors notebook: numeric `TotalCharges`, remove missing values, resulting in 7,032 records.

## Run locally (Windows PowerShell)
```powershell
cd "PATH\TO\customer-churn-dashboard"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

## Run locally (macOS/Linux)
```bash
cd /path/to/customer-churn-dashboard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## API
- `GET /api/analytics` — computed dataset summary and chart series.
- `GET /api/model-info` — saved model metadata + notebook evaluation.
- `POST /api/predict` — JSON payload with the 8 feature names below.

```json
{
  "tenure": 12,
  "MonthlyCharges": 75,
  "TotalCharges": 900,
  "SeniorCitizen": 0,
  "gender_Male": 1,
  "Partner_Yes": 0,
  "Dependents_Yes": 0,
  "PaperlessBilling_Yes": 1
}
```

## Model features (exact order)
`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`, `gender_Male`, `Partner_Yes`, `Dependents_Yes`, `PaperlessBilling_Yes`.

Categorical mapping follows notebook's one-hot columns: Male=1 else 0; Yes=1 else 0. Binary variables are 0/1. Numeric values are scaled with the **persisted** scaler.

## Notes
- This project does not retrain or replace the saved model.
- Predictions are probabilistic estimates, not guarantees or causal explanations.
- Chart.js is loaded from jsDelivr CDN; an internet connection is needed for chart rendering.
- The notebook's saved model was serialized with scikit-learn 1.6.1. The requirements pin that version to minimize pickle compatibility issues.
- Use Python 3.11 recommended.
