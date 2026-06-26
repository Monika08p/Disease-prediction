# Disease Prediction System

A full-stack machine learning web application that predicts the likelihood
of **Heart Disease** or **Diabetes** from patient clinical data, with
explainable AI, risk scoring, PDF reports, and prediction history.

---

## 1. Architecture

```
                ┌─────────────────────┐
                │   Browser (UI)      │
                │  HTML/CSS/JS/Chart  │
                └─────────┬───────────┘
                          │ HTTP
                ┌─────────▼───────────┐
                │   Flask (app.py)    │
                │  routes + API       │
                └─────────┬───────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼──────┐  ┌───────▼───────┐  ┌──────▼───────┐
│ preprocessing │  │    predict     │  │ explainability│
│   .py         │  │     .py        │  │     .py       │
└───────┬──────┘  └───────┬───────┘  └──────┬───────┘
        │                 │                 │
        │         ┌───────▼───────┐         │
        │         │  models/      │         │
        │         │  heart/       │         │
        │         │  diabetes/    │         │
        │         └───────────────┘         │
        │                                    │
┌───────▼──────┐                    ┌────────▼───────┐
│  datasets/    │                    │  pdf_report.py │
│  heart.csv    │                    │  (reportlab)   │
│  diabetes.csv │                    └────────────────┘
└───────────────┘
```

`train_model.py` is run offline (not at request time) to train and save
the best model per disease into `models/<disease>/`. The Flask app only
ever loads already-trained artifacts — it never trains on the fly.

## 2. Folder Structure

```
project/
├── app.py                  # Flask application (routes, API)
├── train_model.py          # Trains & compares all 4 models per disease
├── run_no_debug.py         # Helper to run the server without the debug reloader
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── heart/
│   │   ├── best_model.pkl (or .keras)
│   │   ├── scaler.pkl
│   │   ├── metadata.json
│   │   └── model_comparison.csv
│   └── diabetes/  (same structure)
├── datasets/
│   ├── heart.csv
│   └── diabetes.csv
├── static/
│   ├── css/style.css
│   └── js/{main.js, predict.js}
├── templates/
│   ├── base.html, landing.html, select.html
│   ├── predict.html, history.html, error.html
├── reports/                # Generated PDF reports + prediction history JSON
├── utils/
│   ├── preprocessing.py    # Cleaning, missing-value handling, scaling
│   ├── predict.py          # Loads model+scaler, runs predictions
│   ├── explainability.py   # Feature importance, top factors, explanations
│   ├── validators.py       # Clinical range validation
│   └── pdf_report.py       # PDF report generation
└── tests/
    └── test_core.py        # 19 unit/regression tests
```

## 3. Datasets

| Disease | Source | Rows | Features | Target |
|---|---|---|---|---|
| Heart Disease | UCI Heart Disease dataset | 303 | 13 | `target` (0/1) |
| Diabetes | Pima Indians Diabetes dataset | 768 | 8 | `Outcome` (0/1) |

**Diabetes data quirk handled in preprocessing**: in the raw Pima dataset,
a value of `0` in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`,
or `BMI` is not a real clinical zero — it means "not recorded." These are
converted to `NaN` and imputed with the column median before training.

**Outlier handling**: features are clipped at ±4 standard deviations from
the mean rather than dropping rows, since both datasets are small and every
positive case is valuable.

## 4. Models

For each disease, four models are trained and compared on Accuracy,
Precision, Recall, F1-score, and **ROC-AUC**:

- Logistic Regression
- Random Forest
- XGBoost
- TensorFlow Neural Network (Dense 32→16→1, dropout, early stopping)

The model with the highest ROC-AUC (tie-broken by F1) is automatically
selected and saved as the production model for that disease. Results from
training:

| Disease | Best Model | Accuracy | ROC-AUC |
|---|---|---|---|
| Heart Disease | Random Forest | ~0.80 | ~0.89 |
| Diabetes | XGBoost | ~0.75 | ~0.82 |

(Exact numbers vary slightly between training runs due to model
randomness; see `models/<disease>/model_comparison.csv` for the actual
run that produced your saved model.)

## 5. Explainable AI

Predictions are explained using two transparent, fast techniques (no
SHAP/LIME dependency):

1. **Global feature importance** — from the trained model's native
   `feature_importances_` (tree models) or `coef_` (linear models).
2. **Per-patient deviation** — how far this specific patient's values
   are from the training population mean (z-score), weighted by global
   importance, to surface the "top contributing factors" for *this*
   prediction specifically.

This produces a plain-language explanation shown on the result page and
in the PDF report.

## 6. Setup & Installation

```bash
git clone <your-repo-url>
cd disease-prediction-system
pip install -r requirements.txt
```

### Train the models (required before first run)
```bash
python train_model.py --disease all
# or individually:
python train_model.py --disease heart
python train_model.py --disease diabetes
```
This downloads nothing extra — datasets are already included in
`datasets/`. It trains all 4 models per disease and saves the best one
into `models/<disease>/`.

### Run the app
```bash
python app.py
```
Open **http://127.0.0.1:5000**.

> If you hit issues with Flask's debug auto-reloader in some environments,
> use `python run_no_debug.py` instead.

## 7. Usage

1. Visit the landing page → click **Start Prediction**
2. Choose **Heart Disease** or **Diabetes**
3. Fill in the patient's clinical values (fields show valid ranges)
4. Click **Predict** → view:
   - Verdict (likely / unlikely)
   - Probability gauge (from `predict_proba`)
   - Risk level: **Low / Medium / High**
   - Top contributing factors chart
   - Plain-language explanation
5. Click **Download PDF Report** to save a complete report
6. Visit **History** to see all past predictions

### API usage
```bash
curl -X POST http://127.0.0.1:5000/api/predict/heart \
  -H "Content-Type: application/json" \
  -d '{"age":60,"sex":1,"cp":2,"trestbps":140,"chol":250,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.5,"slope":1,"ca":1,"thal":2}'
```

## 8. Testing

```bash
pip install pytest
pytest tests/ -v
```

19 tests covering:
- Preprocessing (missing-value imputation, outlier clipping, null-free output)
- Input validation (missing fields, non-numeric input, negative values, out-of-range warnings)
- **Prediction regression tests**: different inputs must produce different
  outputs — this directly guards against the scaler-misuse bug class
  (re-fitting a scaler on a single row instead of using the saved fitted
  scaler, which previously caused identical predictions for every input)
- Risk label thresholds
- Explainability output shape and content

## 9. Deployment Guide

### Option A — Simple VM / server (gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```
Put nginx in front for TLS termination and static file serving in production.

### Option B — Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN python train_model.py --disease all
EXPOSE 8000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
```
```bash
docker build -t disease-prediction .
docker run -p 8000:8000 disease-prediction
```

### Option C — Platform-as-a-Service (Render / Railway / Heroku-style)
- Build command: `pip install -r requirements.txt && python train_model.py --disease all`
- Start command: `gunicorn app:app`
- Set `PORT` env var if the platform requires it (modify `app.run(port=...)` accordingly)

## 10. Known Limitations

- Trained on small public research datasets (303 and 768 rows) — accuracy
  reflects dataset size, not production-grade clinical reliability.
- `reports/prediction_history.json` is a simple file store, fine for a
  demo/single-instance deployment; swap for a real database for multi-user
  production use.
- This is an **educational project** and is not a certified medical
  diagnostic tool. Always consult a healthcare professional for real
  medical decisions.

## Author

Monika P — B.E. Computer Science (Cyber Security), Sri Shakthi Institute
of Engineering and Technology
