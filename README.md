# Logistics AI Control Tower

Delivery time prediction system for Brazilian e-commerce. This project analyzes data from Olist - one of the largest marketplace platforms in Brazil.

## Business Problem

E-commerce customers want to know when their package will arrive. Sellers want to optimize logistics. This project tries to answer the question: **can we predict delivery time based on available data?**

Spoiler: yes, but only partially (~41% of variance). The rest depends on things we don't have in the data - weather, traffic, courier availability.

## Results

R² Score - 41.2%
Mean Error - 4.4 days
Accuracy (<3 days error) - 54%

Is this good? For delivery time prediction - yes, it's a decent result. Most factors affecting delivery are outside transactional data.

## Project Structure

```
├── main
├── screenshots/
│   ├── feature-importance.png
│   ├── geographic-distribution-of-anomalies.png
│   └── metrics-and-delivery-simulator.png
├── src/
│   ├── loader.py           # Data fetching from Kaggle
│   ├── processing.py       # Cleaning + feature engineering
│   ├── model.py            # Anomaly detection (Isolation Forest)
│   ├── prediction.py       # XGBoost model
│   └── dashboard.py        # Streamlit interface
├── models/                 # Saved models (.pkl)
├── Dockerfile
└── requirements.txt
```

## Key Model Features

After several iterations, the most important features turned out to be:
- **Distance** (~26%) - calculated using Haversine formula from zip codes
- **Purchase month** (~15%) - seasonality (Black Friday, holidays)
- **Customer location** (~11%) - some regions have weaker infrastructure

## How to Run

### Locally
```bash
pip install -r requirements.txt
uvicorn src.api:app --reload --port 8000  # loads data, runs isolation forest, trains XGBoost
```

Prepare the dashboard (it assumes the FastAPI is accessible via `DELIVERY_API_URL`, default `http://localhost:8000`):

```bash
set DELIVERY_API_URL=http://localhost:8000   # on Windows
streamlit run src/dashboard.py
```

On macOS/Linux use `export DELIVERY_API_URL=http://localhost:8000` if you prefer the dashboard to connect to a different host.

The FastAPI service validates every incoming `DeliveryEstimate` payload through Pydantic, returning a guarded prediction plus MAE/R² and a small warning list when features fall outside the training distribution. Streamlit uses HTTPX to pass simulator inputs to `POST /predict` so the UI stays responsive even if the model lives in a separate process.

### FastAPI interface

- `POST /predict` accepts the full feature set and returns `predicted_days`, `mae`, `r2_score`, and any extreme-value warnings.
- `GET /health` exposes readiness, total records, and the latest MAE/R² so dashboards or observability tools can verify the model is online.
- Every request is validated by Pydantic, which protects against negative distances, out-of-range months, or malformed JSON.

### Docker
```bash
docker build -t delivery-app .
docker run -p 8501:8501 delivery-app
```

Dashboard will be available at `http://localhost:8501`

## Stack

- Python 3.12
- FastAPI + Pydantic (validated prediction API)
- HTTPX (Streamlit ↔ FastAPI communication)
- XGBoost (regression)
- Scikit-learn (preprocessing, Isolation Forest)
- Streamlit + Plotly (dashboard)
- Docker

## Lessons Learned

1. **Data leakage is sneaky** - it's easy to accidentally use information from the future
2. **Removing outliers is a trade-off** - improves metrics, but will the model work on extreme cases?
3. **Feature engineering > more data** - well-designed features give more than raw columns
4. **41% R² is not a failure** - for some problems it's simply the ceiling given the nature of the data

## Dashboard Preview

### Feature Importance
![Feature Importance](screenshots/feature-importance.png)

### Geographic Distribution of Anomalies
![Geographic Distribution of Anomalies](screenshots/geographic-distribution-of-anomalies.png)

### Metrics and Delivery Simulator
![Metrics and Delivery Simulator](screenshots/metrics-and-delivery-simulator.png)

## Dataset

[Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) - public dataset with ~100k orders from 2016-2018.

---

Project created as part of Data Engineering / ML portfolio.
