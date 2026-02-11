# time Series Forecasts

A full-stack **adaptive forecasting system** with portfolio management: time-series forecasts (stocks, forex, crypto), multiple ML models, evaluation, and a React dashboard.

## Features

- **Forecasting** – Multi-horizon forecasts with ARIMA, LSTM, GRU, Transformer, exponential smoothing, and ensemble/adaptive models
- **Data** – Ingestion from Yahoo Finance (yfinance); support for equities, forex, and crypto
- **Evaluation** – Automatic evaluation of forecasts and continuous evaluation in the background
- **Portfolio** – Portfolio management and backtesting with configurable strategies
- **Models** – Model registry, versioning, retraining API, and optional scheduled retraining
- **Dashboard** – React UI (charts, portfolio, settings) plus an optional Dash dashboard at `/dashboard`

## Tech Stack

| Layer    | Stack                                                                                               |
| -------- | --------------------------------------------------------------------------------------------------- |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, Recharts, Plotly.js, React Query               |
| Backend  | FastAPI, Uvicorn                                                                                    |
| Database | MongoDB (primary), SQLite (optional)                                                                |
| ML       | TensorFlow/Keras (LSTM, GRU, Transformer), statsmodels (ARIMA, exponential smoothing), scikit-learn |
| Data     | pandas, numpy, yfinance                                                                             |
| DevOps   | Docker, Docker Compose                                                                              |

## Project Structure

```
forecast-pro-dash/
├── src/                    # React frontend (Vite + TS)
│   ├── components/         # UI components, CandlestickChart, sidebar
│   ├── pages/              # Dashboard, Portfolio, Auth, Settings
│   └── contexts/           # AuthContext
├── backend/                # FastAPI app
│   ├── api/                # data, forecast, models, evaluation, portfolio, retraining
│   ├── models/             # LSTM, ARIMA, ensemble, adaptive forecaster
│   ├── services/           # model service, evaluation, data ingestion, retraining
│   ├── db/                 # MongoDB (unified_db)
│   ├── trading/            # portfolio manager, trading strategy
│   └── scripts/            # populate DB, migrate, upload model
├── notebooks/              # Training scripts (train_*.py)
├── scripts/                # Data population, diagnosis, retrain scripts
├── docker-compose.yml      # MongoDB + backend + frontend
├── Dockerfile.frontend
└── backend/Dockerfile
```

## Prerequisites

- **Node.js** 18+ and npm/bun (for frontend)
- **Python** 3.10+ (for backend)
- **MongoDB** 7.x (local or Docker)

## Quick Start

### Option 1: Docker Compose (recommended)

Runs MongoDB, FastAPI backend, and React frontend:

```bash
docker-compose up --build
```

- API: http://localhost:8000
- Frontend: http://localhost:3000
- MongoDB: localhost:27017

### Option 2: Local development

**1. MongoDB**

Start MongoDB locally on port 27017, or use Docker:

```bash
docker run -d -p 27017:27017 --name forecast_mongodb mongo:7.0
```

**2. Backend**

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**3. Frontend**

```bash
npm install
npm run dev
```

Frontend runs at http://localhost:5173 (Vite default). Set `VITE_API_URL` if your API is elsewhere.

## Environment

Backend settings use the `FORECAST_` prefix. Create a `.env` in `backend/` or set:

| Variable                         | Description                     | Default                      |
| -------------------------------- | ------------------------------- | ---------------------------- |
| `FORECAST_MONGO_URI`             | MongoDB connection string       | `mongodb://localhost:27017/` |
| `FORECAST_MONGO_DB`              | Database name                   | `forecast_db`                |
| `FORECAST_USE_MONGO`             | Use MongoDB                     | `true`                       |
| `FORECAST_SYMBOL`                | Default symbol                  | `AAPL`                       |
| `FORECAST_AUTO_START_RETRAINING` | Auto-start scheduled retraining | `false`                      |

For Docker, the compose file sets `FORECAST_MONGO_URI=mongodb://mongodb:27017/` and `VITE_API_URL=http://localhost:8000` for the frontend.

## API Overview

| Base path         | Description                                         |
| ----------------- | --------------------------------------------------- |
| `/api/data`       | OHLCV data, list symbols, ingestion                 |
| `/api/forecast`   | Generate forecasts (multi-model, horizons, symbols) |
| `/api/models`     | List/register models, model metadata                |
| `/api/evaluation` | Run evaluation, get results                         |
| `/api/portfolio`  | Portfolio state, backtest                           |
| `/api/retraining` | Start/stop scheduled retraining                     |
| `/dashboard`      | Optional Dash dashboard (mounted under FastAPI)     |
| `/health`         | Health check                                        |

Docs: http://localhost:8000/docs when the backend is running.

## Scripts

- **Data**: `populate_data.ps1`, `scripts/populate_mongodb.py`, `scripts/ingest_forex_crypto.py`
- **Training**: `notebooks/train_*.py`, `train_model.ps1`, `reset_and_retrain_aapl.ps1`
- **Backend**: `backend/scripts/populate_database.py`, `backend/scripts/upload_model.py`
- **Diagnosis**: `scripts/diagnose_*.py`, `DIAGNOSIS.md`

## License

Private project. Use and modify as needed for your environment.
