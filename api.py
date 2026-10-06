"""
API FastAPI pour DCForecaster (modèle XGBoost 2 étages, prototype MLOps).

Endpoints:
- GET  /            -> message de bienvenue
- POST /predict      -> prend un CSV d'historique (upload) + horizon_slots
                        renvoie le forecast et logue la prédiction dans predictions_log.csv
"""

import io
import csv
import os
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse

from predictor import DCForecaster

from prometheus_fastapi_instrumentator import Instrumentator



app = FastAPI(title="DCForecaster API", version="0.1.0")
Instrumentator().instrument(app).expose(app)

# Le modèle est chargé une seule fois au démarrage (couteux à recharger à chaque requête)
forecaster = DCForecaster()

LOG_FILE = "predictions_log.csv"
LOG_COLUMNS = [
    "timestamp",
    "reference_time",
    "horizon_slots",
    "n_rows_history",
    "mean_expected_arrival_count",
    "mean_expected_energy_kwh",
]


def _init_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_COLUMNS)


def _log_prediction(reference_time, horizon_slots, n_rows_history, forecast_data):
    _init_log_file()
    arrivals = [slot["expectedArrivalCount"] for slot in forecast_data]
    energies = [slot["expectedEnergyKwh"] for slot in forecast_data]
    mean_arrival = sum(arrivals) / len(arrivals) if arrivals else 0
    mean_energy = sum(energies) / len(energies) if energies else 0

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                reference_time,
                horizon_slots,
                n_rows_history,
                round(mean_arrival, 4),
                round(mean_energy, 4),
            ]
        )


@app.get("/")
def root():
    return {"message": "DCForecaster API is running"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="CSV d'historique: colonnes = timestamp, arrival_count, departure_count, avg_energy_kWh"),
    horizon_slots: int = Query(4, description="Nombre de créneaux de 15 min à prévoir"),
):
    try:
        raw = await file.read()
        history = pd.read_csv(io.BytesIO(raw), parse_dates=[0], index_col=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV invalide: {e}")

    required_cols = {"arrival_count", "departure_count", "avg_energy_kWh"}
    missing = required_cols - set(history.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Colonnes manquantes: {missing}")

    reference_time = history.index.max().ceil("15min")

    try:
        result = forecaster.forecast(
            reference_time=reference_time,
            history=history,
            horizon_slots=horizon_slots,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur du modèle: {e}")

    _log_prediction(
        reference_time=reference_time.isoformat(),
        horizon_slots=horizon_slots,
        n_rows_history=len(history),
        forecast_data=result["forecastData"],
    )

    return JSONResponse(content=result)
