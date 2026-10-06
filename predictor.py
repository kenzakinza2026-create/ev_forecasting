"""
predictor.py — DC charging forecast inference module.
Accepts a reference timestamp and a rolling history buffer,
returns a forecast JSON payload for the requested horizon.

Expected folder structure:
    predictor.py
    models/
      dc_v2/
        clf_arrival.json
        clf_departure.json
        reg_arrival_count.json
        reg_departure_count.json
        reg_energy.json
        reg_duration.json
        feature_cols.json
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

MODELS_DIR        = Path(__file__).parent / "models" / "dc_v2"
RESOLUTION_MINS   = 15
MAX_DURATION_MINS = 240.0

CALENDAR_COLS = [
    "hour", "minute", "day_of_week", "is_weekend", "month",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "day_sin",  "day_cos",  "month_sin",  "month_cos",
]
LAG_DISTANCES = {
    "t_minus_1": 1,  "t_minus_2": 2,  "t_minus_3": 3,
    "1h_ago":    4,  "24h_ago":  96,  "48h_ago": 192,  "1w_ago": 672,
}
LAG_SERIES_COL = {
    "arrivals":   "arrival_count",
    "departures": "departure_count",
    "energy":     "avg_energy_kWh",
}


class DCForecaster:
    def __init__(self):
        with open(MODELS_DIR / "feature_cols.json") as f:
            self._feature_cols = json.load(f)

        self._clf_arrival   = xgb.XGBClassifier()
        self._clf_departure = xgb.XGBClassifier()
        self._clf_arrival.load_model(str(MODELS_DIR / "clf_arrival.json"))
        self._clf_departure.load_model(str(MODELS_DIR / "clf_departure.json"))

        self._regs = {}
        for target, fname in [
            ("arrival_count",     "reg_arrival_count"),
            ("departure_count",   "reg_departure_count"),
            ("avg_energy_kWh",    "reg_energy"),
            ("avg_duration_mins", "reg_duration"),
        ]:
            r = xgb.XGBRegressor()
            r.load_model(str(MODELS_DIR / f"{fname}.json"))
            self._regs[target] = r

    def _time_features(self, ts: pd.Timestamp) -> dict:
        return {
            "hour":        ts.hour,
            "minute":      ts.minute,
            "day_of_week": ts.dayofweek,
            "is_weekend":  int(ts.dayofweek >= 5),
            "month":       ts.month,
            "hour_sin":    np.sin(2 * np.pi * ts.hour      / 24),
            "hour_cos":    np.cos(2 * np.pi * ts.hour      / 24),
            "minute_sin":  np.sin(2 * np.pi * ts.minute    / 60),
            "minute_cos":  np.cos(2 * np.pi * ts.minute    / 60),
            "day_sin":     np.sin(2 * np.pi * ts.dayofweek / 7),
            "day_cos":     np.cos(2 * np.pi * ts.dayofweek / 7),
            "month_sin":   np.sin(2 * np.pi * ts.month     / 12),
            "month_cos":   np.cos(2 * np.pi * ts.month     / 12),
        }

    def _build_row(self, slot_time: pd.Timestamp, buffer: pd.DataFrame) -> pd.DataFrame:
        feat = self._time_features(slot_time)
        for series, col in LAG_SERIES_COL.items():
            for lag_name, n_slots in LAG_DISTANCES.items():
                lag_ts = slot_time - pd.Timedelta(minutes=RESOLUTION_MINS * n_slots)
                feat[f"{series}_{lag_name}"] = (
                    float(buffer.at[lag_ts, col]) if lag_ts in buffer.index else 0.0
                )
        return pd.DataFrame([feat])[self._feature_cols]

    def _predict_slot(self, row: pd.DataFrame) -> dict:
        p_arr = float(self._clf_arrival.predict_proba(row)[:, 1][0])
        p_dep = float(self._clf_departure.predict_proba(row)[:, 1][0])
        return {
            "arrival_count":     p_arr * max(float(self._regs["arrival_count"].predict(row)[0]),     0.0),
            "departure_count":   p_dep * max(float(self._regs["departure_count"].predict(row)[0]),   0.0),
            "avg_energy_kWh":    p_arr * max(float(self._regs["avg_energy_kWh"].predict(row)[0]),    0.0),
            "avg_duration_mins": min(
                p_arr * max(float(self._regs["avg_duration_mins"].predict(row)[0]), 0.0),
                MAX_DURATION_MINS,
            ),
            "p_arrival":   p_arr,
            "p_departure": p_dep,
        }

    def forecast(
        self,
        reference_time: pd.Timestamp,
        history: pd.DataFrame,
        horizon_slots: int = 4,
    ) -> dict:
        """
        Parameters
        ----------
        reference_time : pd.Timestamp
            Start of the first forecast slot.
            Should be floored to the 15-min grid:
              pd.Timestamp.now().floor("15min")

        history : pd.DataFrame
            Timestamp-indexed DataFrame with exactly these 3 columns:
              arrival_count    (float)
              departure_count  (float)
              avg_energy_kWh   (float)
            One row per 15-min slot, covering at least 7 days before
            reference_time (15 days recommended to fill all lag features).
            Missing slots should be filled with 0.0, not NaN.

        horizon_slots : int
            Number of 15-min slots to forecast ahead.
              4  → 1 hour
              8  → 2 hours
             96  → 24 hours

        Returns
        -------
        dict
            Forecast payload ready to send to the THI controller.
            See forecastData entries for field descriptions.
        """
        SIGS   = ["arrival_count", "departure_count", "avg_energy_kWh"]
        buffer = history[SIGS].copy()

        slots = []
        for step in range(horizon_slots):
            slot_start = reference_time + pd.Timedelta(minutes=RESOLUTION_MINS * step)
            row        = self._build_row(slot_start, buffer)
            preds      = self._predict_slot(row)

            # feed prediction back as lag for the next step (autoregressive rollout)
            buffer.loc[slot_start] = [
                preds["arrival_count"],
                preds["departure_count"],
                preds["avg_energy_kWh"],
            ]

            slots.append({
                "timeSlotStart":           slot_start.isoformat(),
                "timeSlotEnd":             (slot_start + pd.Timedelta(minutes=RESOLUTION_MINS)).isoformat(),
                "stepsAhead":              step + 1,
                "minutesAhead":            (step + 1) * RESOLUTION_MINS,
                "expectedArrivalCount":    round(preds["arrival_count"],     4),
                "expectedDepartureCount":  round(preds["departure_count"],   4),
                "expectedEnergyKwh":       round(preds["avg_energy_kWh"],    4),
                "expectedDurationMinutes": round(preds["avg_duration_mins"], 1),
                "_meta": {
                    "pArrivalActive":   round(preds["p_arrival"],  4),
                    "pDepartureActive": round(preds["p_departure"], 4),
                },
            })

        return {
            "forecastTimestamp":    int(reference_time.timestamp() * 1000),
            "forecastTimestampIso": reference_time.isoformat(),
            "resolutionMinutes":    RESOLUTION_MINS,
            "horizonSlots":         horizon_slots,
            "forecastData":         slots,
        }