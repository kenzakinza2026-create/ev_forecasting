"""
serving.py — Inférence à partir d'un historique en mémoire + wrapper MLflow PyFunc.

Pourquoi ce fichier existe
--------------------------
`forecast.generate_forecast` lit son historique dans cfg.data_path, ce qui ne
convient ni à un modèle MLflow (l'historique arrive en entrée) ni au backtest de
track.py (on veut rejouer la même boucle depuis n'importe quelle origine).

`rollout_frame` reproduit donc, ligne pour ligne, la boucle de generate_forecast :
  - buffer = les `history_days` jours strictement avant reference_time
  - build_feature_row -> predict_step
  - réinjection des prévisions (Q95) dans le buffer, avec avg_power_kW recalculé

Il faut le garder synchronisé avec forecast.generate_forecast : c'est ce qui
garantit que les métriques de track.py décrivent le comportement de production.

Ce fichier doit être placé dans pipeline/ (imports nus : `import common`).
"""

from __future__ import annotations

from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd

import common
from common import MixtureConfig, RESOLUTION_MINUTES, SIG_COLS, TARGETS
from forecast import build_feature_row, predict_step

_RES = pd.Timedelta(minutes=RESOLUTION_MINUTES)


# ── Préparation de l'historique ───────────────────────────────────────────────

def prepare_history(history: pd.DataFrame, cap: float) -> pd.DataFrame:
    """
    Applique à un historique brut les mêmes transformations que
    common.load_feature_frame : cap de durée sur la cible, puissance recalculée,
    puis carry-forward de avg_power_kW.

    `history` : index = timestamps, au moins arrival_count, departure_count,
    avg_energy_kWh. Si avg_duration_mins manque, elle est mise à 0 (la durée
    sera alors mal prévue : fournissez-la).
    """
    h = history.copy().sort_index()
    h = h[~h.index.duplicated(keep="last")]
    for c in SIG_COLS:
        if c not in h.columns:
            h[c] = 0.0
    h = h[SIG_COLS].astype(float)
    h = common.apply_duration_cap(h, cap, verbose=False)      # recalcule avg_power_kW
    s = h["avg_power_kW"].replace(0.0, np.nan)
    h["avg_power_kW"] = s.ffill().fillna(0.0)
    return h[SIG_COLS]


# ── Rollout ───────────────────────────────────────────────────────────────────

def rollout_frame(bundle, mix: MixtureConfig, reference_time, history: pd.DataFrame,
                  horizon_slots: int, history_days: int = 15) -> pd.DataFrame:
    """
    Rollout autorégressif depuis reference_time (premier créneau prévu).

    `history` doit déjà être préparé (prepare_history ou load_feature_frame).
    Retourne une ligne par créneau : timestamp, step, p_arrival, p_departure,
    cond_level, et pour chaque cible <t>_pred (Q95 du mélange) et <t>_qmag
    (Q95 conditionnel Stage 2, non plafonné).
    """
    reference_time = pd.Timestamp(reference_time)
    cutoff = reference_time - pd.Timedelta(days=history_days)
    buffer = history[(history.index >= cutoff) & (history.index < reference_time)][SIG_COLS].copy()
    if buffer.empty:
        raise common.DataValidationError(
            f"Aucun historique dans les {history_days} jours avant {reference_time}.")

    recs = []
    for step in range(horizon_slots):
        slot = reference_time + pd.Timedelta(minutes=int(RESOLUTION_MINUTES * step))
        feat = build_feature_row(slot, buffer)
        preds, p_arr, p_dep, q_mags = predict_step(feat, bundle, mix)

        rec = {
            "timestamp": slot, "step": step + 1,
            "p_arrival": p_arr, "p_departure": p_dep,
            "cond_level": float(common.conditional_level(p_arr, mix.alpha)),
        }
        for t in TARGETS:
            rec[f"{t}_pred"] = preds[t]
            rec[f"{t}_qmag"] = q_mags[t]
        recs.append(rec)

        nxt = {t: preds[t] for t in TARGETS}
        dur_h = preds["avg_duration_mins"] / 60.0
        nxt["avg_power_kW"] = (preds["avg_energy_kWh"] / dur_h) if dur_h > 0 else 0.0
        buffer = pd.concat([buffer, pd.DataFrame([nxt], index=[slot])])

    return pd.DataFrame(recs)


def frame_to_payload(frame: pd.DataFrame, *, pipeline: str, zone: str, mix: MixtureConfig,
                     reference_time, cap: float) -> dict:
    """Même schéma JSON que forecast.generate_forecast (contrat avec le contrôleur)."""
    a = int(round(mix.alpha * 100))
    slots = []
    for _, r in frame.iterrows():
        step = int(r["step"])
        slots.append({
            "timeSlotStart": pd.Timestamp(r["timestamp"]).isoformat(),
            "stepsAhead": step,
            "minutesAhead": step * RESOLUTION_MINUTES,
            f"Q{a}_arrivalCount": round(float(r["arrival_count_pred"]), 4),
            f"Q{a}_departureCount": round(float(r["departure_count_pred"]), 4),
            f"Q{a}_energyKwh": round(float(r["avg_energy_kWh_pred"]), 4),
            f"Q{a}_durationMins": round(float(r["avg_duration_mins_pred"]), 1),
            "pArrivalActive": round(float(r["p_arrival"]), 4),
            "pDepartureActive": round(float(r["p_departure"]), 4),
            "conditionalLevelUsed": round(float(r["cond_level"]), 4),
            "stage2_arrivalCount_qmag": round(float(r["arrival_count_qmag"]), 4),
            "stage2_energyKwh_qmag": round(float(r["avg_energy_kWh_qmag"]), 4),
            "stage2_durationMins_qmag": round(float(r["avg_duration_mins_qmag"]), 1),
        })
    return {
        "forecastReferenceTime": pd.Timestamp(reference_time).isoformat(),
        "pipeline": pipeline, "zone": zone, "quantileAlpha": mix.alpha,
        "mixtureInversion": mix.mode, "resolutionMinutes": RESOLUTION_MINUTES,
        "durationCapMins": cap, "forecastData": slots,
    }


# ── Wrapper MLflow ────────────────────────────────────────────────────────────

class EVForecasterModel(mlflow.pyfunc.PythonModel):
    """
    Entrée  : DataFrame indexé par timestamp (ou avec une colonne `timestamp`),
              colonnes arrival_count, departure_count, avg_energy_kWh,
              avg_duration_mins (+ avg_power_kW facultatif), 15 jours au moins.
    Params  : horizon_slots (défaut 4), reference_time (défaut : dernier
              créneau observé + 15 min).
    Sortie  : le même JSON que generate_forecast.
    """

    def __init__(self, pipeline: str):
        self.pipeline = pipeline

    def load_context(self, context):
        import configs
        import forecast

        self.cfg = configs.get_config(self.pipeline)
        self.bundle = forecast.load_models(self.cfg, Path(context.artifacts["models_dir"]))
        self.mix = self.bundle.mixture("exact")

    def predict(self, context, model_input: pd.DataFrame, params=None):
        params = params or {}
        horizon = int(params.get("horizon_slots", 4))

        hist = model_input.copy()
        if "timestamp" in hist.columns:
            hist.index = pd.to_datetime(hist["timestamp"])
        hist = prepare_history(hist, self.bundle.cap)

        ref = pd.Timestamp(params.get("reference_time", hist.index.max() + _RES))
        frame = rollout_frame(self.bundle, self.mix, ref, hist, horizon)
        return frame_to_payload(frame, pipeline=self.cfg.name, zone=self.cfg.default_zone,
                                mix=self.mix, reference_time=ref, cap=self.bundle.cap)