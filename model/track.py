"""
Tracking MLflow pour le forecaster Q95 (AC ou DC).

Ce qui change par rapport à la version dc_v2
--------------------------------------------
Le modèle ne prédit plus une espérance (p x régresseur) mais un QUANTILE 95 % du
mélange zéro-inflaté. NRMSE n'est donc plus une métrique adaptée : un Q95 est
volontairement au-dessus de la réalité, et un RMSE le pénaliserait pour ça.
Les métriques deviennent :

  Stage 1 (classifieurs, p CALIBRÉE)
      PR-AUC, Brier, taux de base.
  Résultat final, 1-step avec vrais lags ("oracle")     préfixe  onestep_
      pinball loss @alpha, couverture (cible ~ alpha), couverture sur slots
      actifs, skill vs prévision "toujours zéro".
  Stage 2 (magnitude, slots actifs)                     préfixe  stage2_
      couverture et pinball du Q95 CONDITIONNEL, NRMSE de la médiane (q50)
      -> seule métrique de continuité avec l'ancien NRMSE des régresseurs.
  Rollout autorégressif multi-origines, par horizon     préfixe  rollout_
      pinball, couverture, skill à h = 1, 2, 4, 8, 16 créneaux (<= --horizon).
      C'est le mode dans lequel le contrôleur consomme le modèle.

Le backtest ne porte plus sur 4 créneaux mais sur les --eval-days derniers jours.
L'historique doit contenir 15 jours AVANT cette fenêtre (features de lag).

Métriques enregistrées
----------------------
Toutes les métriques sont CALCULÉES et affichées dans le résumé final, mais
seules les plus utiles sont enregistrées :
  - MLFLOW_METRICS   (16) : comparaison des versions dans MLflow
  - GRAFANA_METRICS  (10) : surveillance dans Grafana (via Pushgateway)
Le détail par horizon du rollout reste disponible dans l'artefact
rollout_metrics_by_horizon.csv.

Le log de prédictions futures (prediction_logs/<modèle>/<version>.csv) et
l'attribution de l'alias @production sont conservés.

Usage (depuis le dossier qui contient track.py) :
    python track.py --pipeline dc --csv data/processed/ml_features_dc_train_val.csv
    python track.py --pipeline ac --csv data/processed/ml_features_ac_all.csv \
                    --horizon 16 --eval-days 14 --no-promote
"""

import argparse
import dataclasses
import os
import sys
import warnings
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

import common
from common import RESOLUTION_MINUTES, SIG_COLS, TARGETS, load_feature_frame
from configs import get_config
from forecast import MULTISTEP_HORIZONS, _q_grid_real, build_feature_row, load_models, predict_frame
from serving import EVForecasterModel, frame_to_payload, prepare_history, rollout_frame

# Configurable via variable d'environnement :
#   - dans Docker (réseau du docker-compose) : PUSHGATEWAY_URL=pushgateway:9091
#   - en exécution locale hors Docker : PUSHGATEWAY_URL=localhost:9091
PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "pushgateway:9091")
PREDICTION_LOGS_DIR = "prediction_logs"

MODEL_VERSION_TAG = "v3"   # distingue les runs v3 des anciens runs (v2) dans MLflow

RES = pd.Timedelta(minutes=RESOLUTION_MINUTES)
HISTORY_DAYS = 15          # identique à generate_forecast
EPS = 1e-9

# Alias courts : les noms de métriques doivent rester valides pour Prometheus.
ALIAS = {
    "arrival_count": "arrival",
    "departure_count": "departure",
    "avg_energy_kWh": "energy",
    "avg_duration_mins": "duration",
}

# Grafana (via Pushgateway) : surveillance uniquement (10 métriques)
GRAFANA_METRICS = (
    [f"onestep_skill_vs_zero_pct_{a}" for a in ALIAS.values()]
    + [f"onestep_coverage_pct_{a}" for a in ALIAS.values()]
    + ["stage1_pr_auc_arrival", "stage1_pr_auc_departure"]
)

# MLflow : les métriques Grafana + pinball + taux de base (16 métriques)
MLFLOW_METRICS = (
    GRAFANA_METRICS
    + [f"onestep_pinball_{a}" for a in ALIAS.values()]
    + ["stage1_base_rate_arrival", "stage1_base_rate_departure"]
)


# ---------------------------------------------------------------------------
# Métriques élémentaires
# ---------------------------------------------------------------------------
def pinball(y, q, alpha):
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1.0) * d)))


def coverage_pct(y, q):
    y, q = np.asarray(y, float), np.asarray(q, float)
    return float(100.0 * np.mean(y <= q + EPS)) if len(y) else float("nan")


def _target_metrics(y, q, alpha):
    """Pinball, couverture et skill vs 'toujours zéro' (pinball zéro = alpha * mean(y))."""
    y, q = np.asarray(y, float), np.asarray(q, float)
    active = y > 0
    pin = pinball(y, q, alpha)
    zero = alpha * float(np.mean(y))
    return {
        "pinball": pin,
        "coverage_pct": coverage_pct(y, q),
        "coverage_active_pct": coverage_pct(y[active], q[active]) if active.any() else float("nan"),
        "skill_vs_zero_pct": float(100.0 * (1.0 - pin / zero)) if zero > 0 else float("nan"),
        "n_active": int(active.sum()),
    }


def _classifier_metrics(y_true_active, y_prob):
    """PR-AUC et Brier pour un classifieur de stage 1 (p calibrée)."""
    y = np.asarray(y_true_active, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    brier = float(brier_score_loss(y, p))
    if len(np.unique(y)) < 2:
        warnings.warn("Une seule classe dans la fenêtre : PR-AUC non calculable (NaN).")
        return float("nan"), brier
    return float(average_precision_score(y, p)), brier


def _nrmse(y_true, y_pred):
    """NRMSE = RMSE / std(y_true). NaN si std == 0 ou < 2 points."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y_true) < 2:
        return float("nan")
    std = y_true.std()
    if std == 0:
        warnings.warn("std(y_true) == 0 : NRMSE non calculable (NaN).")
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / std)


# ---------------------------------------------------------------------------
# Chargement de l'historique
# ---------------------------------------------------------------------------
def load_history(cfg, csv_path):
    """
    Charge le CSV avec EXACTEMENT le prétraitement de production
    (filtre chargerType + zone, cap de durée sur les cibles, carry-forward).
    Retourne un DataFrame indexé par timestamp, colonnes SIG_COLS, sur une grille
    de 15 min sans trou.
    """
    cfg_csv = dataclasses.replace(cfg, data_path=Path(csv_path))
    frame = load_feature_frame(cfg_csv)
    frame = frame[frame["zone"] == cfg.default_zone]
    if frame.empty:
        raise common.DataValidationError(
            f"Aucune ligne pour la zone {cfg.default_zone!r} dans {csv_path}.")

    history = frame.set_index("timestamp")[SIG_COLS].sort_index()
    if history.index.duplicated().any():
        raise ValueError("Timestamps dupliqués pour cette zone dans le CSV.")

    full = pd.date_range(history.index.min(), history.index.max(), freq=f"{RESOLUTION_MINUTES}min")
    n_gaps = len(full) - len(history)
    if n_gaps:
        warnings.warn(
            f"{n_gaps} créneaux manquants remplis par 0 (le modèle fait pareil en "
            f"production). Les métriques sur ces créneaux sont peu fiables.")
        history = history.reindex(full, fill_value=0.0)
    history.index.name = "timestamp"
    return history, int(n_gaps)


def eval_index(history, eval_days):
    """Créneaux évalués : les eval_days derniers jours, avec 15 jours d'historique avant."""
    first_ok = history.index.min() + pd.Timedelta(days=HISTORY_DAYS)
    start = max(first_ok, history.index.max() - pd.Timedelta(days=eval_days) + RES)
    idx = history.index[history.index >= start]
    if len(idx) == 0:
        raise ValueError(
            f"Historique trop court : il faut plus de {HISTORY_DAYS} jours "
            f"+ la fenêtre d'évaluation.")
    return idx


# ---------------------------------------------------------------------------
# Évaluation 1-step (vrais lags)
# ---------------------------------------------------------------------------
def _build_X(history, index):
    rows = []
    for slot in index:
        past = history.loc[slot - pd.Timedelta(days=HISTORY_DAYS): slot - RES]
        rows.append(build_feature_row(slot, past))
    return pd.concat(rows, ignore_index=True)


def evaluate_one_step(bundle, mix, history, idx):
    """
    Pour chaque créneau de idx, features construites depuis les VRAIES valeurs
    passées, puis prédiction vectorisée. Isole la qualité du modèle de
    l'accumulation d'erreurs du rollout.
    """
    X = _build_X(history, idx)
    out = predict_frame(bundle, X, mix)
    truth = history.loc[idx]
    metrics, n_active = {}, {}

    # Résultat final (tous les slots)
    for tgt in TARGETS:
        a = ALIAS[tgt]
        m = _target_metrics(truth[tgt].to_numpy(float), out["preds"][tgt], mix.alpha)
        metrics[f"onestep_pinball_{a}"] = m["pinball"]
        metrics[f"onestep_coverage_pct_{a}"] = m["coverage_pct"]
        metrics[f"onestep_coverage_active_pct_{a}"] = m["coverage_active_pct"]
        metrics[f"onestep_skill_vs_zero_pct_{a}"] = m["skill_vs_zero_pct"]
        n_active[f"n_active_{a}"] = m["n_active"]

    # Stage 1
    for a, ptag, col in (("arrival", "p_arrival", "arrival_count"),
                         ("departure", "p_departure", "departure_count")):
        y_bin = truth[col].to_numpy(float) > 0
        pr, br = _classifier_metrics(y_bin, out[ptag])
        metrics[f"stage1_pr_auc_{a}"] = pr
        metrics[f"stage1_brier_{a}"] = br
        metrics[f"stage1_base_rate_{a}"] = float(y_bin.mean())

    # Stage 2 (slots actifs uniquement)
    j50 = int(np.argmin(np.abs(np.asarray(mix.grid) - 0.5)))
    for tgt in TARGETS:
        a = ALIAS[tgt]
        y = truth[tgt].to_numpy(float)
        act = y > 0
        if act.sum() < 2:
            warnings.warn(f"< 2 slots actifs pour {tgt} : métriques Stage 2 ignorées.")
            continue
        q50 = _q_grid_real(bundle, tgt, X)[:, j50]
        qm = np.asarray(out["q_mags"][tgt], float)
        if tgt == "avg_duration_mins":            # la vérité est plafonnée : comparer plafonné
            q50, qm = np.minimum(q50, bundle.cap), np.minimum(qm, bundle.cap)
        metrics[f"stage2_coverage_cond_pct_{a}"] = coverage_pct(y[act], qm[act])
        metrics[f"stage2_pinball_cond_{a}"] = pinball(y[act], qm[act], mix.alpha)
        metrics[f"stage2_nrmse_q50_{a}"] = _nrmse(y[act], q50[act])

    return metrics, n_active


# ---------------------------------------------------------------------------
# Évaluation rollout multi-origines
# ---------------------------------------------------------------------------
def evaluate_rollout(bundle, mix, history, idx, max_horizon, method):
    """
    Rollouts autorégressifs depuis plusieurs origines. Le pas des origines est
    premier avec 96 (méthode 'coprime') pour ne pas confondre horizon et heure du
    jour ; 'all' prend chaque créneau (plus lent, plus de slots actifs).
    """
    last_ok = history.index.max() - (max_horizon - 1) * RES
    eligible = idx[idx <= last_ok]
    origins = common.sample_origins(eligible, max_horizon=max_horizon, method=method)
    if len(origins) == 0:
        raise ValueError("Aucune origine éligible pour le rollout (fenêtre trop courte).")
    print(f"[Rollout] {len(origins)} origines x {max_horizon} pas (méthode={method})")

    parts = []
    for i, origin in enumerate(origins, 1):
        fr = rollout_frame(bundle, mix, origin, history, max_horizon, HISTORY_DAYS)
        truth = history.reindex(fr["timestamp"])
        for t in TARGETS:
            fr[f"{t}_true"] = truth[t].to_numpy(float)
        parts.append(fr)
        if i % 25 == 0:
            print(f"[Rollout]   {i}/{len(origins)}")
    roll = pd.concat(parts, ignore_index=True)

    metrics, rows = {}, []
    for h in [x for x in MULTISTEP_HORIZONS if x <= max_horizon]:
        sub = roll[roll["step"] == h]
        for tgt in TARGETS:
            a = ALIAS[tgt]
            m = _target_metrics(sub[f"{tgt}_true"].to_numpy(float),
                                sub[f"{tgt}_pred"].to_numpy(float), mix.alpha)
            metrics[f"rollout_pinball_{a}_h{h}"] = m["pinball"]
            metrics[f"rollout_coverage_pct_{a}_h{h}"] = m["coverage_pct"]
            metrics[f"rollout_skill_vs_zero_pct_{a}_h{h}"] = m["skill_vs_zero_pct"]
            rows.append({"target": tgt, "horizon_slots": h,
                         "horizon_mins": h * RESOLUTION_MINUTES,
                         "n": len(sub), **m})
    return metrics, pd.DataFrame(rows), len(origins), roll


# ---------------------------------------------------------------------------
# Prédictions RÉELLES sur des timestamps futurs (PAS un backtest)
# ---------------------------------------------------------------------------
def predict_future(bundle, mix, history, horizon_slots):
    """Rollout au-delà de la fin de l'historique : aucune vérité disponible."""
    start = history.index.max() + RES
    return start, rollout_frame(bundle, mix, start, history, horizon_slots, HISTORY_DAYS)


def _save_prediction_log(frame, mix, model_name, version):
    """prediction_logs/<model_name>/<version>.csv"""
    a = int(round(mix.alpha * 100))
    out = pd.DataFrame({
        "timestamp": frame["timestamp"],
        f"Q{a}_arrivals": frame["arrival_count_pred"],
        f"Q{a}_departures": frame["departure_count_pred"],
        f"Q{a}_energy_kWh": frame["avg_energy_kWh_pred"],
        f"Q{a}_duration_min": frame["avg_duration_mins_pred"],
        "p_arrival": frame["p_arrival"],
        "p_departure": frame["p_departure"],
        "conditional_level": frame["cond_level"],
    }).set_index("timestamp")

    out_dir = os.path.join(PREDICTION_LOGS_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{version}.csv")
    out.round(3).to_csv(path)
    print(f"[PredictionLog] Sauvegardé : {path}")
    return path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _is_nan(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def _log_metric_safe(name, value):
    if _is_nan(value):
        print(f"[WARNING] Métrique '{name}' = NaN, non loggée.")
        return
    mlflow.log_metric(name, float(value))


def _push_metrics_to_gateway(metrics, job_name):
    registry = CollectorRegistry()
    for name, value in metrics.items():
        if _is_nan(value):
            continue
        Gauge(name, f"Metric {name} from track.py", registry=registry).set(float(value))
    try:
        push_to_gateway(PUSHGATEWAY_URL, job=job_name, registry=registry)
        print(f"[Pushgateway] Métriques poussées vers {PUSHGATEWAY_URL} (job={job_name})")
    except Exception as exc:
        print(f"[WARNING] Échec du push vers Pushgateway : {exc}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_tracking(pipeline, csv_path, horizon_slots, eval_days, origin_method,
                 model_name, promote):
    cfg = get_config(pipeline)
    model_name = model_name or f"{cfg.name}Forecaster_v3"

    bundle = load_models(cfg)              # vérifie que charger_type des modèles == config
    mix = bundle.mixture("exact")

    history, n_gaps = load_history(cfg, csv_path)
    history = prepare_history(history, bundle.cap)   # no-op si déjà préparé, sécurité cap modèle
    idx = eval_index(history, eval_days)

    # ---------------- Backtests ----------------
    one_metrics, n_active = evaluate_one_step(bundle, mix, history, idx)
    roll_metrics, roll_table, n_origins, _ = evaluate_rollout(
        bundle, mix, history, idx, horizon_slots, origin_method)
    metrics = {**one_metrics, **roll_metrics}

    min_active = cfg.min_active_slots
    for k, v in n_active.items():
        if v < min_active:
            print(f"[WARNING] {k}={v} < {min_active} : métriques de cette cible indicatives seulement.")

    # ---------------- Prédictions futures ----------------
    ref_time, future = predict_future(bundle, mix, history, horizon_slots)
    sample_output = frame_to_payload(future, pipeline=cfg.name, zone=cfg.default_zone,
                                     mix=mix, reference_time=ref_time, cap=bundle.cap)
    sample_input = history.tail(200)

    mlflow.set_experiment(f"{cfg.name}Forecaster_production")

    with mlflow.start_run() as run:
        mlflow.set_tag("model_version", MODEL_VERSION_TAG)
        mlflow.log_params({
            "model_name": model_name,
            "pipeline": cfg.name,
            "charger_type": cfg.charger_type,
            "zone": cfg.default_zone,
            "csv_input": os.path.basename(csv_path),
            "horizon_slots": horizon_slots,
            "eval_days": eval_days,
            "origin_method": origin_method,
            "n_origins": n_origins,
            "n_rows_history": len(history),
            "n_gap_slots_filled": n_gaps,
            "n_eval_slots": len(idx),
            "eval_start": str(idx.min()),
            "eval_end": str(idx.max()),
            "quantile_alpha": mix.alpha,
            "mixture_mode": mix.mode,
            "duration_cap_mins": bundle.cap,
        })
        mlflow.log_params({f"train_{k}": v for k, v in bundle.config.get("splits", {}).items()})
        for name, value in n_active.items():
            mlflow.log_param(name, value)

        # MLflow : uniquement les métriques utiles pour comparer les versions
        for name in MLFLOW_METRICS:
            if name in metrics:
                _log_metric_safe(name, metrics[name])

        # Grafana (Pushgateway) : uniquement les métriques de surveillance
        grafana = {k: metrics[k] for k in GRAFANA_METRICS if k in metrics}
        _push_metrics_to_gateway(grafana, job_name=f"{cfg.name.lower()}forecaster_v3_track")

        roll_path = "rollout_metrics_by_horizon.csv"
        roll_table.round(5).to_csv(roll_path, index=False)
        mlflow.log_artifact(roll_path)
        mlflow.log_artifact(str(bundle.dir / "model_config.json"))

        try:
            signature = infer_signature(sample_input, sample_output)
        except Exception as exc:
            print("[WARNING] Signature sortie impossible :", exc)
            signature = infer_signature(sample_input)

        # ---- Enregistrement dans le Model Registry ----
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=EVForecasterModel(cfg.name.lower()),
            artifacts={"models_dir": str(bundle.dir)},
            # fichiers à plat : les modules font des imports nus (import common, ...)
            code_paths=sorted(str(p) for p in PIPELINE_DIR.glob("*.py")),
            pip_requirements=["xgboost==3.3.0", "pandas==2.3.3", "scikit-learn", "numpy"],
            signature=signature,
            registered_model_name=model_name,
        )
        version = model_info.registered_model_version

        if promote:
            MlflowClient().set_registered_model_alias(
                name=model_name, alias="production", version=version)

        log_path = _save_prediction_log(future, mix, model_name, version)
        mlflow.log_artifact(log_path)

        # ---------------- Résumé (toutes les métriques calculées) ----------------
        print("\n==============================")
        print("MLflow Tracking terminé")
        print("==============================")
        print(f"Run ID : {run.info.run_id}")
        print(f"Version MLflow : {version}   alias @production : {'oui' if promote else 'NON (--no-promote)'}")
        print(f"Fenêtre évaluée : {idx.min()} -> {idx.max()}  ({len(idx)} slots)")
        print(f"Slots actifs : {n_active}")
        print("--- Stage 1 (p calibrée) ---")
        for a in ("arrival", "departure"):
            print(f"{a:<10} PR-AUC {metrics[f'stage1_pr_auc_{a}']:.4f}   "
                  f"Brier {metrics[f'stage1_brier_{a}']:.4f}   "
                  f"taux de base {metrics[f'stage1_base_rate_{a}']:.4f}")
        print(f"--- Résultat final 1-step (cible de couverture : {100 * mix.alpha:.0f} %) ---")
        for tgt in TARGETS:
            a = ALIAS[tgt]
            print(f"{a:<10} pinball {metrics[f'onestep_pinball_{a}']:.4f}   "
                  f"couv. {metrics[f'onestep_coverage_pct_{a}']:.2f} %   "
                  f"couv. actifs {metrics[f'onestep_coverage_active_pct_{a}']:.1f} %   "
                  f"skill {metrics[f'onestep_skill_vs_zero_pct_{a}']:.1f} %")
        print("--- Rollout par horizon (skill vs zéro, %) ---")
        pivot = roll_table.pivot(index="target", columns="horizon_mins",
                                 values="skill_vs_zero_pct").round(1)
        print(pivot.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tracking MLflow du forecaster Q95")
    ap.add_argument("--pipeline", choices=["ac", "dc"], default="dc")
    ap.add_argument("--csv", required=True, help="CSV historique (format README §3)")
    ap.add_argument("--horizon", type=int, default=16,
                    help="Créneaux de 15 min : profondeur du rollout et des prédictions futures")
    ap.add_argument("--eval-days", type=int, default=14,
                    help="Nombre de jours de fin d'historique utilisés pour le backtest")
    ap.add_argument("--origin-method", choices=["coprime", "all"], default="coprime")
    ap.add_argument("--model-name", default=None,
                    help="Nom dans le Model Registry (défaut : <AC|DC>Forecaster_v3)")
    ap.add_argument("--no-promote", action="store_true",
                    help="Ne pas déplacer l'alias @production vers cette version")
    args = ap.parse_args()

    run_tracking(args.pipeline, args.csv, args.horizon, args.eval_days,
                 args.origin_method, args.model_name, not args.no_promote)