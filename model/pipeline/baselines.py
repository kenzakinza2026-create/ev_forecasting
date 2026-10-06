"""
baselines.py — The two-stage MSE baseline, trained once and persisted.

Why it lives here rather than inside ablation.py
------------------------------------------------
`two_stage_mse` is the ablation that isolates what the QUANTILE LOSS buys: it has
the same hurdle architecture as the proposed model (a classifier gate times a
regressor on active slots) and differs only in that the second stage targets the
conditional MEAN instead of a conditional quantile.

It is now also the comparator in the multi-step degradation analysis. A baseline
that is trained in two places drifts, and a degradation curve computed against a
differently-fitted model is not the ablation it claims to be — so it is fitted
once by train.py, saved beside the proposed models, and loaded by both
ablation.py and the rollout.

Composition
-----------
    ŷ = P(active | x) · E[y | active, x]

This is the standard hurdle point forecast: the expected value of the mixture.
It is a MEAN, so it is scored on pinball loss knowing it targets a different
estimand — that gap is exactly the thing the ablation is measuring, and the
paper must say so in the caption.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from common import MODEL_KEY, TARGETS, extract_features_for_stage

SUBDIR = "baseline_two_stage_mse"

_FIXED = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=1)


def fit(train: pd.DataFrame, valid: pd.DataFrame, target: str, seed: int = 42):
    """Classifier gate + conditional-mean regressor for one target."""
    X_tr1, cols1 = extract_features_for_stage(train, 1, target)
    X_va1, _ = extract_features_for_stage(valid, 1, target)
    X_tr2, cols2 = extract_features_for_stage(train, 2, target)
    X_va2, _ = extract_features_for_stage(valid, 2, target)
    y_tr = train[target].to_numpy(float)
    y_va = valid[target].to_numpy(float)

    clf = xgb.XGBClassifier(objective="binary:logistic", **_FIXED, random_state=seed,
                            early_stopping_rounds=30, eval_metric="logloss")
    clf.fit(X_tr1, (y_tr > 0).astype(int),
            eval_set=[(X_va1, (y_va > 0).astype(int))], verbose=False)

    a_tr, a_va = y_tr > 0, y_va > 0
    if a_tr.sum() < 10 or a_va.sum() < 2:
        return clf, None, cols1, cols2

    reg = xgb.XGBRegressor(objective="reg:squarederror", **_FIXED, random_state=seed,
                           early_stopping_rounds=30, eval_metric="rmse")
    reg.fit(X_tr2[a_tr], y_tr[a_tr],
            eval_set=[(X_va2[a_va], y_va[a_va])], verbose=False)
    return clf, reg, cols1, cols2


def train_and_save(models_dir: Path, train: pd.DataFrame, valid: pd.DataFrame,
                   seed: int = 42, verbose: bool = True) -> dict:
    out = Path(models_dir) / SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    feature_cols: dict[str, list[str]] = {}
    info: dict[str, dict] = {}

    for tgt in TARGETS:
        clf, reg, cols1, cols2 = fit(train, valid, tgt, seed)
        key = MODEL_KEY[tgt]
        clf.save_model(str(out / f"clf_{key}.json"))
        feature_cols[f"clf_{key}"] = cols1
        if reg is not None:
            reg.save_model(str(out / f"reg_{key}.json"))
            feature_cols[f"reg_{key}"] = cols2
        info[tgt] = {"has_regressor": reg is not None,
                     "n_active_train": int((train[tgt] > 0).sum())}

    (out / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2))
    (out / "info.json").write_text(json.dumps(info, indent=2))
    if verbose:
        print(f"    two-stage MSE baseline saved → {out}")
    return info


class MseBundle:
    """Loaded two-stage MSE baseline. Same call shape as the proposed model."""

    def __init__(self, models_dir: Path):
        d = Path(models_dir) / SUBDIR
        if not (d / "feature_cols.json").exists():
            raise FileNotFoundError(
                f"No two-stage MSE baseline in {d}. Re-run train.py — it is fitted "
                f"there so the ablation and the rollout share one definition."
            )
        self.dir = d
        self.feature_cols = json.loads((d / "feature_cols.json").read_text())
        self.clf: dict[str, xgb.XGBClassifier] = {}
        self.reg: dict[str, xgb.XGBRegressor | None] = {}
        for tgt in TARGETS:
            key = MODEL_KEY[tgt]
            c = xgb.XGBClassifier(); c.load_model(str(d / f"clf_{key}.json"))
            self.clf[tgt] = c
            rp = d / f"reg_{key}.json"
            if rp.exists():
                r = xgb.XGBRegressor(); r.load_model(str(rp))
                self.reg[tgt] = r
            else:
                self.reg[tgt] = None

    def predict(self, X: pd.DataFrame, cap: float | None = None) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for tgt in TARGETS:
            key = MODEL_KEY[tgt]
            p = self.clf[tgt].predict_proba(X[self.feature_cols[f"clf_{key}"]])[:, 1]
            if self.reg[tgt] is None:
                out[tgt] = np.zeros(len(X), float)
                continue
            mu = np.maximum(self.reg[tgt].predict(X[self.feature_cols[f"reg_{key}"]]), 0.0)
            v = np.maximum(p * mu, 0.0)
            if cap is not None and tgt == "avg_duration_mins":
                v = np.minimum(v, cap)
            out[tgt] = v
        return out
