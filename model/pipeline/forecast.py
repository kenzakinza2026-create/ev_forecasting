"""
forecast.py — Inference and evaluation shared by the AC and DC pipelines.

Run as:
    python pipeline/forecast.py --pipeline dc --eval-only
    python pipeline/forecast.py --pipeline ac --multistep-eval --horizon-slots 16
    python pipeline/forecast.py --pipeline ac --time "2026-08-25T08:00" --horizon 4h
    python pipeline/ac_forecast.py --eval-only          # thin wrapper, same thing

What changed
------------
MIX  Inference applies the exact mixture inversion: the Stage-2 quantile function
     is read at level (alpha-1+p)/p rather than at a fixed alpha. `--inversion
     legacy` reproduces the old fixed-alpha behaviour from the same fitted models,
     which is what makes the two comparable in the ablation.

F2   The duration cap is applied to targets at load as well as to predictions, so
     a capped forecast is no longer scored against uncapped ground truth.

SKEW build_feature_row and the vectorised dataframe path now produce the same
     features, including the avg_power_kW carry-forward. Previously the AC
     ablation used the row-by-row builder and the DC ablation read precomputed
     columns, so the two arms were not measuring the same thing.

F3-F6 Reporting uses metrics.compute_metrics and aggregates per zone rather than
     pooling raw rows, so MASE stays defined and all-zero zones are separated out.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore", category=DeprecationWarning)

import baselines
import common
import rollout
from common import (
    LAG_SERIES, LAG_SLOTS, MODEL_KEY, MixtureConfig, PipelineConfig,
    QUANTILE_GRID, RESOLUTION_MINUTES, SEASONAL_PERIOD, SIG_COLS, TARGETS,
    interpolate_quantile, load_feature_frame, mixture_quantile, split_frame,
    zone_activity,
)
from configs import get_config, add_window_flag
import metrics as M

MULTISTEP_HORIZONS = [1, 2, 4, 8, 16]
RULE = "─" * 78


# ── Feature construction ──────────────────────────────────────────────────────

def _time_features(ts: pd.Timestamp) -> dict:
    return {
        "hour": ts.hour, "minute": ts.minute, "day_of_week": ts.dayofweek,
        "is_weekend": int(ts.dayofweek >= 5), "month": ts.month,
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * ts.hour / 24.0),
        "minute_sin": np.sin(2 * np.pi * ts.minute / 60.0),
        "minute_cos": np.cos(2 * np.pi * ts.minute / 60.0),
        "day_sin": np.sin(2 * np.pi * ts.dayofweek / 7.0),
        "day_cos": np.cos(2 * np.pi * ts.dayofweek / 7.0),
        "month_sin": np.sin(2 * np.pi * ts.month / 12.0),
        "month_cos": np.cos(2 * np.pi * ts.month / 12.0),
    }


def _last_nonzero(series: pd.Series) -> float:
    nz = series.replace(0.0, np.nan).dropna()
    return float(nz.iloc[-1]) if len(nz) else 0.0


def build_feature_row(slot_time: pd.Timestamp, history: pd.DataFrame) -> pd.DataFrame:
    """
    One feature row for slot_time, using only history strictly before it.

    Carry-forward is applied to BOTH avg_duration_mins and avg_power_kW so this
    matches the dataframe produced by common.load_feature_frame. That parity is
    the fix for the train/serve skew between the two ablation paths.
    """
    feat = _time_features(slot_time)
    past = history[history.index < slot_time]

    last_dur = _last_nonzero(past["avg_duration_mins"]) if "avg_duration_mins" in past else 0.0

    for series, col in LAG_SERIES.items():
        for lag_name, n_slots in LAG_SLOTS.items():
            lag_time = slot_time - pd.Timedelta(minutes=int(RESOLUTION_MINUTES * n_slots))
            if col in past.columns and lag_time in past.index:
                val = float(past.at[lag_time, col])
                if col == "avg_duration_mins" and val == 0.0:
                    val = last_dur
            else:
                val = last_dur if col == "avg_duration_mins" else 0.0
            feat[f"{series}_{lag_name}"] = val

    if "arrival_count" in past.columns and len(past):
        is_active = (past["arrival_count"] > 0).astype(int)
        nz = np.flatnonzero(past["arrival_count"].to_numpy() > 0)
        feat["bins_since_last_arrival"] = float(len(past) - 1 - nz[-1]) if nz.size else float(len(past))
        feat["rolling_active_4h"] = float(is_active.iloc[-16:].mean())
        feat["rolling_active_24h"] = float(is_active.iloc[-96:].mean())
        same_hour = is_active[is_active.index.hour == slot_time.hour]
        feat["zone_hour_active_rate"] = float(same_hour.mean()) if len(same_hour) else 0.0
    else:
        feat["bins_since_last_arrival"] = 0.0
        feat["rolling_active_4h"] = 0.0
        feat["rolling_active_24h"] = 0.0
        feat["zone_hour_active_rate"] = 0.0

    feat["avg_power_kW"] = _last_nonzero(past["avg_power_kW"]) if "avg_power_kW" in past else 0.0
    return pd.DataFrame([feat])


# ── Model loading ─────────────────────────────────────────────────────────────

class Bundle:
    """Loaded models plus the config they were trained under."""

    def __init__(self, cfg: PipelineConfig, models_dir: Path | None = None):
        md = models_dir or cfg.models_dir
        self.cfg = cfg
        self.dir = md

        self.clf_arrival = xgb.XGBClassifier()
        self.clf_arrival.load_model(str(md / "clf_arrival.json"))
        self.clf_departure = xgb.XGBClassifier()
        self.clf_departure.load_model(str(md / "clf_departure.json"))

        self.regs: dict[str, xgb.XGBRegressor] = {}
        for tgt, key in MODEL_KEY.items():
            r = xgb.XGBRegressor()
            r.load_model(str(md / f"{key}.json"))
            self.regs[tgt] = r

        self.feature_cols = json.loads((md / "feature_cols.json").read_text())

        # Stage-1 isotonic calibrators, if training produced them.
        self.calibration: dict[str, dict | None] = {}
        for key in ("clf_arrival", "clf_departure"):
            path = md / f"calibration_{key}.json"
            self.calibration[key] = json.loads(path.read_text()) if path.exists() else None
        self.config = common.load_model_config(md)
        self.grid = tuple(self.config.get("quantile_grid", QUANTILE_GRID))
        self.alpha = float(self.config.get("quantile_alpha", 0.95))
        log_targets = self.config.get("log_transform_targets", {})
        self.use_log = {tgt: bool(log_targets.get(key, False)) for tgt, key in MODEL_KEY.items()}
        self.cap = float(self.config.get("duration_cap_mins", cfg.duration_cap))

        stored = self.config.get("charger_type")
        if stored and stored != cfg.charger_type:
            raise common.DataValidationError(
                f"Models in {md} were trained on chargerType={stored!r} but this run "
                f"expects {cfg.charger_type!r}. Refusing to mix pipelines."
            )

    def mixture(self, mode: str = "exact") -> MixtureConfig:
        return MixtureConfig(alpha=self.alpha, mode=mode, grid=self.grid)


def load_models(cfg: PipelineConfig, models_dir: Path | None = None) -> Bundle:
    return Bundle(cfg, models_dir)


# ── Prediction ────────────────────────────────────────────────────────────────

def _q_grid_real(bundle: Bundle, target: str, X: pd.DataFrame) -> np.ndarray:
    """Stage-2 grid predictions for one target, in real-world units."""
    raw = np.atleast_2d(bundle.regs[target].predict(X[bundle.feature_cols[MODEL_KEY[target]]]))
    if raw.shape[0] != len(X):
        raw = raw.T
    raw = np.sort(raw, axis=1)                       # rearrangement: no crossing
    if bundle.use_log.get(target, False):
        raw = np.expm1(raw)
    return np.maximum(raw, 0.0)


def predict_frame(bundle: Bundle, X: pd.DataFrame, mix: MixtureConfig) -> dict:
    """
    Vectorised prediction over a dataframe that already carries the feature columns.

    Returns per-target mixture forecasts plus the Stage-1 probabilities and the
    conditional levels actually used, which are what the calibration discussion in
    the paper needs.
    """
    p_arr = bundle.clf_arrival.predict_proba(X[bundle.feature_cols["clf_arrival"]])[:, 1]
    p_dep = bundle.clf_departure.predict_proba(X[bundle.feature_cols["clf_departure"]])[:, 1]
    p_arr = common.apply_isotonic(p_arr, bundle.calibration.get("clf_arrival"))
    p_dep = common.apply_isotonic(p_dep, bundle.calibration.get("clf_departure"))
    p_for = {"arrival_count": p_arr, "departure_count": p_dep,
             "avg_energy_kWh": p_arr, "avg_duration_mins": p_arr}

    preds, q_mags = {}, {}
    for tgt in TARGETS:
        grid_real = _q_grid_real(bundle, tgt, X)
        cap = bundle.cap if tgt == "avg_duration_mins" else None
        preds[tgt] = mixture_quantile(p_for[tgt], grid_real, mix, cap=cap)
        j = int(np.argmin(np.abs(np.asarray(mix.grid) - mix.alpha)))
        q_mags[tgt] = grid_real[:, j]

    return {
        "preds": preds, "q_mags": q_mags, "p_arrival": p_arr, "p_departure": p_dep,
        "cond_level_arrival": common.conditional_level(p_arr, mix.alpha),
        "cond_level_departure": common.conditional_level(p_dep, mix.alpha),
    }


def predict_step(feat_row: pd.DataFrame, bundle: Bundle, mix: MixtureConfig) -> tuple:
    """Single-row prediction, for autoregressive rollout and live forecasting."""
    out = predict_frame(bundle, feat_row, mix)
    preds = {t: float(out["preds"][t][0]) for t in TARGETS}
    q_mags = {t: float(out["q_mags"][t][0]) for t in TARGETS}
    return preds, float(out["p_arrival"][0]), float(out["p_departure"][0]), q_mags


# ── Evaluation ────────────────────────────────────────────────────────────────

def _metric_table(preds_df: pd.DataFrame, cfg: PipelineConfig, alpha: float,
                  suffix: str = "", min_active: int | None = None) -> pd.DataFrame:
    """Per-zone metrics plus an event-weighted aggregate over live zones."""
    rows = []
    for tgt in TARGETS:
        per_zone = []
        for zone, g in preds_df.groupby("zone", observed=True):
            g = g.sort_values("timestamp")
            row = M.compute_metrics(
                g[f"{tgt}_true"].to_numpy(float), g[f"{tgt}_pred"].to_numpy(float),
                alpha, label=f"{tgt}|{zone}{suffix}", is_pooled=False,
                min_active=cfg.min_active_slots if min_active is None else min_active,
            )
            row.update(target=tgt, zone=zone)
            warn = M.saturation_warning(row, cfg.duration_cap if tgt == "avg_duration_mins" else None)
            if warn:
                print(f"    SATURATION: {warn}")
            per_zone.append(row)
        pz = pd.DataFrame(per_zone)
        rows.extend(per_zone)
        agg = M.aggregate_over_zones(pz)
        if not agg.empty:
            agg["target"] = tgt
            agg["label"] = f"{tgt}|AGGREGATE{suffix}"
            rows.extend(agg.to_dict("records"))
    return pd.DataFrame(rows)


def evaluate_1step(cfg: PipelineConfig, bundle: Bundle, mix: MixtureConfig,
                   warmup_days: int = 14, *, n_boot: int = 1000,
                   seed: int = 42) -> pd.DataFrame:
    """
    1-step oracle evaluation over the configured test window.

    Uses the vectorised path on the precomputed feature frame, which is now
    feature-identical to build_feature_row.
    """
    print(f"\n{RULE}\n1-STEP EVALUATION — {cfg.name}  (inversion={mix.mode})\n{RULE}")
    df = load_feature_frame(cfg)
    _, _, test = split_frame(df, cfg)
    if test.empty:
        raise common.DataValidationError(
            f"Test window {cfg.test_start.date()} → {cfg.test_end.date()} is empty for "
            f"chargerType={cfg.charger_type}. Run validate_data.py."
        )

    warm = test["timestamp"].min() + pd.Timedelta(days=warmup_days)
    test = test[test["timestamp"] >= warm].copy()

    act = zone_activity(test, min_active=cfg.min_active_slots)
    live = act.loc[~act["degenerate"], "zone"].tolist()
    dead = act.loc[act["degenerate"], "zone"].tolist()
    print(f"  live zones: {live}")
    if dead:
        print(f"  sparse zones (<{cfg.min_active_slots} active slots) excluded from the aggregate: {dead}")
    if not live:
        raise common.DataValidationError(
            "Test window contains no events for this charger type — this is exactly the "
            "condition that produced the earlier meaningless DC results."
        )

    out = predict_frame(bundle, test, mix)
    preds_df = pd.DataFrame({
        "timestamp": test["timestamp"].to_numpy(),
        "zone": test["zone"].to_numpy(),
        "p_arrival": out["p_arrival"],
        "p_departure": out["p_departure"],
        "cond_level_arrival": out["cond_level_arrival"],
    })
    for tgt in TARGETS:
        preds_df[f"{tgt}_true"] = test[tgt].to_numpy(float)
        preds_df[f"{tgt}_pred"] = out["preds"][tgt]
        preds_df[f"{tgt}_q_mag"] = out["q_mags"][tgt]

    cfg.ensure_dirs()
    preds_df.to_csv(cfg.results_dir / f"1step_predictions_{mix.mode}.csv", index=False)

    mt = _metric_table(preds_df, cfg, mix.alpha)
    mt.to_csv(cfg.results_dir / f"1step_metrics_full_{mix.mode}.csv", index=False)
    M.paper_view(mt).to_csv(cfg.results_dir / f"1step_metrics_{mix.mode}.csv", index=False)

    # G3 — every headline 1-step number now carries a moving-block bootstrap CI,
    # computed on the live-zone pool. Previously only the rollout had intervals,
    # so the table a reviewer reads first was the one without error bars.
    ci_rows = []
    live_mask = preds_df["zone"].isin(live).to_numpy()
    for tgt in TARGETS:
        yv = preds_df.loc[live_mask, f"{tgt}_true"].to_numpy(float)
        fv = preds_df.loc[live_mask, f"{tgt}_pred"].to_numpy(float)
        if yv.size < 2 or not (yv > 0).any():
            continue
        ci = M.headline_cis(yv, fv, mix.alpha, n_boot=n_boot, seed=seed)
        ci.update(target=tgt, model=f"quantile_{mix.mode}", pipeline=cfg.name,
                  scope="1step_live_aggregate", n=int(yv.size))
        ci_rows.append(ci)
    ci_df = pd.DataFrame(ci_rows)
    if not ci_df.empty:
        ci_df.to_csv(cfg.results_dir / f"1step_confidence_intervals_{mix.mode}.csv",
                     index=False)
        print("\n  bootstrap CIs (live-zone pool, moving blocks of one day):")
        for _, r in ci_df.iterrows():
            print(f"    {r['target']:<20} pinball {r['CI_Pinball_All']:.4f} "
                  f"[{r['CI_Pinball_All_lo']:.4f}, {r['CI_Pinball_All_hi']:.4f}]   "
                  f"coverage {r['CI_Coverage_Global_pct']:.2f}% "
                  f"[{r['CI_Coverage_Global_pct_lo']:.2f}, "
                  f"{r['CI_Coverage_Global_pct_hi']:.2f}]")

    print("\n  per-zone and aggregate (reported metrics):")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(M.paper_view(mt).round(4).to_string(index=False))
    print(f"\n  saved → {cfg.results_dir / f'1step_metrics_{mix.mode}.csv'}")
    return mt


def evaluate_multistep(cfg: PipelineConfig, bundle: Bundle, mix: MixtureConfig,
                       max_horizon: int = 16, warmup_days: int = 14,
                       *, origin_method: str = "all",
                       origin_stride: int | None = None,
                       n_origins: int | None = None,
                       compare_mse: bool = True,
                       n_boot: int = 1000, seed: int = 42) -> dict:
    """
    Autoregressive rollout — the mode an EMS actually runs in.

    Rewritten (see rollout.py) to roll every origin forward in lock-step, which
    makes `--origin-method all` affordable. That matters for a reason that is not
    cosmetic: at a stride of ~17 slots a 5%-active DC series yields ~14 active
    slots per horizon bucket, below the 30-slot floor that aggregate_over_zones
    applies, so every bucket was discarded and the degradation table came back
    empty. Dense origins fix that at the source.

    With compare_mse the two-stage MSE baseline is rolled out alongside the
    proposed model, each feeding on its OWN predictions, and the two decay curves
    are compared per horizon (metrics.multistep_degradation).
    """
    print(f"\n{RULE}\nMULTI-STEP ROLLOUT — {cfg.name}  horizons {MULTISTEP_HORIZONS}\n{RULE}")
    df = load_feature_frame(cfg)
    _, _, test = split_frame(df, cfg)
    if test.empty:
        raise common.DataValidationError(
            f"Empty test window for chargerType={cfg.charger_type}.")

    act = zone_activity(test, min_active=cfg.min_active_slots)
    zones = act.loc[~act["degenerate"], "zone"].tolist()
    if not zones:
        raise common.DataValidationError("No live zone in the test window.")

    predictors: dict[str, callable] = {
        "quantile_exact": lambda X: predict_frame(bundle, X, mix)["preds"],
    }
    if compare_mse:
        try:
            mse = baselines.MseBundle(cfg.models_dir)
            predictors["two_stage_mse"] = lambda X: mse.predict(X, cap=bundle.cap)
            print("  comparator: two-stage MSE (loaded from the training run)")
        except FileNotFoundError as exc:
            print(f"  comparator unavailable — {exc}")

    all_rows, balance_rows = [], []
    for zone in zones:
        zdf = df[df["zone"] == zone]
        tmin = test.loc[test["zone"] == zone, "timestamp"].min()
        start = tmin + pd.Timedelta(days=warmup_days)
        eligible = pd.DatetimeIndex(
            test.loc[(test["zone"] == zone) & (test["timestamp"] >= start), "timestamp"])

        origins = common.sample_origins(
            eligible, max_horizon=max_horizon, method=origin_method,
            stride=origin_stride, n_origins=n_origins, seed=seed)
        bal = common.origin_balance(origins)
        bal.update(zone=zone, method=origin_method, max_horizon=max_horizon)
        balance_rows.append(bal)
        print(f"  zone {zone}: {len(origins):,} origins x {max_horizon} steps  "
              f"[clock slots {bal['slots_covered']}/{bal['slots_total']} "
              f"({bal['coverage_pct']:.1f}%), chi2-uniform p={bal['chi2_uniform_p']:.3f}]")

        r = rollout.run_rollout(zdf, origins, max_horizon, predictors, zone)
        if not r.empty:
            all_rows.append(r)

    if not all_rows:
        raise common.DataValidationError(
            "Rollout produced no rows. Every origin lacked either the two weeks of "
            "history the lag features need or the horizon of ground truth ahead of it.")

    preds_df = pd.concat(all_rows, ignore_index=True)
    cfg.ensure_dirs()
    preds_df.to_csv(cfg.results_dir / f"multistep_predictions_{mix.mode}.csv", index=False)

    bal_df = pd.DataFrame(balance_rows)
    bal_df.to_csv(cfg.results_dir / f"multistep_origin_balance_{mix.mode}.csv", index=False)
    print("\n  origin balance (report this in the paper's protocol section):")
    print(bal_df[["zone", "n_origins", "slots_covered", "slots_total",
                  "coverage_pct", "chi2_uniform_p"]].to_string(index=False))

    # ── Per-horizon metrics, per model ────────────────────────────────────────
    # The activity floor is a property of the SAMPLE being summarised, so a
    # per-horizon bucket cannot inherit the whole-window value. It is scaled by
    # the share of the window each bucket represents, and never allowed below 10
    # active slots, which is the point where a coverage estimate stops meaning
    # anything at all.
    rows = []
    n_h = len([x for x in MULTISTEP_HORIZONS if x <= max_horizon])
    for name, g_model in preds_df.groupby("model", observed=True):
        for h in [x for x in MULTISTEP_HORIZONS if x <= max_horizon]:
            sub = g_model[g_model["horizon_slots"] == h]
            if sub.empty:
                continue
            share = len(sub) / max(1, len(test))
            floor = max(10, int(round(cfg.min_active_slots * share)))
            mt = _metric_table(sub, cfg, mix.alpha, suffix=f"|h{h}", min_active=floor)
            mt["model"] = name
            mt["horizon_slots"] = h
            mt["horizon_mins"] = h * RESOLUTION_MINUTES
            mt["min_active_floor"] = floor
            rows.append(mt)

    ms = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    ms.to_csv(cfg.results_dir / f"multistep_metrics_full_{mix.mode}.csv", index=False)
    agg = ms[ms["zone"].astype(str).str.startswith("AGGREGATE")] if not ms.empty else ms
    if not agg.empty:
        M.paper_view(agg, extra=["model"]).to_csv(
            cfg.results_dir / f"multistep_metrics_{mix.mode}.csv", index=False)
        print("\n  degradation with horizon (live-zone aggregate, reported metrics):")
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(M.paper_view(agg, extra=["model"])
                  .sort_values(["target", "model", "horizon_mins"])
                  .round(4).to_string(index=False))

    # ── Degradation curve + per-horizon significance ──────────────────────────
    deg = pd.DataFrame()
    if "two_stage_mse" in preds_df["model"].unique():
        deg = M.multistep_degradation(preds_df, mix.alpha,
                                      reference="quantile_exact",
                                      comparator="two_stage_mse")
        deg.to_csv(cfg.results_dir / f"multistep_degradation_{mix.mode}.csv", index=False)
        print(f"\n{RULE}")
        print("MULTI-STEP DEGRADATION — quantile_exact vs two_stage_mse")
        print(RULE)
        cols = ["target", "horizon_mins",
                "pinball_quantile_exact", "pinball_two_stage_mse",
                "skill_quantile_exact_pct", "skill_two_stage_mse_pct",
                "degradation_quantile_exact", "degradation_two_stage_mse",
                "degradation_ratio", "comparator_at_floor",
                "p_value", "better", "significant_5pct"]
        with pd.option_context("display.width", 240, "display.max_columns", 30):
            print(deg[[c for c in cols if c in deg.columns]].round(4).to_string(index=False))
        print("\n  pinball_<model>     absolute marginal loss at that horizon — the ranking column.")
        print("  skill_<model>_pct   improvement over always-zero AT THAT HORIZON.")
        print("  degradation_<model> loss divided by that model's OWN one-step loss, so the")
        print("                      curve measures decay RATE, not the one-step gap.")
        print("  comparator_at_floor True when the comparator's skill has fallen under 5%:")
        print("                      it has collapsed onto the zero forecast, so its flat")
        print("                      degradation curve means 'nothing left to lose', and the")
        print("                      degradation_ratio on that row must not be quoted.")

    # ── CIs on the proposed model's curve ─────────────────────────────────────
    ci_rows = []
    q = preds_df[preds_df["model"] == "quantile_exact"]
    for h in [x for x in MULTISTEP_HORIZONS if x <= max_horizon]:
        sub = q[q["horizon_slots"] == h]
        for tgt in TARGETS:
            if sub.empty:
                continue
            yv = sub[f"{tgt}_true"].to_numpy(float)
            fv = sub[f"{tgt}_pred"].to_numpy(float)
            if yv.size < 2 or not (yv > 0).any():
                continue
            ci = M.headline_cis(yv, fv, mix.alpha, n_boot=n_boot, seed=seed,
                                block=max(2, min(SEASONAL_PERIOD, yv.size // 10)))
            ci.update(target=tgt, horizon_slots=h, horizon_mins=h * RESOLUTION_MINUTES,
                      model=f"quantile_{mix.mode}", pipeline=cfg.name, n=int(yv.size))
            ci_rows.append(ci)
    if ci_rows:
        pd.DataFrame(ci_rows).to_csv(
            cfg.results_dir / f"multistep_confidence_intervals_{mix.mode}.csv", index=False)

    print(f"\n  saved → {cfg.results_dir / f'multistep_metrics_{mix.mode}.csv'}")
    return {"metrics": ms, "degradation": deg, "predictions": preds_df}


# ── Live forecast ─────────────────────────────────────────────────────────────

def generate_forecast(cfg: PipelineConfig, bundle: Bundle, mix: MixtureConfig,
                      reference_time: pd.Timestamp, horizon_slots: int = 4,
                      zone: str | None = None, history_days: int = 15) -> dict:
    zone = zone or cfg.default_zone
    df = load_feature_frame(cfg, verbose=False)
    df = df[df["zone"] == zone]
    cutoff = reference_time - pd.Timedelta(days=history_days)
    buffer = (df[(df["timestamp"] >= cutoff) & (df["timestamp"] < reference_time)]
              .set_index("timestamp")[SIG_COLS].sort_index())
    if buffer.empty:
        raise common.DataValidationError(
            f"No history for zone {zone!r} in the {history_days} days before {reference_time}."
        )

    slots = []
    for step in range(horizon_slots):
        slot = reference_time + pd.Timedelta(minutes=int(RESOLUTION_MINUTES * step))
        feat = build_feature_row(slot, buffer)
        preds, p_arr, p_dep, q_mags = predict_step(feat, bundle, mix)
        a = int(mix.alpha * 100)
        slots.append({
            "timeSlotStart": slot.isoformat(), "stepsAhead": step + 1,
            "minutesAhead": (step + 1) * RESOLUTION_MINUTES,
            f"Q{a}_arrivalCount": round(preds["arrival_count"], 4),
            f"Q{a}_departureCount": round(preds["departure_count"], 4),
            f"Q{a}_energyKwh": round(preds["avg_energy_kWh"], 4),
            f"Q{a}_durationMins": round(preds["avg_duration_mins"], 1),
            "pArrivalActive": round(p_arr, 4),
            "pDepartureActive": round(p_dep, 4),
            "conditionalLevelUsed": round(float(common.conditional_level(p_arr, mix.alpha)), 4),
            "stage2_arrivalCount_qmag": round(q_mags["arrival_count"], 4),
            "stage2_energyKwh_qmag": round(q_mags["avg_energy_kWh"], 4),
            "stage2_durationMins_qmag": round(q_mags["avg_duration_mins"], 1),
        })
        nxt = {t: preds[t] for t in TARGETS}
        dur_h = preds["avg_duration_mins"] / 60.0
        nxt["avg_power_kW"] = (preds["avg_energy_kWh"] / dur_h) if dur_h > 0 else 0.0
        buffer = pd.concat([buffer, pd.DataFrame([nxt], index=[slot])])

    return {
        "forecastReferenceTime": reference_time.isoformat(),
        "pipeline": cfg.name, "zone": zone, "quantileAlpha": mix.alpha,
        "mixtureInversion": mix.mode, "resolutionMinutes": RESOLUTION_MINUTES,
        "durationCapMins": bundle.cap, "forecastData": slots,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_horizon(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("min"):
        return max(1, int(s[:-3]) // RESOLUTION_MINUTES)
    if s.endswith("h"):
        return max(1, int(float(s[:-1]) * 60) // RESOLUTION_MINUTES)
    if s.endswith("d"):
        return max(1, int(float(s[:-1]) * 1440) // RESOLUTION_MINUTES)
    raise ValueError(f"Unrecognised horizon: {s!r}")


def build_parser(default_pipeline: str | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    if default_pipeline is None:
        ap.add_argument("--pipeline", required=True, choices=["ac", "dc"])
    else:
        ap.add_argument("--pipeline", default=default_pipeline, choices=["ac", "dc"])
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--multistep-eval", action="store_true")
    ap.add_argument("--horizon-slots", type=int, default=16,
                    help="Rollout length in 15-minute slots (16 = 4 hours).")
    ap.add_argument("--no-compare-mse", action="store_true",
                    help="Skip the two-stage MSE comparator in the rollout.")
    ap.add_argument("--origin-method", choices=["all", "coprime", "random"],
                    default="all",
                    help="How rollout origins are sampled. 'all' (default) uses "
                         "every eligible slot, which the batched rollout makes "
                         "affordable and which is what keeps each per-horizon "
                         "bucket above the activity floor on a 95%%-zero series. "
                         "'coprime' uses the smallest stride >= horizon that is "
                         "coprime with the 96-slot day (a stride dividing 96 "
                         "aliases horizon against time of day and is rejected).")
    ap.add_argument("--origin-stride", type=int, default=None,
                    help="Force an explicit stride. Rejected if it shares a factor "
                         "with the daily period.")
    ap.add_argument("--n-origins", type=int, default=None,
                    help="Number of origins for --origin-method random.")
    ap.add_argument("--boot", type=int, default=1000,
                    help="Bootstrap replicates for the evaluation CIs (0 disables).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inversion", choices=["exact", "legacy"], default="exact",
                    help="'legacy' reproduces the old fixed-alpha inversion from the same models.")
    ap.add_argument("--time", type=str, default=None)
    ap.add_argument("--horizon", type=str, default=None)
    ap.add_argument("--zone", type=str, default=None)
    ap.add_argument("--output", type=str, default=None)
    add_window_flag(ap)
    return ap


def run_cli(args) -> None:
    cfg = get_config(args.pipeline)
    cfg.assert_window_confirmed(getattr(args, 'allow_placeholder_window', False))
    print('  ' + cfg.describe_window())
    bundle = load_models(cfg)
    mix = bundle.mixture(args.inversion)

    if args.eval_only:
        evaluate_1step(cfg, bundle, mix, n_boot=args.boot, seed=args.seed)
        return
    if args.multistep_eval:
        evaluate_multistep(cfg, bundle, mix, max_horizon=args.horizon_slots,
                           origin_method=args.origin_method,
                           origin_stride=args.origin_stride,
                           n_origins=args.n_origins,
                           compare_mse=not args.no_compare_mse,
                           n_boot=args.boot, seed=args.seed)
        return

    horizon = parse_horizon(args.horizon) if args.horizon else 4
    ref = pd.Timestamp(args.time) if args.time else pd.Timestamp.now().floor(f"{RESOLUTION_MINUTES}min")
    result = generate_forecast(cfg, bundle, mix, ref, horizon_slots=horizon, zone=args.zone)
    print(json.dumps(result, indent=2))
    if args.output:
        (cfg.results_dir / args.output).write_text(json.dumps(result, indent=2))


def main() -> None:
    run_cli(build_parser().parse_args())


if __name__ == "__main__":
    main()
