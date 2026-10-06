"""
metrics.py — Evaluation metrics and inference statistics shared by both pipelines.

Fixes implemented here
----------------------
F3  Coverage reported honestly. Global Coverage_Q95 on a 97%-zero series is
    dominated by correctly-predicted zeros: an all-zero forecaster scores ~97%.
    Every metric row now carries Coverage_Active_pct, Breach_Rate_pct and the
    coverage of the always-zero reference, so the reader can see the floor the
    metric is measured from.

F4  Uncertainty. `bootstrap_ci` (moving-block, to respect serial correlation) and
    `diebold_mariano` give confidence intervals and a significance test against a
    named baseline, instead of a bare point estimate over a few hundred events.

F5  Active MASE redefined. It previously divided active-slot MAE by a seasonal
    difference taken over the whole zero-dominated series, so the denominator was
    tiny and values landed at 13-24 while the reading guide called <1.0 good. The
    denominator is now the seasonal-naive error on the same active slots, which
    restores the standard "<1 beats naive" reading. The old quantity is kept as
    Active_MASE_hybrid_legacy purely for reconciliation with earlier runs.

Operational metrics
-------------------
Breach rate, mean/max exceedance, mean over-provision and peak capture translate
the forecast into the quantities an EMS operator cares about: how often the bound
was crossed, by how much, and how much headroom was booked and not used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

SEASONAL_PERIOD = 96

# G4 — kept in sync with common.MIN_ACTIVE_SLOTS. Imported lazily to keep this
# module free of a circular import: metrics.py must stay importable on its own.
try:                                                    # pragma: no cover
    from common import MIN_ACTIVE_SLOTS
except Exception:                                       # pragma: no cover
    MIN_ACTIVE_SLOTS = 30


# ── Elementary losses ─────────────────────────────────────────────────────────

def pinball_series(y: np.ndarray, f: np.ndarray, alpha: float) -> np.ndarray:
    e = np.asarray(y, float) - np.asarray(f, float)
    return np.maximum(alpha * e, (alpha - 1.0) * e)


def pinball_loss(y: np.ndarray, f: np.ndarray, alpha: float) -> float:
    if len(y) == 0:
        return float("nan")
    return float(np.mean(pinball_series(y, f, alpha)))


def wape(y: np.ndarray, f: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y)))
    return float(np.sum(np.abs(y - f)) / denom) if denom > 0 else float("nan")


def _seasonal_naive(y: np.ndarray, period: int = SEASONAL_PERIOD) -> np.ndarray:
    """y shifted by one seasonal period; the first `period` entries are undefined."""
    naive = np.full(len(y), np.nan)
    if len(y) > period:
        naive[period:] = y[:-period]
    return naive


def mase_global(y: np.ndarray, f: np.ndarray, period: int = SEASONAL_PERIOD) -> float:
    if len(y) <= period:
        return float("nan")
    denom = float(np.mean(np.abs(y[period:] - y[:-period])))
    if denom <= 0:
        return float("nan")
    return float(np.mean(np.abs(y - f)) / denom)


def mase_active(y: np.ndarray, f: np.ndarray, period: int = SEASONAL_PERIOD) -> float:
    """
    FIX F5 — scale active-slot error by the naive error on those same slots.

    Reads against the conventional threshold: < 1.0 means the model beats a
    24-hour persistence forecast on the slots where an event actually occurred.
    """
    if len(y) <= period:
        return float("nan")
    naive = _seasonal_naive(y, period)
    mask = (y > 0) & np.isfinite(naive)
    if not np.any(mask):
        return float("nan")
    denom = float(np.mean(np.abs(y[mask] - naive[mask])))
    if denom <= 0:
        return float("nan")
    return float(np.mean(np.abs(y[mask] - f[mask])) / denom)


def mase_active_legacy(y: np.ndarray, f: np.ndarray, period: int = SEASONAL_PERIOD) -> float:
    """The previous, non-comparable definition. Retained only for reconciliation."""
    if len(y) <= period:
        return float("nan")
    denom = float(np.mean(np.abs(y[period:] - y[:-period])))
    active = y > 0
    if denom <= 0 or not np.any(active):
        return float("nan")
    return float(np.mean(np.abs(y[active] - f[active])) / denom)


# ── Metric block ──────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
    label: str = "",
    *,
    is_pooled: bool = False,
    period: int = SEASONAL_PERIOD,
    peak_decile: float = 0.10,
    min_active: int = MIN_ACTIVE_SLOTS,
) -> dict:
    """
    Full metric row for one (zone, target, model) cell.

    is_pooled=True suppresses both MASE variants: pooling zones breaks series
    contiguity, so the seasonal difference would straddle a zone boundary. That is
    why the AC arm previously reported no MASE at all — the fix is to compute
    per-zone and aggregate, not to pool and give up (see aggregate_over_zones).

    G4 — `degenerate_zone` is now n_active < min_active (default 30), not
    n_active == 0. Every metric on this row is still computed and reported; the
    flag only controls whether the row enters the event-weighted aggregate. A
    zone with one event produced 0%/100% rates and a one-observation MASE
    denominator, and still carried weight in the headline.
    """
    y = np.asarray(y_true, dtype=float)
    f = np.asarray(y_pred, dtype=float)
    n_tot = int(len(y))
    active = y > 0
    n_act = int(active.sum())

    row: dict[str, object] = {
        "label": label,
        "n_total": n_tot,
        "n_active": n_act,
        "zero_rate_pct": round(100.0 * (1 - n_act / n_tot), 4) if n_tot else np.nan,
        "degenerate_zone": bool(n_act < int(min_active)),
        "empty_zone": bool(n_act == 0),
        "min_active_threshold": int(min_active),
        "exclusion_reason": ("empty" if n_act == 0
                             else (f"sparse (<{int(min_active)} active)"
                                   if n_act < int(min_active) else "")),
    }

    row["Global_RMSE"] = float(np.sqrt(np.mean((y - f) ** 2))) if n_tot else np.nan
    row["Global_MAE"] = float(np.mean(np.abs(y - f))) if n_tot else np.nan
    row["Global_WAPE"] = wape(y, f)
    row["Global_MASE"] = np.nan if is_pooled else mase_global(y, f, period)

    row["Coverage_Global_pct"] = float(100.0 * np.mean(y <= f + 1e-9)) if n_tot else np.nan
    row["Coverage_AlwaysZero_pct"] = row["zero_rate_pct"]      # F3 reference floor
    # Marginal calibration error: how far the realised coverage sits from the
    # nominal level the mixture quantile actually targets. This, not active-slot
    # coverage, is the quantity the estimand is defined against.
    row["Coverage_Gap_pp"] = (
        float(row["Coverage_Global_pct"] - 100.0 * alpha) if n_tot else np.nan
    )
    row["Abs_Coverage_Gap_pp"] = abs(row["Coverage_Gap_pp"]) if n_tot else np.nan
    row["Pinball_All"] = pinball_loss(y, f, alpha)
    row["Pinball_All_AlwaysZero"] = pinball_loss(y, np.zeros_like(y), alpha)
    pz = row["Pinball_All_AlwaysZero"]
    row["Pinball_Skill_vs_Zero_pct"] = (
        float(100.0 * (1.0 - row["Pinball_All"] / pz)) if pz and pz > 0 else np.nan
    )

    if n_act:
        ya, fa = y[active], f[active]
        row["Active_RMSE"] = float(np.sqrt(np.mean((ya - fa) ** 2)))
        row["Active_MAE"] = float(np.mean(np.abs(ya - fa)))
        row["Active_NRMSE_pct"] = float(100.0 * row["Active_RMSE"] / np.mean(ya))
        row["Active_Bias"] = float(np.mean(fa - ya))
        row["Active_MASE"] = np.nan if is_pooled else mase_active(y, f, period)
        row["Active_MASE_hybrid_legacy"] = np.nan if is_pooled else mase_active_legacy(y, f, period)
        row["Pinball_Active"] = pinball_loss(ya, fa, alpha)
        row["Pinball_Active_AlwaysZero"] = pinball_loss(ya, np.zeros_like(ya), alpha)
        pza = row["Pinball_Active_AlwaysZero"]
        row["Pinball_Active_Skill_vs_Zero_pct"] = (
            float(100.0 * (1.0 - row["Pinball_Active"] / pza)) if pza and pza > 0 else np.nan
        )

        # ── Operational quantities ────────────────────────────────────────────
        row["Coverage_Active_pct"] = float(100.0 * np.mean(ya <= fa + 1e-9))
        breach = ya > fa + 1e-9
        row["Breach_Rate_pct"] = float(100.0 * np.mean(breach))
        row["Mean_Exceedance"] = float(np.mean(ya[breach] - fa[breach])) if breach.any() else 0.0
        row["Max_Exceedance"] = float(np.max(ya[breach] - fa[breach])) if breach.any() else 0.0
        over = fa > ya
        row["Mean_OverProvision"] = float(np.mean(fa[over] - ya[over])) if over.any() else 0.0
        row["OverProvision_pct_of_mean"] = (
            float(100.0 * row["Mean_OverProvision"] / np.mean(ya)) if np.mean(ya) > 0 else np.nan
        )
        k = max(1, int(np.ceil(peak_decile * n_act)))
        top = np.argsort(-ya)[:k]
        row["Peak_Capture_pct"] = (
            float(100.0 * np.sum(fa[top]) / np.sum(ya[top])) if np.sum(ya[top]) > 0 else np.nan
        )
        row["Mean_Active_True"] = float(np.mean(ya))
        row["Fire_Rate_pct"] = float(100.0 * np.mean(f > 1e-9))
        row["False_Alarm_pct"] = (
            float(100.0 * np.mean(f[~active] > 1e-9)) if (~active).any() else 0.0
        )
        nz = f[f > 1e-9]
        row["Pred_Distinct_Values"] = int(len(np.unique(np.round(nz, 4))))
        row["Pred_Max"] = float(np.max(f)) if n_tot else np.nan
        row["True_Max"] = float(np.max(y)) if n_tot else np.nan
    else:
        for key in (
            "Active_RMSE", "Active_MAE", "Active_NRMSE_pct", "Active_Bias",
            "Active_MASE", "Active_MASE_hybrid_legacy", "Pinball_Active",
            "Pinball_Active_AlwaysZero", "Pinball_Active_Skill_vs_Zero_pct",
            "Coverage_Active_pct", "Breach_Rate_pct", "Mean_Exceedance",
            "Max_Exceedance", "Mean_OverProvision", "OverProvision_pct_of_mean",
            "Peak_Capture_pct", "Mean_Active_True", "Fire_Rate_pct",
            "False_Alarm_pct", "Pred_Distinct_Values",
        ):
            row[key] = np.nan
        row["Pred_Max"] = float(np.max(f)) if n_tot else np.nan
        row["True_Max"] = float(np.max(y)) if n_tot else np.nan

    return row


def saturation_warning(row: dict, cap: float | None) -> str | None:
    """
    Guard against the failure mode behind F2: a regressor pinned at its own cap.

    A model emitting one or two distinct non-zero values, at the cap, is not fit —
    it is saturated, and no amount of tuning downstream will help.
    """
    n = row.get("Pred_Distinct_Values")
    if not n or (isinstance(n, float) and np.isnan(n)):
        return None
    if n <= 2:
        msg = f"{row.get('label','')}: only {int(n)} distinct non-zero predictions"
        if cap is not None and np.isclose(row.get("Pred_Max", np.nan), cap, rtol=1e-6):
            msg += f" and Pred_Max == cap ({cap:g}) — regressor is saturated at the cap"
        return msg
    return None


# ── Aggregation across zones (F6, F7) ─────────────────────────────────────────

def aggregate_over_zones(rows: pd.DataFrame, weight: str = "n_active") -> pd.DataFrame:
    """
    FIX F6/F7 — aggregate per-zone metrics instead of pooling raw rows.

    Pooling rows across zones made MASE undefined (the AC arm reported none at
    all) and let all-zero zones dilute every average. Computing per zone and then
    taking an event-weighted mean over the non-degenerate zones keeps MASE valid
    and keeps empty zones out of the headline.
    """
    live = rows[~rows["degenerate_zone"]].copy()
    dropped = rows[rows["degenerate_zone"]]
    if live.empty:
        return pd.DataFrame()

    num_cols = [c for c in live.columns
                if live[c].dtype.kind in "fi" and c not in {"n_total", "n_active"}]
    w = live[weight].to_numpy(float)
    if w.sum() <= 0:
        w = np.ones(len(live))

    out: dict[str, object] = {"zone": "AGGREGATE (event-weighted, live zones)"}
    for c in num_cols:
        v = live[c].to_numpy(float)
        m = np.isfinite(v)
        out[c] = float(np.average(v[m], weights=w[m])) if m.any() and w[m].sum() > 0 else np.nan
    out["n_total"] = int(live["n_total"].sum())
    out["n_active"] = int(live["n_active"].sum())
    out["n_zones"] = int(len(live))
    out["n_zones_excluded"] = int(len(dropped))
    out["n_active_excluded"] = int(dropped["n_active"].sum()) if len(dropped) else 0
    out["degenerate_zone"] = False
    out["empty_zone"] = False
    return pd.DataFrame([out])


# ── Uncertainty and significance (F4) ─────────────────────────────────────────

@dataclass
class CI:
    point: float
    lo: float
    hi: float
    level: float = 0.95

    def as_dict(self, prefix: str) -> dict:
        return {f"{prefix}": self.point, f"{prefix}_lo": self.lo, f"{prefix}_hi": self.hi}

    def __str__(self) -> str:
        return f"{self.point:.4g} [{self.lo:.4g}, {self.hi:.4g}]"


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block resample index vector of length n."""
    if block >= n:
        return rng.integers(0, n, size=n)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    return idx[:n]


def bootstrap_ci(
    y: np.ndarray,
    f: np.ndarray,
    statistic,
    *,
    n_boot: int = 2000,
    block: int = SEASONAL_PERIOD,
    level: float = 0.95,
    seed: int = 42,
) -> CI:
    """
    Moving-block bootstrap CI for any statistic(y, f) -> float.

    A plain i.i.d. bootstrap would understate the interval on a serially
    correlated 15-minute series; blocks of one seasonal period preserve the
    daily structure.
    """
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    n = len(y)
    point = float(statistic(y, f))
    if n < 2:
        return CI(point, np.nan, np.nan, level)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = _block_indices(n, block, rng)
        try:
            draws[b] = statistic(y[idx], f[idx])
        except Exception:
            draws[b] = np.nan
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return CI(point, np.nan, np.nan, level)
    a = (1.0 - level) / 2.0
    return CI(point, float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a)), level)


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    h: int = 1,
    harvey_correction: bool = True,
) -> dict:
    """
    Diebold-Mariano test of equal predictive accuracy.

    Negative statistic ⇒ model A has the lower loss. Newey-West HAC variance with
    the Harvey-Leybourne-Newbold small-sample correction, which matters at the few
    hundred events available here.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 8:
        return {"DM_stat": np.nan, "p_value": np.nan, "n": n, "mean_diff": np.nan}

    dbar = float(np.mean(d))
    dc = d - dbar
    gamma0 = float(np.dot(dc, dc) / n)
    lrv = gamma0
    for lag in range(1, h):
        g = float(np.dot(dc[lag:], dc[:-lag]) / n)
        lrv += 2.0 * (1.0 - lag / h) * g
    if lrv <= 0:
        return {"DM_stat": np.nan, "p_value": np.nan, "n": n, "mean_diff": dbar}

    stat = dbar / np.sqrt(lrv / n)
    dof = n - 1
    if harvey_correction:
        adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        stat *= adj

    try:
        from scipy import stats as _st
        p = float(2.0 * (1.0 - _st.t.cdf(abs(stat), df=dof)))
    except Exception:
        from math import erfc, sqrt
        p = float(erfc(abs(stat) / sqrt(2.0)))

    return {"DM_stat": float(stat), "p_value": p, "n": n, "mean_diff": dbar}


def compare_models_dm(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    alpha: float,
    reference: str,
    *,
    active_only: bool = True,
) -> pd.DataFrame:
    """
    Pairwise DM tests of every model against `reference`, on pinball loss.

    IMPORTANT — scope changes the verdict. A marginal Q_alpha forecast is defined
    against the loss over ALL slots; restricting the test to active slots scores a
    different estimand and systematically favours whichever model is more
    conservative. Always report scope="all" as the primary result and scope=
    "active" as a diagnostic. `compare_models_dm_both` returns both.
    """
    y = np.asarray(y, float)
    mask = (y > 0) if active_only else np.ones(len(y), bool)
    if reference not in preds:
        raise KeyError(f"reference model {reference!r} not among {list(preds)}")
    ref_loss = pinball_series(y[mask], np.asarray(preds[reference], float)[mask], alpha)

    out = []
    for name, p in preds.items():
        if name == reference:
            continue
        loss = pinball_series(y[mask], np.asarray(p, float)[mask], alpha)
        res = diebold_mariano(ref_loss, loss, h=1)
        res.update({
            "model": reference,
            "vs": name,
            "scope": "active" if active_only else "all",
            "better": (reference if res["mean_diff"] < 0 else name)
            if np.isfinite(res["mean_diff"]) else "n/a",
            "significant_5pct": bool(np.isfinite(res["p_value"]) and res["p_value"] < 0.05),
        })
        out.append(res)
    return pd.DataFrame(out)


def compare_models_dm_both(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    alpha: float,
    reference: str,
) -> pd.DataFrame:
    """DM tests under both scopes, stacked, so neither can be quoted alone."""
    frames = []
    for active_only in (False, True):
        frames.append(compare_models_dm(y, preds, alpha, reference, active_only=active_only))
    return pd.concat(frames, ignore_index=True)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """
    Binned ECE for the Stage-1 probability.

    The exact mixture inversion reads the Stage-2 quantile function at a level that
    is a function of p, so a miscalibrated p translates directly into a
    miscalibrated forecast. Under the old fixed-alpha inversion p only had to clear
    a threshold, so this never mattered; it does now.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def _stat_bank(alpha: float) -> dict:
    """
    The statistics a reviewer will quote, as (y, f) -> float callables.

    Pinball_All is FIRST because it is the proper scoring rule for the marginal
    Q_alpha estimand and the column the ranking is argued on. It previously had no
    interval at all, which meant the paper's headline ordering was a set of bare
    point estimates.
    """
    def pinball_all(yy, ff):
        return pinball_loss(yy, ff, alpha)

    def cov_global(yy, ff):
        return float(100.0 * np.mean(yy <= ff + 1e-9))

    def cov_gap_pp(yy, ff):
        return float(100.0 * np.mean(yy <= ff + 1e-9) - 100.0 * alpha)

    def pinball_active(yy, ff):
        m = yy > 0
        return pinball_loss(yy[m], ff[m], alpha) if m.any() else np.nan

    def cov_active(yy, ff):
        m = yy > 0
        return float(100.0 * np.mean(yy[m] <= ff[m] + 1e-9)) if m.any() else np.nan

    return {
        "Pinball_All": pinball_all,
        "Coverage_Global_pct": cov_global,
        "Coverage_Gap_pp": cov_gap_pp,
        "Pinball_Active": pinball_active,
        "Coverage_Active_pct": cov_active,
    }


def headline_cis(y: np.ndarray, f: np.ndarray, alpha: float, **kw) -> dict:
    """
    Bootstrap CIs for every number a reviewer will quote, for ONE model.

    Use `paired_bootstrap_cis` when you have several models on the same test
    window — it shares the resample draws, which is both far cheaper and the
    statistically correct thing to do when the intervals will be compared.
    """
    out: dict = {}
    for name, fn in _stat_bank(alpha).items():
        out.update(bootstrap_ci(y, f, fn, **kw).as_dict(f"CI_{name}"))
    return out


def paired_bootstrap_cis(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    alpha: float,
    *,
    n_boot: int = 1000,
    block: int = SEASONAL_PERIOD,
    level: float = 0.95,
    seed: int = 42,
    statistics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    FIX G3 — block-bootstrap CIs for EVERY model, on shared resample draws.

    Two reasons this replaces per-model `headline_cis` calls in the ablation:

    1. Coverage. Previously only the proposed model carried an interval, so the
       baselines it was ranked against were bare point estimates. A table where
       one row has error bars and seven do not invites exactly one question.

    2. Correctness. Independent draws per model would make the intervals
       marginally valid but mutually incomparable — two models could show
       overlapping intervals while one beats the other on every single resample.
       Drawing ONE set of block indices per replicate and evaluating all models on
       it keeps the comparison paired, and lets `win_rate` below report the
       fraction of resamples in which each model beat the reference. That is the
       quantity that actually answers "is this difference real", and it agrees
       with the Diebold-Mariano test rather than contradicting it.

    Cost is ~1 resample per replicate instead of 1 per (model, replicate): with 8
    models this is roughly 8x cheaper than the naive loop as well as more correct.

    Returns one row per (model, statistic) with point, lo, hi and, for every
    non-reference model, the paired win rate against the best-point model.
    """
    y = np.asarray(y, float)
    n = len(y)
    bank = _stat_bank(alpha)
    if statistics:
        bank = {k: v for k, v in bank.items() if k in set(statistics)}
    names = list(preds)
    P = {k: np.asarray(v, float) for k, v in preds.items()}

    point = {m: {s: float(fn(y, P[m])) for s, fn in bank.items()} for m in names}
    if n < 2 or n_boot <= 0:
        return pd.DataFrame([
            {"model": m, "statistic": s, "point": point[m][s], "lo": np.nan,
             "hi": np.nan, "level": level, "n_boot": 0, "n": n}
            for m in names for s in bank
        ])

    rng = np.random.default_rng(seed)
    draws = {m: {s: np.full(n_boot, np.nan) for s in bank} for m in names}
    for b in range(n_boot):
        idx = _block_indices(n, block, rng)          # ONE draw, shared by all models
        yb = y[idx]
        for m in names:
            fb = P[m][idx]
            for s, fn in bank.items():
                try:
                    draws[m][s][b] = fn(yb, fb)
                except Exception:
                    pass

    # Paired win rate on the primary statistic, against the best point estimate.
    primary = "Pinball_All" if "Pinball_All" in bank else next(iter(bank))
    champ = min(names, key=lambda m: (np.inf if not np.isfinite(point[m][primary])
                                      else point[m][primary]))
    a = (1.0 - level) / 2.0
    rows = []
    for m in names:
        for s in bank:
            d = draws[m][s][np.isfinite(draws[m][s])]
            rows.append({
                "model": m, "statistic": s, "point": point[m][s],
                "lo": float(np.quantile(d, a)) if d.size else np.nan,
                "hi": float(np.quantile(d, 1 - a)) if d.size else np.nan,
                "level": level, "n_boot": int(d.size), "n": n,
                "reference": champ,
                "paired_win_rate_vs_reference": (
                    np.nan if m == champ else float(np.mean(
                        draws[m][primary] < draws[champ][primary]))
                ) if s == primary else np.nan,
            })
    return pd.DataFrame(rows)


def cis_to_columns(ci: pd.DataFrame) -> pd.DataFrame:
    """Pivot paired_bootstrap_cis output to one row per model, wide columns."""
    if ci.empty:
        return pd.DataFrame()
    wide = ci.pivot(index="model", columns="statistic",
                    values=["point", "lo", "hi"])
    wide.columns = [f"CI_{stat}{'' if kind == 'point' else '_' + kind}"
                    for kind, stat in wide.columns]
    win = (ci.dropna(subset=["paired_win_rate_vs_reference"])
             .set_index("model")["paired_win_rate_vs_reference"])
    wide["paired_win_rate_vs_reference"] = win
    return wide.reset_index()


# ── Reported metric set (paper-facing) ────────────────────────────────────────
#
# compute_metrics returns ~38 fields. Most exist to catch a specific failure and
# belong in the archive, not in a results table a reader has to hold in their
# head. These are the ones that go in the paper, and why each earns its place:
#
#   Pinball_All              The proper scoring rule for the marginal Q_alpha
#                            estimand. This is what models are RANKED on. Every
#                            other entry is here to interpret it.
#   Pinball_Skill_vs_Zero_pct  The same number as a percentage improvement over
#                            predicting nothing. A raw pinball loss of 0.02 means
#                            nothing to a reader; "46% better than always-zero"
#                            does. Same ordering as Pinball_All by construction.
#   Coverage_Global_pct      Realised coverage: the share of slots at or below the
#                            forecast. The calibration claim.
#   Coverage_AlwaysZero_pct  The coverage a forecaster gets for free by predicting
#                            zero everywhere. Without it a reader cannot tell
#                            whether 98% coverage is good or automatic, and on a
#                            97%-zero series it is very nearly automatic.
#   Coverage_Gap_pp          Realised minus nominal. 0 is perfect; positive means
#                            capacity booked and never used.
#   Breach_Rate_pct          How often the true value exceeded the forecast, on
#                            active slots. The operational risk side.
#   Peak_Capture_pct         Share of the true top decile the forecast reaches.
#                            The operational value side: did it rise with demand?
#
# Deliberately NOT reported: RMSE/MAE/WAPE and both MASE variants (built on
# absolute error, whose minimiser is the median — a Q95 forecast is meant to sit
# above it, so they penalise the estimand rather than the model); active-slot
# pinball and coverage (they condition on the event having happened, which
# reverses several verdicts and is not what the model targets); exceedance,
# over-provision, fire rate and false-alarm rate (useful when sizing a specific
# site, too many columns for a results table). All of them remain in the *_full
# CSVs.
PAPER_METRICS: list[str] = [
    "Pinball_All",
    "Pinball_Skill_vs_Zero_pct",
    "Coverage_Global_pct",
    "Coverage_AlwaysZero_pct",
    "Coverage_Gap_pp",
    "Breach_Rate_pct",
    "Peak_Capture_pct",
]

PAPER_CONTEXT: list[str] = ["n_total", "n_active", "zero_rate_pct"]


def paper_view(rows: pd.DataFrame, extra: Sequence[str] = ()) -> pd.DataFrame:
    """Project a metric frame down to the reported columns, order preserved."""
    if rows.empty:
        return rows
    keys = [c for c in ("pipeline", "target", "zone", "model", "horizon_mins")
            if c in rows.columns]
    cols = keys + list(extra) + [c for c in PAPER_CONTEXT if c in rows.columns] \
           + [c for c in PAPER_METRICS if c in rows.columns]
    seen, ordered = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c); ordered.append(c)
    return rows[ordered]


# ── Multi-step degradation ────────────────────────────────────────────────────

def multistep_degradation(rows: pd.DataFrame, alpha: float,
                          reference: str, comparator: str,
                          targets: Sequence[str] | None = None) -> pd.DataFrame:
    """
    How fast each model's accuracy decays as the forecast reaches further ahead.

    A one-step comparison answers "which model is more accurate right now". An
    energy management system does not run one step ahead: it commits capacity
    hours in advance and feeds each forecast back in as the input to the next, so
    errors compound. The question that matters operationally is whether they
    compound at the SAME RATE for both models.

    `rows` is the wide rollout frame emitted by rollout.run_rollout: one row per
    (model, origin, step), carrying `<target>_true` and `<target>_pred` columns.

    Reported per (target, horizon):

      pinball_<model>       marginal pinball loss at that horizon
      degradation_<model>   that loss divided by the model's OWN loss at the
                            shortest horizon

    degradation is deliberately self-normalised. Dividing each model by its own
    one-step loss removes the level difference between them, so the curve answers
    "how much worse does this model get" rather than restating the one-step gap at
    every horizon. 2.0 at four hours means that model's loss has doubled relative
    to where it started.

      degradation_ratio     degradation_<reference> / degradation_<comparator>.
                            Below 1 means the reference decays more slowly.

                            Read it together with the absolute losses. A model
                            that starts far worse has an easy time posting a flat
                            degradation curve — a forecast that is already near
                            the always-zero floor cannot get much worse — so a
                            ratio above 1 is not by itself evidence against the
                            reference, and the paper must show both columns.

      dm_stat, p_value      Diebold-Mariano on the per-slot loss difference at
                            that horizon, marginal scope. Rollout rows at one
                            horizon are one per origin and serially dependent
                            (overlapping origins share ground truth), which the
                            HAC variance accounts for.
    """
    targets = list(targets) if targets else [
        c[:-5] for c in rows.columns if c.endswith("_true")]
    out = []
    for tgt in targets:
        tcol, pcol = f"{tgt}_true", f"{tgt}_pred"
        if tcol not in rows.columns:
            continue
        for h, g in rows.groupby("horizon_slots", observed=True):
            rec: dict[str, object] = {
                "target": tgt, "horizon_slots": int(h),
                "horizon_mins": int(g["horizon_mins"].iloc[0]),
            }
            losses: dict[str, np.ndarray] = {}
            for name in (reference, comparator):
                sub = g[g["model"] == name]
                if sub.empty:
                    continue
                y = sub[tcol].to_numpy(float)
                f = sub[pcol].to_numpy(float)
                losses[name] = pinball_series(y, f, alpha)
                rec[f"pinball_{name}"] = float(np.mean(losses[name]))
                # Skill against the always-zero forecaster AT THIS HORIZON. This
                # column is what stops degradation_ratio being misread: a model
                # sitting at ~0% skill has already collapsed onto the zero
                # forecast, and a flat decay curve for it means "there was
                # nothing left to lose", not "it is robust to horizon".
                zero = float(np.mean(pinball_series(y, np.zeros_like(y), alpha)))
                rec[f"skill_{name}_pct"] = (
                    float(100.0 * (1.0 - rec[f"pinball_{name}"] / zero)) if zero > 0 else np.nan)
                rec["n"] = int(len(y))
                rec["n_active"] = int((y > 0).sum())
            if reference in losses and comparator in losses:
                m = min(len(losses[reference]), len(losses[comparator]))
                dm = diebold_mariano(losses[reference][:m], losses[comparator][:m], h=1)
                rec["dm_stat"] = dm["DM_stat"]
                rec["p_value"] = dm["p_value"]
                rec["better"] = ((reference if dm["mean_diff"] < 0 else comparator)
                                 if np.isfinite(dm["mean_diff"]) else "n/a")
                rec["significant_5pct"] = bool(
                    np.isfinite(dm["p_value"]) and dm["p_value"] < 0.05)
            out.append(rec)

    deg = pd.DataFrame(out)
    if deg.empty:
        return deg
    deg = deg.sort_values(["target", "horizon_slots"]).reset_index(drop=True)

    h0 = deg["horizon_slots"].min()
    for name in (reference, comparator):
        col = f"pinball_{name}"
        if col not in deg.columns:
            continue
        base = deg[deg["horizon_slots"] == h0].set_index("target")[col]
        deg[f"degradation_{name}"] = [
            (r[col] / base[r["target"]])
            if base.get(r["target"], 0) not in (0, np.nan) and np.isfinite(base.get(r["target"], np.nan))
            else np.nan
            for _, r in deg.iterrows()
        ]

    a, b = f"degradation_{reference}", f"degradation_{comparator}"
    if a in deg.columns and b in deg.columns:
        deg["degradation_ratio"] = deg[a] / deg[b].replace(0, np.nan)

    # Flag rows where the comparator's decay curve is uninformative because it has
    # already collapsed onto the always-zero forecast. Reporting a favourable
    # degradation_ratio from such a row would be an artefact.
    sk = f"skill_{comparator}_pct"
    if sk in deg.columns:
        deg["comparator_at_floor"] = deg[sk] < 5.0
    return deg
