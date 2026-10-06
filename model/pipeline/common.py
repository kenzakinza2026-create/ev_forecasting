"""
common.py — Shared configuration, data handling and mixture-quantile logic for
the AC and DC two-stage forecasting pipelines.

Why this module exists
----------------------
The AC and DC pipelines had drifted apart: different feature sets, different
evaluation code paths, and a cap applied on only one side of the comparison.
Part of the measured AC/DC performance gap was that divergence rather than
charger physics. Everything that must be identical across the two pipelines now
lives here; ac_* and dc_* scripts supply only constants.

Fixes implemented here
----------------------
F1  Charger-type filtering. The DC scripts read a frame containing every zone and
    both charger types and never filtered. DC models were trained and evaluated
    on AC rows. `load_feature_frame` now filters on chargerType and refuses to
    return a frame whose requested type is absent.

F2  Two-sided duration cap. The cap was applied to predictions (np.minimum at
    inference) but never to the targets, so a capped forecast was scored against
    an uncapped 8,120-minute ground truth. `apply_duration_cap` clips the target
    column at load, once, for train / valid / test alike, and reports how many
    rows it touched so the censoring can be declared in the paper.

MIX Exact mixture-quantile inversion. For the zero-inflated mixture
        F(y) = (1-p)*delta_0(y) + p*F_mag(y)
    the alpha-quantile for y > 0 solves (1-p) + p*F_mag(y) = alpha, giving
        Q_alpha = F_mag^{-1}( (alpha - 1 + p) / p )   when p > 1 - alpha
                = 0                                    otherwise.
    The previous code used F_mag^{-1}(alpha) for every p, which is the correct
    answer only at p = 1 and is over-conservative everywhere else. Simulation:
    the exact form holds 95.0% marginal coverage at every p, while the fixed-alpha
    form yields 97.0-99.9%. Set MixtureConfig.mode = "legacy" to reproduce the
    old behaviour for the ablation.

SKEW Feature parity. `apply_carry_forward_power` mirrors, in the dataframe, the
    avg_power_kW carry-forward that build_feature_row already did at inference.
    Without it the ablation scored the quantile model on a feature distribution
    it was never trained on, penalising it against its own baselines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# ── Column vocabulary ─────────────────────────────────────────────────────────

TARGETS: list[str] = [
    "arrival_count",
    "departure_count",
    "avg_energy_kWh",
    "avg_duration_mins",
]
SIG_COLS: list[str] = TARGETS + ["avg_power_kW"]

RESOLUTION_MINUTES = 15
SEASONAL_PERIOD = 96          # 24 h at 15-minute resolution

# G4 — minimum active slots for a zone to enter the headline aggregate.
#
# The previous threshold was `n_active == 0`, which only excluded completely
# empty zones. It let through a zone with a single event in the whole test
# window: every per-zone rate computed there is 0% or 100%, its MASE denominator
# is one absolute difference, and the event-weighted aggregate still gave it a
# non-zero weight. Thirty active slots is the smallest count at which a per-zone
# coverage estimate has a standard error below ~9 pp, which is the resolution at
# which the coverage gaps in this paper are argued. Zones below it are reported
# individually and excluded from the aggregate, and the exclusion is declared.
MIN_ACTIVE_SLOTS = 30

LAG_SLOTS: dict[str, int] = {
    "t_minus_1": 1, "t_minus_2": 2, "t_minus_3": 3,
    "1h_ago": 4, "24h_ago": 96, "48h_ago": 192,
    "1w_ago": 672, "2w_ago": 1344,
}
LAG_SERIES: dict[str, str] = {
    "arrivals": "arrival_count",
    "departures": "departure_count",
    "energy": "avg_energy_kWh",
    "duration": "avg_duration_mins",
}
TARGET_PREFIX: dict[str, str] = {
    "arrival_count": "arrivals",
    "departure_count": "departures",
    "avg_energy_kWh": "energy",
    "avg_duration_mins": "duration",
}
MODEL_KEY: dict[str, str] = {
    "arrival_count": "reg_arrival_count",
    "departure_count": "reg_departure_count",
    "avg_energy_kWh": "reg_energy",
    "avg_duration_mins": "reg_duration",
}
INTERMITTENCY_COLS = {
    "bins_since_last_arrival", "rolling_active_4h",
    "rolling_active_24h", "zone_hour_active_rate",
}

# Stage-2 quantile levels trained as a grid so inference can interpolate to the
# p-dependent conditional level demanded by the exact mixture inversion.
QUANTILE_GRID: tuple[float, ...] = (
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.965, 0.98, 0.99,
)


# ── Pipeline configuration ────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineConfig:
    """Everything that differs between the AC and DC arms."""
    name: str                      # "AC" | "DC"
    charger_type: str              # value of the chargerType column
    duration_cap: float            # minutes
    data_path: Path
    models_dir: Path
    results_dir: Path
    train_end: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    default_zone: str
    experiment_name: str
    optuna_trials: int = 50
    duration_optuna_trials: int = 100
    alpha: float = 0.95
    min_active_slots: int = MIN_ACTIVE_SLOTS
    # Zones this study is scoped to. None = keep every zone in the file.
    #
    # This replaces the previous arrangement, where an out-of-scope zone was
    # loaded, trained on, and then quietly dropped at reporting time by the
    # sparsity rule. That worked, but it made the study's scope an emergent
    # property of a threshold rather than a stated decision, and it left
    # near-empty zones in the training frame and in the rollout's zone loop.
    # Naming the zones here makes the scope explicit, declarable in the paper,
    # and enforced at load.
    zones: tuple[str, ...] | None = None
    # G1 — provenance of the train/valid/test dates. "placeholder" means nobody has
    # confirmed them against the data yet; every entry point refuses to run on that.
    window_source: str = "placeholder"
    window_evidence: dict = field(default_factory=dict)

    def ensure_dirs(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @property
    def window_confirmed(self) -> bool:
        return self.window_source not in ("", "placeholder", None)

    def assert_window_confirmed(self, allow_placeholder: bool = False) -> None:
        """
        FIX G1 — refuse to train or evaluate on an unconfirmed window.

        The DC dates in configs.py were left as a placeholder after the previous
        window (2025-04-02 -> 2025-05-24) was found to contain zero DC events. A
        comment saying "PLACEHOLDER" does not stop a run; this does. Confirm the
        window with

            python pipeline/select_window.py --pipeline dc --write

        which writes data/processed/windows.json from the data itself and is what
        this config reads.
        """
        if self.window_confirmed or allow_placeholder:
            if allow_placeholder and not self.window_confirmed:
                print(f"  [window] WARNING: running {self.name} on an UNCONFIRMED "
                      f"placeholder window because --allow-placeholder-window was "
                      f"given. Do not report these numbers.")
            return
        raise DataValidationError(
            f"{self.name} train/valid/test window is an unconfirmed PLACEHOLDER "
            f"({self.train_end.date()} / {self.valid_end.date()} / "
            f"{self.test_start.date()} -> {self.test_end.date()}).\n"
            f"  Run:  python pipeline/select_window.py --pipeline {self.name.lower()} --write\n"
            f"  then re-run this command. Override with --allow-placeholder-window "
            f"only for smoke tests."
        )

    def describe_window(self) -> str:
        ev = self.window_evidence or {}
        extra = ""
        if ev:
            extra = ("  [" + ", ".join(f"{k}={v}" for k, v in list(ev.items())[:5]) + "]")
        return (f"{self.name} window — train < {self.train_end.date()}, "
                f"valid < {self.valid_end.date()}, test {self.test_start.date()} "
                f"-> {self.test_end.date()}  (source: {self.window_source}){extra}")


@dataclass
class MixtureConfig:
    """
    How Stage-1 probability and Stage-2 magnitude are combined.

    mode = "exact"  → Q_alpha = F_mag^{-1}((alpha-1+p)/p)   [correct]
    mode = "legacy" → Q_alpha = F_mag^{-1}(alpha)           [previous behaviour]
    """
    alpha: float = 0.95
    mode: str = "exact"
    grid: tuple[float, ...] = QUANTILE_GRID

    def __post_init__(self) -> None:
        if self.mode not in {"exact", "legacy"}:
            raise ValueError(f"mode must be 'exact' or 'legacy', got {self.mode!r}")


# ── Data loading ──────────────────────────────────────────────────────────────

class DataValidationError(RuntimeError):
    """Raised when a frame or window cannot support the requested evaluation."""


def read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="mixed")
    if ts.isna().any():
        ts = pd.to_datetime(df["timestamp"], dayfirst=True, format="mixed")
        df["timestamp"] = ts
    else:
        df["timestamp"] = ts.dt.tz_localize(None)
    return df


def apply_duration_cap(df: pd.DataFrame, cap: float, verbose: bool = True) -> pd.DataFrame:
    """
    FIX F2 — clip the duration TARGET at the same cap the forecaster is clipped to.

    Previously the cap was enforced only on predictions, so a forecast pinned at
    240 min was scored against ground-truth values up to 8,120 min and could never
    cover them. The cap is a declared modelling assumption (implausible dwell
    times are censored); it therefore has to apply to both sides of the
    comparison, and the count of affected rows belongs in the paper.
    """
    df = df.copy()
    col = "avg_duration_mins"
    if col not in df.columns:
        return df
    over = int((df[col] > cap).sum())
    df[col] = df[col].clip(upper=cap)
    if "avg_power_kW" in df.columns:
        # Power is derived from energy/duration; recompute where duration changed.
        with np.errstate(divide="ignore", invalid="ignore"):
            hours = df[col].to_numpy(float) / 60.0
            power = np.where(hours > 0, df["avg_energy_kWh"].to_numpy(float) / hours, 0.0)
        df["avg_power_kW"] = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)
    if verbose and over:
        pct = 100.0 * over / max(1, len(df))
        print(f"  [cap] clipped {over:,} duration rows (>{cap:g} min) — {pct:.2f}% of frame")
    return df


def apply_carry_forward_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace zero-filled duration lag columns with the last known non-zero
    duration, per zone x chargerType.

    The preprocessor writes avg_duration_mins = 0 for inactive slots; those zeros
    propagate into duration_* lags, so a Q95 regressor sees a matrix that is 99%
    zeros and learns "duration is about zero".
    """
    df = df.copy().sort_values(["zone", "chargerType", "timestamp"])
    cols = [c for c in df.columns if c.startswith("duration_")]
    if not cols:
        return df
    grouper = ["zone", "chargerType"]
    for col in cols:
        df[col] = df[col].replace(0.0, np.nan)
        df[col] = df.groupby(grouper, observed=True)[col].ffill().fillna(0.0)
    return df


def apply_carry_forward_power(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX (train/serve skew) — carry avg_power_kW forward in the dataframe.

    build_feature_row already carried this forward at inference, but the
    precomputed avg_power_kW column stayed zero-filled. Training and the
    dataframe-based ablation therefore used a different distribution from live
    forecasting, which penalised the quantile model against its own baselines.
    """
    df = df.copy()
    if "avg_power_kW" not in df.columns:
        return df
    df = df.sort_values(["zone", "chargerType", "timestamp"])
    s = df["avg_power_kW"].replace(0.0, np.nan)
    df["avg_power_kW"] = (
        s.groupby([df["zone"], df["chargerType"]], observed=True).ffill().fillna(0.0)
    )
    return df


def load_feature_frame(
    cfg: PipelineConfig,
    *,
    apply_cap: bool = True,
    carry_forward: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the feature frame for one charger type, with every consistency fix applied.

    FIX F1 — the chargerType filter. Without it the DC pipeline silently trained
    and evaluated on AC rows, because the shared CSV holds both.
    """
    df = read_frame(cfg.data_path)

    if "chargerType" not in df.columns:
        raise DataValidationError(
            f"{cfg.data_path.name} has no 'chargerType' column; cannot guarantee "
            f"that only {cfg.charger_type} rows are used. Re-run the preprocessor."
        )

    present = sorted(df["chargerType"].dropna().unique().tolist())
    if cfg.charger_type not in present:
        raise DataValidationError(
            f"chargerType={cfg.charger_type!r} absent from {cfg.data_path.name}. "
            f"Present: {present}. This is the failure that produced a 'DC' ablation "
            f"containing only AC zones — refusing to continue."
        )

    before = len(df)
    df = df[df["chargerType"] == cfg.charger_type].copy()
    if verbose:
        dropped = before - len(df)
        print(f"  [filter] chargerType == {cfg.charger_type}: kept {len(df):,} rows, "
              f"dropped {dropped:,} of other types {[t for t in present if t != cfg.charger_type]}")

    if cfg.zones:
        want = set(cfg.zones)
        have = set(df["zone"].dropna().unique())
        missing = sorted(want - have)
        if missing:
            raise DataValidationError(
                f"Configured zones {missing} are absent from {cfg.data_path.name} for "
                f"chargerType={cfg.charger_type}. Present: {sorted(have)}."
            )
        n_before = len(df)
        df = df[df["zone"].isin(want)].copy()
        if verbose:
            print(f"  [scope] zones {sorted(want)}: kept {len(df):,} rows, "
                  f"dropped {n_before - len(df):,} from {sorted(have - want)}")

    if apply_cap:
        df = apply_duration_cap(df, cfg.duration_cap, verbose=verbose)
    if carry_forward:
        df = apply_carry_forward_duration(df)
        df = apply_carry_forward_power(df)

    return df.sort_values(["zone", "chargerType", "timestamp"]).reset_index(drop=True)


def split_frame(df: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, ...]:
    """Chronological train / valid / test split, each sorted for positional alignment."""
    ts = df["timestamp"]
    train = df[ts < cfg.train_end].copy()
    valid = df[(ts >= cfg.train_end) & (ts < cfg.valid_end)].copy()
    test = df[(ts >= cfg.test_start) & (ts <= cfg.test_end)].copy()
    out = []
    for part in (train, valid, test):
        out.append(part.sort_values(["zone", "chargerType", "timestamp"]).reset_index(drop=True))
    return tuple(out)


# ── Zone activity accounting (F7) ─────────────────────────────────────────────

def zone_activity(
    df: pd.DataFrame,
    target: str = "arrival_count",
    *,
    min_active: int = MIN_ACTIVE_SLOTS,
) -> pd.DataFrame:
    """
    Per-zone slot and event counts.

    `degenerate` now marks zones with FEWER THAN `min_active` events, not only
    zones with none (G4). `reason` distinguishes the two cases so the paper can
    report them separately: an empty zone is a data-coverage fact, a sparse zone
    is an estimation-precision decision.
    """
    g = df.groupby("zone", observed=True)[target]
    out = pd.DataFrame({"n_slots": g.size(), "n_active": (g.apply(lambda s: int((s > 0).sum())))})
    out["zero_rate_pct"] = 100.0 * (1.0 - out["n_active"] / out["n_slots"].clip(lower=1))
    out["degenerate"] = out["n_active"] < int(min_active)
    out["reason"] = np.where(
        out["n_active"] == 0, "empty",
        np.where(out["degenerate"], f"sparse (<{int(min_active)} active)", ""),
    )
    out["min_active_threshold"] = int(min_active)
    return out.reset_index()


def live_zones(
    df: pd.DataFrame,
    target: str = "arrival_count",
    *,
    min_active: int = MIN_ACTIVE_SLOTS,
) -> list[str]:
    """
    FIX F7 + G4 — zones with enough events to support a per-zone estimate.

    All-zero zones were being folded into pooled averages, multiplying n_total and
    dragging every global metric toward zero. Near-empty zones did the same thing
    more subtly. Both are reported separately now.
    """
    act = zone_activity(df, target, min_active=min_active)
    return act.loc[~act["degenerate"], "zone"].tolist()


def describe_zone_activity(act: pd.DataFrame, indent: str = "  ") -> str:
    """One-line-per-zone summary, for the run log and the paper's data section."""
    lines = []
    for _, r in act.sort_values("n_active", ascending=False).iterrows():
        tag = "" if not r["degenerate"] else f"   EXCLUDED: {r['reason']}"
        lines.append(
            f"{indent}{str(r['zone'])[:34]:<34} slots {int(r['n_slots']):>7,}  "
            f"active {int(r['n_active']):>6,}  zero {r['zero_rate_pct']:6.2f}%{tag}"
        )
    return "\n".join(lines)


# ── Feature selection (unified across AC and DC) ──────────────────────────────

BASE_EXCLUDE = [
    "timestamp", "zone", "chargerType", "p_arrival", "p_departure",
    "arrival_count", "departure_count", "avg_energy_kWh", "avg_duration_mins",
    "hour", "minute", "day_of_week", "month",
]


def select_features(df: pd.DataFrame, stage: int, target_col: str) -> list[str]:
    """
    Stage-aware feature list, identical for AC and DC.

    Previously ac_train excluded avg_power_kW outright while dc_train fed it to the
    duration regressor, and only DC let energy lags reach the duration model. The
    documented design (README 3.5) is the DC behaviour, so that is what both arms
    now use.

    Stage 1 (occurrence): calendar + the target's own count lags + intermittency.
    Stage 2 (magnitude):  calendar + the target's own lags; the duration regressor
                          additionally receives energy lags, avg_power_kW and the
                          intermittency block, which is what recovered DC duration
                          coverage in the first place.
    """
    is_duration = target_col == "avg_duration_mins"
    prefix = TARGET_PREFIX.get(target_col, "")
    kept: list[str] = []

    for f in df.columns:
        if f in BASE_EXCLUDE:
            continue

        if f == "avg_power_kW" and not (stage == 2 and is_duration):
            continue

        if stage == 1:
            if "energy_" in f or "duration_" in f:
                continue
            if ("arrivals_" in f or "departures_" in f) and not f.startswith(prefix):
                continue
            if "t_minus_2" in f or "t_minus_3" in f:
                continue

        elif stage == 2:
            if f in INTERMITTENCY_COLS and not is_duration:
                continue
            if any(k in f for k in ("arrivals_", "departures_", "energy_", "duration_")):
                if is_duration and "energy_" in f:
                    pass                      # energy informs dwell time
                elif not f.startswith(prefix):
                    continue
                if "t_minus_2" in f or "t_minus_3" in f:
                    continue
        else:
            raise ValueError(f"stage must be 1 or 2, got {stage}")

        kept.append(f)

    return kept


def extract_features_for_stage(df: pd.DataFrame, stage: int, target_col: str):
    cols = select_features(df, stage, target_col)
    return df[cols], cols


# ── Rollout origin scheduling (G2) ────────────────────────────────────────────

def _coprime_stride(target: int, period: int = SEASONAL_PERIOD) -> int:
    """Smallest stride >= target that is coprime with the daily period."""
    from math import gcd
    k = max(1, int(target))
    while gcd(k, period) != 1:
        k += 1
    return k


def sample_origins(
    index: pd.DatetimeIndex,
    *,
    max_horizon: int,
    method: str = "coprime",
    stride: int | None = None,
    n_origins: int | None = None,
    seed: int = 42,
    period: int = SEASONAL_PERIOD,
) -> pd.DatetimeIndex:
    """
    Choose rollout origins WITHOUT aliasing horizon against time of day.

    The defect this replaces
    ------------------------
    The previous code took `index[::max_horizon]`. With max_horizon = 16 slots
    (4 h) and 96 slots per day, 16 divides 96 exactly, so every origin fell on one
    of only six clock times, and step k of every rollout always landed on the same
    six clock times too. Horizon was therefore perfectly confounded with
    time-of-day: the reported "degradation with horizon" curve was partly a
    daily-profile curve, and the evaluation never scored 90 of the 96 slots in the
    day. Any stride dividing 96 has this property, which is most round numbers an
    author would pick (4, 8, 12, 16, 24, 32, 48).

    Methods
    -------
    coprime  Deterministic stride, chosen as the smallest value >= max_horizon
             that is coprime with `period`. The origin phase then advances by one
             slot every `period / gcd = 96` origins and sweeps all 96 clock
             positions uniformly. Reproducible and reviewer-checkable.
    random   Uniform sample without replacement, fixed seed. Unbiased by
             construction; use when you also want origins non-equispaced.
    all      Every slot is an origin. Correct but ~`max_horizon`x the cost, and
             the resulting rollouts overlap heavily, so the effective sample size
             is far below the row count — quote block-bootstrap CIs if you use it.
    """
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return index

    if method == "all":
        return index
    if method == "random":
        k = int(n_origins) if n_origins else max(1, len(index) // max(1, max_horizon))
        k = min(k, len(index))
        rng = np.random.default_rng(seed)
        pos = np.sort(rng.choice(len(index), size=k, replace=False))
        return index[pos]
    if method != "coprime":
        raise ValueError(f"method must be 'coprime', 'random' or 'all', got {method!r}")

    st = int(stride) if stride else _coprime_stride(max_horizon, period)
    from math import gcd
    if gcd(st, period) != 1:
        raise ValueError(
            f"origin stride {st} shares a factor with the daily period {period}; "
            f"it would alias horizon against time of day. Nearest safe stride: "
            f"{_coprime_stride(st, period)}."
        )
    return index[::st]


def origin_balance(origins: pd.DatetimeIndex, period: int = SEASONAL_PERIOD) -> dict:
    """
    How uniformly the origins cover the day. Report this in the paper's protocol.

    `slots_covered` out of `period` and `chi2_uniform_p` (a plain chi-square
    goodness-of-fit against uniform). Under the old stride the values were 6/96
    and p ~ 0.
    """
    origins = pd.DatetimeIndex(origins)
    if len(origins) == 0:
        return {"n_origins": 0, "slots_covered": 0, "slots_total": period,
                "coverage_pct": 0.0, "chi2_uniform_p": float("nan")}
    slot = (origins.hour * 60 + origins.minute) // RESOLUTION_MINUTES
    counts = np.bincount(np.asarray(slot, int), minlength=period)[:period]
    covered = int((counts > 0).sum())
    exp = counts.sum() / period
    p = float("nan")
    if exp > 0:
        chi2 = float(np.sum((counts - exp) ** 2) / exp)
        try:
            from scipy import stats as _st
            p = float(1.0 - _st.chi2.cdf(chi2, df=period - 1))
        except Exception:
            p = float("nan")
    return {"n_origins": int(len(origins)), "slots_covered": covered,
            "slots_total": int(period),
            "coverage_pct": round(100.0 * covered / period, 2),
            "chi2_uniform_p": p}


# ── Exact mixture-quantile inversion ──────────────────────────────────────────

def conditional_level(p: np.ndarray | float, alpha: float) -> np.ndarray:
    """
    The conditional quantile level the Stage-2 model must be evaluated at.

        level = (alpha - 1 + p) / p     for p > 1 - alpha
        level = 0                       otherwise (the mixture quantile is 0)

    At p = 1 this returns alpha, which is why the old fixed-alpha code looked
    right in the degenerate always-active case and nowhere else.

    Note on attainability: the mixture carries an atom of mass (1-p) at zero. When
    1 - p >= alpha the alpha-quantile IS zero and realised coverage is 1 - p, which
    exceeds alpha. That over-coverage is a property of the distribution, not slack
    in the estimator, and it is unavoidable in the sparse regime. It is worth
    stating explicitly in the paper: on a 97%-zero series, perfect marginal
    calibration at alpha = 0.95 is not achievable slot-by-slot, so the honest
    target is the smallest coverage gap consistent with the atom.
    """
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lvl = (alpha - 1.0 + p) / p
    return np.clip(np.nan_to_num(lvl, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def _sorted_grid_predictions(raw: np.ndarray) -> np.ndarray:
    """
    Enforce monotonicity across quantile levels.

    XGBoost's multi-quantile objective fits each level independently, so predicted
    quantiles can cross. Sorting each row is the standard rearrangement fix
    (Chernozhukov et al., 2010) and never worsens the loss.
    """
    return np.sort(np.asarray(raw, dtype=float), axis=1)


def interpolate_quantile(
    grid_preds: np.ndarray,
    grid_levels: Sequence[float],
    levels: np.ndarray,
) -> np.ndarray:
    """
    Read the Stage-2 quantile function at a per-row level.

    grid_preds : (n, k) predictions at the k trained levels
    levels     : (n,)   the level wanted for each row
    """
    grid_preds = _sorted_grid_predictions(grid_preds)
    gl = np.asarray(grid_levels, dtype=float)
    levels = np.asarray(levels, dtype=float)
    out = np.empty(len(levels), dtype=float)
    for i in range(len(levels)):
        out[i] = np.interp(levels[i], gl, grid_preds[i])
    return np.maximum(out, 0.0)


def mixture_quantile(
    p: np.ndarray | float,
    q_grid: np.ndarray,
    mix: MixtureConfig,
    *,
    cap: float | None = None,
) -> np.ndarray:
    """
    Combine occurrence probability and conditional magnitude into the mixture Q_alpha.

    q_grid : (n, k) Stage-2 predictions at mix.grid levels, in real-world units.
             In legacy mode a single column (the fixed-alpha prediction) is
             accepted so the old behaviour can be reproduced exactly.
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    q_grid = np.asarray(q_grid, dtype=float)
    if q_grid.ndim == 1:
        q_grid = q_grid[:, None]

    active = p > (1.0 - mix.alpha)

    if mix.mode == "legacy":
        if q_grid.shape[1] == 1:
            q = q_grid[:, 0]
        else:
            q = interpolate_quantile(q_grid, mix.grid, np.full(len(p), mix.alpha))
    else:
        levels = conditional_level(p, mix.alpha)
        q = interpolate_quantile(q_grid, mix.grid, levels)

    out = np.where(active, np.maximum(q, 0.0), 0.0)
    if cap is not None:
        out = np.minimum(out, cap)
    return out


# ── Stage-1 probability calibration ───────────────────────────────────────────

def fit_isotonic(p_raw: np.ndarray, y_bin: np.ndarray) -> dict | None:
    """
    Fit an isotonic map from raw classifier scores to calibrated probabilities.

    Why this is now required: the exact mixture inversion reads F_mag^{-1} at level
    (alpha-1+p)/p, so an error in p moves the quantile level directly. The old
    fixed-alpha inversion only compared p against a threshold, so calibration was
    almost irrelevant there. Returned as knots so it can be stored as plain JSON
    beside the XGBoost models rather than pickled.
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:                                    # pragma: no cover
        return None
    p_raw = np.asarray(p_raw, float)
    y_bin = np.asarray(y_bin, float)
    if len(p_raw) < 50 or len(np.unique(y_bin)) < 2:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_raw, y_bin)
    xs = np.unique(np.clip(p_raw, 0.0, 1.0))
    if len(xs) > 512:
        xs = np.quantile(xs, np.linspace(0, 1, 512))
    return {"x": [float(v) for v in xs], "y": [float(v) for v in iso.predict(xs)]}


def apply_isotonic(p_raw: np.ndarray, knots: dict | None) -> np.ndarray:
    """Apply a stored isotonic map; identity when no calibrator was fitted."""
    p_raw = np.asarray(p_raw, float)
    if not knots:
        return p_raw
    x = np.asarray(knots["x"], float)
    y = np.asarray(knots["y"], float)
    return np.clip(np.interp(p_raw, x, y), 0.0, 1.0)


# ── Model config persistence ──────────────────────────────────────────────────

def save_model_config(models_dir: Path, cfg: PipelineConfig, mix: MixtureConfig,
                      log_targets: dict[str, bool], capped_rows: int | None = None) -> None:
    payload = {
        "pipeline": cfg.name,
        "charger_type": cfg.charger_type,
        "quantile_alpha": mix.alpha,
        "mixture_mode": mix.mode,
        "quantile_grid": list(mix.grid),
        "duration_cap_mins": cfg.duration_cap,
        "log_transform_targets": log_targets,
        "splits": {
            "train_end": str(cfg.train_end),
            "valid_end": str(cfg.valid_end),
            "test_start": str(cfg.test_start),
            "test_end": str(cfg.test_end),
        },
    }
    if capped_rows is not None:
        payload["duration_rows_capped"] = int(capped_rows)
    (models_dir / "model_config.json").write_text(json.dumps(payload, indent=2))


def load_model_config(models_dir: Path) -> dict:
    return json.loads((models_dir / "model_config.json").read_text())
