"""
rollout.py — Batched autoregressive multi-step rollout.

Why this module exists
----------------------
The previous rollout lived inside forecast.evaluate_multistep and had three
defects that between them produced an EMPTY degradation table for DC and a
three-row one for AC:

R1  Too few origins. Origins were taken at a stride of ~17 slots, giving ~265
    per zone for DC. At a 5.3% active rate that is ~14 active slots in each
    per-horizon bucket. `aggregate_over_zones` drops any zone with fewer than
    MIN_ACTIVE_SLOTS (30) active slots, so every bucket was dropped and the
    aggregate came back empty. AC sat just either side of 30, which is why
    exactly three of its twenty buckets survived. The threshold was written for
    a whole test window and was being applied to a subsample of it.

R2  Truncated history. The buffer was seeded from the TEST frame only, so an
    origin early in the test window had no 1-week or 2-week lag available and
    silently received zeros. The 1-step path reads those same lags from the full
    precomputed frame, so the two evaluation modes were not scoring the same
    features.

R3  Cost. One `predict_step` call per (origin, step) meant a single-row XGBoost
    predict per slot, which is why a dense origin set was never affordable.

The fix for all three is the same: roll every origin forward TOGETHER. At step k
we assemble one feature matrix holding the k-th slot of every origin and issue a
single batched prediction. That makes `--origin-method all` affordable, which
removes R1 at the source rather than by lowering a threshold, and it lets the
buffer be seeded from the full contiguous series, which removes R2.

How the autoregression is represented
-------------------------------------
For origin o and step k the slot being predicted sits at position
`origin_pos[o] + (k-1)` in the zone's contiguous series. A lag of L slots reads
position `origin_pos[o] + (k-1) - L`, which is

    TRUE history      when L >= k   (the position precedes the origin)
    a PREDICTION      when L <= k-1 (it is this rollout's own step k-L)

That test depends only on k and L, never on o, so each lag column is either one
vectorised gather from the true series or one column of the prediction buffer.
No per-origin Python loop is needed anywhere in the inner step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    LAG_SERIES, LAG_SLOTS, RESOLUTION_MINUTES, SEASONAL_PERIOD, TARGETS,
)

# Longest lag we must be able to look back through (2 weeks = 1344 slots).
MAX_LAG = max(LAG_SLOTS.values())


def _calendar_frame(ts: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Vectorised twin of forecast._time_features."""
    hour = ts.hour.to_numpy(float)
    minute = ts.minute.to_numpy(float)
    dow = ts.dayofweek.to_numpy(float)
    month = ts.month.to_numpy(float)
    return {
        "hour": hour, "minute": minute, "day_of_week": dow,
        "is_weekend": (dow >= 5).astype(float), "month": month,
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "minute_sin": np.sin(2 * np.pi * minute / 60.0),
        "minute_cos": np.cos(2 * np.pi * minute / 60.0),
        "day_sin": np.sin(2 * np.pi * dow / 7.0),
        "day_cos": np.cos(2 * np.pi * dow / 7.0),
        "month_sin": np.sin(2 * np.pi * month / 12.0),
        "month_cos": np.cos(2 * np.pi * month / 12.0),
    }


class ZoneRollout:
    """
    Rolls every origin of one zone forward together.

    series : dict target -> full contiguous array for the zone (train+valid+test)
    index  : DatetimeIndex aligned with those arrays
    origins: integer positions into `index` at which rollouts start
    """

    def __init__(self, series: dict[str, np.ndarray], power: np.ndarray,
                 index: pd.DatetimeIndex, origins: np.ndarray, max_horizon: int):
        self.s = {k: np.asarray(v, float) for k, v in series.items()}
        self.power = np.asarray(power, float)
        self.index = index
        self.o = np.asarray(origins, int)
        self.H = int(max_horizon)
        self.n = len(self.o)

        # Prediction buffers: P[target][o, j] is the value this rollout produced
        # for origin o at step j+1. Filled as the rollout advances.
        self.P = {t: np.zeros((self.n, self.H), float) for t in TARGETS}
        self.P_power = np.zeros((self.n, self.H), float)

        arr = self.s["arrival_count"]
        dur = self.s["avg_duration_mins"]

        # ── State seeded from TRUE history strictly before each origin ────────
        # Last non-zero duration / power, mirroring build_feature_row's
        # carry-forward. Computed once per origin by a running "last non-zero
        # position" scan over the whole series.
        self.last_dur = self._last_nonzero_before(dur, self.o)
        self.last_pow = self._last_nonzero_before(self.power, self.o)

        # Position of the most recent arrival strictly before the origin.
        nz_pos = np.where(arr > 0, np.arange(len(arr)), -1)
        last_nz = np.maximum.accumulate(nz_pos)
        self.last_arr_pos = np.where(self.o > 0, last_nz[self.o - 1], -1).astype(float)

        # Rolling activity window: the 96 activity flags preceding each origin.
        act = (arr > 0).astype(float)
        padded = np.concatenate([np.zeros(SEASONAL_PERIOD), act])
        win = np.lib.stride_tricks.sliding_window_view(padded, SEASONAL_PERIOD)
        self.act_win = win[self.o].copy()          # (n, 96), oldest first

        # Historical active rate at each hour, as of each origin. Exact for the
        # pre-origin history; the <= H predicted slots a rollout adds are not
        # folded back in, which moves this feature by O(1/len(history)).
        self.hour_rate = self._hour_rate_table(arr, self.index, self.o)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _last_nonzero_before(v: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Last strictly-preceding non-zero value of `v` for each position."""
        idx = np.where(v > 0, np.arange(len(v)), -1)
        last = np.maximum.accumulate(idx)
        out = np.zeros(len(pos), float)
        prev = np.where(pos > 0, last[np.maximum(pos - 1, 0)], -1)
        ok = prev >= 0
        out[ok] = v[prev[ok]]
        return out

    @staticmethod
    def _hour_rate_table(arr: np.ndarray, index: pd.DatetimeIndex,
                         origins: np.ndarray) -> np.ndarray:
        """(n_origins, 24) historical active rate per hour, as of each origin."""
        hours = index.hour.to_numpy()
        act = (arr > 0).astype(float)
        n = len(arr)
        cum_a = np.zeros((25, n + 1))
        cum_n = np.zeros((25, n + 1))
        for h in range(24):
            m = (hours == h).astype(float)
            cum_n[h, 1:] = np.cumsum(m)
            cum_a[h, 1:] = np.cumsum(m * act)
        out = np.zeros((len(origins), 24), float)
        for h in range(24):
            a = cum_a[h][origins]
            c = cum_n[h][origins]
            out[:, h] = np.divide(a, c, out=np.zeros_like(a), where=c > 0)
        return out

    def _lag_value(self, target_col: str, k: int, L: int) -> np.ndarray:
        """Lag L of `target_col` for every origin at step k (1-indexed)."""
        if L >= k:
            src = self.o + (k - 1) - L                  # strictly before origin
            vals = np.where(src >= 0, self.s[target_col][np.maximum(src, 0)], 0.0)
        else:
            vals = self.P[target_col][:, (k - 1) - L]   # our own earlier step
        return vals

    # ── the feature matrix for step k ─────────────────────────────────────────

    def features(self, k: int) -> pd.DataFrame:
        pos = self.o + (k - 1)
        ts = self.index[pos]
        cols = _calendar_frame(ts)

        for series, col in LAG_SERIES.items():
            for lag_name, L in LAG_SLOTS.items():
                v = self._lag_value(col, k, L)
                if col == "avg_duration_mins":
                    # build_feature_row substitutes the last non-zero duration
                    # wherever a duration lag is zero; the dataframe path does the
                    # same via apply_carry_forward_duration. Match both.
                    v = np.where(v > 0, v, self.last_dur)
                cols[f"{series}_{lag_name}"] = v

        cols["bins_since_last_arrival"] = np.where(
            self.last_arr_pos >= 0, (pos - 1) - self.last_arr_pos, pos.astype(float))
        cols["rolling_active_4h"] = self.act_win[:, -16:].mean(axis=1)
        cols["rolling_active_24h"] = self.act_win.mean(axis=1)
        cols["zone_hour_active_rate"] = self.hour_rate[
            np.arange(self.n), ts.hour.to_numpy()]
        cols["avg_power_kW"] = self.last_pow
        return pd.DataFrame(cols)

    # ── advance ───────────────────────────────────────────────────────────────

    def commit(self, k: int, preds: dict[str, np.ndarray]) -> None:
        """Record step-k predictions and roll the autoregressive state forward."""
        for t in TARGETS:
            self.P[t][:, k - 1] = preds[t]

        dur_h = preds["avg_duration_mins"] / 60.0
        power = np.divide(preds["avg_energy_kWh"], dur_h,
                          out=np.zeros_like(dur_h), where=dur_h > 0)
        self.P_power[:, k - 1] = power

        # Carry-forward state advances on the value we just produced.
        self.last_dur = np.where(preds["avg_duration_mins"] > 0,
                                 preds["avg_duration_mins"], self.last_dur)
        self.last_pow = np.where(power > 0, power, self.last_pow)

        fired = preds["arrival_count"] > 0
        self.last_arr_pos = np.where(fired, (self.o + (k - 1)).astype(float),
                                     self.last_arr_pos)
        self.act_win = np.concatenate(
            [self.act_win[:, 1:], fired.astype(float)[:, None]], axis=1)


def run_rollout(full_zone: pd.DataFrame, origins_ts: pd.DatetimeIndex,
                max_horizon: int, predictors: dict[str, callable],
                zone: str) -> pd.DataFrame:
    """
    Roll `predictors` forward from every origin, in lock-step.

    predictors : name -> f(feature_frame) -> {target: array}. Each is rolled out
                 independently (its own buffers), so a model's autoregression is
                 fed by its OWN predictions, which is what an EMS would see.

    Returns long-format rows: one per (model, origin, step).
    """
    z = full_zone.sort_values("timestamp").reset_index(drop=True)
    index = pd.DatetimeIndex(z["timestamp"])
    series = {t: z[t].to_numpy(float) for t in TARGETS}
    power = z["avg_power_kW"].to_numpy(float) if "avg_power_kW" in z else np.zeros(len(z))

    pos_of = pd.Series(np.arange(len(index)), index=index)
    origins = pos_of.reindex(origins_ts).dropna().to_numpy(int)
    # An origin needs MAX_LAG slots of real history behind it and max_horizon
    # slots of ground truth ahead of it, or it cannot be scored.
    origins = origins[(origins >= MAX_LAG) & (origins + max_horizon - 1 < len(index))]
    if origins.size == 0:
        return pd.DataFrame()

    frames = []
    for name, predict in predictors.items():
        rl = ZoneRollout(series, power, index, origins, max_horizon)
        for k in range(1, max_horizon + 1):
            X = rl.features(k)
            preds = predict(X)
            rl.commit(k, preds)

        for k in range(1, max_horizon + 1):
            p = rl.o + (k - 1)
            row = {
                "model": name, "zone": zone,
                "origin": index[rl.o], "timestamp": index[p],
                "horizon_slots": k, "horizon_mins": k * RESOLUTION_MINUTES,
            }
            for t in TARGETS:
                row[f"{t}_true"] = series[t][p]
                row[f"{t}_pred"] = rl.P[t][:, k - 1]
            frames.append(pd.DataFrame(row))

    return pd.concat(frames, ignore_index=True)
