"""
configs.py — The AC and DC pipeline constants in one place.

Scope of the study (set here, enforced at load)
-----------------------------------------------
Both arms are single-zone studies, and that is now a stated decision rather than
something a sparsity threshold arrives at by accident:

  DC   PublicParking only, 2021-11-23 → 2024-05-31.
       The file also carries `Non-public Underground garage`, which holds 2
       events in three and a half years, and every month from 2024-06 onward is
       empty for both zones. Loading either would put rows in the training frame
       and the rollout's zone loop that can never support an estimate.

  AC   Employee Parking only, 2024-11-26 → 2026-08-03.
       `Garage` has 27 events, all in 2024-11/12, and none afterwards.

Windows
-------
Chosen to use the whole usable span rather than a short tail. The previous DC
test window (2024-03-01 → 2024-05-01) held 309 events, 239 after the warm-up —
too few to support a per-horizon rollout analysis, which is why the multi-step
degradation table came back empty. The window below holds ~808.

`data/processed/windows.json`, written by select_window.py, still overrides
everything here when present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import PipelineConfig, MixtureConfig, QUANTILE_GRID, MIN_ACTIVE_SLOTS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
WINDOWS_PATH = DATA_DIR / "windows.json"

_PIN_DATE = "2026-08-29"

# Hand-set windows, counted against the monthly event tables before being pinned.
# The counts live in window_evidence so describe_window() prints them on every run
# and a stale window is visible rather than silent.
_PINNED = {
    "ac": dict(
        # Employee Parking: 1,825 events over 2024-11-26 → 2026-08-03.
        train_end="2026-01-01",      # 1,080 events (2024-11 → 2025-12)
        valid_end="2026-04-01",      #   298 events (2026-01 → 2026-03)
        test_start="2026-04-01",
        test_end="2026-08-03",       #   447 events; frame ends here
        source=f"manual (configs.py, pinned {_PIN_DATE})",
        charger_type="AC", zones=["Employee Parking"],
        n_train_events=1080, n_valid_events=298, n_test_events=447,
        warmup_days=14,
    ),
    "dc": dict(
        # PublicParking: 3,383 events over 2021-11-23 → 2024-05-31.
        train_end="2023-09-01",      # 2,197 events (2021-11 → 2023-08)
        valid_end="2023-12-01",      #   378 events (2023-09 → 2023-11)
        test_start="2023-12-01",
        test_end="2024-05-31",       #   808 events; 2024-06 onward is empty
        source=f"manual (configs.py, pinned {_PIN_DATE})",
        charger_type="DC", zones=["PublicParking"],
        n_train_events=2197, n_valid_events=378, n_test_events=808,
        warmup_days=14,
    ),
}

_REQUIRED = ("train_end", "valid_end", "test_start", "test_end")


def load_windows(path: Path = WINDOWS_PATH) -> dict:
    """Read validator-confirmed windows. Missing file → empty dict, not an error."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:                                # pragma: no cover
        print(f"  [window] {path.name} unreadable ({exc}); falling back to pinned windows.")
        return {}
    out = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict) or any(f not in entry for f in _REQUIRED):
            print(f"  [window] {path.name}: entry {key!r} is missing one of "
                  f"{_REQUIRED}; ignoring it.")
            continue
        out[key.strip().lower()] = entry
    return out


_WINDOWS = load_windows()


def _window(key: str) -> tuple[dict, str, dict, tuple[str, ...] | None]:
    """(dates, source, evidence, zones) for one pipeline. windows.json beats _PINNED."""
    entry = _WINDOWS.get(key) or _PINNED[key]
    dates = {f: pd.Timestamp(entry[f]) for f in _REQUIRED}
    source = entry.get("source") or WINDOWS_PATH.name
    evidence = {k: v for k, v in entry.items() if k not in _REQUIRED and k != "source"}
    zones = entry.get("zones") or _PINNED[key].get("zones")
    return dates, source, evidence, (tuple(zones) if zones else None)


_ac_dates, _ac_src, _ac_ev, _ac_zones = _window("ac")
_dc_dates, _dc_src, _dc_ev, _dc_zones = _window("dc")

AC_CONFIG = PipelineConfig(
    name="AC",
    charger_type="AC",
    duration_cap=1440.0,                       # 24 h — workday parking
    data_path=DATA_DIR / "ml_features_ac_all.csv",
    models_dir=BASE_DIR / "new_models_ac",
    results_dir=BASE_DIR / "new_results_ac",
    default_zone="Employee Parking",
    experiment_name="AC_EV_Charging_Forecasting",
    min_active_slots=MIN_ACTIVE_SLOTS,
    zones=_ac_zones,
    window_source=_ac_src,
    window_evidence=_ac_ev,
    **_ac_dates,
)

DC_CONFIG = PipelineConfig(
    name="DC",
    charger_type="DC",
    duration_cap=240.0,                        # 4 h — rapid charging
    data_path=DATA_DIR / "ml_features_dc_train_val.csv",
    models_dir=BASE_DIR / "new_models_dc",
    results_dir=BASE_DIR / "new_results_dc",
    default_zone="PublicParking",
    experiment_name="DC_EV_Charging_Forecasting",
    duration_optuna_trials=100,
    min_active_slots=MIN_ACTIVE_SLOTS,
    zones=_dc_zones,
    window_source=_dc_src,
    window_evidence=_dc_ev,
    **_dc_dates,
)

CONFIGS = {"ac": AC_CONFIG, "dc": DC_CONFIG}

DEFAULT_MIXTURE = MixtureConfig(alpha=0.95, mode="exact", grid=QUANTILE_GRID)
LEGACY_MIXTURE = MixtureConfig(alpha=0.95, mode="legacy", grid=QUANTILE_GRID)


def get_config(name: str) -> PipelineConfig:
    key = name.strip().lower()
    if key not in CONFIGS:
        raise KeyError(f"unknown pipeline {name!r}; choose from {sorted(CONFIGS)}")
    return CONFIGS[key]


def add_window_flag(ap) -> None:
    """Shared CLI flag. Every entry point exposes it so the escape hatch is explicit."""
    ap.add_argument(
        "--allow-placeholder-window", action="store_true",
        help="Run even though the train/valid/test window has not been confirmed "
             "by select_window.py. Smoke tests only — never for reported numbers.")
