# EV Charging Demand Forecaster (AC and DC)

Pre-trained models that forecast EV charging activity at a parking zone, slot by
slot, in 15-minute steps. For every future slot they return a **95th-percentile
(Q95) upper bound** on arrivals, departures, energy per session and plug-in
duration, plus the probability that anything happens at all.

There are two independent models:

| Pipeline | Charger type | Zone it was trained on | Training data | Validation data |
|---|---|---|---|---|
| `ac` | AC (slow charging) | `Employee Parking` | 2024-11-26 → 2025-12-31 | 2026-01-01 → 2026-03-31 |
| `dc` | DC (fast charging) | `PublicParking` | 2021-11-23 → 2023-08-31 | 2023-09-01 → 2023-11-30 |

This package **only produces forecasts**. The preprocessing, training and
evaluation code is not included.

---

## Contents

0. [Quick start](#0-quick-start)
1. [Directory structure](#1-directory-structure)
2. [Installation](#2-installation)
3. [What you need to provide (input)](#3-what-you-need-to-provide-input)
4. [Running a forecast](#4-running-a-forecast)
5. [What you get back (output)](#5-what-you-get-back-output)
6. [How the model works](#6-how-the-model-works)
7. [Limitations and troubleshooting](#7-limitations-and-troubleshooting)

---

## 0. Quick start

```bash
cd ev_forecaster
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux
pip install xgboost==3.3.0 pandas==2.3.3 scikit-learn numpy

# put your history CSV in data/processed/ (see section 3), then:
python pipeline/ac_forecast.py --time "2026-07-20T08:00" --horizon 4h
python pipeline/dc_forecast.py --time "2024-05-20T08:00" --horizon 4h
```

---

## 1. Directory structure

```
ev_forecaster/
├── README.md
│
├── pipeline/                           code
│   ├── ac_forecast.py                  entry point: AC model
│   ├── dc_forecast.py                  entry point: DC model
│   ├── forecast.py                     forecasting logic, shared by AC and DC
│   ├── common.py                       data loading, feature definitions, quantile maths
│   ├── configs.py                      file locations, zone names, duration caps
│   ├── baselines.py                    ┐ not used to forecast, but forecast.py
│   ├── metrics.py                      │ imports them, so the scripts will not
│   └── rollout.py                      ┘ start without them
│
├── new_models_ac/                      trained AC models
│   ├── clf_arrival.json                Stage 1: P(at least one arrival)
│   ├── clf_departure.json              Stage 1: P(at least one departure)
│   ├── calibration_clf_arrival.json    calibration of the Stage 1 probabilities
│   ├── calibration_clf_departure.json
│   ├── reg_arrival_count.json          Stage 2: quantile regressors
│   ├── reg_departure_count.json
│   ├── reg_energy.json
│   ├── reg_duration.json
│   ├── feature_cols.json               exact input columns of each model
│   └── model_config.json               quantile level, duration cap, charger type
│
├── new_models_dc/                      trained DC models: the same 10 files
│   └── ...
│
├── data/
│   └── processed/                      ← YOUR history files go here
│       ├── ml_features_ac_all.csv          history for the AC model
│       └── ml_features_dc_train_val.csv    history for the DC model
│
├── new_results_ac/                     empty; AC files saved with --output land here
└── new_results_dc/                     empty; DC files saved with --output land here
```

Folder and file names are hard-coded in `pipeline/configs.py`. **Do not rename
them.** The two CSV names come from the original research project. Here they
only need to hold your recent history, in the format described in section 3.
You only need the CSV for the model you use.

---

## 2. Installation

**Python 3.12 or newer** (XGBoost 3.3.0 does not install on older versions).
Tested with Python 3.14.2 on Windows 11.

```bash
cd ev_forecaster
python -m venv .venv

# activate it
.venv\Scripts\Activate.ps1          # Windows PowerShell
.venv\Scripts\activate.bat          # Windows cmd
source .venv/bin/activate           # macOS / Linux

pip install xgboost==3.3.0 pandas==2.3.3 scikit-learn numpy
```

On Windows, if `python` is not found, use `py` instead.

| Package | Why it is needed | Tested version |
|---|---|---|
| `xgboost` | loads and runs the models | **3.3.0** |
| `scikit-learn` | required by XGBoost's `XGBClassifier` / `XGBRegressor` classes | 1.9.0 |
| `pandas` | reads the history CSV and builds the features | 2.3.3 |
| `numpy` | numerics | 2.5.1 |

Keep `xgboost` at **3.3.0**. The models were saved with that version, and using
the same version guarantees you get the same predictions.

Check the installation:

```bash
python pipeline/ac_forecast.py --help
```

---

## 3. What you need to provide (input)

The models are **autoregressive**: they forecast the next slots from what
happened at the zone recently. The only input is a CSV of **observed charging
activity at 15-minute resolution**, covering at least the **15 days immediately
before** the time you forecast from. The scripts compute all the model features
(lags, calendar variables, activity rates) from this file.

### 3.1 File format

- AC model: `data/processed/ml_features_ac_all.csv`
- DC model: `data/processed/ml_features_dc_train_val.csv`

```csv
timestamp,zone,chargerType,arrival_count,departure_count,avg_energy_kWh,avg_duration_mins,avg_power_kW
2026-07-05 08:00:00,Employee Parking,AC,0,0,0.0,0.0,0.0
2026-07-05 08:15:00,Employee Parking,AC,1,0,38.441,223.0,10.34287
2026-07-05 08:30:00,Employee Parking,AC,0,0,0.0,0.0,0.0
...
2026-07-20 07:45:00,Employee Parking,AC,0,1,0.0,0.0,0.0
```

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | `YYYY-MM-DD HH:MM:SS` | **Start** of the 15-minute slot (`08:15:00` = 08:15 to 08:30). Minutes must be 00, 15, 30 or 45. |
| `zone` | text | Exactly `Employee Parking` (AC) or `PublicParking` (DC) |
| `chargerType` | text | Exactly `AC` or `DC` |
| `arrival_count` | integer ≥ 0 | Number of charging sessions that **started** (vehicle plugged in) in this slot |
| `departure_count` | integer ≥ 0 | Number of sessions that **ended** (vehicle unplugged) in this slot |
| `avg_energy_kWh` | float ≥ 0 | Mean energy delivered per session, over the sessions that **started** in this slot. `0` if none started. |
| `avg_duration_mins` | float ≥ 0 | Mean plug-in duration (plug-in to unplug) of the sessions that **started** in this slot. `0` if none started. |
| `avg_power_kW` | float ≥ 0 | `avg_energy_kWh / (avg_duration_mins / 60)` if `avg_duration_mins > 0`, otherwise `0` |

A session's energy and duration are recorded in the slot where it **started**,
not where it ended. Slots with only a departure have `avg_energy_kWh = 0` and
`avg_duration_mins = 0`.

### 3.2 Rules the file must follow

1. **One row per slot, with no gaps.** Slots with no activity must still be
   present, filled with zeros. This includes nights and weekends. The features
   count rows (for example, "the last 16 rows = the last 4 hours"), so a missing
   row shifts everything.
2. **At least 15 days of history** (1,440 rows) ending at the slot just before
   `--time`. The models look back up to 2 weeks. **If a needed slot is missing
   it is treated as 0 without any error**, and the forecasts silently get
   worse. More history is fine because only the last 15 days are used, but the
   whole file is read on every run.
3. **Timestamps without a timezone** (`2026-07-20 07:45:00`, not
   `2026-07-20T07:45:00+02:00`). Timestamps with an offset are converted to
   UTC, which shifts the hour of day. Use the same local clock as the site's
   charging records.
4. Rows at or after `--time` are ignored, so you can keep appending to the same
   file.
5. Rows for other zones or charger types, and any extra columns, are ignored.
   Column order does not matter.
6. You do not need to clean long durations. They are clipped automatically at
   the cap the model was trained with (**AC 1440 min, DC 240 min**), and
   `avg_power_kW` is recomputed for those rows.

---

## 4. Running a forecast

Run from the `ev_forecaster/` folder with the virtual environment activated:

```bash
# AC: the 4 hours starting 20 July 2026 08:00
python pipeline/ac_forecast.py --time "2026-07-20T08:00" --horizon 4h

# DC: the next hour, also saved to new_results_dc/dc_forecast.json
python pipeline/dc_forecast.py --time "2024-05-20T08:00" --horizon 1h --output dc_forecast.json
```

`python pipeline/forecast.py --pipeline ac ...` is equivalent to
`python pipeline/ac_forecast.py ...`.

| Option | Default | Description |
|---|---|---|
| `--time` | now, rounded down to the last 15-minute mark | Start of the **first forecast slot**. History before this time is used as input. Format `YYYY-MM-DDTHH:MM`. Must fall on :00, :15, :30 or :45. |
| `--horizon` | `1h` (4 slots) | How far ahead to forecast. Use the suffix `min`, `h` or `d`: `15min`, `30min`, `2h`, `4h`, `1d`. Rounded down to whole 15-minute slots. |
| `--output` | none | Also save the JSON result under this file name in `new_results_ac/` or `new_results_dc/` |
| `--zone` | `Employee Parking` / `PublicParking` | Leave at the default. Each model was trained on that one zone only. |

The script prints one line describing the model's training window, then the
JSON forecast. Use `--output` to get a clean JSON file.

**Recommended horizon: up to 4 h.** The models were evaluated up to 4 hours
(16 slots). Beyond the first slot, each step feeds the model's own earlier
forecasts back in as inputs, so accuracy drops as the horizon grows. For an
updated forecast, add the newly observed slots to the CSV and run again.

**Do not use** `--eval-only`, `--multistep-eval` or `--inversion legacy`. They
reproduce the research evaluation and need the full training datasets, which
are not included.

### From Python

```python
import sys
sys.path.insert(0, "pipeline")            # run from the ev_forecaster/ folder

import pandas as pd
from configs import get_config
from forecast import load_models, generate_forecast

cfg = get_config("ac")                    # or "dc"
bundle = load_models(cfg)                 # load once, reuse
result = generate_forecast(cfg, bundle, bundle.mixture("exact"),
                           pd.Timestamp("2026-07-20 08:00"),
                           horizon_slots=16)  # 16 x 15 min = 4 h
df = pd.DataFrame(result["forecastData"])
```

---

## 5. What you get back (output)

### 5.1 Example

AC output, shortened to the first slot:

```json
{
  "forecastReferenceTime": "2026-07-20T08:00:00",
  "pipeline": "AC",
  "zone": "Employee Parking",
  "quantileAlpha": 0.95,
  "mixtureInversion": "exact",
  "resolutionMinutes": 15,
  "durationCapMins": 1440.0,
  "forecastData": [
    {
      "timeSlotStart": "2026-07-20T08:00:00",
      "stepsAhead": 1,
      "minutesAhead": 15,
      "Q95_arrivalCount": 1.1453,
      "Q95_departureCount": 0.0,
      "Q95_energyKwh": 38.1167,
      "Q95_durationMins": 585.9,
      "pArrivalActive": 0.3469,
      "pDepartureActive": 0.0405,
      "conditionalLevelUsed": 0.8559,
      "stage2_arrivalCount_qmag": 1.9605,
      "stage2_energyKwh_qmag": 45.3569,
      "stage2_durationMins_qmag": 671.4
    }
  ]
}
```

`forecastData` holds one object per slot, so `--horizon 4h` gives 16 objects.

### 5.2 What "Q95" means

A Q95 value is a level the real value will be **at or below 95% of the time**.
It is a conservative planning bound, not an average or a best guess. At these
zones about 97% of 15-minute slots have no activity, so the most likely outcome
of a slot is 0 even when its Q95 is positive.

If the probability of activity is 5% or less (`pArrivalActive ≤ 0.05`), the Q95
is exactly `0`: the model is at least 95% sure nothing happens in that slot.

### 5.3 Header fields

| Field | Meaning |
|---|---|
| `forecastReferenceTime` | The `--time` you passed |
| `pipeline` | `AC` or `DC` |
| `zone` | Zone forecast |
| `quantileAlpha` | Quantile level: `0.95` |
| `mixtureInversion` | How Stage 1 and Stage 2 are combined; always `exact` here |
| `resolutionMinutes` | Slot length: `15` |
| `durationCapMins` | Maximum duration the model can output: AC `1440`, DC `240` |

### 5.4 Per-slot fields

| Field | Unit | Meaning |
|---|---|---|
| `timeSlotStart` | datetime | Start of the forecast slot (the slot lasts 15 min) |
| `stepsAhead` | slots | `1` = first slot starting at `--time` |
| `minutesAhead` | min | End of the slot, measured from `--time` (15, 30, 45, …) |
| **`Q95_arrivalCount`** | sessions | 95% upper bound on the number of sessions starting in the slot |
| **`Q95_departureCount`** | sessions | 95% upper bound on the number of sessions ending in the slot |
| **`Q95_energyKwh`** | kWh **per session** | 95% upper bound on the mean energy of sessions starting in the slot |
| **`Q95_durationMins`** | min **per session** | 95% upper bound on the mean plug-in duration of sessions starting in the slot, capped at `durationCapMins` |
| **`pArrivalActive`** | 0 to 1 | Calibrated probability that at least one session starts in the slot |
| **`pDepartureActive`** | 0 to 1 | Calibrated probability that at least one session ends in the slot |
| `conditionalLevelUsed` | 0 to 1 | Technical: the quantile level read from the Stage 2 distribution, `(0.95 − 1 + p) / p` with `p = pArrivalActive`, or 0 when `p ≤ 0.05` |
| `stage2_arrivalCount_qmag` | sessions | Q95 of arrivals **if at least one arrival happens**, ignoring how likely that is |
| `stage2_energyKwh_qmag` | kWh | Same, for energy per session |
| `stage2_durationMins_qmag` | min | Same, for duration (not capped) |

The fields in **bold** are the main outputs.

### 5.5 Reading the numbers correctly

- **Values are continuous.** `Q95_arrivalCount = 1.15` is a valid output,
  because quantiles are interpolated between trained levels. Round up if you
  need a whole number of vehicles.
- **Energy and duration are per session**, not slot totals. The energy and
  duration forecasts depend on the arrival probability, because they only exist
  when a session starts.
- **Do not add or multiply Q95 values** to get other Q95 values. The sum of
  per-slot Q95s is not the Q95 of the total over the period, and
  `Q95_arrivalCount × Q95_energyKwh` is not the Q95 of total energy.
- The `stage2_*_qmag` fields tell you how big activity would be if it happens.
  The `Q95_*` fields also account for how likely it is to happen.

---

## 6. How the model works

Most slots are empty, so each quantity is modelled in two stages:

1. **Stage 1: will anything happen?** An XGBoost classifier gives the
   probability `p` of at least one arrival (or departure) in the slot. The
   probability is then calibrated with isotonic regression
   (`calibration_*.json`).
2. **Stage 2: if something happens, how big?** An XGBoost quantile regressor,
   trained only on active slots, predicts the size distribution at 17 quantile
   levels from 0.05 to 0.99. The duration model works on a log scale.
3. **Combination.** Both stages are merged into one zero-inflated distribution,
   and its 95th percentile is returned:

   ```
   Q95 = Stage-2 quantile at level (0.95 − 1 + p) / p     if p > 0.05
       = 0                                                 otherwise
   ```

   Energy and duration use the arrival probability.

**Model inputs**, all computed from your CSV (the exact list per model is in
`feature_cols.json`):

- **Calendar:** hour, minute, day of week and month as sin/cos pairs, plus a
  weekend flag.
- **Lags:** each quantity's value 15 min, 1 h, 24 h, 48 h, 1 week and 2 weeks
  earlier.
- **Activity pattern:** slots since the last arrival, share of active slots in
  the last 4 h and 24 h, and the historical active rate at this hour of day.
- **Charging power:** the last observed `avg_power_kW`, used by the duration
  model only.

For multi-slot forecasts, the forecast for slot *k* becomes the "15 minutes
earlier" input for slot *k + 1*.

---

## 7.2 Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: Data file not found: …/data/processed/ml_features_ac_all.csv` | CSV missing, misnamed or in the wrong folder | Check the name and location in section 1 |
| `DataValidationError: No history for zone '…' in the 15 days before …` | No rows between `--time − 15 days` and `--time` | Check `--time` and the dates in the CSV |
| `DataValidationError: Configured zones [...] are absent from …` | `zone` value spelled differently | Use exactly `Employee Parking` / `PublicParking` |
| `DataValidationError: chargerType='AC' absent from …` or `has no 'chargerType' column` | `chargerType` column wrong or missing | Add it with `AC` / `DC` |
| `KeyError: "[...] not in index"` | A required column is missing | Include all 8 columns from section 3.1 |
| `ValueError: Unrecognised horizon: '…'` | Wrong `--horizon` format | Use `15min`, `4h`, `1d` |
| `ImportError: sklearn needs to be installed in order to use this module` | scikit-learn not installed | `pip install scikit-learn` |
| Error while loading a `.json` model | Different XGBoost version or damaged file | `pip install xgboost==3.3.0`; re-copy `new_models_*` |
| No error, but forecasts look wrong (all zero, flat) | Less than 14 days of history, missing slot rows, or `--time` not on :00/:15/:30/:45 | Check rules 1 to 3 in section 3.2 |
