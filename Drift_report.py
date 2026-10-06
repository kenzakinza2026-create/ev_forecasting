"""
Drift_report.py — Monitoring DCForecaster avec Evidently : data quality,
data drift.

Compatible evidently==0.4.17 (ancienne API : evidently.report.Report,
evidently.metric_preset.*, evidently.metrics.*, evidently.pipeline.column_mapping.ColumnMapping).

Concept drift retiré de cette version (voir historique du projet) :
seuls Data Quality et Data Drift sont produits.

Les histogrammes par colonne (widget ColumnSummaryMetric) ont aussi été
retirés de la section Data Quality : sur des colonnes très déséquilibrées
(ex: avg_energy_kWh à ~97% de zéros), ce widget rend un graphique quasi
plat/vide. La section Data Quality ne garde donc que des métriques
"dataset-level" (DatasetSummaryMetric, DatasetMissingValuesMetric) plus le
bloc HTML fait main qui donne déjà les chiffres utiles en clair.

Usage:
    python Drift_report.py --csv history.csv --current-slots 96
    python Drift_report.py --csv history.csv --current-slots 96 --log-mlflow
"""

import argparse
import os
import re

import pandas as pd
import mlflow

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import (
    DatasetMissingValuesMetric,
    DatasetSummaryMetric,
)
from evidently.pipeline.column_mapping import ColumnMapping

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


NUMERICAL_COLS = [
    "arrival_count",
    "departure_count",
    "avg_energy_kWh",
    "avg_duration_mins",
]
CATEGORICAL_COLS = ["zone", "chargerType"]

OUTPUT_DIR = "Drift_reports"
DRIFT_SHARE_THRESHOLD = 0.5  # alerte si >= 50% des colonnes ont drifté

# Configurable via variable d'environnement :
#   - dans Docker (réseau du docker-compose) : PUSHGATEWAY_URL=pushgateway:9091
#   - en exécution locale hors Docker : laisser vide pour utiliser localhost:9091
PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "localhost:9091")

# ---------------------------------------------------------------------------
# CONFIG — Data quality : plages valides et catégories attendues
# ---------------------------------------------------------------------------
VALID_RANGES = {
    "arrival_count": (0, None),
    "departure_count": (0, None),
    "avg_energy_kWh": (0, None),
    "avg_duration_mins": (0, None),
}
EXPECTED_CATEGORIES = {
    "chargerType": {"DC"},
}
EXPECTED_SCHEMA = set(NUMERICAL_COLS + CATEGORICAL_COLS + ["timestamp"])


def load_data(csv_path: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    return df


def split_reference_current(df: pd.DataFrame, current_slots: int, timestamp_col: str = "timestamp"):
    """Découpe les données en reference / current sur la base d'un nombre de
    lignes (slots), et non plus d'une fenêtre calendaire en jours.

    Le DataFrame est supposé trié par ordre croissant de `timestamp_col`
    (c'est le cas après `load_data`) et à pas de temps régulier (ex: 15
    min). Les `current_slots` dernières lignes deviennent la période
    "current" (ex: 96 slots de 15 min = les dernières 24h), tout ce qui
    précède devient la "reference".
    """
    if len(df) <= current_slots:
        raise ValueError(
            f"Not enough rows ({len(df)}) to isolate {current_slots} slots as "
            "'current' while keeping a non-empty reference period. Check "
            "--current-slots or the depth of your history."
        )

    reference = df.iloc[:-current_slots].copy()
    current = df.iloc[-current_slots:].copy()

    if reference.empty:
        raise ValueError(
            "No reference data available before the current period. "
            "Check --current-slots or the depth of your history."
        )
    if current.empty:
        raise ValueError("No data in the current period.")

    return reference, current


def check_data_quality(df: pd.DataFrame) -> dict:
    issues = {
        "missing_columns": sorted(EXPECTED_SCHEMA - set(df.columns)),
        "unexpected_columns": sorted(set(df.columns) - EXPECTED_SCHEMA),
        "missing_values": {},
        "out_of_range": {},
        "unexpected_categories": {},
    }

    for col in NUMERICAL_COLS:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing:
            issues["missing_values"][col] = n_missing

        lo, hi = VALID_RANGES.get(col, (None, None))
        mask = pd.Series(False, index=df.index)
        if lo is not None:
            mask |= df[col] < lo
        if hi is not None:
            mask |= df[col] > hi
        n_out = int(mask.sum())
        if n_out:
            issues["out_of_range"][col] = n_out

    for col, allowed in EXPECTED_CATEGORIES.items():
        if col not in df.columns:
            continue
        unexpected = set(df[col].dropna().unique()) - allowed
        if unexpected:
            issues["unexpected_categories"][col] = sorted(unexpected)

    issues["n_rows"] = len(df)
    return issues


def data_quality_html_block(label: str, issues: dict) -> str:
    def fmt_dict(d):
        if not d:
            return "<li>None</li>"
        return "".join(f"<li>{k}: {v}</li>" for k, v in d.items())

    return f"""
    <div style="font-family:sans-serif;border:1px solid #ccc;padding:16px;margin-bottom:16px;">
      <h2>Data Quality — {label} ({issues['n_rows']} rows)</h2>
      <b>Missing columns (schema):</b> {issues['missing_columns'] or 'None'}<br>
      <b>Unexpected columns (schema):</b> {issues['unexpected_columns'] or 'None'}<br>
      <b>Missing values per column:</b>
      <ul>{fmt_dict(issues['missing_values'])}</ul>
      <b>Out-of-range values:</b>
      <ul>{fmt_dict(issues['out_of_range'])}</ul>
      <b>Unexpected categories:</b>
      <ul>{fmt_dict(issues['unexpected_categories'])}</ul>
    </div>
    """


def inject_html_block(html_path: str, block: str):
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "<body>" in content:
        content = content.replace("<body>", "<body>" + block, 1)
    else:
        content = block + content

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)


def build_column_mapping() -> ColumnMapping:
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = NUMERICAL_COLS
    column_mapping.categorical_features = CATEGORICAL_COLS
    return column_mapping


def build_quality_metrics() -> list:
    """Métriques Data Quality "dataset-level" uniquement (pas de widget par
    colonne) : DatasetSummaryMetric + DatasetMissingValuesMetric.

    Volontairement exclus :
      - ColumnSummaryMetric (par colonne) : rend un histogramme par colonne
        qui apparaît quasi vide/plat sur les colonnes très déséquilibrées
        (ex: avg_energy_kWh dominé par des zéros).
      - DatasetCorrelationsMetric : exclu depuis le début du projet.
    """
    return [
        DatasetSummaryMetric(),
        DatasetMissingValuesMetric(),
    ]


def run_drift_and_quality_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Report:
    column_mapping = build_column_mapping()
    columns = NUMERICAL_COLS + CATEGORICAL_COLS

    quality_metrics = build_quality_metrics()
    report = Report(metrics=quality_metrics + [DataDriftPreset()])
    report.run(
        reference_data=reference_df[columns],
        current_data=current_df[columns],
        column_mapping=column_mapping,
    )
    return report


def extract_drift_summary(report: Report) -> dict:
    result = report.as_dict()
    metrics = result["metrics"]

    drift_dataset_metric = next(
        m for m in metrics if m["metric"] == "DatasetDriftMetric"
    )
    drift_table_metric = next(
        m for m in metrics if m["metric"] == "DataDriftTable"
    )

    dataset_drift_result = drift_dataset_metric["result"]
    summary = {
        "dataset_drift": dataset_drift_result.get("dataset_drift"),
        "n_drifted_columns": dataset_drift_result.get("number_of_drifted_columns"),
        "n_total_columns": dataset_drift_result.get("number_of_columns"),
        "per_column": {},
    }

    drift_by_columns = drift_table_metric["result"].get("drift_by_columns", {})
    for col, col_result in drift_by_columns.items():
        summary["per_column"][col] = {
            "drift_score": col_result.get("drift_score"),
            "drift_detected": col_result.get("drift_detected"),
            "stattest": col_result.get("stattest_name"),
        }

    summary["drift_share"] = (
        summary["n_drifted_columns"] / summary["n_total_columns"]
        if summary["n_total_columns"] else 0.0
    )
    return summary


def print_summary(label: str, summary: dict):
    print(f"\n=== Drift Report: {label} ===")
    print(f"Drifted columns: {summary['n_drifted_columns']} / {summary['n_total_columns']}")
    print(f"Share of drifted columns: {summary['drift_share']:.1%}")
    print(f"Dataset drift detected: {summary['dataset_drift']}")
    print("--- Per-column detail ---")
    for col, info in summary["per_column"].items():
        flag = "DRIFT" if info["drift_detected"] else "OK"
        score = info["drift_score"]
        score_str = f"{score:.4f}" if score is not None else "N/A"
        stattest = info["stattest"] or "N/A"
        print(f"  {col:25s} score={score_str:>8s}  test={stattest:10s}  {flag}")

    if summary["drift_share"] >= DRIFT_SHARE_THRESHOLD:
        print(
            f"\n[ALERT] {summary['drift_share']:.1%} of columns have drifted "
            f"(threshold: {DRIFT_SHARE_THRESHOLD:.0%}). The model may be operating "
            "outside its training environment."
        )


def log_summary_to_mlflow(label: str, summary: dict, quality_issues: dict, ref_len: int, cur_len: int):
    mlflow.set_experiment("drift_monitoring_rolling_2026")
    with mlflow.start_run(run_name=f"drift_monitor_{label}"):
        mlflow.log_param("segment", label)
        mlflow.log_param("n_reference_rows", ref_len)
        mlflow.log_param("n_current_rows", cur_len)

        mlflow.log_metric("n_drifted_columns", summary["n_drifted_columns"])
        mlflow.log_metric("drift_share", summary["drift_share"])
        mlflow.log_metric("dataset_drift", int(bool(summary["dataset_drift"])))

        for col, info in summary["per_column"].items():
            if info["drift_score"] is not None:
                mlflow.log_metric(f"drift_score_{col}", info["drift_score"])
            mlflow.log_metric(f"drift_detected_{col}", int(bool(info["drift_detected"])))

        mlflow.log_metric("n_missing_values_total", sum(quality_issues["missing_values"].values()))
        mlflow.log_metric("n_out_of_range_total", sum(quality_issues["out_of_range"].values()))
        mlflow.log_metric("n_unexpected_categories", len(quality_issues["unexpected_categories"]))
        mlflow.log_metric("n_schema_issues", len(quality_issues["missing_columns"]) + len(quality_issues["unexpected_columns"]))


def push_drift_metrics_to_gateway(label: str, summary: dict, quality_issues: dict,
                                   job_name: str = "dcforecaster_drift"):
    registry = CollectorRegistry()

    Gauge("drift_share", "Share of drifted columns", registry=registry).set(summary["drift_share"])
    Gauge("dataset_drift", "Dataset-level drift detected (0/1)", registry=registry).set(
        int(bool(summary["dataset_drift"]))
    )
    Gauge("n_drifted_columns", "Number of drifted columns", registry=registry).set(
        summary["n_drifted_columns"]
    )

    for col, info in summary["per_column"].items():
        if info["drift_score"] is not None:
            safe_col = col.replace(" ", "_").replace("-", "_")
            Gauge(f"drift_score_{safe_col}", f"Drift score for column {col}", registry=registry).set(
                info["drift_score"]
            )

    Gauge("n_missing_values_total", "Total number of missing values", registry=registry).set(
        sum(quality_issues["missing_values"].values())
    )
    Gauge("n_out_of_range_total", "Total number of out-of-range values", registry=registry).set(
        sum(quality_issues["out_of_range"].values())
    )
    Gauge("n_unexpected_categories", "Number of columns with unexpected categories", registry=registry).set(
        len(quality_issues["unexpected_categories"])
    )

    try:
        push_to_gateway(PUSHGATEWAY_URL, job=f"{job_name}_{label}", registry=registry)
        print(f"[Pushgateway] Metrics pushed (job={job_name}_{label})")
    except Exception as exc:
        print(f"[WARNING] Push to Pushgateway failed: {exc}")


def run_monitoring(csv_path: str, current_slots: int, output_dir: str, log_mlflow: bool):
    df = load_data(csv_path)
    reference, current = split_reference_current(df, current_slots)

    label = "global"
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"drift_{label}.html")

    report = run_drift_and_quality_report(reference, current)
    summary = extract_drift_summary(report)
    quality_issues = check_data_quality(current)
    report.save_html(html_path)
    print_summary(label, summary)

    inject_html_block(html_path, data_quality_html_block(label, quality_issues))
    print(f"\nRapport HTML unique généré (data quality + data drift) : {html_path}")

    if log_mlflow:
        log_summary_to_mlflow(label, summary, quality_issues, len(reference), len(current))

    push_drift_metrics_to_gateway(label, summary, quality_issues)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCForecaster monitoring (Evidently): data quality, data drift — un seul fichier HTML")
    parser.add_argument("--csv", required=True, help="Path to the history CSV (features)")
    parser.add_argument("--current-slots", type=int, default=96,
                         help="Number of most recent rows (time slots) considered as the "
                              "'current' period, e.g. 96 slots of 15 min = last 24h")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                         help="Output directory for the HTML report")
    parser.add_argument("--log-mlflow", action="store_true",
                         help="Log results to MLflow")

    args = parser.parse_args()

    run_monitoring(
        csv_path=args.csv,
        current_slots=args.current_slots,
        output_dir=args.output_dir,
        log_mlflow=args.log_mlflow,
    )