"""
dc_forecast.py — DC entry point for forecasting and evaluation.

Logic lives in forecast.py. Kept so existing commands keep working:
    python pipeline/dc_forecast.py --eval-only
    python pipeline/dc_forecast.py --time "2024-11-15T08:00" --horizon 4h
"""

from forecast import (                                  # noqa: F401  (re-exported)
    Bundle, build_feature_row, build_parser, evaluate_1step, evaluate_multistep,
    generate_forecast, load_models, parse_horizon, predict_frame, predict_step, run_cli,
)


def main() -> None:
    run_cli(build_parser(default_pipeline="dc").parse_args())


if __name__ == "__main__":
    main()
