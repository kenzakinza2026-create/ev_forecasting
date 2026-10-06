"""
Script CLI: lance une prédiction DCForecaster à partir d'un CSV local,
sans passer par l'API. Utile pour des tests rapides ou du batch.

Usage:
    python predict.py --csv history.csv --horizon 4
"""

import argparse

import pandas as pd

from predictor import DCForecaster


def main():
    parser = argparse.ArgumentParser(description="Prédiction DCForecaster en ligne de commande")
    parser.add_argument("--csv", required=True, help="Chemin vers le CSV d'historique")
    parser.add_argument("--horizon", type=int, default=4, help="Nombre de créneaux de 15 min à prévoir")
    args = parser.parse_args()

    history = pd.read_csv(args.csv, parse_dates=[0], index_col=0)
    reference_time = history.index.max().ceil("15min")

    forecaster = DCForecaster()
    result = forecaster.forecast(
        reference_time=reference_time,
        history=history,
        horizon_slots=args.horizon,
    )

    for slot in result["forecastData"]:
        print(
            f"{slot['timeSlotStart']}  "
            f"arrivals={slot['expectedArrivalCount']:.3f}  "
            f"energy={slot['expectedEnergyKwh']:.1f} kWh  "
            f"duration={slot['expectedDurationMinutes']:.1f} min  "
            f"p_arrival={slot['_meta']['pArrivalActive']:.3f}"
        )


if __name__ == "__main__":
    main()