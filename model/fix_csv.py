import pandas as pd

path = "ev_forecaster/data/processed/ml_features_dc_train_val.csv"
df = pd.read_csv(path)

h = df["avg_duration_mins"] / 60
df["avg_power_kW"] = (df["avg_energy_kWh"] / h).where(h > 0, 0.0)

df.to_csv(path, index=False)
print(df.columns.tolist())
print(df["timestamp"].min(), "->", df["timestamp"].max(), "|", len(df), "lignes")