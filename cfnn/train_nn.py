import pandas as pd
import numpy as np
import torch
from neuralforecast.models import Informer
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import random
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

SCALER_MAP = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler
}

def train_nn_model(config: dict):
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    LOOKBACK = config.get("lookback", 30)
    HORIZON = config.get("horizon", 7)
    dataset_path = config["dataset"]
    validation_years = config.get("years", list(range(2020, 2025)))
    scaler_name = config.get("scaler", "standard").lower()
    ScalerClass = SCALER_MAP.get(scaler_name, StandardScaler)

    df = pd.read_csv(dataset_path, parse_dates=["ds"], thousands=',')
    df = df.dropna(subset=["y"]).copy()
    df = df.sort_values("ds").reset_index(drop=True)
    df["unique_id"] = "main_series"

    scaler_target = ScalerClass()
    df["y"] = scaler_target.fit_transform(df["y"].values.reshape(-1,1))

    all_preds = []

    print("Starting Rolling Walk-Forward Validation with Informer...")

    for year in validation_years:
        print(f"\n--- Validation Year: {year} ---")

        train_end_date = pd.to_datetime(f"{year}-01-01")
        train_df = df[df["ds"] < train_end_date].copy()

        test_start = pd.to_datetime(f"{year}-01-01")
        test_end = pd.to_datetime(f"{year+1}-01-01") if (year+1) <= df["ds"].dt.year.max() else df["ds"].max()
        test_df = df[(df["ds"] >= test_start) & (df["ds"] < test_end)].copy()

        if train_df.empty or test_df.empty:
            print(f"Skipping year {year}: insufficient train/test data.")
            continue

        print(f" Training on data before {train_end_date.date()} — predicting daily in {year}...")

        model = Informer(
            h=HORIZON,
            input_size=LOOKBACK,
            loss=MSE(),
            learning_rate=config.get("learning_rate", 0.001),
            scaler_type=scaler_name,
            max_steps=config.get("max_steps", 200),
            batch_size=config.get("batch_size", 64),
            random_seed=SEED,
        )

        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df=train_df, val_size=HORIZON)

        rolling_preds = []

        for i in range(len(test_df)):
            current_date = test_df.iloc[i]["ds"]
            lookback_data = df[df["ds"] < current_date].tail(LOOKBACK).copy()

            if len(lookback_data) < LOOKBACK:
                continue

            try:
                forecast = nf.predict(df=lookback_data)
                y_pred = forecast[model.__class__.__name__].iloc[0]
                y_true = test_df.iloc[i]["y"]
                rolling_preds.append({
                    "ds": current_date,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "year": year
                })
            except Exception as e:
                print(f" Prediction error on {current_date.date()}: {e}")
                continue

        if rolling_preds:
            all_preds.append(pd.DataFrame(rolling_preds))
        else:
            print(f" No predictions for year {year}.")

    if all_preds:
        all_preds_df = pd.concat(all_preds, ignore_index=True)

        all_preds_df["y_true"] = scaler_target.inverse_transform(all_preds_df["y_true"].values.reshape(-1,1)).flatten()
        all_preds_df["y_pred"] = scaler_target.inverse_transform(all_preds_df["y_pred"].values.reshape(-1,1)).flatten()

        y_true = all_preds_df["y_true"]
        y_pred = all_preds_df["y_pred"]

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        da = (np.sign(y_true) == np.sign(y_pred)).mean()

        print("\n--- Evaluation Metrics (All Years) ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MSE : {mse:.6f}")
        print(f"MAE : {mae:.4f}")
        print(f"R²  : {r2:.4f}")
        print(f"Directional Accuracy: {da:.2%}")

        os.makedirs("output/metrics", exist_ok=True)
        results_df = all_preds_df.groupby("year").apply(
            lambda df: pd.Series({
                "MSE": mean_squared_error(df["y_true"], df["y_pred"]),
                "RMSE": mean_squared_error(df["y_true"], df["y_pred"], squared=False),
                "MAE": mean_absolute_error(df["y_true"], df["y_pred"]),
                "R2": r2_score(df["y_true"], df["y_pred"]),
                "Directional_Accuracy": (np.sign(df["y_true"]) == np.sign(df["y_pred"])).mean()
            })
        ).reset_index()
        results_df.to_csv("output/metrics/metrics_by_year_nn.csv", index=False)

        os.makedirs("output/plots", exist_ok=True)
        plt.figure(figsize=(18, 6))
        plt.plot(all_preds_df["ds"], all_preds_df["y_true"], label="Actual")
        plt.plot(all_preds_df["ds"], all_preds_df["y_pred"], label="Forecast", linestyle="--")
        plt.title("Rolling Forecast (All Years) — Informer")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("output/plots/forecast_all_years_nn.png")
        plt.close()

        for year in validation_years:
            df_year = all_preds_df[all_preds_df["year"] == year]
            if not df_year.empty:
                plt.figure(figsize=(14, 5))
                plt.plot(df_year["ds"], df_year["y_true"], label="Actual")
                plt.plot(df_year["ds"], df_year["y_pred"], label="Forecast", linestyle="--")
                plt.title(f"Rolling Forecast — {year}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"output/plots/forecast_{year}_nn.png")
                plt.close()

    else:
        print(" No predictions were generated. Check your training/test splits and data continuity.")
