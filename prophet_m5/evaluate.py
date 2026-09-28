import os
import argparse
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEFAULT_SALES_FILE = "data/processed/sales.csv"
DEFAULT_HOLIDAYS_FILE = "data/processed/holidays.csv"
DEFAULT_OUTPUT_DIR = "output"


def load_sales_data(sales_file, sku_id):
    if not os.path.exists(sales_file):
        raise FileNotFoundError(f"Sales file not found: {sales_file}")
    df = pd.read_csv(sales_file)
    required = {"date", "sku_id", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in sales.csv: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Sales data contains invalid dates.")
    df = df[df["sku_id"] == sku_id].copy()
    if df.empty:
        raise ValueError(f"SKU not found: {sku_id}")
    df = df.groupby("date", as_index=False)["sales"].sum().sort_values("date")
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_dates).rename_axis("date").reset_index()
    df["sales"] = df["sales"].fillna(0)
    return df


def load_holidays(holidays_file):
    if not holidays_file or not os.path.exists(holidays_file):
        print(f"Holiday file not found: {holidays_file}; continue without holidays.")
        return None
    holidays = pd.read_csv(holidays_file)
    holidays["ds"] = pd.to_datetime(holidays["ds"], errors="coerce")
    if holidays["ds"].isna().any():
        raise ValueError("Holiday file contains invalid dates.")
    return holidays


def create_model(holidays=None):
    return Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
        seasonality_mode="multiplicative",
    )


def calculate_wape(actual, predicted):
    denominator = np.sum(np.abs(actual))
    if denominator == 0:
        return np.nan
    return np.sum(np.abs(actual - predicted)) / denominator


def evaluate_window(train_df, validation_df, holidays):
    train_prophet = train_df.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]
    model = create_model(holidays)
    model.fit(train_prophet)
    validation_days = len(validation_df)
    future = model.make_future_dataframe(periods=validation_days, freq="D")
    forecast = model.predict(future)
    prediction = forecast[forecast["ds"].isin(validation_df["date"])][["ds", "yhat"]]
    result = validation_df.merge(prediction, left_on="date", right_on="ds", how="left")
    if result["yhat"].isna().any():
        raise ValueError("Backtest contains missing predictions.")
    result["yhat"] = result["yhat"].clip(lower=0)
    actual = result["sales"].to_numpy()
    predicted = result["yhat"].to_numpy()
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    wape = calculate_wape(actual, predicted)
    return {"MAE": mae, "RMSE": rmse, "WAPE": wape, "result": result}


def rolling_evaluate(df, holidays, validation_days, windows):
    total_days = len(df)
    if validation_days <= 0 or windows <= 0:
        raise ValueError("validation_days and windows must be greater than 0.")

    minimum_training_days = 30
    max_possible_windows = (total_days - minimum_training_days) // validation_days
    if max_possible_windows <= 0:
        raise ValueError(
            "Not enough data for backtest. "
            f"Available days: {total_days}; validation_days: {validation_days}; "
            f"minimum training days: {minimum_training_days}."
        )

    actual_windows = min(windows, max_possible_windows)
    if actual_windows < windows:
        print(f"WARNING: Requested {windows} windows, but only {actual_windows} are possible. Auto-reducing.")

    results = []
    last_prediction = None

    for i in range(actual_windows):
        validation_end = total_days - i * validation_days
        validation_start = validation_end - validation_days
        if validation_start < minimum_training_days:
            continue

        train_df = df.iloc[:validation_start].copy()
        validation_df = df.iloc[validation_start:validation_end].copy()

        print("\n" + "=" * 60)
        print(f"Backtest Window {i + 1}")
        print(f"Train:      {train_df['date'].min().date()} -> {train_df['date'].max().date()} ({len(train_df)} days)")
        print(f"Validation: {validation_df['date'].min().date()} -> {validation_df['date'].max().date()} ({len(validation_df)} days)")

        try:
            evaluation = evaluate_window(train_df, validation_df, holidays)
        except Exception as exc:
            print(f"ERROR in window {i + 1}: {exc}")
            continue

        mae = evaluation["MAE"]
        rmse = evaluation["RMSE"]
        wape = evaluation["WAPE"]
        results.append(
            {
                "window": i + 1,
                "train_start": train_df["date"].min(),
                "train_end": train_df["date"].max(),
                "validation_start": validation_df["date"].min(),
                "validation_end": validation_df["date"].max(),
                "train_days": len(train_df),
                "validation_days": len(validation_df),
                "MAE": mae,
                "RMSE": rmse,
                "WAPE": wape,
            }
        )
        last_prediction = evaluation["result"]
        print(f"MAE  = {mae:.4f}")
        print(f"RMSE = {rmse:.4f}")
        print(f"WAPE = {'NaN' if np.isnan(wape) else f'{wape:.2%}'}")

    metrics = pd.DataFrame(results)
    if metrics.empty:
        raise ValueError("Backtest produced no valid results. Check data volume and Prophet errors.")
    return metrics, last_prediction


def evaluate(sales_file, holidays_file, output_dir, sku_id, validation_days, windows):
    print("=" * 60)
    print("Prophet Evaluation")
    print(f"SKU: {sku_id}")
    print("=" * 60)

    df = load_sales_data(sales_file, sku_id)
    holidays = load_holidays(holidays_file)
    metrics, last_prediction = rolling_evaluate(df, holidays, validation_days, windows)

    avg_mae = metrics["MAE"].mean()
    avg_rmse = metrics["RMSE"].mean()
    avg_wape = metrics["WAPE"].mean()

    print("\n" + "=" * 60)
    print("Average Metrics")
    print("=" * 60)
    print(f"Average MAE  = {avg_mae:.4f}")
    print(f"Average RMSE = {avg_rmse:.4f}")
    print(f"Average WAPE = {'NaN' if np.isnan(avg_wape) else f'{avg_wape:.2%}'}")

    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, f"{sku_id}_metrics.csv")
    metrics.to_csv(metrics_file, index=False)

    summary = pd.DataFrame(
        [
            {
                "sku_id": sku_id,
                "data_start": df["date"].min(),
                "data_end": df["date"].max(),
                "data_days": len(df),
                "backtest_windows": len(metrics),
                "validation_days": validation_days,
                "average_mae": avg_mae,
                "average_rmse": avg_rmse,
                "average_wape": avg_wape,
            }
        ]
    )
    summary_file = os.path.join(output_dir, f"{sku_id}_metrics_summary.csv")
    summary.to_csv(summary_file, index=False)

    if last_prediction is not None:
        prediction_file = os.path.join(output_dir, f"{sku_id}_backtest_prediction.csv")
        prediction_output = last_prediction[["date", "sales", "yhat"]].rename(
            columns={"sales": "actual_sales", "yhat": "predicted_sales"}
        )
        prediction_output.to_csv(prediction_file, index=False)

    print(f"Metrics saved to: {metrics_file}")
    print(f"Summary saved to: {summary_file}")
    print("Evaluation finished.")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Prophet sales forecasting model.")
    parser.add_argument("--sales", default=DEFAULT_SALES_FILE)
    parser.add_argument("--holidays", default=DEFAULT_HOLIDAYS_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sku", required=True)
    parser.add_argument("--validation-days", type=int, default=28)
    parser.add_argument("--windows", type=int, default=4)
    args = parser.parse_args()
    evaluate(args.sales, args.holidays, args.output_dir, args.sku, args.validation_days, args.windows)
