import os
import argparse
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_wape(actual, predicted):
    denominator = np.sum(np.abs(actual))
    if denominator == 0:
        return np.nan
    return np.sum(np.abs(actual - predicted)) / denominator


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


def load_m5(raw_dir):
    sales_path = os.path.join(raw_dir, "sales_train_validation.csv")
    calendar_path = os.path.join(raw_dir, "calendar.csv")
    if not os.path.exists(sales_path):
        raise FileNotFoundError(f"Not found: {sales_path}")
    if not os.path.exists(calendar_path):
        raise FileNotFoundError(f"Not found: {calendar_path}")
    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    calendar["date"] = pd.to_datetime(calendar["date"])
    date_columns = [c for c in sales.columns if c.startswith("d_")]
    return sales, calendar, date_columns


def make_holidays(calendar, cutoff_date=None):
    rows = []
    for _, row in calendar.iterrows():
        for event_col in ("event_name_1", "event_name_2"):
            if pd.notna(row.get(event_col)):
                ds = row["date"]
                if cutoff_date is None or ds <= cutoff_date:
                    rows.append({"holiday": row[event_col], "ds": ds, "lower_window": 0, "upper_window": 0})
    return pd.DataFrame(rows, columns=["holiday", "ds", "lower_window", "upper_window"])


def find_sku(sales, sku_id):
    row = sales[sales["id"] == sku_id]
    if row.empty:
        raise ValueError(f"SKU not found in M5 validation file: {sku_id}")
    return row.iloc[0]


def evaluate_sku(raw_dir, sku_id, output_dir):
    sales, calendar, date_columns = load_m5(raw_dir)
    row = find_sku(sales, sku_id)
    if len(date_columns) < 60:
        raise ValueError("Too few d_xxx columns for a 28-day holdout.")

    holdout_days = 28
    train_columns = date_columns[:-holdout_days]
    holdout_columns = date_columns[-holdout_days:]

    # Map M5 d_xxx to actual dates.
    d_to_date = dict(zip(calendar["d"], calendar["date"]))
    train = pd.DataFrame(
        {
            "ds": [d_to_date[d] for d in train_columns],
            "y": [row[d] for d in train_columns],
        }
    )
    holdout = pd.DataFrame(
        {
            "date": [d_to_date[d] for d in holdout_columns],
            "actual_sales": [row[d] for d in holdout_columns],
        }
    )

    holidays = make_holidays(calendar, cutoff_date=train["ds"].max())
    model = create_model(holidays if not holidays.empty else None)
    print(f"Training {sku_id} with {len(train)} days...")
    model.fit(train)

    future = model.make_future_dataframe(periods=holdout_days, freq="D")
    pred = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    pred = pred[pred["ds"].isin(holdout["date"])].copy()
    pred["yhat"] = pred["yhat"].clip(lower=0)
    pred["yhat_lower"] = pred["yhat_lower"].clip(lower=0)
    pred["yhat_upper"] = pred["yhat_upper"].clip(lower=0)

    result = holdout.merge(pred, left_on="date", right_on="ds", how="left").drop(columns=["ds"])
    if result["yhat"].isna().any():
        raise ValueError("Holdout predictions are incomplete.")

    actual = result["actual_sales"].to_numpy()
    predicted = result["yhat"].to_numpy()
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    wape = calculate_wape(actual, predicted)

    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"{sku_id}_m5_holdout.csv")
    result.to_csv(result_file, index=False)

    summary = pd.DataFrame([
        {
            "sku_id": sku_id,
            "train_start": train["ds"].min(),
            "train_end": train["ds"].max(),
            "holdout_start": holdout["date"].min(),
            "holdout_end": holdout["date"].max(),
            "holdout_days": holdout_days,
            "MAE": mae,
            "RMSE": rmse,
            "WAPE": wape,
        }
    ])
    summary_file = os.path.join(output_dir, f"{sku_id}_m5_holdout_summary.csv")
    summary.to_csv(summary_file, index=False)

    print("=" * 60)
    print(f"M5 Holdout: {sku_id}")
    print(f"Train:   {train['ds'].min().date()} -> {train['ds'].max().date()}")
    print(f"Holdout: {holdout['date'].min().date()} -> {holdout['date'].max().date()}")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"WAPE = {'NaN' if np.isnan(wape) else f'{wape:.2%}'}")
    print(f"Saved: {result_file}")
    print(f"Saved: {summary_file}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Prophet on the final 28-day M5 validation holdout.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--sku", required=True)
    args = parser.parse_args()
    evaluate_sku(args.raw_dir, args.sku, args.output_dir)
