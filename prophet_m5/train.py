import os
import argparse
import joblib
import pandas as pd
from prophet import Prophet

DEFAULT_SALES_FILE = "data/processed/sales.csv"
DEFAULT_HOLIDAYS_FILE = "data/processed/holidays.csv"
DEFAULT_MODEL_DIR = "models"


def load_holidays(holidays_file):
    if not os.path.exists(holidays_file):
        print(f"Holiday file not found: {holidays_file}")
        return None
    holidays = pd.read_csv(holidays_file)
    holidays["ds"] = pd.to_datetime(holidays["ds"])
    return holidays


def load_sales_data(sales_file, sku_id):
    if not os.path.exists(sales_file):
        raise FileNotFoundError(f"Sales file not found: {sales_file}")
    df = pd.read_csv(sales_file)
    required = {"date", "sku_id", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Sales data contains invalid dates.")
    df = df[df["sku_id"] == sku_id].copy()
    if df.empty:
        raise ValueError(f"SKU not found: {sku_id}")
    df = df.groupby("date", as_index=False)["sales"].sum().sort_values("date")
    return df


def fill_missing_dates(df):
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = (
        df.set_index("date")
        .reindex(full_dates)
        .rename_axis("date")
        .reset_index()
    )
    df["sales"] = df["sales"].fillna(0)
    return df


def prepare_prophet_data(df):
    return df.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]


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


def train(sales_file, holidays_file, sku_id, model_dir):
    print("=" * 60)
    print("Prophet Training")
    print("=" * 60)
    print(f"SKU: {sku_id}")

    df = fill_missing_dates(load_sales_data(sales_file, sku_id))
    print(f"Training days: {len(df)}")
    prophet_df = prepare_prophet_data(df)
    holidays = load_holidays(holidays_file)
    model = create_model(holidays)

    print("Training Prophet...")
    model.fit(prophet_df)
    print("Training finished.")

    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, f"{sku_id}.pkl")
    joblib.dump(model, model_file)
    print(f"Model saved to: {model_file}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", default=DEFAULT_SALES_FILE)
    parser.add_argument("--holidays", default=DEFAULT_HOLIDAYS_FILE)
    parser.add_argument("--sku", required=True)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    train(args.sales, args.holidays, args.sku, args.model_dir)
