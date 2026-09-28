import os
import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_SALES_FILE = "data/processed/sales.csv"
DEFAULT_MODEL_DIR = "models"
DEFAULT_OUTPUT_DIR = "output"


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
    return df.groupby("date", as_index=False)["sales"].sum().sort_values("date")


def forecast(sales_file, model_dir, output_dir, sku_id, days):
    if days <= 0:
        raise ValueError("days must be greater than 0.")

    print("=" * 60)
    print("Prophet Forecast")
    print("=" * 60)

    model_file = os.path.join(model_dir, f"{sku_id}.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}")

    print(f"Loading model: {model_file}")
    model = joblib.load(model_file)
    df = load_sales_data(sales_file, sku_id)
    last_date = df["date"].max()
    print(f"Last historical date: {last_date.date()}")

    future = model.make_future_dataframe(periods=days, freq="D")
    forecast_df = model.predict(future)
    future_forecast = forecast_df[forecast_df["ds"] > last_date].copy()
    future_forecast["yhat"] = future_forecast["yhat"].clip(lower=0)
    future_forecast["yhat_lower"] = future_forecast["yhat_lower"].clip(lower=0)
    future_forecast["yhat_upper"] = future_forecast["yhat_upper"].clip(lower=0)

    result = future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
        columns={
            "ds": "date",
            "yhat": "predicted_sales",
            "yhat_lower": "lower_bound",
            "yhat_upper": "upper_bound",
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{sku_id}_forecast.csv")
    result.to_csv(output_file, index=False)
    print(f"Forecast saved to: {output_file}")
    print(result.to_string(index=False))

    model.plot(forecast_df)
    plt.title(f"Sales Forecast - {sku_id}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    figure_file = os.path.join(output_dir, f"{sku_id}_forecast.png")
    plt.savefig(figure_file, dpi=150)
    plt.close()

    model.plot_components(forecast_df)
    plt.tight_layout()
    component_file = os.path.join(output_dir, f"{sku_id}_components.png")
    plt.savefig(component_file, dpi=150)
    plt.close()

    print(f"Forecast figure saved to: {figure_file}")
    print(f"Components saved to: {component_file}")
    print("=" * 60)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast future sales with Prophet.")
    parser.add_argument("--sales", default=DEFAULT_SALES_FILE)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sku", required=True)
    parser.add_argument("--days", type=int, default=28)
    args = parser.parse_args()
    forecast(args.sales, args.model_dir, args.output_dir, args.sku, args.days)
