# forecast.py

import os
import argparse
import joblib

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 默认配置
# ============================================================

DEFAULT_SALES_FILE = "data/sales.csv"
DEFAULT_MODEL_DIR = "models"
DEFAULT_OUTPUT_DIR = "output"


# ============================================================
# 加载销售数据
# ============================================================

def load_sales_data(
    sales_file,
    sku_id
):

    df = pd.read_csv(
        sales_file
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df[
        df["sku_id"] == sku_id
    ].copy()

    if df.empty:
        raise ValueError(
            f"SKU not found: {sku_id}"
        )

    df = (
        df.groupby(
            "date",
            as_index=False
        )["sales"]
        .sum()
    )

    df = df.sort_values(
        "date"
    )

    return df


# ============================================================
# 预测
# ============================================================

def forecast(
    sales_file,
    model_dir,
    output_dir,
    sku_id,
    days
):

    print("=" * 60)
    print("Prophet Forecast")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. 加载模型
    # --------------------------------------------------------

    model_file = os.path.join(
        model_dir,
        f"{sku_id}.pkl"
    )

    if not os.path.exists(
        model_file
    ):
        raise FileNotFoundError(
            f"Model not found: {model_file}"
        )

    print(
        f"Loading model: {model_file}"
    )

    model = joblib.load(
        model_file
    )

    # --------------------------------------------------------
    # 2. 加载历史数据
    # --------------------------------------------------------

    df = load_sales_data(
        sales_file,
        sku_id
    )

    last_date = df["date"].max()

    print(
        f"Last historical date: "
        f"{last_date.date()}"
    )

    # --------------------------------------------------------
    # 3. 创建未来日期
    # --------------------------------------------------------

    future = model.make_future_dataframe(
        periods=days,
        freq="D"
    )

    # --------------------------------------------------------
    # 4. 预测
    # --------------------------------------------------------

    forecast_df = model.predict(
        future
    )

    # --------------------------------------------------------
    # 5. 只保留未来数据
    # --------------------------------------------------------

    future_forecast = forecast_df[
        forecast_df["ds"] > last_date
    ].copy()

    # --------------------------------------------------------
    # 6. 销量不能为负
    # --------------------------------------------------------

    future_forecast["yhat"] = (
        future_forecast["yhat"]
        .clip(lower=0)
    )

    future_forecast["yhat_lower"] = (
        future_forecast["yhat_lower"]
        .clip(lower=0)
    )

    future_forecast["yhat_upper"] = (
        future_forecast["yhat_upper"]
        .clip(lower=0)
    )

    # --------------------------------------------------------
    # 7. 输出结果
    # --------------------------------------------------------

    result = future_forecast[
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper"
        ]
    ].copy()

    result = result.rename(
        columns={
            "ds": "date",
            "yhat": "predicted_sales",
            "yhat_lower": "lower_bound",
            "yhat_upper": "upper_bound"
        }
    )

    # --------------------------------------------------------
    # 8. 保存 CSV
    # --------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        output_dir,
        f"{sku_id}_forecast.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nForecast saved to:"
        f" {output_file}"
    )

    # --------------------------------------------------------
    # 9. 打印预测结果
    # --------------------------------------------------------

    print("\nFuture forecast:")
    print(
        result.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 10. 预测图
    # --------------------------------------------------------

    fig = model.plot(
        forecast_df
    )

    plt.title(
        f"Sales Forecast - {sku_id}"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Sales"
    )

    plt.tight_layout()

    figure_file = os.path.join(
        output_dir,
        f"{sku_id}_forecast.png"
    )

    plt.savefig(
        figure_file,
        dpi=150
    )

    plt.show()

    print(
        f"Forecast figure saved to:"
        f" {figure_file}"
    )

    # --------------------------------------------------------
    # 11. Prophet Components
    # --------------------------------------------------------

    model.plot_components(
        forecast_df
    )

    plt.tight_layout()

    component_file = os.path.join(
        output_dir,
        f"{sku_id}_components.png"
    )

    plt.savefig(
        component_file,
        dpi=150
    )

    plt.show()

    print(
        f"Components saved to:"
        f" {component_file}"
    )

    print("=" * 60)

    return result


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sales",
        default=DEFAULT_SALES_FILE
    )

    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR
    )

    parser.add_argument(
        "--sku",
        required=True
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30
    )

    args = parser.parse_args()

    forecast(
        sales_file=args.sales,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        sku_id=args.sku,
        days=args.days
    )
