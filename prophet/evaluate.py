# evaluate.py

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# 默认配置
# ============================================================

DEFAULT_SALES_FILE = "data/sales.csv"
DEFAULT_HOLIDAYS_FILE = "data/holidays.csv"
DEFAULT_OUTPUT_DIR = "output"


# ============================================================
# 读取数据
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

    # 补齐日期
    full_dates = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="D"
    )

    df = (
        df.set_index("date")
        .reindex(full_dates)
        .rename_axis("date")
        .reset_index()
    )

    df["sales"] = (
        df["sales"]
        .fillna(0)
    )

    return df


# ============================================================
# Holiday
# ============================================================

def load_holidays(
    holidays_file
):

    if not os.path.exists(
        holidays_file
    ):
        return None

    holidays = pd.read_csv(
        holidays_file
    )

    holidays["ds"] = pd.to_datetime(
        holidays["ds"]
    )

    return holidays


# ============================================================
# 创建 Prophet
# ============================================================

def create_model(
    holidays
):

    model = Prophet(

        yearly_seasonality=True,

        weekly_seasonality=True,

        daily_seasonality=False,

        holidays=holidays,

        changepoint_prior_scale=0.05,

        seasonality_prior_scale=10,

        holidays_prior_scale=10,

        seasonality_mode="multiplicative"
    )

    return model


# ============================================================
# WAPE
# ============================================================

def calculate_wape(
    actual,
    predicted
):

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:
        return np.nan

    return (
        np.sum(
            np.abs(
                actual - predicted
            )
        )
        / denominator
    )


# ============================================================
# 单次回测
# ============================================================

def evaluate_once(
    df,
    holidays,
    validation_days
):

    if len(df) <= validation_days:
        raise ValueError(
            "Not enough data for validation."
        )

    # --------------------------------------------------------
    # 训练集
    # --------------------------------------------------------

    train_df = df.iloc[
        :-validation_days
    ].copy()

    # --------------------------------------------------------
    # 验证集
    # --------------------------------------------------------

    validation_df = df.iloc[
        -validation_days:
    ].copy()

    # --------------------------------------------------------
    # 转换 Prophet 格式
    # --------------------------------------------------------

    train_prophet = train_df.rename(
        columns={
            "date": "ds",
            "sales": "y"
        }
    )

    train_prophet = train_prophet[
        ["ds", "y"]
    ]

    # --------------------------------------------------------
    # 创建模型
    # --------------------------------------------------------

    model = create_model(
        holidays
    )

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------

    model.fit(
        train_prophet
    )

    # --------------------------------------------------------
    # 创建未来日期
    # --------------------------------------------------------

    future = model.make_future_dataframe(
        periods=validation_days,
        freq="D"
    )

    # --------------------------------------------------------
    # 预测
    # --------------------------------------------------------

    forecast = model.predict(
        future
    )

    prediction = forecast[
        forecast["ds"].isin(
            validation_df["date"]
        )
    ][
        ["ds", "yhat"]
    ]

    # --------------------------------------------------------
    # 合并真实值
    # --------------------------------------------------------

    result = validation_df.merge(
        prediction,
        left_on="date",
        right_on="ds"
    )

    # --------------------------------------------------------
    # 防止负数
    # --------------------------------------------------------

    result["yhat"] = (
        result["yhat"]
        .clip(lower=0)
    )

    actual = result[
        "sales"
    ].values

    predicted = result[
        "yhat"
    ].values

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual,
        predicted
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    # --------------------------------------------------------
    # WAPE
    # --------------------------------------------------------

    wape = calculate_wape(
        actual,
        predicted
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "result": result
    }


# ============================================================
# 滚动回测
# ============================================================

def rolling_evaluate(
    df,
    holidays,
    validation_days,
    windows
):

    results = []

    total_days = len(df)

    for i in range(windows):

        # 当前 validation 的结束位置
        validation_end = (
            total_days
            - i * validation_days
        )

        validation_start = (
            validation_end
            - validation_days
        )

        if validation_start <= 0:
            break

        train_df = df.iloc[
            :validation_start
        ].copy()

        validation_df = df.iloc[
            validation_start:
            validation_end
        ].copy()

        print()
        print(
            "=" * 60
        )

        print(
            f"Backtest window {i + 1}"
        )

        print(
            f"Train:"
            f" {train_df['date'].min().date()}"
            f" → "
            f"{train_df['date'].max().date()}"
        )

        print(
            f"Validation:"
            f" {validation_df['date'].min().date()}"
            f" → "
            f"{validation_df['date'].max().date()}"
        )

        # ----------------------------------------------------
        # Prophet 数据
        # ----------------------------------------------------

        train_prophet = train_df.rename(
            columns={
                "date": "ds",
                "sales": "y"
            }
        )

        train_prophet = train_prophet[
            ["ds", "y"]
        ]

        # ----------------------------------------------------
        # 模型
        # ----------------------------------------------------

        model = create_model(
            holidays
        )

        model.fit(
            train_prophet
        )

        # ----------------------------------------------------
        # 预测
        # ----------------------------------------------------

        future = model.make_future_dataframe(
            periods=validation_days,
            freq="D"
        )

        forecast = model.predict(
            future
        )

        prediction = forecast[
            forecast["ds"].isin(
                validation_df["date"]
            )
        ][
            ["ds", "yhat"]
        ]

        # ----------------------------------------------------
        # 合并
        # ----------------------------------------------------

        result = validation_df.merge(
            prediction,
            left_on="date",
            right_on="ds"
        )

        result["yhat"] = (
            result["yhat"]
            .clip(lower=0)
        )

        actual = result[
            "sales"
        ].values

        predicted = result[
            "yhat"
        ].values

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        mae = mean_absolute_error(
            actual,
            predicted
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        wape = calculate_wape(
            actual,
            predicted
        )

        results.append({

            "window": i + 1,

            "train_start":
                train_df["date"].min(),

            "train_end":
                train_df["date"].max(),

            "validation_start":
                validation_df["date"].min(),

            "validation_end":
                validation_df["date"].max(),

            "MAE": mae,

            "RMSE": rmse,

            "WAPE": wape
        })

        print(
            f"MAE  = {mae:.4f}"
        )

        print(
            f"RMSE = {rmse:.4f}"
        )

        print(
            f"WAPE = {wape:.2%}"
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# 主函数
# ============================================================

def evaluate(
    sales_file,
    holidays_file,
    output_dir,
    sku_id,
    validation_days,
    windows
):

    print("=" * 60)
    print("Prophet Evaluation")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. 数据
    # --------------------------------------------------------

    df = load_sales_data(
        sales_file,
        sku_id
    )

    # --------------------------------------------------------
    # 2. Holiday
    # --------------------------------------------------------

    holidays = load_holidays(
        holidays_file
    )

    # --------------------------------------------------------
    # 3. Rolling Backtest
    # --------------------------------------------------------

    metrics = rolling_evaluate(
        df=df,
        holidays=holidays,
        validation_days=validation_days,
        windows=windows
    )

    # --------------------------------------------------------
    # 4. 平均指标
    # --------------------------------------------------------

    avg_mae = metrics[
        "MAE"
    ].mean()

    avg_rmse = metrics[
        "RMSE"
    ].mean()

    avg_wape = metrics[
        "WAPE"
    ].mean()

    print()
    print("=" * 60)
    print("Average Metrics")
    print("=" * 60)

    print(
        f"Average MAE  = {avg_mae:.4f}"
    )

    print(
        f"Average RMSE = {avg_rmse:.4f}"
    )

    print(
        f"Average WAPE = {avg_wape:.2%}"
    )

    # --------------------------------------------------------
    # 5. 保存 metrics
    # --------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    metrics_file = os.path.join(
        output_dir,
        f"{sku_id}_metrics.csv"
    )

    metrics.to_csv(
        metrics_file,
        index=False
    )

    print(
        f"\nMetrics saved to:"
        f" {metrics_file}"
    )

    # --------------------------------------------------------
    # 6. 保存总结
    # --------------------------------------------------------

    summary = pd.DataFrame([{
        "sku_id": sku_id,
        "average_mae": avg_mae,
        "average_rmse": avg_rmse,
        "average_wape": avg_wape
    }])

    summary_file = os.path.join(
        output_dir,
        f"{sku_id}_metrics_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(
        f"Summary saved to:"
        f" {summary_file}"
    )

    print("=" * 60)

    return metrics


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
        "--holidays",
        default=DEFAULT_HOLIDAYS_FILE
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
        "--validation-days",
        type=int,
        default=30
    )

    parser.add_argument(
        "--windows",
        type=int,
        default=4
    )

    args = parser.parse_args()

    evaluate(
        sales_file=args.sales,
        holidays_file=args.holidays,
        output_dir=args.output_dir,
        sku_id=args.sku,
        validation_days=args.validation_days,
        windows=args.windows
    )
