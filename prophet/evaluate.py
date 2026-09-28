import os
import argparse
import numpy as np
import pandas as pd

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


DEFAULT_SALES_FILE = "data/sales.csv"
DEFAULT_HOLIDAYS_FILE = "data/holidays.csv"
DEFAULT_OUTPUT_DIR = "output"


# ============================================================
# 1. 读取销售数据
# ============================================================

def load_sales_data(sales_file, sku_id):
    """
    读取指定 SKU 的销售数据，并补齐日期。

    注意：
    这里假设缺失日期 = 当天真实销量为 0。

    如果你的业务中：
        缺失日期 = 数据缺失
    那么不要使用 fillna(0)。
    """

    if not os.path.exists(sales_file):
        raise FileNotFoundError(
            f"Sales file not found: {sales_file}"
        )

    df = pd.read_csv(sales_file)

    required_columns = {"date", "sku_id", "sales"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in sales.csv: {missing_columns}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # 检查非法日期
    invalid_dates = df["date"].isna().sum()

    if invalid_dates > 0:
        raise ValueError(
            f"Found {invalid_dates} invalid date records."
        )

    # 过滤 SKU
    df = df[df["sku_id"] == sku_id].copy()

    if df.empty:
        raise ValueError(
            f"SKU not found: {sku_id}"
        )

    # 同一天如果存在多条记录，进行聚合
    df = (
        df.groupby("date", as_index=False)["sales"]
        .sum()
    )

    df = df.sort_values("date")

    # 补齐完整日期
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

    # 假设缺失日期销量为 0
    df["sales"] = df["sales"].fillna(0)

    return df


# ============================================================
# 2. 读取节假日
# ============================================================

def load_holidays(holidays_file):

    if not holidays_file:
        return None

    if not os.path.exists(holidays_file):
        print(
            f"Holiday file not found: {holidays_file}"
        )
        print("Continue without holidays.")
        return None

    holidays = pd.read_csv(holidays_file)

    if "ds" not in holidays.columns:
        raise ValueError(
            "Holiday file must contain 'ds' column."
        )

    holidays["ds"] = pd.to_datetime(
        holidays["ds"],
        errors="coerce"
    )

    if holidays["ds"].isna().any():
        raise ValueError(
            "Holiday file contains invalid dates."
        )

    return holidays


# ============================================================
# 3. 创建 Prophet 模型
# ============================================================

def create_model(holidays=None):

    return Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,

        holidays=holidays,

        changepoint_prior_scale=0.05,

        seasonality_prior_scale=10,

        holidays_prior_scale=10,

        seasonality_mode="multiplicative"
    )


# ============================================================
# 4. WAPE
# ============================================================

def calculate_wape(actual, predicted):

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:
        return np.nan

    return (
        np.sum(
            np.abs(actual - predicted)
        )
        / denominator
    )


# ============================================================
# 5. 单个 Backtest
# ============================================================

def evaluate_window(
    train_df,
    validation_df,
    holidays
):

    # 转换成 Prophet 格式

    train_prophet = (
        train_df
        .rename(
            columns={
                "date": "ds",
                "sales": "y"
            }
        )
        [["ds", "y"]]
    )

    # 创建模型

    model = create_model(
        holidays
    )

    # 训练

    model.fit(
        train_prophet
    )

    validation_days = len(
        validation_df
    )

    # 创建未来日期

    future = (
        model.make_future_dataframe(
            periods=validation_days,
            freq="D"
        )
    )

    # 预测

    forecast = model.predict(
        future
    )

    # 只取验证区间

    prediction = (
        forecast[
            forecast["ds"].isin(
                validation_df["date"]
            )
        ]
        [["ds", "yhat"]]
    )

    # 合并真实值和预测值

    result = validation_df.merge(
        prediction,
        left_on="date",
        right_on="ds",
        how="left"
    )

    # 检查预测是否完整

    if result["yhat"].isna().any():

        missing_count = (
            result["yhat"]
            .isna()
            .sum()
        )

        raise ValueError(
            f"Missing {missing_count} prediction records."
        )

    # 销量不能为负

    result["yhat"] = (
        result["yhat"]
        .clip(lower=0)
    )

    actual = (
        result["sales"]
        .values
    )

    predicted = (
        result["yhat"]
        .values
    )

    # MAE

    mae = mean_absolute_error(
        actual,
        predicted
    )

    # RMSE

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    # WAPE

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
# 6. Rolling Backtest
# ============================================================

def rolling_evaluate(
    df,
    holidays,
    validation_days,
    windows
):

    total_days = len(df)

    print()
    print("=" * 60)
    print("Backtest Configuration")
    print("=" * 60)

    print(
        f"Total data days : {total_days}"
    )

    print(
        f"Validation days : {validation_days}"
    )

    print(
        f"Requested windows: {windows}"
    )

    # --------------------------------------------------------
    # 参数检查
    # --------------------------------------------------------

    if validation_days <= 0:
        raise ValueError(
            "validation_days must be greater than 0."
        )

    if windows <= 0:
        raise ValueError(
            "windows must be greater than 0."
        )

    # --------------------------------------------------------
    # 至少需要训练数据
    #
    # Prophet 至少需要一些历史数据。
    # 这里人为要求训练集 >= 30 天。
    # --------------------------------------------------------

    minimum_training_days = 30

    max_possible_windows = (
        total_days - minimum_training_days
    ) // validation_days

    if max_possible_windows <= 0:

        raise ValueError(
            "\nNot enough data for backtest.\n"
            f"Available days: {total_days}\n"
            f"Validation days: {validation_days}\n"
            f"Minimum training days: "
            f"{minimum_training_days}\n\n"
            "Please either:\n"
            "1. provide more historical data, or\n"
            "2. reduce --validation-days."
        )

    # --------------------------------------------------------
    # 自动调整 windows
    # --------------------------------------------------------

    actual_windows = min(
        windows,
        max_possible_windows
    )

    if actual_windows < windows:

        print()
        print(
            f"WARNING: Requested {windows} windows, "
            f"but only {actual_windows} windows "
            f"are possible."
        )

        print(
            "Automatically reducing windows."
        )

    print(
        f"Actual windows : {actual_windows}"
    )

    print("=" * 60)

    # ========================================================
    # 开始 Rolling Backtest
    # ========================================================

    results = []

    last_prediction = None

    for i in range(
        actual_windows
    ):

        # ----------------------------------------------------
        # 计算窗口
        # ----------------------------------------------------

        validation_end = (
            total_days
            - i * validation_days
        )

        validation_start = (
            validation_end
            - validation_days
        )

        # ----------------------------------------------------
        # 检查训练数据
        # ----------------------------------------------------

        if validation_start < minimum_training_days:

            print(
                f"Skip window {i + 1}: "
                f"not enough training data."
            )

            continue

        train_df = (
            df.iloc[
                :validation_start
            ]
            .copy()
        )

        validation_df = (
            df.iloc[
                validation_start:
                validation_end
            ]
            .copy()
        )

        # ----------------------------------------------------
        # 输出窗口信息
        # ----------------------------------------------------

        print()
        print("=" * 60)

        print(
            f"Backtest Window {i + 1}"
        )

        print(
            f"Train      : "
            f"{train_df['date'].min().date()} "
            f"→ "
            f"{train_df['date'].max().date()}"
        )

        print(
            f"Validation : "
            f"{validation_df['date'].min().date()} "
            f"→ "
            f"{validation_df['date'].max().date()}"
        )

        print(
            f"Train days : {len(train_df)}"
        )

        print(
            f"Validation days : "
            f"{len(validation_df)}"
        )

        # ----------------------------------------------------
        # 执行预测
        # ----------------------------------------------------

        try:

            evaluation = evaluate_window(
                train_df=train_df,
                validation_df=validation_df,
                holidays=holidays
            )

        except Exception as e:

            print()
            print(
                f"ERROR in window {i + 1}: "
                f"{e}"
            )

            continue

        # ----------------------------------------------------
        # 获取指标
        # ----------------------------------------------------

        mae = evaluation["MAE"]

        rmse = evaluation["RMSE"]

        wape = evaluation["WAPE"]

        # ----------------------------------------------------
        # 保存结果
        # ----------------------------------------------------

        results.append(
            {
                "window": i + 1,

                "train_start":
                    train_df["date"].min(),

                "train_end":
                    train_df["date"].max(),

                "validation_start":
                    validation_df["date"].min(),

                "validation_end":
                    validation_df["date"].max(),

                "train_days":
                    len(train_df),

                "validation_days":
                    len(validation_df),

                "MAE":
                    mae,

                "RMSE":
                    rmse,

                "WAPE":
                    wape
            }
        )

        last_prediction = (
            evaluation["result"]
        )

        # ----------------------------------------------------
        # 输出指标
        # ----------------------------------------------------

        print()

        print(
            f"MAE  = {mae:.4f}"
        )

        print(
            f"RMSE = {rmse:.4f}"
        )

        if np.isnan(wape):

            print(
                "WAPE = NaN"
            )

        else:

            print(
                f"WAPE = {wape:.2%}"
            )

    # ========================================================
    # 创建 DataFrame
    # ========================================================

    metrics = pd.DataFrame(
        results
    )

    # ========================================================
    # 最终安全检查
    # ========================================================

    if metrics.empty:

        raise ValueError(
            "\nBacktest produced no valid results.\n\n"
            f"Total data days: {total_days}\n"
            f"Validation days: {validation_days}\n"
            f"Requested windows: {windows}\n\n"
            "Please check:\n"
            "- SKU has enough historical data\n"
            "- validation_days is not too large\n"
            "- Prophet training did not fail\n"
        )

    # 检查关键列

    required_metric_columns = {
        "MAE",
        "RMSE",
        "WAPE"
    }

    missing_columns = (
        required_metric_columns
        - set(metrics.columns)
    )

    if missing_columns:

        raise ValueError(
            "Backtest result is missing columns: "
            f"{missing_columns}"
        )

    return (
        metrics,
        last_prediction
    )


# ============================================================
# 7. 主评估函数
# ============================================================

def evaluate(
    sales_file,
    holidays_file,
    output_dir,
    sku_id,
    validation_days,
    windows
):

    print()
    print("=" * 60)
    print("Prophet Evaluation")
    print("=" * 60)

    print(
        f"SKU: {sku_id}"
    )

    # --------------------------------------------------------
    # 读取数据
    # --------------------------------------------------------

    df = load_sales_data(
        sales_file,
        sku_id
    )

    print(
        f"Data start: "
        f"{df['date'].min().date()}"
    )

    print(
        f"Data end: "
        f"{df['date'].max().date()}"
    )

    print(
        f"Total days: {len(df)}"
    )

    print(
        f"Total sales: "
        f"{df['sales'].sum():.2f}"
    )

    # --------------------------------------------------------
    # Holiday
    # --------------------------------------------------------

    holidays = load_holidays(
        holidays_file
    )

    if holidays is not None:

        print(
            f"Holidays: "
            f"{len(holidays)}"
        )

    else:

        print(
            "Holidays: None"
        )

    # --------------------------------------------------------
    # Rolling Backtest
    # --------------------------------------------------------

    metrics, last_prediction = (
        rolling_evaluate(
            df=df,
            holidays=holidays,
            validation_days=validation_days,
            windows=windows
        )
    )

    # ========================================================
    # 平均指标
    # ========================================================

    avg_mae = (
        metrics["MAE"]
        .mean()
    )

    avg_rmse = (
        metrics["RMSE"]
        .mean()
    )

    avg_wape = (
        metrics["WAPE"]
        .mean()
    )

    # ========================================================
    # 输出平均指标
    # ========================================================

    print()
    print("=" * 60)
    print("Average Metrics")
    print("=" * 60)

    print(
        f"Average MAE  = "
        f"{avg_mae:.4f}"
    )

    print(
        f"Average RMSE = "
        f"{avg_rmse:.4f}"
    )

    if np.isnan(avg_wape):

        print(
            "Average WAPE = NaN"
        )

    else:

        print(
            f"Average WAPE = "
            f"{avg_wape:.2%}"
        )

    # ========================================================
    # 保存结果
    # ========================================================

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 每个 Backtest 窗口
    # --------------------------------------------------------

    metrics_file = os.path.join(
        output_dir,
        f"{sku_id}_metrics.csv"
    )

    metrics.to_csv(
        metrics_file,
        index=False
    )

    print()
    print(
        f"Metrics saved to: "
        f"{metrics_file}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "sku_id": sku_id,

                "data_start":
                    df["date"].min(),

                "data_end":
                    df["date"].max(),

                "data_days":
                    len(df),

                "backtest_windows":
                    len(metrics),

                "validation_days":
                    validation_days,

                "average_mae":
                    avg_mae,

                "average_rmse":
                    avg_rmse,

                "average_wape":
                    avg_wape
            }
        ]
    )

    summary_file = os.path.join(
        output_dir,
        f"{sku_id}_metrics_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(
        f"Summary saved to: "
        f"{summary_file}"
    )

    # --------------------------------------------------------
    # 保存最后一个预测窗口
    # --------------------------------------------------------

    if last_prediction is not None:

        prediction_file = os.path.join(
            output_dir,
            f"{sku_id}_backtest_prediction.csv"
        )

        prediction_output = (
            last_prediction[
                [
                    "date",
                    "sales",
                    "yhat"
                ]
            ]
            .rename(
                columns={
                    "sales":
                        "actual_sales",

                    "yhat":
                        "predicted_sales"
                }
            )
        )

        prediction_output.to_csv(
            prediction_file,
            index=False
        )

        print(
            f"Backtest prediction saved to: "
            f"{prediction_file}"
        )

    print()
    print("=" * 60)
    print("Evaluation finished.")
    print("=" * 60)

    return metrics


# ============================================================
# 8. Command Line
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "Evaluate Prophet sales forecasting model."
    )

    parser.add_argument(
        "--sales",
        default=DEFAULT_SALES_FILE,
        help="Sales CSV file."
    )

    parser.add_argument(
        "--holidays",
        default=DEFAULT_HOLIDAYS_FILE,
        help="Holiday CSV file."
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory."
    )

    parser.add_argument(
        "--sku",
        required=True,
        help="SKU ID."
    )

    parser.add_argument(
        "--validation-days",
        type=int,
        default=30,
        help="Number of validation days."
    )

    parser.add_argument(
        "--windows",
        type=int,
        default=4,
        help="Number of rolling backtest windows."
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
