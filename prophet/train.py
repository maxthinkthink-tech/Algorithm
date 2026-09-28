# train.py

import os
import argparse
import joblib
import pandas as pd

from prophet import Prophet


# ============================================================
# 配置
# ============================================================

DEFAULT_SALES_FILE = "data/sales.csv"
DEFAULT_HOLIDAYS_FILE = "data/holidays.csv"
DEFAULT_MODEL_DIR = "models"


# ============================================================
# 读取节假日
# ============================================================

def load_holidays(holidays_file):
    """
    读取节假日 / 促销活动配置
    """

    if not os.path.exists(holidays_file):
        print(
            f"Holiday file not found: {holidays_file}"
        )

        return None

    holidays = pd.read_csv(
        holidays_file
    )

    holidays["ds"] = pd.to_datetime(
        holidays["ds"]
    )

    return holidays


# ============================================================
# 读取销售数据
# ============================================================

def load_sales_data(
    sales_file,
    sku_id
):
    """
    读取指定 SKU 的销售数据
    """

    df = pd.read_csv(
        sales_file
    )

    # 日期
    df["date"] = pd.to_datetime(
        df["date"]
    )

    # SKU
    df = df[
        df["sku_id"] == sku_id
    ].copy()

    if df.empty:
        raise ValueError(
            f"SKU not found: {sku_id}"
        )

    # 如果一天存在多条销售记录
    # 先按照日期聚合
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
# 补齐日期
# ============================================================

def fill_missing_dates(df):
    """
    补齐连续日期

    注意：
    这里假设缺失日期代表销量为 0。
    如果你的缺失日期代表“没有数据”，
    不应该直接 fillna(0)。
    """

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
# 转换成 Prophet 数据格式
# ============================================================

def prepare_prophet_data(df):

    prophet_df = df.rename(
        columns={
            "date": "ds",
            "sales": "y"
        }
    )

    prophet_df = prophet_df[
        ["ds", "y"]
    ]

    return prophet_df


# ============================================================
# 创建 Prophet 模型
# ============================================================

def create_model(holidays=None):

    model = Prophet(

        # 年季节性
        yearly_seasonality=True,

        # 周季节性
        weekly_seasonality=True,

        # 日季节性
        daily_seasonality=False,

        # 节假日
        holidays=holidays,

        # 趋势变化程度
        changepoint_prior_scale=0.05,

        # 季节性程度
        seasonality_prior_scale=10,

        # 节假日影响程度
        holidays_prior_scale=10,

        # 乘法季节性
        seasonality_mode="multiplicative"
    )

    return model


# ============================================================
# 主训练函数
# ============================================================

def train(
    sales_file,
    holidays_file,
    sku_id,
    model_dir
):

    print("=" * 60)
    print("Prophet Training")
    print("=" * 60)

    print(
        f"SKU: {sku_id}"
    )

    # --------------------------------------------------------
    # 1. 读取销售数据
    # --------------------------------------------------------

    df = load_sales_data(
        sales_file,
        sku_id
    )

    print(
        f"Raw records: {len(df)}"
    )

    # --------------------------------------------------------
    # 2. 补齐日期
    # --------------------------------------------------------

    df = fill_missing_dates(
        df
    )

    print(
        f"Records after filling dates: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # 3. Prophet 数据
    # --------------------------------------------------------

    prophet_df = prepare_prophet_data(
        df
    )

    # --------------------------------------------------------
    # 4. Holiday
    # --------------------------------------------------------

    holidays = load_holidays(
        holidays_file
    )

    # --------------------------------------------------------
    # 5. 创建模型
    # --------------------------------------------------------

    model = create_model(
        holidays
    )

    # --------------------------------------------------------
    # 6. 训练
    # --------------------------------------------------------

    print(
        "Training Prophet..."
    )

    model.fit(
        prophet_df
    )

    print(
        "Training finished."
    )

    # --------------------------------------------------------
    # 7. 保存模型
    # --------------------------------------------------------

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    model_file = os.path.join(
        model_dir,
        f"{sku_id}.pkl"
    )

    joblib.dump(
        model,
        model_file
    )

    print(
        f"Model saved to: {model_file}"
    )

    print("=" * 60)


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
        "--sku",
        required=True
    )

    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR
    )

    args = parser.parse_args()

    train(
        sales_file=args.sales,
        holidays_file=args.holidays,
        sku_id=args.sku,
        model_dir=args.model_dir
    )
