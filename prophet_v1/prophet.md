可以。下面这版我会以**前面那套完整、可实际运行的 `train.py / forecast.py / evaluate.py` 为准**，不再为了精简而改变代码逻辑。文档本身可以直接保存为 `README.md`。

# Prophet 服装商品销量预测完整实战指南

> 使用 Prophet 对服装 SKU 的历史日销量进行预测。
> 项目包含：数据准备、Prophet 原理、依赖安装、模型训练、未来销量预测、Rolling Backtest、MAE/RMSE/WAPE 评估以及项目运行方式。

---

# 1. 项目简介

Prophet 是 Meta 开源的时间序列预测算法。

它适合处理具有以下特点的业务数据：

* 明显的长期趋势
* 周期性变化
* 节假日影响
* 促销活动影响
* 时间序列数据存在部分缺失
* 需要较强的模型可解释性

对于服装商品，可以首先建立：

```text
历史 SKU 日销量
       ↓
    Prophet
       ↓
未来 30 天销量
```

本项目进一步加入：

```text
Holiday / Promotion
        ↓
Rolling Backtest
        ↓
MAE / RMSE / WAPE
```

用于判断 Prophet 是否适合实际业务。

---

# 2. Prophet 原理

Prophet 可以将时间序列表示为：

```text
y(t) = g(t) + s(t) + h(t) + ε(t)
```

其中：

| 符号     | 含义               |
| ------ | ---------------- |
| `g(t)` | Trend，长期趋势       |
| `s(t)` | Seasonality，季节性  |
| `h(t)` | Holiday，节假日/促销活动 |
| `ε(t)` | 随机误差             |

例如服装销量：

```text
销量
 ↑
 │                         /\
 │              /\        /  \
 │       /\    /  \______/    \
 │______/  \__/                 \___
 │
 └──────────────────────────────────→ 时间

       趋势 + 周期 + 活动
```

---

# 3. Trend：趋势

Trend 描述销量长期变化。

例如：

```text
新品上市
   ↓
销量快速增长
   ↓
销量稳定
   ↓
商品生命周期下降
```

Prophet 可以自动检测趋势变化点（Changepoint）。

主要参数：

```python
changepoint_prior_scale=0.05
```

该参数控制趋势变化的灵活程度。

值越大：

```text
模型更加容易发生趋势变化
```

值越小：

```text
模型趋势更加平滑
```

---

# 4. Seasonality：季节性

服装商品通常存在明显的季节性。

## 4.1 周季节性

例如：

```text
周一  低
周二  低
周三  中
周四  中
周五  高
周六  高
周日  高
```

Prophet：

```python
weekly_seasonality=True
```

---

## 4.2 年季节性

例如：

```text
春季 → 春装
夏季 → T 恤 / 短裤
秋季 → 外套
冬季 → 羽绒服
```

Prophet：

```python
yearly_seasonality=True
```

---

# 5. Holiday：节假日和促销

服装销售中促销活动非常重要。

例如：

```text
618
双11
双12
春节
国庆
品牌大促
会员日
```

促销不仅影响活动当天，还可能影响前后几天。

例如双11：

```text
          双11
           ↓
───────┬───┼───────┬────→ 时间
      -7   0       +3
```

因此 Holiday 数据可以设置：

```text
lower_window = -7
upper_window = 3
```

表示活动前 7 天到活动后 3 天都可能受到影响。

---

# 6. 服装销量预测中的一个关键问题

必须区分：

```text
Sales ≠ Demand
```

例如：

```text
真实需求 = 100
库存     = 30
实际销售 = 30
```

Prophet 看到的是：

```text
Sales = 30
```

但真实需求可能是：

```text
Demand = 100
```

因此如果商品经常缺货，仅使用销量训练 Prophet 会低估真实需求。

生产环境建议进一步增加：

```text
inventory
stockout_flag
```

例如：

```text
date        sales    inventory
2026-01-01  100      500
2026-01-02  120      450
2026-01-03   30        0
```

其中 2026-01-03 很可能是缺货导致销量下降。

---

# 7. 项目结构

```text
prophet_sales_forecast/
│
├── data/
│   ├── sales.csv
│   └── holidays.csv
│
├── models/
│
├── output/
│
├── train.py
├── forecast.py
├── evaluate.py
└── requirements.txt
```

三个核心程序：

```text
train.py
    ↓
训练 Prophet 模型

forecast.py
    ↓
预测未来销量

evaluate.py
    ↓
Rolling Backtest
```

---

# 8. Python 环境

推荐使用虚拟环境。

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

# 9. 安装依赖

创建：

```text
requirements.txt
```

内容：

```text
prophet
pandas
numpy
matplotlib
scikit-learn
joblib
```

安装：

```bash
pip install -r requirements.txt
```

也可以直接：

```bash
pip install prophet pandas numpy matplotlib scikit-learn joblib
```

---

# 10. 输入数据

## 10.1 sales.csv

最基本的数据结构：

```csv
date,sku_id,sales
2026-01-01,SKU001,12
2026-01-02,SKU001,15
2026-01-03,SKU001,18
2026-01-04,SKU001,21
2026-01-05,SKU001,16
2026-01-06,SKU001,14
2026-01-07,SKU001,20
```

可以包含多个 SKU：

```csv
date,sku_id,sales
2026-01-01,SKU001,12
2026-01-01,SKU002,8
2026-01-01,SKU003,21
2026-01-02,SKU001,15
2026-01-02,SKU002,9
2026-01-02,SKU003,25
```

---

# 11. holidays.csv

创建：

```text
data/holidays.csv
```

例如：

```csv
holiday,ds,lower_window,upper_window
618,2026-06-18,-3,2
double_11,2026-11-11,-7,3
double_12,2026-12-12,-3,1
national_day,2026-10-01,0,7
```

字段：

| 字段             | 含义      |
| -------------- | ------- |
| `holiday`      | 活动名称    |
| `ds`           | 活动日期    |
| `lower_window` | 活动前影响天数 |
| `upper_window` | 活动后影响天数 |

---

# 12. train.py

`train.py` 负责：

```text
sales.csv
    ↓
读取指定 SKU
    ↓
聚合日销量
    ↓
补齐日期
    ↓
加载 Holiday
    ↓
创建 Prophet
    ↓
训练
    ↓
保存模型
```

完整代码：

```python
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

    # 如果一天存在多条销售记录，
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
    补齐连续日期。

    注意：
    这里假设缺失日期代表销量为 0。
    如果缺失日期代表“没有数据”，
    不应该直接填充 0。
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
```

---

# 13. forecast.py

`forecast.py` 负责：

```text
加载模型
    ↓
读取历史数据
    ↓
确定最后历史日期
    ↓
生成未来日期
    ↓
预测
    ↓
截断负销量
    ↓
保存 CSV
    ↓
生成预测图
```

完整代码：

```python
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
```

---

# 14. evaluate.py

模型不能只训练后直接预测。

需要回答：

> Prophet 在历史数据上的预测效果到底怎么样？

因此采用 Rolling Backtest。

例如：

```text
Window 1

Train                    Validation
─────────────────────────┬────────────
                         30 天


Window 2

Train                              Validation
───────────────────────────────────┬──────────
                                   30 天


Window 3

Train                                        Validation
─────────────────────────────────────────────┬──────────
                                             30 天
```

这种方法比随机划分训练集/测试集更适合时间序列。

---

# 15. evaluate.py 完整代码

```python
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
```

---

# 16. 运行项目

## 16.1 训练

```bash
python train.py --sku SKU001
```

输出：

```text
models/
└── SKU001.pkl
```

---

## 16.2 评估

建议先进行评估：

```bash
python evaluate.py \
    --sku SKU001 \
    --validation-days 30 \
    --windows 4
```

输出类似：

```text
============================================================
Backtest window 1
============================================================
Train:
2026-01-01 → 2026-05-31

Validation:
2026-06-01 → 2026-06-30

MAE  = 8.2134
RMSE = 11.3421
WAPE = 18.32%

============================================================
Backtest window 2
============================================================
...
```

最终：

```text
Average MAE  = 7.84
Average RMSE = 11.02
Average WAPE = 17.43%
```

---

# 17. 预测未来 30 天

```bash
python forecast.py \
    --sku SKU001 \
    --days 30
```

生成：

```text
output/
├── SKU001_forecast.csv
├── SKU001_forecast.png
└── SKU001_components.png
```

---

# 18. 预测结果

`SKU001_forecast.csv`：

```csv
date,predicted_sales,lower_bound,upper_bound
2026-10-01,125,110,141
2026-10-02,132,116,149
2026-10-03,145,128,163
```

字段：

| 字段                | 含义   |
| ----------------- | ---- |
| `date`            | 预测日期 |
| `predicted_sales` | 预测销量 |
| `lower_bound`     | 预测下界 |
| `upper_bound`     | 预测上界 |

---

# 19. Prophet Components

运行：

```python
model.plot_components(forecast)
```

可以看到：

```text
Trend
Weekly
Yearly
Holiday
```

例如：

```text
Trend
  ↑
  │       /
  │     /
  │___/
  └────────────→


Weekly
  ↑
  │   /\      /\
  │  /  \____/  \
  └──────────────→


Yearly
  ↑
  │       /\
  │      /  \
  │_____/    \____
  └──────────────→
```

这对于分析服装商品的销量规律非常有价值。

---

# 20. 评估指标

## MAE

Mean Absolute Error：

```text
MAE = mean(|y - ŷ|)
```

含义：

```text
平均每天预测错多少件
```

例如：

```text
MAE = 8
```

表示平均误差约 8 件。

---

## RMSE

```text
RMSE = sqrt(mean((y - ŷ)^2))
```

相比 MAE，RMSE 对大的预测误差更加敏感。

---

## WAPE

```text
WAPE =
Σ|实际销量 - 预测销量|
----------------------
Σ|实际销量|
```

例如：

```text
WAPE = 18%
```

表示总体预测误差规模约占实际销量的 18%。

对于商品销量预测，可以重点观察 WAPE。

---

# 21. 为什么使用 Rolling Backtest

不能简单随机划分：

```text
❌ Random Split

Train:
1月 3月 5月 7月

Test:
2月 4月 6月
```

因为时间序列必须保证：

```text
训练数据发生在过去
测试数据发生在未来
```

正确方式：

```text
Train
───────────────────────
                        ↓
                    Validation
                    ──────────
```

Rolling Backtest：

```text
Window 1

Train ────────── Validation


Window 2

Train ───────────────── Validation


Window 3

Train ──────────────────────── Validation
```

这样可以检验模型在不同历史时间段上的稳定性。

---

# 22. 完整运行流程

推荐按照：

```text
                    sales.csv
                       │
                       ↓
                 数据预处理
                       │
                       ↓
                   train.py
                       │
                       ↓
                 SKU001.pkl
                       │
              ┌────────┴────────┐
              ↓                 ↓
        evaluate.py       forecast.py
              │                 │
              ↓                 ↓
       Rolling Backtest      未来30天
              │                 │
              ↓                 ↓
       MAE/RMSE/WAPE       forecast.csv
```

命令：

```bash
# 1. 训练
python train.py --sku SKU001

# 2. 回测
python evaluate.py \
    --sku SKU001 \
    --validation-days 30 \
    --windows 4

# 3. 预测
python forecast.py \
    --sku SKU001 \
    --days 30
```

---

# 23. 输出目录

最终：

```text
prophet_sales_forecast/
│
├── data/
│   ├── sales.csv
│   └── holidays.csv
│
├── models/
│   └── SKU001.pkl
│
├── output/
│   ├── SKU001_forecast.csv
│   ├── SKU001_forecast.png
│   ├── SKU001_components.png
│   ├── SKU001_metrics.csv
│   └── SKU001_metrics_summary.csv
│
├── train.py
├── forecast.py
├── evaluate.py
└── requirements.txt
```

---

# 24. Prophet 模型参数

当前项目使用：

```python
Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,

    holidays=holidays,

    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    holidays_prior_scale=10,

    seasonality_mode="multiplicative"
)
```

## `changepoint_prior_scale`

控制趋势变化。

```text
小 → 趋势平滑
大 → 趋势更加灵活
```

## `seasonality_prior_scale`

控制季节性影响。

```text
小 → 季节性弱
大 → 季节性强
```

## `holidays_prior_scale`

控制节假日/促销活动影响。

## `seasonality_mode`

目前使用：

```python
seasonality_mode="multiplicative"
```

适合销量规模越大、季节性波动幅度也越大的场景。

例如：

```text
低销量时期：

100 → 120

高销量时期：

500 → 600
```

变化比例比较稳定。

---

# 25. 当前项目的一个重要假设

代码中：

```python
df["sales"] = df["sales"].fillna(0)
```

意味着：

```text
缺失日期 = 当天销量为 0
```

这只有在业务数据满足以下条件时才合理：

```text
数据库每天都有完整记录
没有记录 = 当天确实没有销售
```

如果：

```text
没有记录
```

实际上意味着：

```text
数据同步失败
```

则不能填 0。

生产环境需要区分：

```text
0 销量
```

和：

```text
缺失数据
```

---

# 26. 服装行业生产环境建议

基础版本：

```text
SKU
 ↓
历史销量
 ↓
Prophet
 ↓
未来销量
```

生产环境通常还需要：

```text
SKU
│
├── 历史销量
├── 库存
├── 价格
├── 折扣
├── 促销
├── 流量
├── 广告
├── 商品生命周期
├── 品类
├── 季节
└── 渠道
```

例如：

```text
日期
SKU
销量
库存
价格
折扣
流量
广告
促销
```

这些变量可以帮助解释：

```text
为什么销量发生变化
```

---

# 27. Prophet 与 LightGBM

Prophet 更擅长：

```text
Trend
Seasonality
Holiday
```

LightGBM 更适合：

```text
价格
折扣
库存
流量
广告
商品属性
滞后特征
滚动统计特征
```

因此可以进一步构建：

```text
                 历史销量
                    │
                    ↓
                 Prophet
                    │
                    ↓
              基础销量预测
                    │
                    ↓
                 Residual
                    │
                    ↓
                 LightGBM
                    ↑
       ┌────────────┼────────────┐
       │            │            │
      价格         库存         折扣
       │            │            │
       └────────────┼────────────┘
                    ↓
                 最终预测
```

也就是：

```text
Prophet
+
LightGBM
```

但第一阶段建议先把 Prophet 单独跑通，并通过 WAPE 验证效果。

---

# 28. 多 SKU

当前代码采用：

```text
一个 SKU
    ↓
一个 Prophet 模型
```

例如：

```bash
python train.py --sku SKU001
python train.py --sku SKU002
python train.py --sku SKU003
```

得到：

```text
models/
├── SKU001.pkl
├── SKU002.pkl
└── SKU003.pkl
```

对于 SKU 数量较少的场景，这种方式简单直接。

如果 SKU 数量达到：

```text
1,000
10,000
100,000
```

则需要进一步解决：

* 批量训练
* 并行训练
* 模型参数自动调优
* SKU 分层
* 新品冷启动
* 低销量 SKU
* 缺货处理
* 模型监控
* 模型版本管理

---

# 29. 推荐的生产化演进路线

## 第一阶段

先实现：

```text
单 SKU
    ↓
Prophet
    ↓
未来30天
```

---

## 第二阶段

加入：

```text
Rolling Backtest
    ↓
WAPE
    ↓
参数调优
```

---

## 第三阶段

扩展：

```text
多 SKU
    ↓
批量训练
    ↓
批量预测
```

---

## 第四阶段

增加业务特征：

```text
价格
折扣
库存
流量
促销
```

---

## 第五阶段

尝试：

```text
Prophet
   +
LightGBM
```

---

# 30. 最终模型结构

第一版：

```text
              历史销量
                  │
                  ↓
              Prophet
                  │
       ┌──────────┼──────────┐
       ↓          ↓          ↓
     Trend     Weekly      Yearly
       │          │          │
       └──────────┼──────────┘
                  ↓
              Holidays
                  │
                  ↓
              Forecast
                  │
                  ↓
             未来30天销量
```

完整项目：

```text
                  sales.csv
                      │
                      ↓
                 Data Cleaning
                      │
                      ↓
                    Prophet
                      │
              ┌───────┴───────┐
              ↓               ↓
        Rolling Backtest    Forecast
              │               │
              ↓               ↓
       MAE/RMSE/WAPE      未来30/60/90天
```

---

# 31. 核心结论

Prophet 的核心可以理解为：

```text
销量
 =
趋势
 +
季节性
 +
节假日/促销
 +
随机误差
```

服装销量预测的第一版推荐流程：

```text
历史销量
   ↓
数据清洗
   ↓
Prophet
   ↓
Rolling Backtest
   ↓
WAPE / MAE / RMSE
   ↓
确认模型效果
   ↓
未来销量预测
```

当仅依靠时间序列信息无法达到业务要求时，再增加：

```text
价格
折扣
库存
流量
广告
促销
商品属性
```

并进一步演进到：

```text
Prophet
     +
LightGBM
```

这样可以从一个简单的时间序列预测程序，逐步演进为完整的服装商品销量预测系统。

