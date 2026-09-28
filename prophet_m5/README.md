# Prophet + M5 零售 SKU 销量预测验证项目

这是一个基于 M5 Forecasting Accuracy 数据集的 Prophet 零售 SKU 销量预测实验项目。

目标流程：

M5 原始数据 → `prepare_m5.py` → `sales.csv` / `holidays.csv` → `train.py` → `evaluate.py` / `m5_holdout_evaluate.py` → `forecast.py`

## 1. 项目结构

```text
prophet_m5_project/
├── data/
│   ├── raw/
│   │   ├── calendar.csv
│   │   ├── sales_train_validation.csv
│   │   ├── sales_train_evaluation.csv
│   │   └── sell_prices.csv
│   └── processed/
│       ├── sales.csv
│       ├── holidays.csv
│       └── smoke_test_sales.csv
├── models/
├── output/
├── prepare_m5.py
├── train.py
├── evaluate.py
├── m5_holdout_evaluate.py
├── forecast.py
├── make_smoke_test_data.py
├── download_m5.sh
├── requirements.txt
└── README.md
```

## 2. Python 环境

推荐 Python 3.9+。例如使用当前项目虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

验证依赖：

```bash
python -c "import prophet, pandas, numpy, matplotlib, plotly, sklearn, joblib; print('All dependencies OK')"
```

## 3. 获取 M5 原始数据

完整 M5 原始数据不能随项目重新分发，请从 Kaggle 官方比赛页面下载：

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

下载并解压后，把至少以下文件放到 `data/raw/`：

```text
calendar.csv
sales_train_validation.csv
sales_train_evaluation.csv
sell_prices.csv
```

### Kaggle CLI（可选）

先完成 Kaggle API 认证，然后执行：

```bash
./download_m5.sh
```

等价命令：

```bash
kaggle competitions download -c m5-forecasting-accuracy -p data/raw
unzip -o data/raw/m5-forecasting-accuracy.zip -d data/raw
```

## 4. 从 M5 生成 Prophet 输入数据

默认从 M5 中选 10 个真实 `item × store` SKU：

```bash
python prepare_m5.py
```

也可以指定数量：

```bash
python prepare_m5.py --sku-count 5
```

指定路径：

```bash
python prepare_m5.py \
  --raw-dir data/raw \
  --output-dir data/processed \
  --sku-count 10
```

生成：

```text
data/processed/sales.csv
data/processed/holidays.csv
```

`prepare_m5.py` 会把 M5 的宽表 `d_1, d_2, ...` 转成：

```csv
date,sku_id,sales
2011-01-29,FOODS_1_001_CA_1,3
...
```

## 5. 训练 Prophet

基本命令：

```bash
python train.py --sku FOODS_1_001_CA_1 \
  --sales data/processed/sales.csv \
  --holidays data/processed/holidays.csv
```

生成：

```text
models/FOODS_1_001_CA_1.pkl
```

## 6. Rolling Backtest

使用多个时间窗口评估 MAE / RMSE / WAPE：

```bash
python evaluate.py \
  --sku FOODS_1_001_CA_1 \
  --sales data/processed/sales.csv \
  --holidays data/processed/holidays.csv \
  --validation-days 28 \
  --windows 4 \
  --output-dir output
```

如果历史数据不足，脚本会自动减少可用窗口，并在没有任何有效窗口时给出明确错误，而不是产生 `KeyError: 'MAE'`。

输出：

```text
output/FOODS_1_001_CA_1_metrics.csv
output/FOODS_1_001_CA_1_metrics_summary.csv
output/FOODS_1_001_CA_1_backtest_prediction.csv
```

## 7. M5 官方风格 28 天 Holdout 验证

M5 验证脚本会以 `sales_train_validation.csv` 的最后 28 天作为 Holdout，在前面的历史数据上训练 Prophet，然后预测这 28 天并与真实值比较。

先确保已经下载 M5 原始数据，然后运行：

```bash
python m5_holdout_evaluate.py \
  --sku FOODS_1_001_CA_1 \
  --raw-dir data/raw \
  --output-dir output
```

也可指定 SKU：

```bash
python m5_holdout_evaluate.py \
  --sku HOBBIES_1_001_CA_1 \
  --raw-dir data/raw \
  --output-dir output
```

输出：

```text
output/FOODS_1_001_CA_1_m5_holdout.csv
output/FOODS_1_001_CA_1_m5_holdout_summary.csv
```

## 8. 未来预测

先执行 `train.py`，然后：

```bash
python forecast.py \
  --sku FOODS_1_001_CA_1 \
  --sales data/processed/sales.csv \
  --model-dir models \
  --output-dir output \
  --days 28
```

也可以预测 7 / 30 / 90 天：

```bash
python forecast.py --sku FOODS_1_001_CA_1 --days 7
python forecast.py --sku FOODS_1_001_CA_1 --days 30
python forecast.py --sku FOODS_1_001_CA_1 --days 90
```

## 9. 一条龙流程

真实 M5：

```bash
python prepare_m5.py --sku-count 10
python train.py --sku FOODS_1_001_CA_1
python evaluate.py --sku FOODS_1_001_CA_1 --validation-days 28 --windows 4
python m5_holdout_evaluate.py --sku FOODS_1_001_CA_1
python forecast.py --sku FOODS_1_001_CA_1 --days 28
```

## 10. 不下载 M5 也可以先做 Smoke Test

项目附带 `make_smoke_test_data.py`，生成一个小型可运行数据集，只用于验证代码链路，不是真实 M5 数据：

```bash
python make_smoke_test_data.py
```

然后：

```bash
python train.py \
  --sales data/processed/smoke_test_sales.csv \
  --holidays data/processed/holidays.csv \
  --sku SKU001

python evaluate.py \
  --sales data/processed/smoke_test_sales.csv \
  --holidays data/processed/holidays.csv \
  --sku SKU001 \
  --validation-days 14 \
  --windows 2

python forecast.py \
  --sales data/processed/smoke_test_sales.csv \
  --model-dir models \
  --output-dir output \
  --sku SKU001 \
  --days 14
```

## 11. 注意事项

### 缺失日期

当前 baseline 把缺失日期视为销量 0。这只有在业务上“没有记录 = 真正零销量”时才成立。如果是数据丢失，应改成其他缺失值处理方式。

### Stockout

M5 的销量是观测销量，不一定等于真实需求。发生缺货时：

```text
真实需求 = 100
库存 = 30
观测销量 = 30
```

模型看到的是 30，因此可能系统性低估需求。

### Prophet baseline 的边界

当前模型主要使用：

- trend
- weekly seasonality
- yearly seasonality
- holiday effects

暂时没有把价格、促销、库存、广告等变量加入模型。后续可以升级到 Prophet + LightGBM 的混合方案。

## 12. 推荐学习顺序

1. 先运行 smoke test，验证环境与脚本。
2. 下载 M5，运行 `prepare_m5.py`。
3. 选 1 个真实 SKU 跑 `train.py`。
4. 跑 `evaluate.py`，观察 Rolling Backtest。
5. 跑 `m5_holdout_evaluate.py`，与固定 28 天验证窗口对照。
6. 跑 `forecast.py`，查看未来预测和 Prophet components。
7. 再加入 price / event / SNAP / store / item 等特征。
