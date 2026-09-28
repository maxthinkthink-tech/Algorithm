可以。按照我们目前这套 `train.py`、`evaluate.py`、`forecast.py`，三个脚本的命令行参数可以整理成下面这份使用手册。

## 1. `train.py` —— 训练模型

### 基本用法

```bash
python train.py --sku SKU001
```

这会使用默认配置：

```text
销售数据：data/sales.csv
节假日：data/holidays.csv
模型目录：models/
```

最终生成：

```text
models/SKU001.pkl
```

### 支持的参数

| 参数            | 默认值                 | 说明     |
| ------------- | ------------------- | ------ |
| `--sales`     | `data/sales.csv`    | 销售数据   |
| `--holidays`  | `data/holidays.csv` | 节假日数据  |
| `--sku`       | 必填                  | SKU ID |
| `--model-dir` | `models`            | 模型保存目录 |

### 完整示例

```bash
python train.py \
    --sales data/sales.csv \
    --holidays data/holidays.csv \
    --sku SKU001 \
    --model-dir models
```

如果你的数据文件不同：

```bash
python train.py \
    --sales data/clothing_sales.csv \
    --holidays data/holidays.csv \
    --sku NIKE_001 \
    --model-dir models
```

---

# 2. `evaluate.py` —— 模型评估

`evaluate.py` 用于做 **Rolling Backtest**，也就是：

```text
历史数据
    ↓
训练
    ↓
预测未来 N 天
    ↓
和真实销量比较
    ↓
MAE / RMSE / WAPE
    ↓
移动到更早的数据
    ↓
再次训练和预测
```

### 基本用法

```bash
python evaluate.py --sku SKU001
```

默认：

```text
validation-days = 30
windows = 4
```

也就是：

```text
每次预测 30 天
进行最多 4 个 Backtest 窗口
```

### 支持的参数

| 参数                  | 默认值                 | 说明            |
| ------------------- | ------------------- | ------------- |
| `--sales`           | `data/sales.csv`    | 销售数据          |
| `--holidays`        | `data/holidays.csv` | 节假日数据         |
| `--output-dir`      | `output`            | 评估结果目录        |
| `--sku`             | 必填                  | SKU ID        |
| `--validation-days` | `30`                | 每个验证窗口的天数     |
| `--windows`         | `4`                 | Backtest 窗口数量 |

### 常用命令

#### 30 天 × 4 个窗口

```bash
python evaluate.py \
    --sku SKU001 \
    --validation-days 30 \
    --windows 4
```

#### 14 天 × 4 个窗口

如果历史数据比较短：

```bash
python evaluate.py \
    --sku SKU001 \
    --validation-days 14 \
    --windows 4
```

#### 7 天 × 8 个窗口

如果想观察短期预测能力：

```bash
python evaluate.py \
    --sku SKU001 \
    --validation-days 7 \
    --windows 8
```

#### 指定数据文件

```bash
python evaluate.py \
    --sales data/sales.csv \
    --holidays data/holidays.csv \
    --sku SKU001 \
    --validation-days 30 \
    --windows 4 \
    --output-dir output
```

### 输出

例如：

```text
output/
├── SKU001_metrics.csv
├── SKU001_metrics_summary.csv
└── SKU001_backtest_prediction.csv
```

其中：

```text
SKU001_metrics.csv
```

保存每个 Backtest 窗口：

```text
window
train_start
train_end
validation_start
validation_end
train_days
validation_days
MAE
RMSE
WAPE
```

---

# 3. `forecast.py` —— 未来销量预测

这个脚本使用已经训练好的：

```text
models/SKU001.pkl
```

预测未来 N 天。

### 基本用法

```bash
python forecast.py --sku SKU001
```

默认预测：

```text
未来 30 天
```

### 支持的参数

| 参数             | 默认值              | 说明      |
| -------------- | ---------------- | ------- |
| `--sales`      | `data/sales.csv` | 历史销售数据  |
| `--model-dir`  | `models`         | 模型所在目录  |
| `--output-dir` | `output`         | 预测结果目录  |
| `--sku`        | 必填               | SKU ID  |
| `--days`       | `30`             | 预测未来多少天 |

### 预测未来 30 天

```bash
python forecast.py \
    --sku SKU001 \
    --days 30
```

### 预测未来 7 天

```bash
python forecast.py \
    --sku SKU001 \
    --days 7
```

### 预测未来 90 天

```bash
python forecast.py \
    --sku SKU001 \
    --days 90
```

### 指定模型和输出目录

```bash
python forecast.py \
    --sales data/sales.csv \
    --model-dir models \
    --output-dir output \
    --sku SKU001 \
    --days 30
```

---

# 4. 三个脚本完整工作流程

实际使用的时候，推荐按照：

```text
sales.csv
    │
    ▼
┌──────────────┐
│   train.py   │
│    训练模型   │
└──────┬───────┘
       │
       ▼
models/SKU001.pkl
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ evaluate.py  │      │ forecast.py  │
│    模型评估   │      │   未来预测    │
└──────┬───────┘      └──────┬───────┘
       │                     │
       ▼                     ▼
 MAE/RMSE/WAPE          未来销量预测
```

---

# 5. 推荐的实际操作顺序

### 第一步：训练

```bash
python train.py --sku SKU001
```

得到：

```text
models/SKU001.pkl
```

### 第二步：评估

```bash
python evaluate.py \
    --sku SKU001 \
    --validation-days 30 \
    --windows 4
```

查看：

```text
Average MAE
Average RMSE
Average WAPE
```

### 第三步：预测

确认模型效果满足要求后：

```bash
python forecast.py \
    --sku SKU001 \
    --days 30
```

得到：

```text
output/SKU001_forecast.csv
```

以及：

```text
output/SKU001_forecast.png
output/SKU001_components.png
```

---

# 6. 三个脚本的命令速查

```bash
# 训练
python train.py --sku SKU001

# 训练 + 指定数据
python train.py \
    --sales data/sales.csv \
    --holidays data/holidays.csv \
    --sku SKU001 \
    --model-dir models


# 评估
python evaluate.py --sku SKU001

# 评估：30天 × 4窗口
python evaluate.py \
    --sku SKU001 \
    --validation-days 30 \
    --windows 4


# 预测
python forecast.py --sku SKU001

# 预测未来7天
python forecast.py \
    --sku SKU001 \
    --days 7

# 预测未来30天
python forecast.py \
    --sku SKU001 \
    --days 30

# 预测未来90天
python forecast.py \
    --sku SKU001 \
    --days 90
```

**最核心的三个命令就是：**

```bash
python train.py --sku SKU001

python evaluate.py --sku SKU001 --validation-days 30 --windows 4

python forecast.py --sku SKU001 --days 30
```

也就是 **训练 → 评估 → 预测**。

[M5 Walmart 测试数据集](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data?utm_source=chatgpt.com)
