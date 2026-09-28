import os
import numpy as np
import pandas as pd

OUT = "data/processed/smoke_test_sales.csv"
HOLIDAYS = "data/processed/holidays.csv"


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    dates = pd.date_range("2024-01-01", periods=240, freq="D")
    rows = []
    rng = np.random.default_rng(42)

    for sku_id, base, trend in [
        ("SKU001", 100, 0.08),
        ("SKU002", 60, 0.03),
    ]:
        for i, date in enumerate(dates):
            weekly = 12 * np.sin(2 * np.pi * i / 7)
            yearly_like = 8 * np.sin(2 * np.pi * i / 60)
            noise = rng.normal(0, 4)
            sales = max(0, base + trend * i + weekly + yearly_like + noise)
            rows.append({"date": date, "sku_id": sku_id, "sales": round(float(sales), 2)})

    pd.DataFrame(rows).to_csv(OUT, index=False)

    holiday_dates = pd.to_datetime(["2024-02-10", "2024-06-18", "2024-11-11"])
    holiday_df = pd.DataFrame(
        {
            "holiday": ["spring_festival", "618", "double_11"],
            "ds": holiday_dates,
            "lower_window": [0, -3, -7],
            "upper_window": [2, 2, 3],
        }
    )
    holiday_df.to_csv(HOLIDAYS, index=False)

    print(f"Created {OUT}")
    print(f"Created {HOLIDAYS}")


if __name__ == "__main__":
    main()
