import os
import argparse
import pandas as pd

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_OUTPUT_DIR = "data/processed"


def load_calendar(raw_dir):
    calendar_file = os.path.join(raw_dir, "calendar.csv")
    if not os.path.exists(calendar_file):
        raise FileNotFoundError(f"Calendar file not found: {calendar_file}")
    calendar = pd.read_csv(calendar_file)
    calendar["date"] = pd.to_datetime(calendar["date"])
    return calendar


def load_sales(raw_dir):
    sales_file = os.path.join(raw_dir, "sales_train_validation.csv")
    if not os.path.exists(sales_file):
        raise FileNotFoundError(f"Sales file not found: {sales_file}")
    return pd.read_csv(sales_file)


def prepare_sales(sales, calendar, sku_count):
    date_columns = [c for c in sales.columns if c.startswith("d_")]
    if not date_columns:
        raise ValueError("No d_xxx columns found in sales_train_validation.csv")

    selected = sales.sort_values("id").head(sku_count).copy()
    if selected.empty:
        raise ValueError("No SKU rows were selected.")

    print(f"Selected SKU count: {len(selected)}")
    for sku in selected["id"]:
        print(f"  {sku}")

    long_df = selected.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=date_columns,
        var_name="d",
        value_name="sales",
    )

    long_df = long_df.merge(calendar[["d", "date"]], on="d", how="left")
    long_df = long_df.rename(columns={"id": "sku_id"})
    long_df = long_df[["date", "sku_id", "sales"]]
    long_df = long_df.sort_values(["sku_id", "date"])
    return long_df


def prepare_holidays(calendar):
    holidays = []
    for _, row in calendar.iterrows():
        for event_col in ("event_name_1", "event_name_2"):
            if pd.notna(row.get(event_col)):
                holidays.append(
                    {
                        "holiday": row[event_col],
                        "ds": row["date"],
                        "lower_window": 0,
                        "upper_window": 0,
                    }
                )
    return pd.DataFrame(holidays, columns=["holiday", "ds", "lower_window", "upper_window"])


def main(raw_dir, output_dir, sku_count):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("M5 Data Preparation")
    print("=" * 60)

    calendar = load_calendar(raw_dir)
    print(f"Calendar rows: {len(calendar)}")

    sales = load_sales(raw_dir)
    print(f"Sales rows: {len(sales)}")

    sales_output = prepare_sales(sales, calendar, sku_count)
    sales_file = os.path.join(output_dir, "sales.csv")
    sales_output.to_csv(sales_file, index=False)
    print(f"Sales saved to: {sales_file}")

    holidays = prepare_holidays(calendar)
    holidays_file = os.path.join(output_dir, "holidays.csv")
    holidays.to_csv(holidays_file, index=False)
    print(f"Holidays saved to: {holidays_file}")

    print("=" * 60)
    print("Preparation finished.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sku-count", type=int, default=10)
    args = parser.parse_args()
    main(args.raw_dir, args.output_dir, args.sku_count)
