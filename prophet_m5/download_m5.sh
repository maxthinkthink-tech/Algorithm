#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install it with: python -m pip install kaggle"
  exit 1
fi

echo "Downloading M5 Forecasting Accuracy dataset..."
kaggle competitions download -c m5-forecasting-accuracy -p data/raw

ZIP="data/raw/m5-forecasting-accuracy.zip"
if [ -f "$ZIP" ]; then
  echo "Extracting..."
  unzip -o "$ZIP" -d data/raw
  echo "Done. Raw files are under data/raw/"
else
  echo "Download finished but $ZIP was not found. Check data/raw/"
fi
