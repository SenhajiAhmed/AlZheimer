#!/usr/bin/env bash
# =============================================================================
# download_data.sh — Download the Alzheimer's 4-class MRI dataset from Kaggle
#
# PREREQUISITES:
#   1. Install the Kaggle CLI:   pip install kaggle
#   2. Get your API key from:   https://www.kaggle.com/settings → API → Create Token
#   3. Place kaggle.json at:    ~/.kaggle/kaggle.json
#   4. Set permissions:         chmod 600 ~/.kaggle/kaggle.json
#
# USAGE:
#   chmod +x data/download_data.sh
#   ./data/download_data.sh
# =============================================================================

set -euo pipefail

DATASET="sachinkumar413/alzheimer-mri-dataset"
RAW_DIR="$(dirname "$0")/raw"

echo "==> Checking Kaggle CLI..."
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: kaggle CLI not found. Run: pip install kaggle"
    exit 1
fi

echo "==> Checking ~/.kaggle/kaggle.json..."
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "ERROR: kaggle.json not found at ~/.kaggle/kaggle.json"
    echo "       Download it from https://www.kaggle.com/settings → API → Create Token"
    exit 1
fi

echo "==> Downloading dataset: $DATASET"
mkdir -p "$RAW_DIR"
kaggle datasets download -d "$DATASET" -p "$RAW_DIR" --unzip

echo ""
echo "==> Done. Dataset extracted to: $RAW_DIR"
echo "    Expected structure:"
echo "    raw/"
echo "    ├── Alzheimer_s Dataset/"
echo "    │   ├── train/"
echo "    │   │   ├── NonDemented/"
echo "    │   │   ├── VeryMildDemented/"
echo "    │   │   ├── MildDemented/"
echo "    │   │   └── ModerateDemented/"
echo "    │   └── test/"
echo "    │       └── ..."
