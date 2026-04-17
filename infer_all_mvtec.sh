#!/bin/bash
# Infer RefField on all MVTec AD categories

set -e  # Exit on error

# Configuration
CONFIG="ref_field/configs/mvtec.yaml"
SAVE_VIS="${1:---save-vis}"  # Pass --save-vis or --no-save-vis as argument

echo "========================================"
echo "RefField MVTec AD Inference"
echo "========================================"
echo "Config: $CONFIG"
echo "Save visualizations: $SAVE_VIS"
echo "Start time: $(date)"
echo "========================================"
echo ""

# Run benchmark in conda environment
conda run -n reffield python benchmark.py --config "$CONFIG" $SAVE_VIS

echo ""
echo "========================================"
echo "All inference completed!"
echo "End time: $(date)"
echo "========================================"
