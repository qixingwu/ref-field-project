#!/bin/bash
# Train RefField on all MVTec AD categories

set -e  # Exit on error

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate reffield

# Configuration
CONFIG="ref_field/configs/mvtec.yaml"
CATEGORIES=("bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")

echo "========================================"
echo "RefField MVTec AD Training"
echo "Categories: ${#CATEGORIES[@]}"
echo "========================================"

for category in "${CATEGORIES[@]}"; do
    echo ""
    echo "========================================"
    echo "Training category: $category"
    echo "========================================"

    # Stage 1: Context Pretraining
    echo "[Stage 1] Context Pretraining..."
    python train_context.py --config "$CONFIG" --category "$category"

    # Stage 2: Gate Training
    echo "[Stage 2] Gate Training..."
    python train_gate.py --config "$CONFIG" --category "$category"

    echo "[Done] $category completed!"
done

echo ""
echo "========================================"
echo "All categories trained successfully!"
echo "========================================"
