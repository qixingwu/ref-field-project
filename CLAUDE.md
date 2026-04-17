# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RefField-v1 is a research implementation for **Multi-Reference Competitive Reasoning for Anomaly Detection**. It uses a frozen ViT encoder with three complementary experts (intra-image recurrence, external prototype retrieval, masked context reconstruction) combined via a learned reliability gate.

## Development Commands

### Environment Setup

**Conda Environment**: `reffield` (already created locally)

```bash
# Activate environment before running any commands
conda activate reffield

# Install dependencies (if needed)
pip install -r requirements.txt
```

### Training Pipeline

**Stage 1 - Context Pretraining** (trains only the context expert's masked reconstruction):
```bash
python train_context.py --config ref_field/configs/mvtec.yaml --category bottle
```

**Stage 2 - Gate Training** (trains the reliability gate using pseudo-anomalies):
```bash
# Auto-loads context checkpoint from default path
python train_gate.py --config ref_field/configs/mvtec.yaml --category bottle

# Or explicitly specify context checkpoint
python train_gate.py --config ref_field/configs/mvtec.yaml --category bottle --resume_context work_dirs/ref_field_mvtec/bottle/checkpoints/context_last.pt
```

### Inference / Evaluation
```bash
# Single category (auto-loads gate checkpoint)
python infer.py --config ref_field/configs/mvtec.yaml --category bottle --split test

# All MVTec categories
python benchmark.py --config ref_field/configs/mvtec.yaml

# With visualization
python infer.py --config ref_field/configs/mvtec.yaml --category bottle --split test --save_vis
```

## Architecture Overview

### Core Model (`ref_field/models/ref_field.py`)

The `RefField` class orchestrates three experts that compete and collaborate:

1. **IntraExpert** (`intra_expert.py`): Computes anomaly scores based on local patch recurrence within the same image, with spatial exclusion to avoid trivial self-matches.

2. **ExternalExpert** (`external_expert.py`): Compares each patch against retrieved normal reference patches from a memory bank built during training.

3. **ContextExpert** (`context_expert.py`): Uses a Transformer encoder to predict masked tokens; high reconstruction error indicates anomalies.

4. **ReliabilityGate** (`gate.py`): Learns to weight each expert's contribution and detect conflicts between experts. Takes a 10D feature vector (3 expert scores + 3 reliabilities + invariance + 3 pairwise differences).

### Data Flow

```
Image → PatchEncoder (frozen timm ViT) → Projector → Tokens [B,N,D]
                                                           ↓
                                ┌──────────────────────────────────────────┐
                                ↓              ↓              ↓              ↓
                         IntraExpert   ExternalExpert   ContextExpert   NuisanceViews
                         (local recur) (retrieval)     (masked recon)   (invariance)
                                ↓              ↓              ↓              ↓
                            scores        scores          scores      reliability
                                └──────────────┬───────────────┴──────────────┘
                                               ↓
                                         ReliabilityGate
                                               ↓
                              alpha (3 expert weights) + lambda_conf (conflict)
                                               ↓
                         Final Score = -log(normality) + lambda_conf * conflict
```

### Memory System (`ref_field/memory/`)

- **TokenStore**: Stores global descriptors and patch tokens for all training images
- **ImageIndex**: FAISS-based retrieval (with NumPy fallback if FAISS not installed) for finding top-k similar normal images

Memory is built on each inference run from training images. The first run caches results; subsequent runs reuse if memory files exist.

### Two-Stage Training

**Why two stages?** The context expert needs pretraining before the gate can meaningfully weigh expert outputs.

1. **Context Pretraining** (`train_context.py`): Only trains the ContextExpert's Transformer encoder using masked token reconstruction on normal training images. Also trains the projector if it exists.

2. **Gate Training** (`train_gate.py`): Uses `PseudoAnomalyGenerator` to create synthetic anomalies (local shuffle, color shift, cutpaste, patch relocate) and trains the gate with BCE + ranking loss. The context checkpoint is auto-loaded from the default path if not explicitly provided.

### Datasets (`ref_field/datasets/`)

- `MVTecADDataset`: Standard MVTec AD loader. Expects data in `mvtec_ad/<category>/train|test/` format.
- `PseudoAnomalyGenerator`: Creates synthetic anomalies for gate training (local shuffle, color shift, cutpaste, patch relocate, nuisance-only).

## Configuration

All hyperparameters are in YAML configs (e.g., `ref_field/configs/mvtec.yaml`). Key sections:

- `data.root`: Path to MVTec AD dataset
- `model.name`: timm model name (e.g., `vit_small_patch14_dinov2.lvd142m`)
- `memory.image_top_r`: Number of reference images to retrieve
- `memory.patch_top_k`: Patches per reference for external expert
- `train.epochs_context` / `epochs_gate`: Training epochs for each stage

## Important Implementation Details

- **Projector initialization**: The projector is created lazily on first forward pass because the encoder's output dimension is inferred from the backbone. A dummy forward pass is done in training scripts to initialize it before optimizer creation.

- **Checkpoint auto-loading**: Both gate training and inference scripts automatically find checkpoints at `<work_dir>/<category>/checkpoints/`. Explicit `--resume_context` or `--checkpoint` arguments override this behavior.

- **FAISS optional**: The code falls back to NumPy-based retrieval if FAISS is not installed (handled in `faiss_index.py`).

- **Nuisance views**: During gate training and inference, augmented views (gamma jitter + optional shift) are created to compute invariance reliability.

- **Memory caching**: Token banks are padded to the maximum size in each batch for efficient tensor operations.

- **Evaluation protocol**: Image-level scores use max pooling on original patch scores (decoupled from segmentation smoothing). Pixel-level scores are upsampled to GT resolution, then Gaussian smoothed with `sigma=4.0`.

## Extending to New Datasets

To add VisA, BTAD, or other datasets:

1. Create a new dataset class in `ref_field/datasets/` following the pattern in `mvtec.py`
2. Inherit from `BaseADDataset` and implement `_build_samples()` to return `Sample` objects
3. The rest of the pipeline (training, inference, memory building) works unchanged

## Work Directory Structure

```
work_dirs/ref_field_mvtec/
└── <category>/
    ├── checkpoints/
    │   ├── context_last.pt
    │   └── gate_last.pt
    ├── visualizations/
    │   └── <subfolder>_<filename>.png
    └── metrics.json
```

## MVTec Categories

```
bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut,
pill, screw, tile, toothbrush, transistor, wood, zipper
```
