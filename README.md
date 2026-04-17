# RefField-v1

Reference PyTorch implementation of **RefField-v1: Multi-Reference Competitive Reasoning for Anomaly Detection**.

This repository is a **strong baseline / research scaffold** for the idea discussed in chat. It is designed to be readable, hackable, and runnable on MVTec AD first, then extended to VisA / BTAD / MPDD / Real-IAD.

## What is implemented

- Frozen image encoder from `timm`
- Patch-token projector
- Intra-image recurrence expert
- External prototype expert with image-level retrieval and patch-level matching
- Context expert (masked token reconstruction head)
- Reliability gate and conflict-aware fusion
- Pseudo-anomaly generator for gate training
- MVTec AD dataset loader
- Context pretraining script
- Gate training script
- Inference / evaluation script
- FAISS-based image retrieval with a NumPy fallback

## What this is not

- Not a polished production repo
- Not benchmarked by us end-to-end on every dataset yet
- Not a final SOTA recipe

It is intended to give you a **complete, modifiable codebase** that matches the algorithm design and can be used as the starting point for your paper/code experiments.

## Recommended environment

- Python 3.10+
- PyTorch 2.2+
- CUDA optional but recommended

## Install

```bash
pip install -r requirements.txt
```

## Project structure

```text
ref_field_project/
├── README.md
├── requirements.txt
├── train_context.py
├── train_gate.py
├── infer.py
├── benchmark.py
└── ref_field/
    ├── configs/
    │   └── mvtec.yaml
    ├── datasets/
    │   ├── __init__.py
    │   ├── base_dataset.py
    │   ├── mvtec.py
    │   ├── pseudo_anomaly.py
    │   └── transforms.py
    ├── memory/
    │   ├── __init__.py
    │   ├── faiss_index.py
    │   └── token_store.py
    ├── models/
    │   ├── __init__.py
    │   ├── encoder.py
    │   ├── projector.py
    │   ├── intra_expert.py
    │   ├── external_expert.py
    │   ├── context_expert.py
    │   ├── gate.py
    │   └── ref_field.py
    └── utils/
        ├── __init__.py
        ├── config.py
        ├── metrics.py
        ├── misc.py
        └── visualize.py
```

## Data layout for MVTec AD

Point `data.root` in `ref_field/configs/mvtec.yaml` to your MVTec AD root.

Expected layout:

```text
mvtec_ad/
├── bottle/
│   ├── train/good/*.png
│   ├── test/good/*.png
│   ├── test/broken_large/*.png
│   └── ground_truth/broken_large/*.png
├── cable/
... 
```

## Quick start

This is the category-specific context workflow. It remains supported for compatibility and for per-category experiments. The repository also supports MVTec dataset-shared context pretraining; see the next section for the recommended workflow when you want to reduce per-category context overfitting.

### 1. Context pretraining

```bash
python train_context.py --config ref_field/configs/mvtec.yaml --category bottle
```

### 2. Gate training

```bash
python train_gate.py --config ref_field/configs/mvtec.yaml --category bottle
```

### 3. Inference / evaluation

```bash
python infer.py --config ref_field/configs/mvtec.yaml --category bottle --split test
```

## Recommended MVTec workflow: dataset-shared context

This workflow pretrains one context expert on all MVTec categories, then trains and evaluates the gate per category. It is recommended when you want the context branch to rely less on category-specific reconstruction shortcuts.

### 1. Shared context pretraining

```bash
python train_context.py --config ref_field/configs/mvtec.yaml --category bottle --context_scope dataset_shared
```

In dataset-shared mode, `--category` is accepted for CLI consistency; the shared context is trained across all MVTec categories.

### 2. Gate training

```bash
python train_gate.py --config ref_field/configs/mvtec.yaml --category bottle --context_scope dataset_shared
```

### 3. Inference / evaluation

```bash
python infer.py --config ref_field/configs/mvtec.yaml --category bottle --split test --context_scope dataset_shared
```

With `context_scope=dataset_shared`, the shared MVTec context checkpoint is saved to and auto-loaded from ``[work_dir]/_shared_context/mvtec/checkpoints/context_last.pt``.

## Recommended unified multi-class workflow on MVTec

This workflow pretrains one MVTec shared context expert, then trains and evaluates one unified gate across all MVTec categories.

### 1. Shared context pretraining

```bash
python train_context.py --config ref_field/configs/mvtec.yaml --category bottle --context_scope dataset_shared
```

`--category` is kept only for CLI compatibility in dataset-shared mode.

### 2. Unified multi-class gate training

```bash
python train_gate.py --config ref_field/configs/mvtec.yaml --category bottle --context_scope dataset_shared --gate_scope dataset_shared
```

### 3. Unified multi-class inference / evaluation

```bash
python infer.py --config ref_field/configs/mvtec.yaml --category bottle --split test --context_scope dataset_shared --gate_scope dataset_shared
```

### 4. Unified multi-class benchmark

```bash
python benchmark.py --config ref_field/configs/mvtec.yaml --context_scope dataset_shared --gate_scope dataset_shared
```

With `gate_scope=dataset_shared`, inference and benchmark results are written under ``[work_dir]/_shared_gate/mvtec`` instead of a single category directory. Metrics are saved to ``[work_dir]/_shared_gate/mvtec/metrics.json`` and visualizations, when enabled, are saved under ``[work_dir]/_shared_gate/mvtec/visualizations/``.

**Checkpoint auto-loading**:
- Context checkpoints are selected by `context_scope`. `context_scope=category` uses ``[work_dir]/[category]/checkpoints/context_last.pt``; `context_scope=dataset_shared` uses ``[work_dir]/_shared_context/mvtec/checkpoints/context_last.pt``.
- Gate checkpoints are selected by `gate_scope`. `gate_scope=category` uses ``[work_dir]/[category]/checkpoints/gate_last.pt``; `gate_scope=dataset_shared` uses ``[work_dir]/_shared_gate/mvtec/checkpoints/gate_last.pt``.
- `--resume_context` explicitly overrides the context checkpoint path for `train_gate.py` and `infer.py`. `--checkpoint` explicitly overrides the gate checkpoint path for `infer.py` and `benchmark.py`.

## Notes

### Context and gate scope

- `context_scope` decides where context checkpoints are trained or loaded from: `category` or `dataset_shared`.
- `gate_scope` decides the data and result scope for gate training, inference, and benchmark: `category` or `dataset_shared`.
- Do not mix them up: shared context can be used with either category-specific gate training or unified multi-class gate training.
- `context_scope=category`: uses ``[work_dir]/[category]/checkpoints/context_last.pt``.
- `context_scope=dataset_shared`: uses ``[work_dir]/_shared_context/mvtec/checkpoints/context_last.pt``.
- `gate_scope=category`: uses ``[work_dir]/[category]/checkpoints/gate_last.pt`` and writes inference results under ``[work_dir]/[category]``.
- `gate_scope=dataset_shared`: uses ``[work_dir]/_shared_gate/mvtec/checkpoints/gate_last.pt`` and writes unified inference / benchmark results under ``[work_dir]/_shared_gate/mvtec``.

### Evaluation protocol

The evaluation protocol is aligned more closely with PatchCore-style scoring:
- **Image-level score**: aggregated from original patch-level anomaly scores (max pooling), decoupled from segmentation smoothing
- **Pixel-level score**: upsampled to GT mask resolution, then Gaussian smoothed with `sigma=4.0` (default, matching PatchCore)

This separation ensures image-level metrics are not affected by segmentation post-processing.

### Encoder choices

Default config uses a `timm` ViT. If your local `timm` version uses a slightly different model name, change `model.name` in the config.

### Memory building

The first evaluation run builds and caches the normal-image retrieval memory under ``[work_dir]/[category]/memory/``.

### Extending to VisA / Real-IAD

- Copy `ref_field/datasets/mvtec.py`
- Implement dataset parsing
- Keep the rest of the pipeline unchanged

### Suggested first ablations

1. External-only (`use_intra=false`, `use_context=false`)
2. External + Intra
3. External + Intra + Gate
4. Full model with context expert

For context-branch ablations, compare `context_scope=category` against `context_scope=dataset_shared`.

For gate ablations, compare category-specific `gate_scope=category` against unified multi-class `gate_scope=dataset_shared`.

## Citation

This code was generated as a reference implementation for the research idea discussed in chat. Please adapt, verify, and benchmark before publication.
