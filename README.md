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

**Checkpoint auto-loading**: The scripts automatically find checkpoints at `<work_dir>/<category>/checkpoints/`. You can still manually specify `--resume_context` or `--checkpoint` to override.

## Notes

### Inference smoothing

Default `smoothing_sigma=0.0` preserves pixel-level anomaly boundaries, which typically improves pixel-level AP.

### Encoder choices

Default config uses a `timm` ViT. If your local `timm` version uses a slightly different model name, change `model.name` in the config.

### Memory building

The first evaluation run builds and caches the normal-image retrieval memory under `work_dir/<category>/memory/`.

### Extending to VisA / Real-IAD

- Copy `ref_field/datasets/mvtec.py`
- Implement dataset parsing
- Keep the rest of the pipeline unchanged

### Suggested first ablations

1. External-only (`use_intra=false`, `use_context=false`)
2. External + Intra
3. External + Intra + Gate
4. Full model with context expert

## Citation

This code was generated as a reference implementation for the research idea discussed in chat. Please adapt, verify, and benchmark before publication.
