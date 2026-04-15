#!/usr/bin/env bash
set -e
python train_context.py --config ref_field/configs/mvtec.yaml --category bottle
python train_gate.py --config ref_field/configs/mvtec.yaml --category bottle
python infer.py --config ref_field/configs/mvtec.yaml --category bottle --split test --save_vis
