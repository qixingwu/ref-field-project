from __future__ import annotations

import argparse
import subprocess


MVTec_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--resume_context", type=str, default="")
    ap.add_argument("--context_scope", type=str, choices=["category", "dataset_shared"], default="category")
    ap.add_argument("--gate_scope", type=str, choices=["category", "dataset_shared"], default="category")
    ap.add_argument("--save_vis", action="store_true", help="Save visualization images")
    args = ap.parse_args()

    print(f"gate_scope: {args.gate_scope}")
    print(f"context_scope: {args.context_scope}")
    if args.resume_context:
        print(f"resume_context: {args.resume_context}")
    if args.checkpoint:
        print(f"checkpoint: {args.checkpoint}")

    if args.gate_scope == "dataset_shared":
        print("benchmark will run a single unified multi-class inference call")
        print("unified mode: --category bottle is only for CLI compatibility and will be ignored by infer.py")
        cmd = [
            "python", "infer.py",
            "--config", args.config,
            "--category", "bottle",
            "--split", "test",
            "--gate_scope", args.gate_scope,
            "--context_scope", args.context_scope,
        ]
        if args.resume_context:
            cmd += ["--resume_context", args.resume_context]
        if args.checkpoint:
            cmd += ["--checkpoint", args.checkpoint]
        if args.save_vis:
            cmd += ["--save_vis"]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=False)
        return

    print("benchmark will iterate over all MVTec categories")
    for cat in MVTec_CATEGORIES:
        cmd = [
            "python", "infer.py",
            "--config", args.config,
            "--category", cat,
            "--split", "test",
            "--context_scope", args.context_scope,
        ]
        if args.resume_context:
            cmd += ["--resume_context", args.resume_context]
        if args.checkpoint:
            cmd += ["--checkpoint", args.checkpoint]
        if args.save_vis:
            cmd += ["--save_vis"]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
