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
    args = ap.parse_args()

    for cat in MVTec_CATEGORIES:
        cmd = [
            "python", "infer.py",
            "--config", args.config,
            "--category", cat,
            "--split", "test",
        ]
        if args.checkpoint:
            cmd += ["--checkpoint", args.checkpoint]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
