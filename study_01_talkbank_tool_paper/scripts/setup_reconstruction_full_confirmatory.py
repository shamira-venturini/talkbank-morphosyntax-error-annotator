from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List

from common import resolve_path


VARIANTS = {
    "recon_full_curr_nonword_only": {
        "reconstruction_mode": "nonword_only",
        "description": "Full-data confirmatory baseline with curriculum fixed and nonword-only reconstruction.",
    },
    "recon_full_curr_preserve": {
        "reconstruction_mode": "preserve",
        "description": "Full-data confirmatory comparison with curriculum fixed and preserved reconstruction markers.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the full-data confirmatory reconstruction comparison with curriculum fixed."
    )
    parser.add_argument(
        "--input",
        default="data/processed/master_training.jsonl",
        help="Master enriched input JSONL.",
    )
    parser.add_argument(
        "--experiments-root",
        default="experiments",
        help="Root directory where experiment folders are created.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Split seed.")
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if completed.stdout.strip():
        print(completed.stdout.strip())


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    root = resolve_path(args.experiments_root)
    root.mkdir(parents=True, exist_ok=True)

    overall_summary = {
        "experiment_family": "reconstruction_full_confirmatory",
        "input_file": str(input_path),
        "experiments_root": str(root),
        "seed": args.seed,
        "notes": [
            "Curriculum is fixed across both conditions and is treated as a methodological choice.",
            "Reconstruction is the only manipulated factor in this confirmatory setup.",
            "Both packages are intended for full curriculum runs using stages 1, 2, and 3.",
        ],
        "variants": {},
    }

    for name, cfg in VARIANTS.items():
        out_dir = root / name
        build_cmd = [
            "python3",
            "scripts/build_acl_splits.py",
            "--input",
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--seed",
            str(args.seed),
            "--train-synthetic-ratio",
            "-1",
            "--reconstruction-mode",
            cfg["reconstruction_mode"],
        ]
        if cfg["reconstruction_mode"] == "preserve":
            build_cmd.append("--autofill-recon-03s-baseed")
        run_cmd(build_cmd)

        prompt_cmd = [
            "python3",
            "scripts/set_experiment_prompts.py",
            "--experiment-dir",
            str(out_dir),
            "--label-source-splits",
            "train",
            "--holdout-label-source-splits",
            "train,holdout",
            "--reconstruction-mode",
            cfg["reconstruction_mode"],
        ]
        run_cmd(prompt_cmd)

        summary_path = out_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        overall_summary["variants"][name] = {
            "description": cfg["description"],
            "reconstruction_mode": cfg["reconstruction_mode"],
            "split_sizes_stage3": summary.get("split_sizes_stage3", {}),
            "source_by_split_stage3": summary.get("source_by_split_stage3", {}),
            "holdout_labels": summary.get("holdout_labels", []),
        }

    out_summary = root / "reconstruction_full_confirmatory_summary.json"
    out_summary.write_text(json.dumps(overall_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Full confirmatory reconstruction setup complete: {out_summary}")


if __name__ == "__main__":
    main()
