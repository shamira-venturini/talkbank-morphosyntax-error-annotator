from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List

from common import resolve_path


VARIANTS = {
    "recon_full_comp_nonword_only": {
        "reconstruction_mode": "nonword_only",
        "chat_token_strategy": "components",
        "prompt_style": "compositional",
        "description": "Exploratory compositional probe with nonword-only reconstruction.",
    },
    "recon_full_comp_preserve": {
        "reconstruction_mode": "preserve",
        "chat_token_strategy": "components",
        "prompt_style": "compositional",
        "description": "Exploratory compositional probe with preserved reconstruction markers.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the minimal exploratory compositional reconstruction probe."
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
        "experiment_family": "compositional_reconstruction_probe",
        "input_file": str(input_path),
        "experiments_root": str(root),
        "seed": args.seed,
        "notes": [
            "Exploratory only: focuses on the combined compositional package rather than a full ablation grid.",
            "Controls already exist in the repo as the current full-curr hybrid+standard runs.",
            "This probe asks whether component tokens plus compositional prompting interact with reconstruction mode.",
        ],
        "existing_controls": {
            "nonword_only_control": "recon_full_curr_nonwordonly_seed3407",
            "preserve_control": "recon_full_curr_preserve_seed3407",
        },
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
            "--chat-token-strategy",
            cfg["chat_token_strategy"],
        ]
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
            "--prompt-style",
            cfg["prompt_style"],
        ]
        run_cmd(prompt_cmd)

        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        prompt_summary = json.loads((out_dir / "prompt_update_summary.json").read_text(encoding="utf-8"))
        overall_summary["variants"][name] = {
            "description": cfg["description"],
            "reconstruction_mode": cfg["reconstruction_mode"],
            "chat_token_strategy": cfg["chat_token_strategy"],
            "prompt_style": cfg["prompt_style"],
            "split_sizes_stage3": summary.get("split_sizes_stage3", {}),
            "source_by_split_stage3": summary.get("source_by_split_stage3", {}),
            "holdout_labels": summary.get("holdout_labels", []),
            "prompt_preview_stage3": prompt_summary.get("prompt_preview_stage3", ""),
        }

    out_summary = root / "compositional_reconstruction_probe_summary.json"
    out_summary.write_text(json.dumps(overall_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Compositional reconstruction probe setup complete: {out_summary}")


if __name__ == "__main__":
    main()
