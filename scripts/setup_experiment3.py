import argparse
import json
import subprocess
from pathlib import Path
from typing import List

from common import resolve_path


VARIANTS = {
    "exp3_abs_on_manual": {
        "reconstruction_mode": "preserve",
        "description": "AbS-on baseline: preserve manual distinction between [:] and [::].",
    },
    "exp3_abs_off_no_recon": {
        "reconstruction_mode": "drop_all",
        "description": "AbS-off: remove all reconstruction markers from training/eval targets.",
    },
    "exp3_abs_diag_nonword_only": {
        "reconstruction_mode": "nonword_only",
        "description": "Diagnostic: keep [:] for nonword-like corrections, remove [::].",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Experiment 3 reconstruction ablation packages.")
    parser.add_argument(
        "--input",
        default="FT-3/df_master_training_v3_with_provenance_errorcount.jsonl",
        help="Master enriched input JSONL.",
    )
    parser.add_argument(
        "--experiments-root",
        default="FT-3/experiments",
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
        "input_file": str(input_path),
        "experiments_root": str(root),
        "seed": args.seed,
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
            "--reconstruction-mode",
            cfg["reconstruction_mode"],
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
        ]
        run_cmd(prompt_cmd)

        summary_path = out_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        overall_summary["variants"][name] = {
            "description": cfg["description"],
            "reconstruction_mode": cfg["reconstruction_mode"],
            "split_sizes_stage3": summary.get("split_sizes_stage3", {}),
            "source_by_split_stage3": summary.get("source_by_split_stage3", {}),
        }

    out_summary = root / "experiment3_setup_summary.json"
    out_summary.write_text(json.dumps(overall_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Experiment 3 setup complete: {out_summary}")


if __name__ == "__main__":
    main()
