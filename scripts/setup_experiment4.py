import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List

from common import resolve_path


TAG_PATTERN = re.compile(r"\[\*\s*[ms](?::[^\]]+)?\]")
DEFAULT_HOLDOUT_LABELS = ["[* m:++er]", "[* m:++est]", "[* m:0er]", "[* m:0est]"]


def canonical_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    return f"[* {body}]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Experiment 4 unseen-tag generalization package.")
    parser.add_argument(
        "--input",
        default="data/processed/master_training.jsonl",
        help="Master enriched input JSONL.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/exp4_unseen_tags",
        help="Output experiment directory.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Split seed.")
    parser.add_argument(
        "--holdout-label",
        action="append",
        default=None,
        help="Hold out detailed labels for compositional generalization (repeat flag).",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if completed.stdout.strip():
        print(completed.stdout.strip())


def count_label_occurrences(path: Path) -> Counter:
    counts: Counter = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            output = row.get("output", "")
            for tag in TAG_PATTERN.findall(output):
                counts[canonical_tag(tag)] += 1
    return counts


def count_label_row_support(path: Path) -> Counter:
    counts: Counter = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            output = row.get("output", "")
            present = {canonical_tag(tag) for tag in TAG_PATTERN.findall(output)}
            for tag in present:
                counts[tag] += 1
    return counts


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout_labels = args.holdout_label if args.holdout_label else DEFAULT_HOLDOUT_LABELS

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
        "preserve",
    ]
    for label in holdout_labels:
        build_cmd.extend(["--holdout-label", label])
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
        "preserve",
    ]
    run_cmd(prompt_cmd)

    stage3_train = out_dir / "stage3_train.jsonl"
    stage3_holdout = out_dir / "stage3_holdout.jsonl"
    train_occ = count_label_occurrences(stage3_train)
    holdout_occ = count_label_occurrences(stage3_holdout)
    train_rows = count_label_row_support(stage3_train)
    holdout_rows = count_label_row_support(stage3_holdout)

    withheld_summary: Dict[str, Dict[str, int]] = {}
    for label in holdout_labels:
        withheld_summary[label] = {
            "train_row_support": train_rows.get(label, 0),
            "holdout_row_support": holdout_rows.get(label, 0),
            "train_occurrences": train_occ.get(label, 0),
            "holdout_occurrences": holdout_occ.get(label, 0),
        }

    base_summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    exp4_summary = {
        "experiment_id": "exp4_unseen_tags",
        "input_file": str(input_path),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "holdout_labels": holdout_labels,
        "withheld_label_counts": withheld_summary,
        "split_sizes_stage3": base_summary.get("split_sizes_stage3", {}),
        "source_by_split_stage3": base_summary.get("source_by_split_stage3", {}),
        "notes": [
            "Holdout labels must have zero support in stage3_train.",
            "Generalization is evaluated on stage3_holdout only.",
            "Main confirmatory metrics still come from real-only test.",
        ],
    }
    (out_dir / "experiment4_summary.json").write_text(
        json.dumps(exp4_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Experiment 4 setup complete: {out_dir / 'experiment4_summary.json'}")


if __name__ == "__main__":
    main()
