import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence

from build_acl_splits import (
    DEFAULT_HOLDOUT_LABELS,
    apply_reconstruction_mode,
    build_chat_tokens,
    sample_counts_by_split,
    transform_rows,
)
from common import resolve_path, write_jsonl
from set_experiment_prompts import update_file, build_prompt


TAG_RE = re.compile(r"\[\*\s*[ms](?::[^\]]+)?\]")

VARIANTS = {
    "exp3_recon_mini_nonword_only": {
        "reconstruction_mode": "nonword_only",
        "description": "Fixed baseline for reconstruction ablation: keep [:] for nonword corrections only.",
    },
    "exp3_recon_mini_preserve": {
        "reconstruction_mode": "preserve",
        "description": "Reconstruction comparison condition: preserve [:] / [::] distinction.",
    },
}


def canonical_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    return f"[* {body}]"


def extract_tags(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [canonical_tag(tag) for tag in TAG_RE.findall(text)]


def primary_tag(row: Dict) -> str:
    tags = extract_tags(row.get("output", ""))
    return tags[0] if tags else "CLEAN"


def load_split(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sample_balanced_train(
    rows: Sequence[Dict],
    train_per_label: int,
    clean_count: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict]] = {}
    for row in rows:
        by_label.setdefault(primary_tag(row), []).append(row)

    sampled: List[Dict] = []
    for label in sorted(by_label):
        bucket = list(by_label[label])
        rng.shuffle(bucket)
        if label == "CLEAN":
            if clean_count > len(bucket):
                raise ValueError(f"Requested clean_count={clean_count}, but only {len(bucket)} clean rows are available.")
            sampled.extend(bucket[:clean_count])
            continue
        if train_per_label > len(bucket):
            raise ValueError(
                f"Requested train_per_label={train_per_label} for {label}, but only {len(bucket)} rows are available."
            )
        sampled.extend(bucket[:train_per_label])

    rng.shuffle(sampled)
    return sampled


def write_manifest(path: Path, split_to_rows: Dict[str, Sequence[Dict]]) -> None:
    headers = ["row_id", "split", "source_group", "error_count", "trace_ambiguous", "tags"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for split_name in ["train", "eval", "test", "eval_coverage", "test_coverage", "holdout"]:
            for row in split_to_rows.get(split_name, []):
                writer.writerow(
                    {
                        "row_id": row.get("row_id"),
                        "split": split_name,
                        "source_group": row.get("provenance_label", ""),
                        "error_count": row.get("error_count", 0),
                        "trace_ambiguous": row.get("trace_ambiguous", False),
                        "tags": "|".join(extract_tags(row.get("output", ""))),
                    }
                )


def refresh_prompts(exp_dir: Path, reconstruction_mode: str) -> None:
    stage_files: Dict[int, List[Path]] = {1: [], 2: [], 3: []}
    for path in sorted(exp_dir.glob("stage*_*.jsonl")):
        if path.name.startswith("stage1_"):
            stage_files[1].append(path)
        elif path.name.startswith("stage2_"):
            stage_files[2].append(path)
        elif path.name.startswith("stage3_"):
            stage_files[3].append(path)

    prompts = {}
    holdout_prompts = {}
    for stage in [1, 2, 3]:
        train_labels = set()
        holdout_labels = set()
        for file_path in stage_files[stage]:
            split_name = file_path.stem.split("_", 1)[1]
            rows = load_split(file_path)
            labels = set()
            for row in rows:
                labels.update(extract_tags(row.get("output", "")))
            if split_name == "train":
                train_labels.update(labels)
            if split_name in {"train", "holdout"}:
                holdout_labels.update(labels)
        prompts[stage] = build_prompt(sorted(train_labels), reconstruction_mode)
        holdout_prompts[stage] = build_prompt(sorted(holdout_labels), reconstruction_mode)

    for stage in [1, 2, 3]:
        for file_path in stage_files[stage]:
            split_name = file_path.stem.split("_", 1)[1]
            prompt = holdout_prompts[stage] if split_name == "holdout" else prompts[stage]
            update_file(file_path, prompt)

    summary = {
        "experiment_dir": str(exp_dir),
        "reconstruction_mode": reconstruction_mode,
        "prompt_preview_stage1": prompts[1][:300],
        "prompt_preview_stage2": prompts[2][:300],
        "prompt_preview_stage3": prompts[3][:300],
        "holdout_prompt_preview_stage1": holdout_prompts[1][:300],
        "holdout_prompt_preview_stage2": holdout_prompts[2][:300],
        "holdout_prompt_preview_stage3": holdout_prompts[3][:300],
    }
    (exp_dir / "prompt_update_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small reconstruction-only ablation package with a balanced mini-train split."
    )
    parser.add_argument(
        "--source-experiment-dir",
        default="experiments/exp3_abs_on_manual",
        help="Canonical source experiment package used for row membership and non-train splits.",
    )
    parser.add_argument(
        "--out-root",
        default="experiments",
        help="Output root for the mini reconstruction experiment packages.",
    )
    parser.add_argument(
        "--train-per-label",
        type=int,
        default=24,
        help="Number of training rows sampled per non-clean primary label.",
    )
    parser.add_argument(
        "--clean-train-count",
        type=int,
        default=288,
        help="Number of clean training rows sampled for the mini-train split.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=3407,
        help="Sampling seed used to choose the balanced mini-train subset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = resolve_path(args.source_experiment_dir)
    out_root = resolve_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    source_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    holdout_labels = source_summary.get("holdout_labels", DEFAULT_HOLDOUT_LABELS)

    source_stage3 = {
        split: load_split(source_dir / f"stage3_{split}.jsonl")
        for split in ["train", "eval", "test", "eval_coverage", "test_coverage", "holdout"]
    }

    sampled_train = sample_balanced_train(
        rows=source_stage3["train"],
        train_per_label=args.train_per_label,
        clean_count=args.clean_train_count,
        seed=args.sampling_seed,
    )

    overall_summary = {
        "source_experiment_dir": str(source_dir),
        "sampling_seed": args.sampling_seed,
        "train_per_label": args.train_per_label,
        "clean_train_count": args.clean_train_count,
        "train_primary_label_counts": sample_counts_by_split(sampled_train),
        "variants": {},
        "recommended_training_seeds": [3407, 3408, 3409],
    }

    primary_label_counts: Dict[str, int] = {}
    for row in sampled_train:
        label = primary_tag(row)
        primary_label_counts[label] = primary_label_counts.get(label, 0) + 1
    overall_summary["train_primary_label_counts"] = dict(sorted(primary_label_counts.items()))

    for exp_name, cfg in VARIANTS.items():
        exp_dir = out_root / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        stage3_rows = {
            "train": list(sampled_train),
            "eval": list(source_stage3["eval"]),
            "test": list(source_stage3["test"]),
            "eval_coverage": list(source_stage3["eval_coverage"]),
            "test_coverage": list(source_stage3["test_coverage"]),
            "holdout": list(source_stage3["holdout"]),
        }
        stage3_rows = {
            split: apply_reconstruction_mode(rows, cfg["reconstruction_mode"])
            for split, rows in stage3_rows.items()
        }
        stage2_rows = {split: transform_rows(rows, stage=2) for split, rows in stage3_rows.items()}
        stage1_rows = {split: transform_rows(rows, stage=1) for split, rows in stage3_rows.items()}

        for stage, split_map in [(1, stage1_rows), (2, stage2_rows), (3, stage3_rows)]:
            for split_name, rows in split_map.items():
                write_jsonl(exp_dir / f"stage{stage}_{split_name}.jsonl", rows)

        write_manifest(exp_dir / "split_manifest.csv", stage3_rows)

        chat_tokens = build_chat_tokens(stage3_rows["train"], extra_labels=holdout_labels)
        (exp_dir / "chat_tokens.json").write_text(
            json.dumps(chat_tokens, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        summary = {
            "experiment_id": exp_name,
            "description": cfg["description"],
            "source_experiment_dir": str(source_dir),
            "sampling_seed": args.sampling_seed,
            "train_per_label": args.train_per_label,
            "clean_train_count": args.clean_train_count,
            "holdout_labels": holdout_labels,
            "split_sizes_stage3": {split: len(rows) for split, rows in stage3_rows.items()},
            "source_by_split_stage3": {split: sample_counts_by_split(rows) for split, rows in stage3_rows.items()},
            "train_primary_label_counts": dict(sorted(primary_label_counts.items())),
            "reconstruction_mode": cfg["reconstruction_mode"],
            "notes": [
                "Mini reconstruction ablation package: balanced training subset, canonical non-train splits.",
                "Primary-label balancing is applied only to the training split.",
                "Eval/test/eval_coverage/test_coverage/holdout inherit the canonical row membership from source_experiment_dir.",
                "This package is intended for no-curriculum stage-3 runs only.",
            ],
        }
        (exp_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        refresh_prompts(exp_dir, cfg["reconstruction_mode"])
        overall_summary["variants"][exp_name] = summary

    out_summary = out_root / "reconstruction_mini_ablation_summary.json"
    out_summary.write_text(json.dumps(overall_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote reconstruction mini-ablation packages to: {out_root}")
    print(f"Summary: {out_summary}")


if __name__ == "__main__":
    main()
