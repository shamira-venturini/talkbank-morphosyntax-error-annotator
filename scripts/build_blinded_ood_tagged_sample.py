from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List

from common import resolve_path


TAG_RE = re.compile(r"\[\* [^\]]+\]")
INPUT_ANNOTATION_RE = re.compile(r"\s*\[(?::|::)\s+[^\]]+\]|\s*\[\*\s*[^\]]+\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a blinded annotation sheet from tagged rows in an OOD review CSV."
    )
    parser.add_argument(
        "--input-csv",
        default="reviews/ood/enni/enni_review_utterance_only.csv",
        help="Reviewed OOD utterance-level CSV containing model predictions.",
    )
    parser.add_argument(
        "--out-dir",
        default="reviews/study_02/enni_blinded_tagged_annotation_sample",
        help="Output directory for the blinded sheet and hidden answer key.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of tagged rows to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Shuffle seed for reproducible sampling.",
    )
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        default=False,
        help="Require a non-empty review_decision before a tagged row can be sampled.",
    )
    parser.add_argument(
        "--review-decision",
        action="append",
        default=[],
        help="Optional review_decision filter. Repeat flag to allow multiple values, e.g. --review-decision CORRECT.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def is_tagged(row: Dict[str, str]) -> bool:
    return bool(TAG_RE.search((row.get("model_prediction") or "").strip()))


def is_reviewed(row: Dict[str, str]) -> bool:
    return bool((row.get("review_decision") or "").strip())


def review_decision(row: Dict[str, str]) -> str:
    return (row.get("review_decision") or "").strip().upper()


def strip_input_annotations(text: str) -> str:
    cleaned = INPUT_ANNOTATION_RE.sub("", (text or "").strip())
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def build_annotation_sheet(rows: List[Dict[str, str]], out_dir: Path) -> Dict[str, str]:
    review_path = out_dir / "blinded_annotation_sheet.csv"
    key_path = out_dir / "answer_key.csv"

    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_id",
                "input",
                "human_annotation",
                "notes",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            review_id = f"ENNI_TAG_{idx:03d}"
            row["review_id"] = review_id
            writer.writerow(
                {
                    "review_id": review_id,
                    "input": strip_input_annotations(row.get("input", "")),
                    "human_annotation": "",
                    "notes": "",
                }
            )

    with key_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_id",
                "row_id",
                "file_name",
                "speaker",
                "line_no",
                "input",
                "blinded_input",
                "model_prediction",
                "corrected_prediction",
                "review_decision",
                "labels_or_clean",
                "n_labels",
                "missed_errors",
                "well-formed",
                "input_annotation",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "review_id": row.get("review_id", ""),
                    "row_id": row.get("row_id", ""),
                    "file_name": row.get("file_name", ""),
                    "speaker": row.get("speaker", ""),
                    "line_no": row.get("line_no", ""),
                    "input": row.get("input", ""),
                    "blinded_input": strip_input_annotations(row.get("input", "")),
                    "model_prediction": row.get("model_prediction", ""),
                    "corrected_prediction": row.get("corrected_prediction", ""),
                    "review_decision": row.get("review_decision", ""),
                    "labels_or_clean": row.get("labels_or_clean", ""),
                    "n_labels": row.get("n_labels", ""),
                    "missed_errors": row.get("missed_errors", ""),
                    "well-formed": row.get("well-formed", ""),
                    "input_annotation": row.get("input_annotation", ""),
                    "notes": row.get("notes", ""),
                }
            )

    return {
        "review_sheet": str(review_path),
        "answer_key": str(key_path),
    }


def main() -> None:
    args = parse_args()
    input_csv = resolve_path(args.input_csv)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_csv(input_csv)
    eligible = [row for row in all_rows if is_tagged(row)]
    if args.reviewed_only:
        eligible = [row for row in eligible if is_reviewed(row)]
    if args.review_decision:
        allowed = {value.strip().upper() for value in args.review_decision if value.strip()}
        eligible = [row for row in eligible if review_decision(row) in allowed]

    if not eligible:
        raise ValueError("No eligible tagged rows found with the current filters.")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")
    if args.sample_size > len(eligible):
        raise ValueError(
            f"Requested sample_size={args.sample_size}, but only {len(eligible)} eligible rows are available."
        )

    rng = random.Random(args.seed)
    sampled = rng.sample(eligible, args.sample_size)
    sampled.sort(key=lambda row: row.get("review_id", ""))

    paths = build_annotation_sheet(sampled, out_dir)
    summary = {
        "input_csv": str(input_csv),
        "eligible_rows": len(eligible),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "reviewed_only": args.reviewed_only,
        "review_decision_filter": sorted({value.strip().upper() for value in args.review_decision if value.strip()}),
        "review_sheet": paths["review_sheet"],
        "answer_key": paths["answer_key"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
