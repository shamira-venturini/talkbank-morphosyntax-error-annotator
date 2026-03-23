from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from common import iter_jsonl, resolve_path, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a blind manual-reannotation JSONL from the frozen final training split."
    )
    parser.add_argument(
        "--train-split",
        default="experiments/recon_full_comp_preserve/stage3_train.jsonl",
        help="Frozen stage3_train JSONL.",
    )
    parser.add_argument(
        "--out-jsonl",
        default=(
            "studies/02_uncertainty_and_feedback/audits/"
            "training_missing_error_audit_real_only_v1/manual_reannotation.jsonl"
        ),
        help="Blind JSONL to fill in manually.",
    )
    parser.add_argument(
        "--answer-key-jsonl",
        default=(
            "studies/02_uncertainty_and_feedback/audits/"
            "training_missing_error_audit_real_only_v1/manual_reannotation_answer_key.jsonl"
        ),
        help="Hidden answer key with the original frozen outputs.",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        default=False,
        help="Include synthetic rows.",
    )
    parser.add_argument(
        "--include-ambiguous",
        action="store_true",
        default=False,
        help="Include trace_ambiguous rows.",
    )
    parser.add_argument(
        "--zero-error-only",
        action="store_true",
        default=False,
        help="Keep only rows whose original frozen label is clean.",
    )
    return parser.parse_args()


def should_keep(row: Dict, include_synthetic: bool, include_ambiguous: bool, zero_error_only: bool) -> bool:
    if not include_synthetic and row.get("provenance_label") == "synthetic":
        return False
    if not include_ambiguous and bool(row.get("trace_ambiguous", False)):
        return False
    if zero_error_only and int(row.get("error_count", 0) or 0) != 0:
        return False
    return True


def build_review_row(row: Dict) -> Dict:
    return {
        "row_id": row.get("row_id"),
        "input": row.get("input", ""),
        "reannotated_output": "",
        "notes": "",
    }


def build_answer_key_row(row: Dict) -> Dict:
    return {
        "row_id": row.get("row_id"),
        "input": row.get("input", ""),
        "original_output": row.get("output", ""),
        "provenance_label": row.get("provenance_label", ""),
        "trace_method": row.get("trace_method", ""),
        "trace_ambiguous": row.get("trace_ambiguous", False),
        "error_count": row.get("error_count", 0),
        "source_files": row.get("source_files", []),
    }


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    train_split = resolve_path(args.train_split)
    out_jsonl = resolve_path(args.out_jsonl)
    answer_key_jsonl = resolve_path(args.answer_key_jsonl)

    source_rows = [
        row
        for row in iter_jsonl(train_split)
        if should_keep(
            row,
            include_synthetic=args.include_synthetic,
            include_ambiguous=args.include_ambiguous,
            zero_error_only=args.zero_error_only,
        )
    ]

    review_rows: List[Dict] = [build_review_row(row) for row in source_rows]
    answer_key_rows: List[Dict] = [build_answer_key_row(row) for row in source_rows]
    write_jsonl(out_jsonl, review_rows)
    write_jsonl(answer_key_jsonl, answer_key_rows)
    review_csv = out_jsonl.with_suffix(".csv")
    answer_key_csv = answer_key_jsonl.with_suffix(".csv")
    write_csv(
        review_csv,
        review_rows,
        fieldnames=["row_id", "input", "reannotated_output", "notes"],
    )
    write_csv(
        answer_key_csv,
        answer_key_rows,
        fieldnames=[
            "row_id",
            "input",
            "original_output",
            "provenance_label",
            "trace_method",
            "trace_ambiguous",
            "error_count",
            "source_files",
        ],
    )

    summary = {
        "train_split": str(train_split),
        "out_jsonl": str(out_jsonl),
        "answer_key_jsonl": str(answer_key_jsonl),
        "out_csv": str(review_csv),
        "answer_key_csv": str(answer_key_csv),
        "rows_written": len(review_rows),
        "include_synthetic": args.include_synthetic,
        "include_ambiguous": args.include_ambiguous,
        "zero_error_only": args.zero_error_only,
    }
    out_jsonl.with_name("manual_reannotation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
