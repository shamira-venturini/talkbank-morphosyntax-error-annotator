from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from common import iter_jsonl, resolve_path, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build utterance-only audit input JSONL from the frozen training split."
    )
    parser.add_argument(
        "--train-split",
        default="experiments/recon_full_comp_preserve/stage3_train.jsonl",
        help="Frozen stage3_train JSONL from the final experiment package.",
    )
    parser.add_argument(
        "--out-jsonl",
        default=(
            "studies/02_uncertainty_and_feedback/audits/"
            "training_missing_error_audit_real_only_v1/input_utterance_only.jsonl"
        ),
        help="Output JSONL in the format expected by run_ood_context_inference.py.",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        default=False,
        help="Include synthetic rows. Default keeps only original-corpus real rows.",
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
        help="Keep only rows whose gold annotation has zero in-scope error tags.",
    )
    return parser.parse_args()


def basename_or_empty(values: Iterable[str]) -> str:
    for value in values:
        if value:
            return Path(value).name
    return ""


def to_audit_row(index: int, row: Dict) -> Dict:
    source_files = row.get("source_files") or []
    file_path = source_files[0] if source_files else ""
    file_name = basename_or_empty(source_files)
    return {
        "row_id": row.get("row_id"),
        "file_name": file_name or f"train_row_{row.get('row_id')}.txt",
        "file_path": file_path,
        "speaker": "UNK",
        "line_no": "",
        "utterance_index_raw": index,
        "input": row.get("input", ""),
    }


def should_keep(row: Dict, include_synthetic: bool, include_ambiguous: bool, zero_error_only: bool) -> bool:
    if not include_synthetic and row.get("provenance_label") == "synthetic":
        return False
    if not include_ambiguous and bool(row.get("trace_ambiguous", False)):
        return False
    if zero_error_only and int(row.get("error_count", 0) or 0) != 0:
        return False
    return True


def main() -> None:
    args = parse_args()
    train_split = resolve_path(args.train_split)
    out_jsonl = resolve_path(args.out_jsonl)

    source_rows = list(iter_jsonl(train_split))
    kept_rows: List[Dict] = []
    for idx, row in enumerate(source_rows, start=1):
        if should_keep(
            row,
            include_synthetic=args.include_synthetic,
            include_ambiguous=args.include_ambiguous,
            zero_error_only=args.zero_error_only,
        ):
            kept_rows.append(to_audit_row(idx, row))

    write_jsonl(out_jsonl, kept_rows)

    summary = {
        "train_split": str(train_split),
        "out_jsonl": str(out_jsonl),
        "rows_total": len(source_rows),
        "rows_written": len(kept_rows),
        "include_synthetic": args.include_synthetic,
        "include_ambiguous": args.include_ambiguous,
        "zero_error_only": args.zero_error_only,
    }
    summary_path = out_jsonl.with_name("input_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
