from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from common import iter_jsonl, resolve_path


TAG_RE = re.compile(r"\[\*\s*[^\]]+\]")


def canonical_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    return f"[* {body}]"


def extract_tags(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [canonical_tag(tag) for tag in TAG_RE.findall(text)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare manual reannotations against the frozen original training outputs."
    )
    parser.add_argument(
        "--reannotations",
        default=(
            "studies/02_uncertainty_and_feedback/audits/"
            "training_missing_error_audit_real_only_v1/manual_reannotation.jsonl"
        ),
        help="Manual reannotation JSONL with row_id and reannotated_output filled in.",
    )
    parser.add_argument(
        "--answer-key-jsonl",
        default=(
            "studies/02_uncertainty_and_feedback/audits/"
            "training_missing_error_audit_real_only_v1/manual_reannotation_answer_key.jsonl"
        ),
        help="Hidden answer key JSONL built from the frozen training split.",
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "studies/02_uncertainty_and_feedback/audits/"
            "training_missing_error_audit_real_only_v1/manual_reannotation_comparison"
        ),
        help="Directory for comparison outputs.",
    )
    return parser.parse_args()


def load_jsonl_by_row_id(path: Path) -> Dict[str, Dict]:
    return {str(row.get("row_id")): row for row in iter_jsonl(path)}


def write_csv(path: Path, rows: Iterable[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    reannotations_path = resolve_path(args.reannotations)
    answer_key_path = resolve_path(args.answer_key_jsonl)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reannotations = load_jsonl_by_row_id(reannotations_path)
    answer_key = load_jsonl_by_row_id(answer_key_path)

    compared_rows: List[Dict] = []
    counts = Counter()

    for row_id, gold in answer_key.items():
        reannotated = reannotations.get(row_id)
        if reannotated is None:
            counts["missing_reannotation_row"] += 1
            continue

        proposed = (reannotated.get("reannotated_output") or "").strip()
        if not proposed:
            counts["blank_reannotation"] += 1
            continue

        original = (gold.get("original_output") or "").strip()
        original_tags = sorted(set(extract_tags(original)))
        proposed_tags = sorted(set(extract_tags(proposed)))
        extra_proposed_tags = sorted(set(proposed_tags) - set(original_tags))
        missing_proposed_tags = sorted(set(original_tags) - set(proposed_tags))

        exact_match = proposed == original
        tag_set_match = set(proposed_tags) == set(original_tags)

        if exact_match:
            counts["exact_match"] += 1
        else:
            counts["changed_output"] += 1

        if tag_set_match:
            counts["tag_set_match"] += 1
        else:
            counts["tag_set_changed"] += 1

        if not original_tags and proposed_tags:
            counts["clean_to_tagged"] += 1
        if original_tags and not proposed_tags:
            counts["tagged_to_clean"] += 1
        if extra_proposed_tags:
            counts["extra_proposed_tags"] += 1
        if missing_proposed_tags:
            counts["missing_original_tags"] += 1

        compared_rows.append(
            {
                "row_id": row_id,
                "provenance_label": gold.get("provenance_label", ""),
                "trace_method": gold.get("trace_method", ""),
                "error_count_original": gold.get("error_count", 0),
                "input": gold.get("input", ""),
                "original_output": original,
                "reannotated_output": proposed,
                "exact_match": exact_match,
                "tag_set_match": tag_set_match,
                "original_tags": "|".join(original_tags),
                "reannotated_tags": "|".join(proposed_tags),
                "extra_proposed_tags": "|".join(extra_proposed_tags),
                "missing_original_tags": "|".join(missing_proposed_tags),
                "notes": reannotated.get("notes", ""),
            }
        )

    comparison_csv = out_dir / "comparison.csv"
    write_csv(
        comparison_csv,
        compared_rows,
        fieldnames=[
            "row_id",
            "provenance_label",
            "trace_method",
            "error_count_original",
            "input",
            "original_output",
            "reannotated_output",
            "exact_match",
            "tag_set_match",
            "original_tags",
            "reannotated_tags",
            "extra_proposed_tags",
            "missing_original_tags",
            "notes",
        ],
    )

    summary = {
        "reannotations": str(reannotations_path),
        "answer_key_jsonl": str(answer_key_path),
        "rows_in_answer_key": len(answer_key),
        "rows_with_reannotation": len(compared_rows),
        "blank_reannotation": counts["blank_reannotation"],
        "missing_reannotation_row": counts["missing_reannotation_row"],
        "exact_match": counts["exact_match"],
        "changed_output": counts["changed_output"],
        "tag_set_match": counts["tag_set_match"],
        "tag_set_changed": counts["tag_set_changed"],
        "clean_to_tagged": counts["clean_to_tagged"],
        "tagged_to_clean": counts["tagged_to_clean"],
        "extra_proposed_tags": counts["extra_proposed_tags"],
        "missing_original_tags": counts["missing_original_tags"],
        "comparison_csv": str(comparison_csv),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
