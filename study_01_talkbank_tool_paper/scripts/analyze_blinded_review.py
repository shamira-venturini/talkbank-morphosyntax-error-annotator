from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from common import resolve_path


SCORE_FIELDS = [
    "score_correct",
    "score_incorrect",
    "score_ambiguous",
    "score_unsure",
]
DECISION_MAP = {
    "score_correct": "correct",
    "score_incorrect": "incorrect",
    "score_ambiguous": "ambiguous",
    "score_unsure": "unsure",
}
FALSEY_VALUES = {"", "0", "false", "f", "no", "n", "none", "na", "n/a"}


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def marked(value: object) -> bool:
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized not in FALSEY_VALUES


def parse_review_decision(row: Dict[str, str]) -> str:
    active = [field for field in SCORE_FIELDS if marked(row.get(field, ""))]
    if not active:
        return "unscored"
    if len(active) > 1:
        return "invalid_multiple"
    return DECISION_MAP[active[0]]


def merge_review_and_key(review_rows: Sequence[Dict[str, str]], key_rows: Sequence[Dict[str, str]]) -> List[Dict]:
    review_by_id = {row["review_id"]: row for row in review_rows}
    merged: List[Dict] = []
    for key_row in key_rows:
        review = review_by_id.get(key_row["review_id"])
        if review is None:
            continue
        decision = parse_review_decision(review)
        merged.append(
            {
                "review_id": review["review_id"],
                "utterance_id": review.get("utterance_id", ""),
                "row_id": key_row.get("row_id", ""),
                "source": key_row.get("source", ""),
                "original_category": key_row.get("original_category", ""),
                "decision": decision,
                "input": review.get("input", ""),
                "candidate_annotation": review.get("candidate_annotation", ""),
                "notes": review.get("notes", ""),
                "score_correct": review.get("score_correct", ""),
                "score_incorrect": review.get("score_incorrect", ""),
                "score_ambiguous": review.get("score_ambiguous", ""),
                "score_unsure": review.get("score_unsure", ""),
            }
        )
    return merged


def summarize_counts(
    rows: Sequence[Dict],
    group_fields: Sequence[str],
    label_field: str,
    labels: Sequence[str],
) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    out: List[Dict] = []
    for key, bucket in sorted(grouped.items()):
        counts = Counter(row.get(label_field, "") for row in bucket)
        total = len(bucket)
        base = {field: value for field, value in zip(group_fields, key)}
        for decision in labels:
            base[f"{decision}_count"] = counts.get(decision, 0)
            base[f"{decision}_rate"] = round(counts.get(decision, 0) / total, 6) if total else 0.0
        base["n"] = total
        out.append(base)
    return out


def collapse_source_decisions(rows: Sequence[Dict]) -> str:
    decisions = {row.get("decision", "") for row in rows}
    if len(decisions) == 1:
        return next(iter(decisions))
    if "correct" in decisions and "incorrect" in decisions:
        return "mixed_correct_incorrect"
    if "ambiguous" in decisions:
        return "ambiguous"
    if "unsure" in decisions:
        return "unsure"
    if "invalid_multiple" in decisions:
        return "invalid_multiple"
    if "unscored" in decisions:
        return "unscored"
    return "mixed_other"


def compare_to_gold(model_decision: str, gold_decision: str) -> str:
    unresolved = {
        "ambiguous",
        "unsure",
        "unscored",
        "invalid_multiple",
        "mixed_correct_incorrect",
        "mixed_other",
    }
    if model_decision in unresolved or gold_decision in unresolved:
        return "unresolved_review"
    if model_decision == "correct" and gold_decision == "incorrect":
        return "model_preferred"
    if model_decision == "incorrect" and gold_decision == "correct":
        return "gold_preferred"
    if model_decision == "correct" and gold_decision == "correct":
        return "both_acceptable"
    if model_decision == "incorrect" and gold_decision == "incorrect":
        return "neither_acceptable"
    return "other_mixed"


def build_head_to_head_rows(source_rows: Sequence[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in source_rows:
        grouped[(row.get("utterance_id", ""), row.get("source", ""))].append(row)

    utterance_sources = defaultdict(dict)
    for (utterance_id, source), bucket in grouped.items():
        row0 = bucket[0]
        utterance_sources[utterance_id][source] = {
            "row_id": row0.get("row_id", ""),
            "decision": collapse_source_decisions(bucket),
        }

    out: List[Dict] = []
    for utterance_id, source_map in sorted(utterance_sources.items()):
        gold_info = source_map.get("gold")
        if not gold_info:
            continue
        for source, info in sorted(source_map.items()):
            if source == "gold":
                continue
            out.append(
                {
                    "utterance_id": utterance_id,
                    "row_id": info.get("row_id", "") or gold_info.get("row_id", ""),
                    "source": source,
                    "source_decision": info["decision"],
                    "gold_decision": gold_info["decision"],
                    "pair_outcome": compare_to_gold(info["decision"], gold_info["decision"]),
                }
            )
    return out


def write_csv(path: Path, rows: Iterable[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze_review_bundle(review_sheet: Path, answer_key: Path, out_dir: Path) -> Dict:
    review_rows = load_csv(review_sheet)
    key_rows = load_csv(answer_key)
    merged_rows = merge_review_and_key(review_rows, key_rows)
    source_summary = summarize_counts(
        merged_rows,
        ["source"],
        "decision",
        ["correct", "incorrect", "ambiguous", "unsure", "unscored", "invalid_multiple"],
    )
    category_summary = summarize_counts(
        merged_rows,
        ["source", "original_category"],
        "decision",
        ["correct", "incorrect", "ambiguous", "unsure", "unscored", "invalid_multiple"],
    )
    head_to_head_rows = build_head_to_head_rows(merged_rows)
    head_to_head_summary = (
        summarize_counts(
            head_to_head_rows,
            ["source"],
            "pair_outcome",
            [
                "model_preferred",
                "gold_preferred",
                "both_acceptable",
                "neither_acceptable",
                "unresolved_review",
                "other_mixed",
            ],
        )
        if head_to_head_rows
        else []
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "posthoc_review_items.csv",
        merged_rows,
        [
            "review_id",
            "utterance_id",
            "row_id",
            "source",
            "original_category",
            "decision",
            "input",
            "candidate_annotation",
            "notes",
            "score_correct",
            "score_incorrect",
            "score_ambiguous",
            "score_unsure",
        ],
    )
    write_csv(
        out_dir / "posthoc_review_by_source.csv",
        source_summary,
        [
            "source",
            "n",
            "correct_count",
            "correct_rate",
            "incorrect_count",
            "incorrect_rate",
            "ambiguous_count",
            "ambiguous_rate",
            "unsure_count",
            "unsure_rate",
            "unscored_count",
            "unscored_rate",
            "invalid_multiple_count",
            "invalid_multiple_rate",
        ],
    )
    write_csv(
        out_dir / "posthoc_review_by_category.csv",
        category_summary,
        [
            "source",
            "original_category",
            "n",
            "correct_count",
            "correct_rate",
            "incorrect_count",
            "incorrect_rate",
            "ambiguous_count",
            "ambiguous_rate",
            "unsure_count",
            "unsure_rate",
            "unscored_count",
            "unscored_rate",
            "invalid_multiple_count",
            "invalid_multiple_rate",
        ],
    )

    if head_to_head_rows:
        write_csv(
            out_dir / "posthoc_review_gold_comparison_items.csv",
            head_to_head_rows,
            [
                "utterance_id",
                "row_id",
                "source",
                "source_decision",
                "gold_decision",
                "pair_outcome",
            ],
        )
        write_csv(
            out_dir / "posthoc_review_gold_comparison_summary.csv",
            head_to_head_summary,
            [
                "source",
                "n",
                "model_preferred_count",
                "model_preferred_rate",
                "gold_preferred_count",
                "gold_preferred_rate",
                "both_acceptable_count",
                "both_acceptable_rate",
                "neither_acceptable_count",
                "neither_acceptable_rate",
                "unresolved_review_count",
                "unresolved_review_rate",
                "other_mixed_count",
                "other_mixed_rate",
            ],
        )

    review_decisions = Counter(row["decision"] for row in merged_rows)
    pair_outcomes = Counter(row["pair_outcome"] for row in head_to_head_rows)
    summary = {
        "review_sheet": str(review_sheet),
        "answer_key": str(answer_key),
        "n_review_rows": len(review_rows),
        "n_answer_key_rows": len(key_rows),
        "n_merged_source_rows": len(merged_rows),
        "n_scored_source_rows": sum(
            1 for row in merged_rows if row["decision"] not in {"unscored", "invalid_multiple"}
        ),
        "decision_counts": dict(review_decisions),
        "head_to_head_pair_outcomes": dict(pair_outcomes),
        "sources": sorted({row.get("source", "") for row in merged_rows}),
    }
    (out_dir / "posthoc_review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a completed blinded review sheet plus answer key.")
    parser.add_argument("--review-sheet", required=True, help="CSV created by a blinded review builder.")
    parser.add_argument("--answer-key", required=True, help="Answer key CSV paired with the blinded review sheet.")
    parser.add_argument("--out-dir", default="artifacts/posthoc_review_analysis", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_review_bundle(
        review_sheet=resolve_path(args.review_sheet),
        answer_key=resolve_path(args.answer_key),
        out_dir=resolve_path(args.out_dir),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
