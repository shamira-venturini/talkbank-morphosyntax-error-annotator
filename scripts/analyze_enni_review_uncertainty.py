from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from common import resolve_path


KNOWN_UNCERTAINTY_FIELDS = [
    "uncertainty_mean_token_logprob",
    "uncertainty_min_token_logprob",
    "uncertainty_mean_token_margin",
    "uncertainty_min_token_margin",
]
DEFAULT_REVIEW_DECISIONS = ["CORRECT", "WRONG", "UNSURE", "AMBIGUOUS", "MIXED"]
OPERATING_POINT_TARGETS = [0.9, 0.95, 0.97, 0.99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relate ENNI OOD review uncertainty fields to human-reviewed correctness."
    )
    parser.add_argument(
        "--review-csv",
        default="reviews/ood/enni/enni_review_utterance_only.csv",
        help="Reviewed ENNI utterance-level CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/OOD_eval/enni_ood_chat/uncertainty_review_analysis",
        help="Directory for summary JSON and CSV outputs.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=4,
        help="Number of ranked confidence bins to report per metric.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pairwise_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    paired = sorted(zip(scores, labels), key=lambda item: item[0])
    pos_total = sum(1 for _, label in paired if label == 1)
    neg_total = len(paired) - pos_total
    if not pos_total or not neg_total:
        return None
    wins = 0.0
    neg_seen = 0
    idx = 0
    while idx < len(paired):
        score = paired[idx][0]
        pos_ties = 0
        neg_ties = 0
        while idx < len(paired) and paired[idx][0] == score:
            if paired[idx][1] == 1:
                pos_ties += 1
            else:
                neg_ties += 1
            idx += 1
        wins += pos_ties * neg_seen
        wins += 0.5 * pos_ties * neg_ties
        neg_seen += neg_ties
    return wins / (pos_total * neg_total)


def ranked_bins(values: Sequence[float], n_bins: int) -> List[int]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    bins = [0] * len(values)
    for rank, idx in enumerate(order):
        bins[idx] = min(n_bins - 1, (rank * n_bins) // max(1, len(values)))
    return bins


def safe_int(text: str) -> int:
    text = (text or "").strip()
    return int(text) if text else 0


def safe_float(text: str) -> float | None:
    text = (text or "").strip()
    return float(text) if text else None


def output_family(row: Dict[str, str]) -> str:
    return "tagged_prediction" if safe_int(row.get("n_labels", "")) > 0 else "untagged_prediction"


def strict_correct(review_decision: str) -> int | None:
    decision = (review_decision or "").strip().upper()
    if decision not in DEFAULT_REVIEW_DECISIONS:
        return None
    return int(decision == "CORRECT")


def lenient_correct(review_decision: str) -> int | None:
    decision = (review_decision or "").strip().upper()
    if decision not in DEFAULT_REVIEW_DECISIONS:
        return None
    return int(decision in {"CORRECT", "AMBIGUOUS"})


def build_item_rows(rows: Sequence[Dict[str, str]]) -> List[Dict]:
    items: List[Dict] = []
    for row in rows:
        review = (row.get("review_decision") or "").strip().upper()
        item = {
            "row_id": row.get("row_id", ""),
            "prediction_status": row.get("prediction_status", ""),
            "output_family": output_family(row),
            "labels_or_clean": row.get("labels_or_clean", ""),
            "n_labels": safe_int(row.get("n_labels", "")),
            "review_decision": review,
            "strict_correct": strict_correct(review),
            "lenient_correct": lenient_correct(review),
            "missed_errors": row.get("missed_errors", ""),
            "well_formed": row.get("well-formed", ""),
            "preannotated_input": row.get("preannotated_input", ""),
            "input_annotation": row.get("input_annotation", ""),
            "notes": row.get("notes", ""),
            "input": row.get("input", ""),
            "model_prediction": row.get("model_prediction", ""),
        }
        for field in KNOWN_UNCERTAINTY_FIELDS:
            item[field] = safe_float(row.get(field, ""))
        items.append(item)
    return items


def summarize_metric(
    items: Sequence[Dict],
    metric: str,
    correctness_key: str,
    n_bins: int,
    group_name: str,
) -> Dict:
    usable = [item for item in items if item.get(metric) is not None and item.get(correctness_key) is not None]
    if not usable:
        return {
            "group": group_name,
            "metric": metric,
            "n_items": 0,
            "n_bins": n_bins,
            "auc": None,
            "mean_correct": None,
            "mean_incorrect": None,
            "top_quartile_accuracy": None,
            "bottom_quartile_accuracy": None,
            "bins": [],
            "operating_points": [],
        }

    scores = [float(item[metric]) for item in usable]
    labels = [int(item[correctness_key]) for item in usable]
    assigned_bins = ranked_bins(scores, n_bins)
    grouped_rows: List[Dict] = []
    for bin_idx in range(n_bins):
        bucket = [usable[idx] for idx, assigned in enumerate(assigned_bins) if assigned == bin_idx]
        if not bucket:
            continue
        grouped_rows.append(
            {
                "group": group_name,
                "correctness_target": correctness_key,
                "metric": metric,
                "bin_index": bin_idx,
                "bin_label": f"Q{bin_idx + 1}",
                "n_items": len(bucket),
                "score_min": min(float(item[metric]) for item in bucket),
                "score_max": max(float(item[metric]) for item in bucket),
                "score_mean": mean(float(item[metric]) for item in bucket),
                "accuracy": mean(int(item[correctness_key]) for item in bucket),
            }
        )

    sorted_usable = sorted(usable, key=lambda item: float(item[metric]), reverse=True)
    top_cut = max(1, len(sorted_usable) // 4)
    top = sorted_usable[:top_cut]
    bottom = sorted_usable[-top_cut:]
    correct_scores = [float(item[metric]) for item in usable if int(item[correctness_key]) == 1]
    incorrect_scores = [float(item[metric]) for item in usable if int(item[correctness_key]) == 0]

    operating_points: List[Dict] = []
    prefix_best: Dict[float, Dict] = {}
    running_correct = 0
    for prefix_len, item in enumerate(sorted_usable, start=1):
        running_correct += int(item[correctness_key])
        acc = running_correct / prefix_len
        for target in OPERATING_POINT_TARGETS:
            if acc >= target:
                prefix_best[target] = {
                    "group": group_name,
                    "correctness_target": correctness_key,
                    "metric": metric,
                    "target_accuracy": target,
                    "accepted_n": prefix_len,
                    "accepted_fraction": prefix_len / len(sorted_usable),
                    "accepted_accuracy": acc,
                    "threshold": float(item[metric]),
                }
    for target in OPERATING_POINT_TARGETS:
        best_prefix = prefix_best.get(target)
        if best_prefix is None:
            best_prefix = {
                "group": group_name,
                "correctness_target": correctness_key,
                "metric": metric,
                "target_accuracy": target,
                "accepted_n": 0,
                "accepted_fraction": 0.0,
                "accepted_accuracy": None,
                "threshold": None,
            }
        operating_points.append(best_prefix)

    return {
        "group": group_name,
        "metric": metric,
        "n_items": len(usable),
        "n_bins": n_bins,
        "auc": pairwise_auc(scores, labels),
        "mean_correct": mean(correct_scores) if correct_scores else None,
        "mean_incorrect": mean(incorrect_scores) if incorrect_scores else None,
        "top_quartile_accuracy": mean(int(item[correctness_key]) for item in top),
        "bottom_quartile_accuracy": mean(int(item[correctness_key]) for item in bottom),
        "bins": grouped_rows,
        "operating_points": operating_points,
    }


def group_rows(rows: Sequence[Dict]) -> Dict[str, List[Dict]]:
    return {
        "all_reviewed": list(rows),
        "untagged_prediction": [row for row in rows if row["output_family"] == "untagged_prediction"],
        "tagged_prediction": [row for row in rows if row["output_family"] == "tagged_prediction"],
    }


def build_summary(item_rows: Sequence[Dict], n_bins: int) -> Dict:
    grouped = group_rows(item_rows)
    out = {
        "n_reviewed_rows": len(item_rows),
        "review_decision_counts": {},
        "groups": {},
    }
    for group_name, rows in grouped.items():
        out["groups"][group_name] = {
            "n_rows": len(rows),
            "strict_accuracy": mean(int(row["strict_correct"]) for row in rows if row["strict_correct"] is not None),
            "lenient_accuracy": mean(int(row["lenient_correct"]) for row in rows if row["lenient_correct"] is not None),
            "review_decision_counts": {},
            "metrics": [],
        }
        decisions = {}
        for decision in DEFAULT_REVIEW_DECISIONS:
            count = sum(1 for row in rows if row["review_decision"] == decision)
            if count:
                decisions[decision] = count
        out["groups"][group_name]["review_decision_counts"] = decisions
        for correctness_key in ["strict_correct", "lenient_correct"]:
            metric_summaries = [
                summarize_metric(rows, metric=metric, correctness_key=correctness_key, n_bins=n_bins, group_name=group_name)
                for metric in KNOWN_UNCERTAINTY_FIELDS
            ]
            out["groups"][group_name]["metrics"].append(
                {
                    "correctness_target": correctness_key,
                    "summaries": [
                        {key: value for key, value in summary.items() if key not in {"bins", "operating_points"}}
                        for summary in metric_summaries
                    ],
                }
            )
    return out


def main() -> None:
    args = parse_args()
    review_csv = resolve_path(args.review_csv)
    out_dir = resolve_path(args.out_dir)

    with review_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    item_rows = build_item_rows(rows)
    reviewed_rows = [row for row in item_rows if row["strict_correct"] is not None]
    if not reviewed_rows:
        raise SystemExit(f"No reviewed non-out-of-scope rows found in {review_csv}")

    summary = build_summary(reviewed_rows, n_bins=args.n_bins)
    summary["review_csv"] = str(review_csv)
    summary["out_dir"] = str(out_dir)

    grouped_reviewed_rows = group_rows(reviewed_rows)
    metric_rows: List[Dict] = []
    bin_rows: List[Dict] = []
    operating_point_rows: List[Dict] = []
    for group_name, group_summary in summary["groups"].items():
        for target_block in group_summary["metrics"]:
            correctness_key = target_block["correctness_target"]
            metric_summaries = [
                summarize_metric(
                    grouped_reviewed_rows[group_name],
                    metric=metric,
                    correctness_key=correctness_key,
                    n_bins=args.n_bins,
                    group_name=group_name,
                )
                for metric in KNOWN_UNCERTAINTY_FIELDS
            ]
            metric_rows.extend(
                [
                    {
                        "group": group_name,
                        "correctness_target": correctness_key,
                        **{key: value for key, value in metric_summary.items() if key not in {"bins", "operating_points", "group"}},
                    }
                    for metric_summary in metric_summaries
                ]
            )
            for metric_summary in metric_summaries:
                bin_rows.extend(metric_summary["bins"])
                operating_point_rows.extend(metric_summary["operating_points"])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "uncertainty_review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(
        out_dir / "uncertainty_review_items.csv",
        reviewed_rows,
        fieldnames=[
            "row_id",
            "prediction_status",
            "output_family",
            "labels_or_clean",
            "n_labels",
            "review_decision",
            "strict_correct",
            "lenient_correct",
            "missed_errors",
            "well_formed",
            "preannotated_input",
            "input_annotation",
            "notes",
            *KNOWN_UNCERTAINTY_FIELDS,
            "input",
            "model_prediction",
        ],
    )
    write_csv(
        out_dir / "uncertainty_review_metrics.csv",
        metric_rows,
        fieldnames=[
            "group",
            "correctness_target",
            "metric",
            "n_items",
            "n_bins",
            "auc",
            "mean_correct",
            "mean_incorrect",
            "top_quartile_accuracy",
            "bottom_quartile_accuracy",
        ],
    )
    write_csv(
        out_dir / "uncertainty_review_bins.csv",
        bin_rows,
        fieldnames=[
            "group",
            "correctness_target",
            "metric",
            "bin_index",
            "bin_label",
            "n_items",
            "score_min",
            "score_max",
            "score_mean",
            "accuracy",
        ],
    )
    write_csv(
        out_dir / "uncertainty_review_operating_points.csv",
        operating_point_rows,
        fieldnames=[
            "group",
            "correctness_target",
            "metric",
            "target_accuracy",
            "accepted_n",
            "accepted_fraction",
            "accepted_accuracy",
            "threshold",
        ],
    )
    print(
        json.dumps(
            {
                "review_csv": str(review_csv),
                "out_dir": str(out_dir),
                "n_reviewed_rows": len(reviewed_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
