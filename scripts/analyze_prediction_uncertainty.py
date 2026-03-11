from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from common import iter_jsonl, resolve_path


TAG_RE = __import__("re").compile(r"\[\*\s*[ms](?::[^\]]+)?\]")
KNOWN_UNCERTAINTY_FIELDS = [
    "uncertainty_seq_logprob",
    "uncertainty_mean_token_logprob",
    "uncertainty_min_token_logprob",
    "uncertainty_mean_token_margin",
    "uncertainty_min_token_margin",
]


def canonical_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    return f"[* {body}]"


def tag_set(text: str) -> set[str]:
    if not isinstance(text, str):
        return set()
    return {canonical_tag(tag) for tag in TAG_RE.findall(text)}


def has_uncertainty_fields(row: Dict) -> bool:
    return any(field in row for field in KNOWN_UNCERTAINTY_FIELDS)


def exact_text_correct(row: Dict) -> int:
    return int((row.get("human_gold") or "").strip() == (row.get("model_prediction") or "").strip())


def exact_tag_set_correct(row: Dict) -> int:
    return int(tag_set(row.get("human_gold", "")) == tag_set(row.get("model_prediction", "")))


def tag_micro_components(row: Dict) -> Dict[str, int]:
    gold = tag_set(row.get("human_gold", ""))
    pred = tag_set(row.get("model_prediction", ""))
    return {
        "tp": len(gold & pred),
        "fp": len(pred - gold),
        "fn": len(gold - pred),
    }


def pairwise_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def ranked_bins(values: Sequence[float], n_bins: int) -> List[int]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    bins = [0] * len(values)
    for rank, idx in enumerate(order):
        bins[idx] = min(n_bins - 1, (rank * n_bins) // max(1, len(values)))
    return bins


def summarize_items(items: Sequence[Dict], metric: str, correctness_key: str, n_bins: int) -> Dict:
    usable = [item for item in items if item.get(metric) is not None]
    if not usable:
        return {
            "metric": metric,
            "n_items": 0,
            "n_bins": n_bins,
            "auc": None,
            "mean_correct": None,
            "mean_incorrect": None,
            "top_quartile_accuracy": None,
            "bottom_quartile_accuracy": None,
            "bins": [],
        }

    scores = [float(item[metric]) for item in usable]
    labels = [int(item[correctness_key]) for item in usable]
    bins = ranked_bins(scores, n_bins)
    grouped: List[Dict] = []
    for bin_idx in range(n_bins):
        bucket = [usable[idx] for idx, assigned in enumerate(bins) if assigned == bin_idx]
        if not bucket:
            continue
        grouped.append(
            {
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

    top_cut = max(1, len(usable) // 4)
    sorted_usable = sorted(usable, key=lambda item: float(item[metric]))
    bottom = sorted_usable[:top_cut]
    top = sorted_usable[-top_cut:]
    correct_scores = [float(item[metric]) for item in usable if int(item[correctness_key]) == 1]
    incorrect_scores = [float(item[metric]) for item in usable if int(item[correctness_key]) == 0]

    return {
        "metric": metric,
        "n_items": len(usable),
        "n_bins": n_bins,
        "auc": pairwise_auc(scores, labels),
        "mean_correct": mean(correct_scores) if correct_scores else None,
        "mean_incorrect": mean(incorrect_scores) if incorrect_scores else None,
        "top_quartile_accuracy": mean(int(item[correctness_key]) for item in top),
        "bottom_quartile_accuracy": mean(int(item[correctness_key]) for item in bottom),
        "bins": grouped,
    }


def write_csv(path: Path, rows: Iterable[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_item_rows(rows: Sequence[Dict], correctness_key: str) -> List[Dict]:
    items = []
    for row in rows:
        micro = tag_micro_components(row)
        item = {
            "row_id": row.get("row_id"),
            "provenance_label": row.get("provenance_label"),
            "input": row.get("input"),
            "human_gold": row.get("human_gold"),
            "model_prediction": row.get("model_prediction"),
            "exact_text_correct": exact_text_correct(row),
            "exact_tag_set_correct": exact_tag_set_correct(row),
            "tag_tp": micro["tp"],
            "tag_fp": micro["fp"],
            "tag_fn": micro["fn"],
            "gold_tag_count": len(tag_set(row.get("human_gold", ""))),
            "pred_tag_count": len(tag_set(row.get("model_prediction", ""))),
            "uncertainty_num_tokens": row.get("uncertainty_num_tokens"),
        }
        for field in KNOWN_UNCERTAINTY_FIELDS:
            item[field] = row.get(field)
        item["selected_correctness"] = item[correctness_key]
        items.append(item)
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze uncertainty fields saved with evaluation predictions.")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Prediction JSONL with uncertainty fields exported during generation.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/uncertainty_analysis",
        help="Directory for summary JSON and CSV outputs.",
    )
    parser.add_argument(
        "--correctness",
        choices=["exact_tag_set_correct", "exact_text_correct"],
        default="exact_tag_set_correct",
        help="Correctness target to relate uncertainty to.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=4,
        help="Number of ranked confidence bins to report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = resolve_path(args.predictions)
    out_dir = resolve_path(args.out_dir)
    rows = list(iter_jsonl(pred_path))
    if not rows:
        raise SystemExit(f"No prediction rows found in {pred_path}")
    if not has_uncertainty_fields(rows[0]):
        raise SystemExit(
            "Prediction file does not contain uncertainty fields. Re-run evaluation with "
            "`cfg.save_prediction_uncertainty=True` in the generated notebook."
        )

    item_rows = build_item_rows(rows, correctness_key=args.correctness)
    metric_summaries = [
        summarize_items(item_rows, metric=field, correctness_key=args.correctness, n_bins=args.n_bins)
        for field in KNOWN_UNCERTAINTY_FIELDS
    ]
    available = [summary for summary in metric_summaries if summary["n_items"] > 0]

    summary = {
        "predictions_file": str(pred_path),
        "out_dir": str(out_dir),
        "n_rows": len(rows),
        "correctness_target": args.correctness,
        "available_uncertainty_metrics": [summary["metric"] for summary in available],
        "overall_accuracy": mean(int(item[args.correctness]) for item in item_rows),
        "metrics": [
            {
                key: value
                for key, value in summary_row.items()
                if key != "bins"
            }
            for summary_row in available
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "uncertainty_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(
        out_dir / "uncertainty_items.csv",
        item_rows,
        fieldnames=[
            "row_id",
            "provenance_label",
            "exact_text_correct",
            "exact_tag_set_correct",
            "selected_correctness",
            "tag_tp",
            "tag_fp",
            "tag_fn",
            "gold_tag_count",
            "pred_tag_count",
            "uncertainty_num_tokens",
            *KNOWN_UNCERTAINTY_FIELDS,
            "input",
            "human_gold",
            "model_prediction",
        ],
    )
    write_csv(
        out_dir / "uncertainty_bins.csv",
        [row for summary_row in available for row in summary_row["bins"]],
        fieldnames=[
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

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
