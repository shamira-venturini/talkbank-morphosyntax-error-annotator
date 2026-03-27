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
RECON_RE = re.compile(r"\[(?::|::)\s+[^\]]+\]")


def canonical_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    return f"[* {body}]"


def extract_tags(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [canonical_tag(tag) for tag in TAG_RE.findall(text)]


def extract_recon_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(RECON_RE.findall(text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze final-model training-set audit predictions for possible missed gold errors."
    )
    parser.add_argument(
        "--train-split",
        default="experiments/recon_full_comp_preserve/stage3_train.jsonl",
        help="Frozen stage3_train JSONL used as the gold reference for the audit.",
    )
    parser.add_argument(
        "--predictions",
        default=(
            "studies/02_uncertainty_and_feedback/audits/"
            "training_missing_error_audit_real_only_v1/inference/predictions_utterance_only.jsonl"
        ),
        help="Prediction JSONL produced by run_ood_context_inference.py.",
    )
    parser.add_argument(
        "--out-dir",
        default="studies/02_uncertainty_and_feedback/audits/training_missing_error_audit_real_only_v1/analysis",
        help="Output directory for CSV and JSON summaries.",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        default=False,
        help="Include synthetic rows in the reference set.",
    )
    parser.add_argument(
        "--include-ambiguous",
        action="store_true",
        default=False,
        help="Include trace_ambiguous rows in the reference set.",
    )
    return parser.parse_args()


def load_jsonl_by_row_id(path: Path) -> Dict[str, Dict]:
    return {str(row.get("row_id")): row for row in iter_jsonl(path)}


def should_keep(row: Dict, include_synthetic: bool, include_ambiguous: bool) -> bool:
    if not include_synthetic and row.get("provenance_label") == "synthetic":
        return False
    if not include_ambiguous and bool(row.get("trace_ambiguous", False)):
        return False
    return True


def write_csv(path: Path, rows: Iterable[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    train_split = resolve_path(args.train_split)
    predictions_path = resolve_path(args.predictions)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_rows = {
        str(row.get("row_id")): row
        for row in iter_jsonl(train_split)
        if should_keep(row, include_synthetic=args.include_synthetic, include_ambiguous=args.include_ambiguous)
    }
    pred_rows = load_jsonl_by_row_id(predictions_path)

    candidates: List[Dict] = []
    summary_counts = Counter()
    extra_tag_counter = Counter()

    for row_id, gold in gold_rows.items():
        pred = pred_rows.get(row_id)
        if pred is None:
            summary_counts["missing_prediction_rows"] += 1
            continue

        gold_tags = sorted(set(extract_tags(gold.get("output", ""))))
        pred_tags = sorted(set(extract_tags(pred.get("model_prediction", ""))))
        extra_pred_tags = sorted(set(pred_tags) - set(gold_tags))
        missing_gold_tags = sorted(set(gold_tags) - set(pred_tags))

        if not extra_pred_tags:
            continue

        if not gold_tags:
            candidate_type = "clean_to_tagged"
        else:
            candidate_type = "tagged_with_extra_prediction"

        for tag in extra_pred_tags:
            extra_tag_counter[tag] += 1

        summary_counts["candidate_rows"] += 1
        summary_counts[candidate_type] += 1

        candidates.append(
            {
                "row_id": row_id,
                "candidate_type": candidate_type,
                "provenance_label": gold.get("provenance_label", ""),
                "trace_method": gold.get("trace_method", ""),
                "trace_ambiguous": gold.get("trace_ambiguous", False),
                "error_count_gold": gold.get("error_count", 0),
                "file_path": (gold.get("source_files") or [""])[0] if gold.get("source_files") else "",
                "input": gold.get("input", ""),
                "gold_output": gold.get("output", ""),
                "model_prediction": pred.get("model_prediction", ""),
                "gold_tags": "|".join(gold_tags),
                "pred_tags": "|".join(pred_tags),
                "extra_pred_tags": "|".join(extra_pred_tags),
                "missing_gold_tags": "|".join(missing_gold_tags),
                "gold_recon_count": extract_recon_count(gold.get("output", "")),
                "pred_recon_count": extract_recon_count(pred.get("model_prediction", "")),
                "uncertainty_mean_token_logprob": pred.get("uncertainty_mean_token_logprob"),
                "uncertainty_min_token_logprob": pred.get("uncertainty_min_token_logprob"),
                "uncertainty_mean_token_margin": pred.get("uncertainty_mean_token_margin"),
                "uncertainty_min_token_margin": pred.get("uncertainty_min_token_margin"),
            }
        )

    candidates.sort(
        key=lambda row: (
            row["candidate_type"],
            row["provenance_label"],
            row["row_id"],
        )
    )

    candidates_csv = out_dir / "missed_error_candidates.csv"
    write_csv(
        candidates_csv,
        candidates,
        fieldnames=[
            "row_id",
            "candidate_type",
            "provenance_label",
            "trace_method",
            "trace_ambiguous",
            "error_count_gold",
            "file_path",
            "input",
            "gold_output",
            "model_prediction",
            "gold_tags",
            "pred_tags",
            "extra_pred_tags",
            "missing_gold_tags",
            "gold_recon_count",
            "pred_recon_count",
            "uncertainty_mean_token_logprob",
            "uncertainty_min_token_logprob",
            "uncertainty_mean_token_margin",
            "uncertainty_min_token_margin",
        ],
    )

    summary = {
        "train_split": str(train_split),
        "predictions": str(predictions_path),
        "rows_in_scope": len(gold_rows),
        "predictions_available": len(pred_rows),
        "candidate_rows": summary_counts["candidate_rows"],
        "clean_to_tagged": summary_counts["clean_to_tagged"],
        "tagged_with_extra_prediction": summary_counts["tagged_with_extra_prediction"],
        "missing_prediction_rows": summary_counts["missing_prediction_rows"],
        "top_extra_pred_tags": [
            {"tag": tag, "count": count}
            for tag, count in extra_tag_counter.most_common(20)
        ],
        "candidate_csv": str(candidates_csv),
        "notes": [
            "This is a training-set audit, not an unbiased evaluation.",
            "Candidate rows are cases where the final model predicted at least one tag not present in the gold training annotation.",
            "Clean-to-tagged rows are the strongest missed-error candidates and should be manually reviewed first.",
        ],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
