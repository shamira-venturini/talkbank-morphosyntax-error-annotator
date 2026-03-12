from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from common import resolve_path
from normalize_ood_review_csv import classify_hallucination, normalize_review_decision


TAG_RE = re.compile(r"\[\*\s*[ms](?::[^\]]+)?\]")
MODES = ["utterance_only", "local_prev", "full_prev", "full_document"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge reviewed OOD rows with all context-mode predictions.")
    parser.add_argument(
        "--review-csv",
        default="results/ood_vercellotti/review_TAGGED_utterance_only.csv",
        help="Reviewed utterance_only CSV.",
    )
    parser.add_argument(
        "--predictions-dir",
        default="results/ood_vercellotti 4",
        help="Directory containing predictions_<mode>.jsonl for all context modes.",
    )
    parser.add_argument(
        "--out-csv",
        default="results/ood_vercellotti/review_TAGGED_cross_mode_assist.csv",
        help="Merged assist CSV output.",
    )
    parser.add_argument(
        "--priority-csv",
        default="results/ood_vercellotti/review_TAGGED_cross_mode_priority.csv",
        help="Priority subset CSV output.",
    )
    parser.add_argument(
        "--summary-json",
        default="results/ood_vercellotti/review_TAGGED_cross_mode_assist_summary.json",
        help="Summary JSON output.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl_lookup(path: Path) -> Dict[str, Dict]:
    lookup: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            lookup[str(row["row_id"])] = row
    return lookup


def write_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def canonical_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    return f"[* {body}]"


def extract_tags(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return sorted(dict.fromkeys(canonical_tag(tag) for tag in TAG_RE.findall(text)))


def normalize_text(text: str) -> str:
    text = (text or "").replace("\xa0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def determine_target_annotation(review_row: Dict[str, str]) -> Tuple[str, str]:
    corrected = normalize_text(review_row.get("corrected_annotation", ""))
    if corrected:
        return corrected, "corrected_annotation"

    decision = normalize_review_decision(review_row.get("review_decision", ""))
    pred = normalize_text(review_row.get("model_prediction", ""))
    hallucinated, _ = classify_hallucination(pred)
    if decision["review_primary_decision"] == "CORRECT" and pred and hallucinated == "0":
        return pred, "utterance_only_reviewed_correct"

    return "", "unavailable"


def compare_prediction_to_target(prediction: str, target: str) -> Tuple[str, str]:
    if not target:
        return "", ""
    pred_norm = normalize_text(prediction)
    target_norm = normalize_text(target)
    exact_match = "1" if pred_norm == target_norm else "0"
    tag_match = "1" if extract_tags(pred_norm) == extract_tags(target_norm) else "0"
    return exact_match, tag_match


def triage_status(row: Dict[str, str], modes_with_exact: List[str], modes_with_tag: List[str], hallucination_modes: List[str]) -> str:
    target_source = row["target_source"]
    if target_source == "unavailable":
        if row["distinct_prediction_count"] != "1" or row["distinct_tagset_count"] != "1":
            return "NEEDS_MANUAL_DISCREPANCY_REVIEW"
        return "NO_TARGET_ALL_MODES_AGREE"

    if row["utterance_only_exact_match_target"] == "1":
        if len(modes_with_exact) == 1:
            return "BASELINE_ONLY_CORRECT"
        return "MULTIPLE_MODES_MATCH_TARGET"

    if modes_with_exact:
        return "ALTERNATIVE_MODE_MATCHES_TARGET"

    if modes_with_tag:
        return "TAGSET_ONLY_MATCH"

    if hallucination_modes:
        return "NO_MATCH_HALLUCINATION_PRESENT"

    return "NO_MODE_MATCHES_TARGET"


def build_rows(review_rows: List[Dict[str, str]], prediction_lookups: Dict[str, Dict[str, Dict]]) -> List[Dict[str, str]]:
    out_rows: List[Dict[str, str]] = []
    for review_row in review_rows:
        row_id = str(review_row["row_id"])
        decision_info = normalize_review_decision(review_row.get("review_decision", ""))
        target_annotation, target_source = determine_target_annotation(review_row)

        merged = dict(review_row)
        merged.update(decision_info)
        merged["target_annotation"] = target_annotation
        merged["target_source"] = target_source
        merged["target_labels"] = " ; ".join(extract_tags(target_annotation))

        predictions_by_mode: Dict[str, str] = {}
        hallucination_modes: List[str] = []
        exact_modes: List[str] = []
        tag_modes: List[str] = []
        distinct_predictions = set()
        distinct_tagsets = set()

        for mode in MODES:
            pred_row = prediction_lookups.get(mode, {}).get(row_id, {})
            prediction = normalize_text(pred_row.get("model_prediction", ""))
            predictions_by_mode[mode] = prediction
            merged[f"{mode}_prediction"] = pred_row.get("model_prediction", "")
            merged[f"{mode}_labels"] = " ; ".join(extract_tags(prediction))
            merged[f"{mode}_mean_token_logprob"] = pred_row.get("uncertainty_mean_token_logprob", "")
            merged[f"{mode}_mean_token_margin"] = pred_row.get("uncertainty_mean_token_margin", "")

            hallucinated, hallucination_type = classify_hallucination(prediction)
            merged[f"{mode}_hallucination_flag"] = hallucinated
            merged[f"{mode}_hallucination_type"] = hallucination_type
            if hallucinated == "1":
                hallucination_modes.append(mode)

            exact_match, tag_match = compare_prediction_to_target(prediction, target_annotation)
            merged[f"{mode}_exact_match_target"] = exact_match
            merged[f"{mode}_tag_match_target"] = tag_match
            if exact_match == "1":
                exact_modes.append(mode)
            if tag_match == "1":
                tag_modes.append(mode)

            distinct_predictions.add(prediction)
            distinct_tagsets.add(tuple(extract_tags(prediction)))

        merged["modes_matching_target_exact"] = " ; ".join(exact_modes)
        merged["modes_matching_target_tags"] = " ; ".join(tag_modes)
        merged["n_modes_matching_target_exact"] = str(len(exact_modes))
        merged["n_modes_matching_target_tags"] = str(len(tag_modes))
        merged["hallucination_modes"] = " ; ".join(hallucination_modes)
        merged["distinct_prediction_count"] = str(len(distinct_predictions))
        merged["distinct_tagset_count"] = str(len(distinct_tagsets))
        merged["cross_mode_discrepancy_flag"] = "1" if len(distinct_predictions) > 1 else "0"
        merged["cross_mode_tag_discrepancy_flag"] = "1" if len(distinct_tagsets) > 1 else "0"
        merged["triage_status"] = triage_status(merged, exact_modes, tag_modes, hallucination_modes)

        out_rows.append(merged)
    return out_rows


def priority_subset(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    priority_statuses = {
        "ALTERNATIVE_MODE_MATCHES_TARGET",
        "TAGSET_ONLY_MATCH",
        "NEEDS_MANUAL_DISCREPANCY_REVIEW",
        "NO_MODE_MATCHES_TARGET",
        "NO_MATCH_HALLUCINATION_PRESENT",
    }
    return [row for row in rows if row.get("triage_status") in priority_statuses]


def build_summary(rows: List[Dict[str, str]]) -> Dict[str, object]:
    return {
        "rows": len(rows),
        "target_available_rows": sum(1 for row in rows if row.get("target_source") != "unavailable"),
        "triage_status_counts": dict(Counter(row.get("triage_status", "") for row in rows)),
        "rows_with_cross_mode_discrepancy": sum(1 for row in rows if row.get("cross_mode_discrepancy_flag") == "1"),
        "rows_with_cross_mode_tag_discrepancy": sum(1 for row in rows if row.get("cross_mode_tag_discrepancy_flag") == "1"),
        "rows_with_hallucination_in_any_mode": sum(1 for row in rows if (row.get("hallucination_modes") or "").strip()),
        "exact_target_match_counts_by_mode": {
            mode: sum(1 for row in rows if row.get(f"{mode}_exact_match_target") == "1") for mode in MODES
        },
        "tag_target_match_counts_by_mode": {
            mode: sum(1 for row in rows if row.get(f"{mode}_tag_match_target") == "1") for mode in MODES
        },
    }


def main() -> None:
    args = parse_args()
    review_csv = resolve_path(args.review_csv)
    predictions_dir = resolve_path(args.predictions_dir)
    out_csv = resolve_path(args.out_csv)
    priority_csv = resolve_path(args.priority_csv)
    summary_json = resolve_path(args.summary_json)

    review_rows = load_csv(review_csv)
    reviewed_tagged_rows = [row for row in review_rows if row.get("prediction_status") == "TAGGED"]

    prediction_lookups: Dict[str, Dict[str, Dict]] = {}
    for mode in MODES:
        path = predictions_dir / f"predictions_{mode}.jsonl"
        if path.exists():
            prediction_lookups[mode] = load_jsonl_lookup(path)
        else:
            prediction_lookups[mode] = {}

    merged_rows = build_rows(reviewed_tagged_rows, prediction_lookups)
    priority_rows = priority_subset(merged_rows)

    fieldnames = list(merged_rows[0].keys()) if merged_rows else []
    write_csv(out_csv, merged_rows, fieldnames)
    write_csv(priority_csv, priority_rows, fieldnames)

    summary = build_summary(merged_rows)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out_csv": str(out_csv), "priority_csv": str(priority_csv), "summary_json": str(summary_json), **summary}, indent=2))


if __name__ == "__main__":
    main()
