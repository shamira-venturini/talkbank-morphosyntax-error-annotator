from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from common import resolve_path
from normalize_ood_review_csv import classify_hallucination, normalize_review_decision


MODES = ["utterance_only", "local_prev", "full_prev", "full_document"]
TAG_RE = re.compile(r"\[\*\s*[ms](?::[^\]]+)?\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single master OOD review CSV across all context modes.")
    parser.add_argument(
        "--predictions-dir",
        default="results/ood_vercellotti 4",
        help="Directory containing predictions_<mode>.jsonl files.",
    )
    parser.add_argument(
        "--utterance-only-review-csv",
        default="results/ood_vercellotti/review_TAGGED_utterance_only.csv",
        help="Existing reviewed utterance_only CSV.",
    )
    parser.add_argument(
        "--suspicious-csv",
        default="results/ood_vercellotti/suspicious_missed_errors_utterance_only.csv",
        help="Suspicious missed-error shortlist CSV.",
    )
    parser.add_argument(
        "--out-csv",
        default="results/ood_vercellotti/master_review_union.csv",
        help="Master review CSV path.",
    )
    parser.add_argument(
        "--priority-csv",
        default="results/ood_vercellotti/master_review_union_priority.csv",
        help="Priority subset CSV path.",
    )
    parser.add_argument(
        "--focus-csv",
        default="results/ood_vercellotti/master_review_union_focus.csv",
        help="Focused subset CSV path for immediate manual review.",
    )
    parser.add_argument(
        "--summary-json",
        default="results/ood_vercellotti/master_review_union_summary.json",
        help="Summary JSON path.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl_lookup(path: Path) -> Dict[str, Dict]:
    rows: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[str(row["row_id"])] = row
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: Sequence[str]) -> None:
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


def priority_bucket(score: int) -> str:
    if score >= 8:
        return "A"
    if score >= 5:
        return "B"
    return "C"


def build_master_rows(
    prediction_lookups: Dict[str, Dict[str, Dict]],
    utterance_review_rows: Dict[str, Dict[str, str]],
    suspicious_rows: Dict[str, Dict[str, str]],
) -> List[Dict[str, str]]:
    tagged_union = set()
    for mode in MODES:
        tagged_union.update(
            row_id
            for row_id, row in prediction_lookups[mode].items()
            if extract_tags(row.get("model_prediction", ""))
        )
    suspicious_ids = set(suspicious_rows)
    row_ids = sorted(tagged_union | suspicious_ids, key=lambda x: int(x))

    master_rows: List[Dict[str, str]] = []
    for row_id in row_ids:
        base_row = prediction_lookups["utterance_only"].get(row_id)
        if base_row is None:
            # Fall back to any available mode row for metadata.
            for mode in MODES[1:]:
                base_row = prediction_lookups[mode].get(row_id)
                if base_row is not None:
                    break
        if base_row is None:
            continue

        review_row = utterance_review_rows.get(row_id, {})
        suspicious_row = suspicious_rows.get(row_id, {})
        decision_info = normalize_review_decision(review_row.get("review_decision", ""))

        out: Dict[str, str] = {
            "row_id": row_id,
            "file_name": str(base_row.get("file_name", "")),
            "speaker": str(base_row.get("speaker", "")),
            "line_no": str(base_row.get("line_no", "")),
            "utterance_index_raw": str(base_row.get("utterance_index_raw", "")),
            "input": str(base_row.get("input", "")),
            "included_because_tagged": "1" if row_id in tagged_union else "0",
            "included_because_suspicious": "1" if row_id in suspicious_ids else "0",
            "existing_utterance_only_review_decision": review_row.get("review_decision", ""),
            "existing_utterance_only_review_normalized": decision_info["review_decision_normalized"],
            "existing_utterance_only_corrected_annotation": review_row.get("corrected_annotation", ""),
            "existing_utterance_only_notes": review_row.get("notes", ""),
            "suspicious_score": suspicious_row.get("suspicion_score", ""),
            "suspicious_flags": suspicious_row.get("heuristic_flags", ""),
        }

        tagged_modes: List[str] = []
        hallucination_modes: List[str] = []
        distinct_predictions = set()
        distinct_tagsets = set()
        priority_score = 0
        priority_reasons: List[str] = []

        for mode in MODES:
            pred_row = prediction_lookups[mode].get(row_id, {})
            prediction = normalize_text(pred_row.get("model_prediction", ""))
            labels = extract_tags(prediction)
            hall_flag, hall_type = classify_hallucination(prediction)

            out[f"{mode}_prediction"] = str(pred_row.get("model_prediction", ""))
            out[f"{mode}_labels"] = " ; ".join(labels)
            out[f"{mode}_tagged"] = "1" if labels else "0"
            out[f"{mode}_hallucination_flag"] = hall_flag
            out[f"{mode}_hallucination_type"] = hall_type
            out[f"{mode}_mean_token_logprob"] = str(pred_row.get("uncertainty_mean_token_logprob", ""))
            out[f"{mode}_mean_token_margin"] = str(pred_row.get("uncertainty_mean_token_margin", ""))

            if labels:
                tagged_modes.append(mode)
            if hall_flag == "1":
                hallucination_modes.append(mode)

            distinct_predictions.add(prediction)
            distinct_tagsets.add(tuple(labels))

        if len(set(tuple(extract_tags(normalize_text(prediction_lookups[mode].get(row_id, {}).get("model_prediction", "")))) for mode in MODES)) > 1:
            priority_score += 3
            priority_reasons.append("TAGSET_DISAGREEMENT")

        if len(distinct_predictions) > 1:
            priority_score += 2
            priority_reasons.append("SURFACE_DISAGREEMENT")

        if hallucination_modes:
            priority_score += 3
            priority_reasons.append("HALLUCINATION_PRESENT")

        if row_id in suspicious_ids:
            priority_score += 4
            priority_reasons.append("SUSPICIOUS_MISSED_ERROR")

        primary_decision = decision_info["review_primary_decision"]
        if primary_decision in {"WRONG", "PARTIAL_MIXED", "UNCERTAIN", "AMBIGUOUS"}:
            priority_score += 4
            priority_reasons.append("BASELINE_REVIEW_NOT_SETTLED")

        if primary_decision == "CORRECT":
            priority_score += 1
            priority_reasons.append("BASELINE_ALREADY_REVIEWED_CORRECT")

        if "utterance_only" not in tagged_modes and any(mode != "utterance_only" for mode in tagged_modes):
            priority_score += 3
            priority_reasons.append("OTHER_MODE_ONLY_TAGGED")

        if len(tagged_modes) == 1 and tagged_modes[0] == "utterance_only":
            priority_score += 1
            priority_reasons.append("BASELINE_ONLY_TAGGED")

        out["tagged_modes"] = " ; ".join(tagged_modes)
        out["n_tagged_modes"] = str(len(tagged_modes))
        out["hallucination_modes"] = " ; ".join(hallucination_modes)
        out["n_hallucination_modes"] = str(len(hallucination_modes))
        out["distinct_prediction_count"] = str(len(distinct_predictions))
        out["distinct_tagset_count"] = str(len(distinct_tagsets))
        out["priority_score"] = str(priority_score)
        out["priority_bucket"] = priority_bucket(priority_score)
        out["priority_reasons"] = " ; ".join(dict.fromkeys(priority_reasons))
        # Carry forward previously reviewed corrected outputs to avoid duplicate manual work.
        existing_corrected = normalize_text(review_row.get("corrected_annotation", ""))
        if existing_corrected:
            out["final_corrected_annotation"] = review_row.get("corrected_annotation", "")
            out["final_corrected_prefill_source"] = "existing_utterance_only_corrected_annotation"
        elif decision_info["review_primary_decision"] == "CORRECT":
            out["final_corrected_annotation"] = out["utterance_only_prediction"]
            out["final_corrected_prefill_source"] = "utterance_only_prediction_from_correct_review"
        else:
            out["final_corrected_annotation"] = ""
            out["final_corrected_prefill_source"] = ""

        out["final_review_decision"] = ""
        out["preferred_mode"] = ""
        out["final_notes"] = ""

        master_rows.append(out)

    master_rows.sort(
        key=lambda row: (
            row["priority_bucket"],
            -int(row["priority_score"]),
            row["file_name"],
            int(row["line_no"] or 0),
        )
    )
    return master_rows


def build_summary(rows: List[Dict[str, str]]) -> Dict[str, object]:
    return {
        "rows": len(rows),
        "priority_bucket_counts": dict(Counter(row["priority_bucket"] for row in rows)),
        "priority_reason_counts": dict(
            Counter(
                reason
                for row in rows
                for reason in [part.strip() for part in row["priority_reasons"].split(";")]
                if reason
            )
        ),
        "tagged_mode_count_distribution": dict(Counter(row["n_tagged_modes"] for row in rows)),
        "included_because_suspicious_count": sum(1 for row in rows if row["included_because_suspicious"] == "1"),
        "included_because_tagged_count": sum(1 for row in rows if row["included_because_tagged"] == "1"),
    }


def priority_subset(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [row for row in rows if row["priority_bucket"] in {"A", "B"}]


def focus_subset(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        row
        for row in rows
        if row["existing_utterance_only_review_decision"]
        or row["included_because_suspicious"] == "1"
        or row["hallucination_modes"]
    ]


def main() -> None:
    args = parse_args()
    predictions_dir = resolve_path(args.predictions_dir)
    utterance_only_review_csv = resolve_path(args.utterance_only_review_csv)
    suspicious_csv = resolve_path(args.suspicious_csv)
    out_csv = resolve_path(args.out_csv)
    priority_csv = resolve_path(args.priority_csv)
    focus_csv = resolve_path(args.focus_csv)
    summary_json = resolve_path(args.summary_json)

    prediction_lookups = {
        mode: load_jsonl_lookup(predictions_dir / f"predictions_{mode}.jsonl")
        for mode in MODES
    }
    utterance_review_rows = {
        row["row_id"]: row for row in load_csv(utterance_only_review_csv)
    }
    suspicious_rows = {
        row["row_id"]: row for row in load_csv(suspicious_csv)
    }

    rows = build_master_rows(prediction_lookups, utterance_review_rows, suspicious_rows)
    priority_rows = priority_subset(rows)
    focus_rows = focus_subset(rows)

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(out_csv, rows, fieldnames)
    write_csv(priority_csv, priority_rows, fieldnames)
    write_csv(focus_csv, focus_rows, fieldnames)

    summary = build_summary(rows)
    summary["priority_rows"] = len(priority_rows)
    summary["focus_rows"] = len(focus_rows)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_csv": str(out_csv),
                "priority_csv": str(priority_csv),
                "focus_csv": str(focus_csv),
                "summary_json": str(summary_json),
                **summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
