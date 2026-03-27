from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from common import resolve_path


PROMPT_LEAKAGE_PATTERNS: List[Tuple[str, str]] = [
    ("PROMPT_MARKER", r"###\s*(Instruction|Input|Response)\s*:"),
    ("PROMPT_RULE_COPY", r"Output exactly one annotated utterance line"),
    ("PROMPT_RULE_COPY", r"licensed CHAT tags"),
    ("PROMPT_RULE_COPY", r"unsupported combinations"),
    ("PROMPT_RULE_COPY", r"do not invent unattested"),
    ("PROMPT_RULE_COPY", r"Task:\s*Annotate the input utterance"),
]

NOTE_CATEGORY_PATTERNS: List[Tuple[str, List[str]]] = [
    ("ADDED_ERROR", ["ADDED ERROR", "ADDED ERROR LABEL", "ADDED ERROR LABELS", "ADDED (1)", "ADDED (", "CORRECT BUT ADDS ERROR"]),
    ("MISSED_ERROR", ["MISSED", "MISSING ONE LABEL", "ONE MISSING ERROR LABEL", "PARTIAL_MISSING"]),
    ("WRONG_LABEL", ["WRONG LABEL", "good logic but no", "wrong correction correct label"]),
    ("WRONG_RECONSTRUCTION", ["RECONSTRUCTION WRONG", "ADDED RECONSTRUCTIONS", "CORRECT LABEL BUT RECONSTRUCTION IS WRONG", "MISSING_RECONSTRUCTION"]),
    ("FRAGMENT_CORRECTION", ["MODEL CORRECTS FRAGMENTS", "LABELLED FRAGMENT"]),
    ("NO_ERROR", ["NO ERROR HERE", "no error label needed", "NO LABEL NECESSARY"]),
    ("UNSUPPORTED_ERROR_TYPE", ["DOESNT HAVE THE TOOLS", "DOESN'T HAVE THE TOOLS", "doesnt have the tools", "did not know how to tag", "cant label", "can't label", "WE DONT HAVE A LABEL", "proxy", "closest tag"]),
    ("CONTEXT_DEPENDENT", ["CONTEXT DEPENDENT"]),
    ("AMBIGUOUS_CASE", ["AMBIGUOUS"]),
    ("DOUBLE_LABEL", ["double label"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize manual OOD review CSV labels and notes.")
    parser.add_argument(
        "--input-csv",
        default="results/ood_vercellotti/review_TAGGED_utterance_only.csv",
        help="Reviewed CSV to normalize.",
    )
    parser.add_argument(
        "--out-csv",
        default="results/ood_vercellotti/review_TAGGED_utterance_only_normalized.csv",
        help="Output normalized CSV path.",
    )
    parser.add_argument(
        "--summary-json",
        default="results/ood_vercellotti/review_TAGGED_utterance_only_normalized_summary.json",
        help="Summary JSON path.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_review_decision(raw: str) -> Dict[str, str]:
    text = (raw or "").strip()
    if not text:
        return {
            "review_decision_normalized": "",
            "review_primary_decision": "",
            "review_has_uncertain_flag": "0",
            "review_has_ambiguous_flag": "0",
        }

    text = text.replace("WONG", "WRONG")
    parts = [part.strip().upper() for part in text.split(",") if part.strip()]
    parts = ["PARTIAL_MIXED" if part in {"PARTIAL_MIX", "PARTIAL_MIXED"} else part for part in parts]
    parts = ["UNCERTAIN" if part == "UNCERTAIN" else part for part in parts]
    parts = ["AMBIGUOUS" if part == "AMBIGUOUS" else part for part in parts]
    parts = ["WRONG" if part == "WRONG" else part for part in parts]
    parts = ["CORRECT" if part == "CORRECT" else part for part in parts]

    has_uncertain = "UNCERTAIN" in parts
    has_ambiguous = "AMBIGUOUS" in parts

    primary_candidates = [p for p in parts if p not in {"UNCERTAIN", "AMBIGUOUS"}]
    if not primary_candidates:
        primary = "AMBIGUOUS" if has_ambiguous else "UNCERTAIN"
    else:
        primary = primary_candidates[0]

    normalized = primary
    if has_ambiguous and primary not in {"AMBIGUOUS"}:
        normalized = f"{normalized}_AMBIGUOUS"
    if has_uncertain and primary not in {"UNCERTAIN"}:
        normalized = f"{normalized}_UNCERTAIN"

    return {
        "review_decision_normalized": normalized,
        "review_primary_decision": primary,
        "review_has_uncertain_flag": "1" if has_uncertain else "0",
        "review_has_ambiguous_flag": "1" if has_ambiguous else "0",
    }


def classify_hallucination(prediction: str) -> Tuple[str, str]:
    text = prediction or ""
    matched = []
    for category, pattern in PROMPT_LEAKAGE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matched.append(category)
    if matched:
        # Prefer more concrete prompt leakage labels over the generic marker.
        if "PROMPT_RULE_COPY" in matched:
            return "1", "PROMPT_RULE_COPY"
        return "1", matched[0]
    return "0", ""


def classify_note_categories(note: str) -> List[str]:
    text = (note or "").strip()
    if not text:
        return []
    upper = text.upper()
    categories: List[str] = []
    for category, needles in NOTE_CATEGORY_PATTERNS:
        if any(needle.upper() in upper for needle in needles):
            categories.append(category)
    if "CORRECT (" in upper or "WRONG (" in upper:
        if "CORRECT" in upper and "WRONG" in upper:
            categories.append("MIXED_ISSUES")
    return sorted(dict.fromkeys(categories))


def normalize_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out_rows: List[Dict[str, str]] = []
    for row in rows:
        decision_info = normalize_review_decision(row.get("review_decision", ""))
        hallucinated, hallucination_type = classify_hallucination(row.get("model_prediction", ""))
        note_categories = classify_note_categories(row.get("notes", ""))

        normalized = dict(row)
        normalized.update(decision_info)
        normalized["hallucination_flag"] = hallucinated
        normalized["hallucination_type"] = hallucination_type
        normalized["note_categories"] = " ; ".join(note_categories)

        if normalized.get("prediction_status", "") == "NON_TAGGED_CHANGED" and hallucinated == "1":
            normalized["auto_issue_primary"] = "HALLUCINATION"
        elif note_categories:
            normalized["auto_issue_primary"] = note_categories[0]
        else:
            normalized["auto_issue_primary"] = ""

        if len(note_categories) > 1:
            normalized["auto_issue_secondary"] = " ; ".join(note_categories[1:])
        else:
            normalized["auto_issue_secondary"] = ""

        out_rows.append(normalized)
    return out_rows


def build_summary(rows: List[Dict[str, str]]) -> Dict[str, object]:
    return {
        "rows": len(rows),
        "normalized_decision_counts": dict(
            Counter((row.get("review_decision_normalized") or "").strip() for row in rows if (row.get("review_decision_normalized") or "").strip())
        ),
        "hallucination_flag_counts": dict(Counter(row.get("hallucination_flag", "0") for row in rows)),
        "hallucination_type_counts": dict(
            Counter((row.get("hallucination_type") or "").strip() for row in rows if (row.get("hallucination_type") or "").strip())
        ),
        "auto_issue_primary_counts": dict(
            Counter((row.get("auto_issue_primary") or "").strip() for row in rows if (row.get("auto_issue_primary") or "").strip())
        ),
        "note_category_counts": dict(
            Counter(
                category
                for row in rows
                for category in [c.strip() for c in (row.get("note_categories") or "").split(";")]
                if category
            )
        ),
    }


def main() -> None:
    args = parse_args()
    input_csv = resolve_path(args.input_csv)
    out_csv = resolve_path(args.out_csv)
    summary_json = resolve_path(args.summary_json)

    rows = load_csv(input_csv)
    normalized_rows = normalize_rows(rows)

    fieldnames = list(normalized_rows[0].keys()) if normalized_rows else []
    write_csv(out_csv, normalized_rows, fieldnames)

    summary = build_summary(normalized_rows)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out_csv": str(out_csv), "summary_json": str(summary_json), **summary}, indent=2))


if __name__ == "__main__":
    main()
