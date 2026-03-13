from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from common import resolve_path
from normalize_ood_review_csv import classify_hallucination


DEFAULT_REVIEW_CSVS = [
    "results/ood_vercellotti/ENNI_review_utterance_only.csv",
    "results/ood_vercellotti/review_TAGGED_utterance_only.csv",
    "results/ood_vercellotti/review_TAGGED_utterance_only_normalized.csv",
    "results/ood_vercellotti/review_TAGGED_cross_mode_assist.csv",
    "results/ood_vercellotti/review_TAGGED_cross_mode_priority.csv",
    "results/ood_vercellotti/master_review_union.csv",
    "results/ood_vercellotti/master_review_union_focus.csv",
    "results/ood_vercellotti/master_review_union_priority.csv",
    "results/ood_vercellotti/master_review_union_priority_pending.csv",
    "results/ood_vercellotti/context_analysis/master_review_union_focus.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-flag prompt-fragment hallucinations in review CSVs."
    )
    parser.add_argument(
        "--csv",
        action="append",
        default=[],
        help="CSV path to process. Can be specified multiple times. If omitted, defaults are used.",
    )
    parser.add_argument(
        "--summary-json",
        default="results/ood_vercellotti/auto_hallucination_flagging_summary.json",
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--flag-shorter-than-input",
        action="store_true",
        help="Also auto-flag HALLUCINATION when prediction text is shorter than input (decision column must be empty).",
    )
    parser.add_argument(
        "--shorter-prediction-col",
        default="utterance_only_prediction",
        help="Prediction column used for the shorter-than-input rule.",
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


def clean(value: str) -> str:
    return (value or "").strip()


def detect_prediction_columns(fieldnames: List[str]) -> List[str]:
    cols: List[str] = []
    for col in fieldnames:
        lc = col.lower()
        if lc == "model_prediction":
            cols.append(col)
        elif lc == "prediction":
            cols.append(col)
        elif lc.endswith("_prediction"):
            cols.append(col)
        elif lc in {"baseline_prediction", "mode_prediction"}:
            cols.append(col)
    return cols


def pick_decision_column(fieldnames: List[str]) -> str:
    if "final_review_decision" in fieldnames:
        return "final_review_decision"
    if "review_decision" in fieldnames:
        return "review_decision"
    return ""


def pick_notes_column(fieldnames: List[str]) -> str:
    if "final_notes" in fieldnames:
        return "final_notes"
    if "notes" in fieldnames:
        return "notes"
    return ""


def normalize_spaces(text: str) -> str:
    return " ".join((text or "").split())


def process_csv(path: Path, flag_shorter_than_input: bool, shorter_prediction_col: str) -> Dict[str, object]:
    rows = load_csv(path)
    if not rows:
        return {"path": str(path), "rows": 0, "updated": 0, "auto_hallucination_rows": 0}

    fieldnames = list(rows[0].keys())
    decision_col = pick_decision_column(fieldnames)
    notes_col = pick_notes_column(fieldnames)
    prediction_cols = detect_prediction_columns(fieldnames)

    if "auto_hallucination_flag" not in fieldnames:
        fieldnames.append("auto_hallucination_flag")
    if "auto_hallucination_prediction_cols" not in fieldnames:
        fieldnames.append("auto_hallucination_prediction_cols")
    if "auto_hallucination_types" not in fieldnames:
        fieldnames.append("auto_hallucination_types")

    updated = 0
    auto_hall_rows = 0
    updated_shorter = 0

    for row in rows:
        hallucinated_cols: List[str] = []
        hallucination_types: List[str] = []
        for col in prediction_cols:
            flag, htype = classify_hallucination(row.get(col, ""))
            if flag == "1":
                hallucinated_cols.append(col)
                if htype:
                    hallucination_types.append(htype)

        if hallucinated_cols:
            auto_hall_rows += 1
            row["auto_hallucination_flag"] = "1"
            row["auto_hallucination_prediction_cols"] = " ; ".join(hallucinated_cols)
            row["auto_hallucination_types"] = " ; ".join(sorted(dict.fromkeys(hallucination_types)))
        else:
            row["auto_hallucination_flag"] = "0"
            row["auto_hallucination_prediction_cols"] = ""
            row["auto_hallucination_types"] = ""

        if not decision_col:
            continue
        if clean(row.get(decision_col, "")):
            continue
        if not hallucinated_cols:
            continue

        row[decision_col] = "HALLUCINATION"
        if notes_col and not clean(row.get(notes_col, "")):
            row[notes_col] = "AUTO_PROMPT_FRAGMENT"
        updated += 1

    if flag_shorter_than_input and decision_col and "input" in fieldnames and shorter_prediction_col in fieldnames:
        for row in rows:
            if clean(row.get(decision_col, "")):
                continue
            input_len = len(normalize_spaces(row.get("input", "")))
            pred_len = len(normalize_spaces(row.get(shorter_prediction_col, "")))
            if input_len > 0 and pred_len < input_len:
                row[decision_col] = "HALLUCINATION"
                if notes_col and not clean(row.get(notes_col, "")):
                    row[notes_col] = "AUTO_SHORTER_THAN_INPUT"
                updated += 1
                updated_shorter += 1

    write_csv(path, rows, fieldnames)
    return {
        "path": str(path),
        "rows": len(rows),
        "decision_column": decision_col,
        "prediction_columns": prediction_cols,
        "auto_hallucination_rows": auto_hall_rows,
        "updated_decisions": updated,
        "updated_decisions_shorter_rule": updated_shorter,
    }


def main() -> None:
    args = parse_args()
    csv_paths = [resolve_path(p) for p in (args.csv or DEFAULT_REVIEW_CSVS)]
    summary_path = resolve_path(args.summary_json)

    results: List[Dict[str, object]] = []
    for path in csv_paths:
        if not path.exists():
            results.append({"path": str(path), "missing": True})
            continue
        results.append(
            process_csv(
                path,
                flag_shorter_than_input=args.flag_shorter_than_input,
                shorter_prediction_col=args.shorter_prediction_col,
            )
        )

    summary = {
        "processed_files": len(results),
        "total_rows": sum(int(r.get("rows", 0)) for r in results if not r.get("missing")),
        "total_auto_hallucination_rows": sum(
            int(r.get("auto_hallucination_rows", 0)) for r in results if not r.get("missing")
        ),
        "total_updated_decisions": sum(
            int(r.get("updated_decisions", 0)) for r in results if not r.get("missing")
        ),
        "total_updated_decisions_shorter_rule": sum(
            int(r.get("updated_decisions_shorter_rule", 0)) for r in results if not r.get("missing")
        ),
        "files": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
