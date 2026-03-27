from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from common import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge reviewed suspicious missed-error rows back into utterance-only review CSV."
    )
    parser.add_argument(
        "--suspicious-csv",
        default="results/ood_vercellotti/suspicious_missed_errors_utterance_only.csv",
        help="Reviewed suspicious shortlist CSV.",
    )
    parser.add_argument(
        "--review-csv",
        default="results/ood_vercellotti/ENNI_review_utterance_only.csv",
        help="Main utterance-only review CSV to update in place.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def clean(value: str) -> str:
    return (value or "").strip()


def main() -> None:
    args = parse_args()
    suspicious_csv = resolve_path(args.suspicious_csv)
    review_csv = resolve_path(args.review_csv)

    suspicious_rows = load_csv(suspicious_csv)
    review_rows = load_csv(review_csv)

    updates = {
        clean(row["row_id"]): {
            "review_decision": clean(row.get("review_decision", "")),
            "corrected_annotation": clean(row.get("corrected_annotation", "")),
            "notes": clean(row.get("notes", "")),
        }
        for row in suspicious_rows
        if clean(row.get("row_id", "")) and any(clean(row.get(k, "")) for k in ["review_decision", "corrected_annotation", "notes"])
    }

    touched = 0
    fieldnames = list(review_rows[0].keys()) if review_rows else []
    for row in review_rows:
        row_id = clean(row.get("row_id", ""))
        if row_id not in updates:
            continue
        payload = updates[row_id]
        changed = False
        for key, value in payload.items():
            if key not in fieldnames:
                continue
            if value and clean(row.get(key, "")) != value:
                row[key] = value
                changed = True
        if changed:
            touched += 1

    write_csv(review_csv, review_rows, fieldnames)

    print(
        json.dumps(
            {
                "suspicious_csv": str(suspicious_csv),
                "review_csv": str(review_csv),
                "updates_available": len(updates),
                "rows_touched": touched,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
