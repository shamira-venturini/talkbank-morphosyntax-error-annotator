from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from common import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge completed focus review fields into canonical master review CSVs."
    )
    parser.add_argument(
        "--reviewed-focus-csv",
        default="results/ood_vercellotti/context_analysis/master_review_union_focus.csv",
        help="Reviewed focus CSV containing completed decisions/corrections.",
    )
    parser.add_argument(
        "--master-union-csv",
        default="results/ood_vercellotti/master_review_union.csv",
        help="Canonical full union CSV to update in place.",
    )
    parser.add_argument(
        "--master-focus-csv",
        default="results/ood_vercellotti/master_review_union_focus.csv",
        help="Canonical focus CSV to update in place.",
    )
    parser.add_argument(
        "--master-priority-csv",
        default="results/ood_vercellotti/master_review_union_priority.csv",
        help="Canonical priority CSV to update in place.",
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
            writer.writerow(row)


def clean(value: str) -> str:
    return (value or "").strip()


def pick_first_nonempty(row: Dict[str, str], candidates: List[str]) -> str:
    for key in candidates:
        value = clean(row.get(key, ""))
        if value:
            return value
    return ""


def build_review_updates(reviewed_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    updates: Dict[str, Dict[str, str]] = {}
    for row in reviewed_rows:
        row_id = clean(row.get("row_id", ""))
        if not row_id:
            continue

        decision = pick_first_nonempty(
            row,
            [
                "final_review_decision",
                "review_decision",
                "review_decision_normalized",
            ],
        )
        preferred_mode = pick_first_nonempty(row, ["preferred_mode"])
        corrected = pick_first_nonempty(
            row,
            [
                "final_corrected_annotation",
                "corrected_output",
                "corrected_annotation",
            ],
        )
        notes = pick_first_nonempty(row, ["final_notes", "notes"])

        if decision or preferred_mode or corrected or notes:
            updates[row_id] = {
                "final_review_decision": decision,
                "preferred_mode": preferred_mode,
                "final_corrected_annotation": corrected,
                "final_notes": notes,
            }
    return updates


def apply_updates(rows: List[Dict[str, str]], updates: Dict[str, Dict[str, str]]) -> int:
    touched = 0
    for row in rows:
        row_id = clean(row.get("row_id", ""))
        if not row_id or row_id not in updates:
            continue
        upd = updates[row_id]
        changed = False
        for key, value in upd.items():
            if clean(value):
                if clean(row.get(key, "")) != value:
                    row[key] = value
                    changed = True
        if changed:
            touched += 1
    return touched


def main() -> None:
    args = parse_args()
    reviewed_focus_csv = resolve_path(args.reviewed_focus_csv)
    master_union_csv = resolve_path(args.master_union_csv)
    master_focus_csv = resolve_path(args.master_focus_csv)
    master_priority_csv = resolve_path(args.master_priority_csv)

    reviewed_rows = load_csv(reviewed_focus_csv)
    updates = build_review_updates(reviewed_rows)

    union_rows = load_csv(master_union_csv)
    focus_rows = load_csv(master_focus_csv)
    priority_rows = load_csv(master_priority_csv) if master_priority_csv.exists() else []

    union_touched = apply_updates(union_rows, updates)
    focus_touched = apply_updates(focus_rows, updates)
    priority_touched = apply_updates(priority_rows, updates) if priority_rows else 0

    union_fields = list(union_rows[0].keys()) if union_rows else []
    focus_fields = list(focus_rows[0].keys()) if focus_rows else []
    priority_fields = list(priority_rows[0].keys()) if priority_rows else []
    write_csv(master_union_csv, union_rows, union_fields)
    write_csv(master_focus_csv, focus_rows, focus_fields)
    if priority_rows:
        write_csv(master_priority_csv, priority_rows, priority_fields)

    print(
        json.dumps(
            {
                "reviewed_focus_csv": str(reviewed_focus_csv),
                "master_union_csv": str(master_union_csv),
                "master_focus_csv": str(master_focus_csv),
                "master_priority_csv": str(master_priority_csv),
                "updates_available_rows": len(updates),
                "union_rows_touched": union_touched,
                "focus_rows_touched": focus_touched,
                "priority_rows_touched": priority_touched,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
