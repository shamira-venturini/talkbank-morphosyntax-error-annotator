from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from common import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Propagate final review fields from master priority CSV into union/focus CSVs."
    )
    parser.add_argument(
        "--priority-csv",
        default="results/ood_vercellotti/master_review_union_priority.csv",
        help="Priority CSV containing latest manual review updates.",
    )
    parser.add_argument(
        "--union-csv",
        default="results/ood_vercellotti/master_review_union.csv",
        help="Union CSV to update in place.",
    )
    parser.add_argument(
        "--focus-csv",
        default="results/ood_vercellotti/master_review_union_focus.csv",
        help="Focus CSV to update in place.",
    )
    parser.add_argument(
        "--context-focus-csv",
        default="results/ood_vercellotti/context_analysis/master_review_union_focus.csv",
        help="Context-analysis focus CSV to update in place.",
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


def build_update_map(priority_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    fields = [
        "final_review_decision",
        "preferred_mode",
        "final_corrected_annotation",
        "final_notes",
    ]
    out: Dict[str, Dict[str, str]] = {}
    for row in priority_rows:
        row_id = clean(row.get("row_id", ""))
        if not row_id:
            continue
        payload = {field: clean(row.get(field, "")) for field in fields}
        if any(payload.values()):
            out[row_id] = payload
    return out


def apply_map(rows: List[Dict[str, str]], updates: Dict[str, Dict[str, str]]) -> int:
    touched = 0
    for row in rows:
        row_id = clean(row.get("row_id", ""))
        if not row_id or row_id not in updates:
            continue
        payload = updates[row_id]
        changed = False
        for key, value in payload.items():
            if value and clean(row.get(key, "")) != value:
                row[key] = value
                changed = True
        if changed:
            touched += 1
    return touched


def main() -> None:
    args = parse_args()
    priority_csv = resolve_path(args.priority_csv)
    union_csv = resolve_path(args.union_csv)
    focus_csv = resolve_path(args.focus_csv)
    context_focus_csv = resolve_path(args.context_focus_csv)

    priority_rows = load_csv(priority_csv)
    updates = build_update_map(priority_rows)

    report = {
        "priority_csv": str(priority_csv),
        "updates_available_rows": len(updates),
        "targets": [],
    }

    for target in [union_csv, focus_csv, context_focus_csv]:
        if not target.exists():
            report["targets"].append({"path": str(target), "missing": True})
            continue
        rows = load_csv(target)
        touched = apply_map(rows, updates)
        fields = list(rows[0].keys()) if rows else []
        write_csv(target, rows, fields)
        report["targets"].append({"path": str(target), "rows": len(rows), "touched": touched})

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
