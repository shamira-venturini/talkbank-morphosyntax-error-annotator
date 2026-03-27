from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from common import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare utterance_only vs prev_same_speaker ENNI predictions against gold output."
    )
    parser.add_argument("--utterance-only", required=True, help="predictions_utterance_only.jsonl")
    parser.add_argument("--prev-same-speaker", required=True, help="predictions_prev_same_speaker.jsonl")
    parser.add_argument(
        "--prepared-input-jsonl",
        default="",
        help="Optional prepared eval JSONL used to recover prev_same_speaker_text for review exports.",
    )
    parser.add_argument(
        "--out-dir",
        default="study_04_context_windows/results/enni_context_ablation",
        help="Directory for summary outputs.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def exact_match(prediction: str, gold: str) -> bool:
    return str(prediction or "").strip() == str(gold or "").strip()


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def row_key(row: Dict) -> Tuple[str, str, int, int]:
    return (
        str(row.get("file_name", "") or ""),
        str(row.get("speaker", "") or ""),
        int(row.get("line_no", 0) or 0),
        int(row.get("utterance_index_raw", 0) or 0),
    )


def choose_alignment(
    utterance_only_rows: List[Dict],
    prev_rows: List[Dict],
) -> Tuple[str, List[Tuple[Optional[int], Dict, Optional[int], Dict]], Dict[str, int]]:
    by_row_id_uo = {int(row["row_id"]): row for row in utterance_only_rows}
    by_row_id_prev = {int(row["row_id"]): row for row in prev_rows}
    shared_row_ids = sorted(set(by_row_id_uo) & set(by_row_id_prev))

    row_id_matches: List[Tuple[Optional[int], Dict, Optional[int], Dict]] = []
    row_id_consistent = 0
    for row_id in shared_row_ids:
        base = by_row_id_uo[row_id]
        alt = by_row_id_prev[row_id]
        row_id_matches.append((row_id, base, row_id, alt))
        if row_key(base) == row_key(alt):
            row_id_consistent += 1

    by_struct_uo = {row_key(row): row for row in utterance_only_rows}
    by_struct_prev = {row_key(row): row for row in prev_rows}
    shared_struct_keys = sorted(set(by_struct_uo) & set(by_struct_prev))
    structural_matches = [
        (
            int(by_struct_uo[key]["row_id"]),
            by_struct_uo[key],
            int(by_struct_prev[key]["row_id"]),
            by_struct_prev[key],
        )
        for key in shared_struct_keys
    ]

    if shared_row_ids and row_id_consistent == len(shared_row_ids):
        strategy = "row_id"
        matches = row_id_matches
    elif shared_struct_keys:
        strategy = "structural_key"
        matches = structural_matches
    elif shared_row_ids:
        strategy = "row_id"
        matches = row_id_matches
    else:
        raise SystemExit("No shared row_ids or structural keys between the two prediction files.")

    diagnostics = {
        "shared_row_ids": len(shared_row_ids),
        "row_id_consistent_keys": row_id_consistent,
        "shared_structural_keys": len(shared_struct_keys),
    }
    return strategy, matches, diagnostics


def index_by_structural_key(rows: List[Dict]) -> Dict[Tuple[str, str, int, int], Dict]:
    return {row_key(row): row for row in rows}


def main() -> None:
    args = parse_args()
    utterance_only_path = resolve_path(args.utterance_only)
    prev_path = resolve_path(args.prev_same_speaker)
    prepared_input_path = resolve_path(args.prepared_input_jsonl) if args.prepared_input_jsonl else None
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    utterance_only_rows = load_jsonl(utterance_only_path)
    prev_rows = load_jsonl(prev_path)
    alignment_strategy, matches, diagnostics = choose_alignment(utterance_only_rows, prev_rows)
    prepared_rows_by_key = {}
    if prepared_input_path:
        prepared_rows_by_key = index_by_structural_key(load_jsonl(prepared_input_path))

    comparison_rows: List[Dict] = []
    uo_correct = 0
    prev_correct = 0
    changed_predictions = 0
    improved = 0
    worsened = 0

    for base_row_id, base, prev_row_id, alt in matches:
        key = row_key(alt)
        prepared_row = prepared_rows_by_key.get(key, {})
        gold = str(base.get("gold_output", alt.get("gold_output", "")) or "").strip()
        if not gold:
            raise SystemExit(
                "Missing gold_output for aligned rows: "
                f"utterance_only_row_id={base_row_id}, prev_same_speaker_row_id={prev_row_id}"
            )

        base_pred = str(base.get("model_prediction", "") or "").strip()
        alt_pred = str(alt.get("model_prediction", "") or "").strip()
        base_ok = exact_match(base_pred, gold)
        alt_ok = exact_match(alt_pred, gold)

        uo_correct += int(base_ok)
        prev_correct += int(alt_ok)
        changed = base_pred != alt_pred
        changed_predictions += int(changed)
        if (not base_ok) and alt_ok:
            delta = "improved"
            improved += 1
        elif base_ok and (not alt_ok):
            delta = "worsened"
            worsened += 1
        elif base_ok and alt_ok:
            delta = "both_correct"
        else:
            delta = "both_incorrect"

        comparison_rows.append(
            {
                "utterance_only_row_id": base_row_id,
                "prev_same_speaker_row_id": prev_row_id,
                "file_name": base.get("file_name", ""),
                "speaker": base.get("speaker", ""),
                "line_no": base.get("line_no"),
                "utterance_index_raw": base.get("utterance_index_raw"),
                "input": base.get("input", ""),
                "prev_same_speaker_text": (
                    alt.get("prev_same_speaker_text", "")
                    or prepared_row.get("prev_same_speaker_text", "")
                ),
                "gold_output": gold,
                "pred_utterance_only": base_pred,
                "pred_prev_same_speaker": alt_pred,
                "utterance_only_correct": int(base_ok),
                "prev_same_speaker_correct": int(alt_ok),
                "prediction_changed": int(changed),
                "delta": delta,
                "prev_same_speaker_available": int(bool(alt.get("context_utterance_count", 0))),
            }
        )

    changed_rows = [row for row in comparison_rows if row["prediction_changed"]]
    changed_improved = sum(1 for row in changed_rows if row["delta"] == "improved")
    changed_worsened = sum(1 for row in changed_rows if row["delta"] == "worsened")

    changed_csv = out_dir / "changed_items.csv"
    with changed_csv.open("w", encoding="utf-8") as handle:
        if changed_rows:
            fieldnames = list(changed_rows[0].keys())
            handle.write(",".join(fieldnames) + "\n")
            for row in changed_rows:
                values = []
                for field in fieldnames:
                    value = str(row.get(field, "")).replace('"', '""')
                    values.append(f'"{value}"')
                handle.write(",".join(values) + "\n")
        else:
            handle.write("row_id\n")

    summary = {
        "alignment_strategy": alignment_strategy,
        "rows_shared": len(matches),
        "utterance_only_exact_match": ratio(uo_correct, len(matches)),
        "prev_same_speaker_exact_match": ratio(prev_correct, len(matches)),
        "prediction_changed_rows": changed_predictions,
        "prediction_changed_rate": ratio(changed_predictions, len(matches)),
        "improved_rows": improved,
        "worsened_rows": worsened,
        "changed_rows_improved": changed_improved,
        "changed_rows_worsened": changed_worsened,
        "changed_rows_improved_rate": ratio(changed_improved, len(changed_rows)),
        "changed_rows_worsened_rate": ratio(changed_worsened, len(changed_rows)),
        "alignment_diagnostics": diagnostics,
        "prepared_input_jsonl": str(prepared_input_path) if prepared_input_path else "",
        "outputs": {
            "changed_items_csv": str(changed_csv),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
