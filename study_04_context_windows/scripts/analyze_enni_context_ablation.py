from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from common import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare utterance_only vs prev_same_speaker ENNI predictions against gold output."
    )
    parser.add_argument("--utterance-only", required=True, help="predictions_utterance_only.jsonl")
    parser.add_argument("--prev-same-speaker", required=True, help="predictions_prev_same_speaker.jsonl")
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


def main() -> None:
    args = parse_args()
    utterance_only_path = resolve_path(args.utterance_only)
    prev_path = resolve_path(args.prev_same_speaker)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    utterance_only_rows = {int(row["row_id"]): row for row in load_jsonl(utterance_only_path)}
    prev_rows = {int(row["row_id"]): row for row in load_jsonl(prev_path)}
    shared_ids = sorted(set(utterance_only_rows) & set(prev_rows))
    if not shared_ids:
        raise SystemExit("No shared row_ids between the two prediction files.")

    comparison_rows: List[Dict] = []
    uo_correct = 0
    prev_correct = 0
    changed_predictions = 0
    improved = 0
    worsened = 0

    for row_id in shared_ids:
        base = utterance_only_rows[row_id]
        alt = prev_rows[row_id]
        gold = str(base.get("gold_output", alt.get("gold_output", "")) or "").strip()
        if not gold:
            raise SystemExit(f"Missing gold_output for row_id={row_id}")

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
                "row_id": row_id,
                "file_name": base.get("file_name", ""),
                "speaker": base.get("speaker", ""),
                "line_no": base.get("line_no"),
                "input": base.get("input", ""),
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
        "rows_shared": len(shared_ids),
        "utterance_only_exact_match": ratio(uo_correct, len(shared_ids)),
        "prev_same_speaker_exact_match": ratio(prev_correct, len(shared_ids)),
        "prediction_changed_rows": changed_predictions,
        "prediction_changed_rate": ratio(changed_predictions, len(shared_ids)),
        "improved_rows": improved,
        "worsened_rows": worsened,
        "changed_rows_improved": changed_improved,
        "changed_rows_worsened": changed_worsened,
        "changed_rows_improved_rate": ratio(changed_improved, len(changed_rows)),
        "changed_rows_worsened_rate": ratio(changed_worsened, len(changed_rows)),
        "outputs": {
            "changed_items_csv": str(changed_csv),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
