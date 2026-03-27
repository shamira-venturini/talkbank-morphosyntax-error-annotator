from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from common import resolve_path


TAG_RE = re.compile(r"\[\*\s*[ms](?::[^\]]+)?\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a flat review CSV from OOD prediction JSONL.")
    parser.add_argument(
        "--predictions-jsonl",
        default="results/ood_vercellotti/predictions_utterance_only.jsonl",
        help="Prediction JSONL produced by run_ood_context_inference.py",
    )
    parser.add_argument(
        "--out-csv",
        default="results/ood_vercellotti/ENNI_review_utterance_only.csv",
        help="Output review CSV path.",
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


def canonical_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    return f"[* {body}]"


def extract_tags(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    tags = [canonical_tag(tag) for tag in TAG_RE.findall(text)]
    return sorted(dict.fromkeys(tags))


def prediction_status(input_text: str, pred_text: str, tags: List[str]) -> str:
    if tags:
        return "TAGGED"
    if (pred_text or "").strip() == (input_text or "").strip():
        return "CLEAN"
    return "NON_TAGGED_CHANGED"


def build_review_rows(rows: Iterable[Dict]) -> List[Dict]:
    review_rows: List[Dict] = []
    for row in rows:
        input_text = (row.get("input") or "").strip()
        pred_text = (row.get("model_prediction") or "").strip()
        tags = extract_tags(pred_text)
        status = prediction_status(input_text, pred_text, tags)
        review_rows.append(
            {
                "row_id": row.get("row_id", ""),
                "file_name": row.get("file_name", ""),
                "speaker": row.get("speaker", ""),
                "line_no": row.get("line_no", ""),
                "utterance_index_raw": row.get("utterance_index_raw", ""),
                "prediction_status": status,
                "labels_or_clean": " ; ".join(tags) if tags else "CLEAN",
                "n_labels": len(tags),
                "input": row.get("input", ""),
                "model_prediction": row.get("model_prediction", ""),
                "uncertainty_mean_token_logprob": row.get("uncertainty_mean_token_logprob", ""),
                "uncertainty_min_token_logprob": row.get("uncertainty_min_token_logprob", ""),
                "uncertainty_mean_token_margin": row.get("uncertainty_mean_token_margin", ""),
                "uncertainty_min_token_margin": row.get("uncertainty_min_token_margin", ""),
                "review_decision": "",
                "notes": "",
            }
        )
    return review_rows


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    fieldnames = [
        "row_id",
        "file_name",
        "speaker",
        "line_no",
        "utterance_index_raw",
        "prediction_status",
        "labels_or_clean",
        "n_labels",
        "input",
        "model_prediction",
        "uncertainty_mean_token_logprob",
        "uncertainty_min_token_logprob",
        "uncertainty_mean_token_margin",
        "uncertainty_min_token_margin",
        "review_decision",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    predictions_path = resolve_path(args.predictions_jsonl)
    out_csv = resolve_path(args.out_csv)
    rows = load_jsonl(predictions_path)
    review_rows = build_review_rows(rows)
    write_csv(out_csv, review_rows)
    print(
        json.dumps(
            {
                "predictions_jsonl": str(predictions_path),
                "out_csv": str(out_csv),
                "rows": len(review_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
