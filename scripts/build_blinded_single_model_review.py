from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List

from common import resolve_path


TAG_RE = re.compile(r"\[\* [^\]]+\]")


def has_tag(text: str) -> bool:
    return bool(TAG_RE.search(text or ""))


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def classify_candidate(gold: str, candidate: str, source: str) -> str:
    if source == "gold":
        return "gold_reference"
    gold_has_tag = has_tag(gold)
    cand_has_tag = has_tag(candidate)
    if candidate == gold:
        if gold_has_tag:
            return "true_positive_exact"
        return "true_negative_exact"
    if (not gold_has_tag) and cand_has_tag:
        return "false_positive"
    if gold_has_tag and (not cand_has_tag):
        return "false_negative"
    if gold_has_tag and cand_has_tag:
        return "wrong_label"
    return "other_clean_mismatch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a blinded manual-review sheet for a single model against gold."
    )
    parser.add_argument(
        "--predictions",
        default="results/recon_full_comp_preserve_seed3407/eval_outputs/predictions_test_real.jsonl",
        help="Predictions JSONL for the model under review.",
    )
    parser.add_argument(
        "--model-label",
        default="best_model",
        help="Source label to store in the answer key for the model outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/manual_review_test_real_best_model_vs_gold",
        help="Output directory for blinded review files.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Shuffle seed for blinded row order.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = resolve_path(args.predictions)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_rows = load_jsonl(pred_path)
    blinded_rows = []
    group_counter = 0

    for row in pred_rows:
        gold = row["human_gold"]
        model_pred = row["model_prediction"]
        if gold == model_pred:
            continue

        group_counter += 1
        group_id = f"U{group_counter:03d}"
        candidate_map: Dict[str, Dict] = {}
        candidates = [("gold", gold), (args.model_label, model_pred)]
        for source, candidate in candidates:
            entry = candidate_map.setdefault(
                candidate,
                {
                    "group_id": group_id,
                    "row_id": row["row_id"],
                    "input": row["input"],
                    "candidate_annotation": candidate,
                    "members": [],
                },
            )
            entry["members"].append(
                {
                    "source": source,
                    "category": classify_candidate(gold, candidate, source),
                }
            )
        blinded_rows.extend(candidate_map.values())

    rng = random.Random(args.seed)
    rng.shuffle(blinded_rows)

    review_path = out_dir / "blinded_review_sheet.csv"
    key_path = out_dir / "answer_key.csv"

    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "review_id",
                "utterance_id",
                "input",
                "candidate_annotation",
                "score_correct",
                "score_incorrect",
                "score_ambiguous",
                "score_unsure",
                "notes",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(blinded_rows, start=1):
            review_id = f"R{idx:03d}"
            row["review_id"] = review_id
            writer.writerow(
                {
                    "review_id": review_id,
                    "utterance_id": row["group_id"],
                    "input": row["input"],
                    "candidate_annotation": row["candidate_annotation"],
                    "score_correct": "",
                    "score_incorrect": "",
                    "score_ambiguous": "",
                    "score_unsure": "",
                    "notes": "",
                }
            )

    with key_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["review_id", "utterance_id", "row_id", "source", "original_category"],
        )
        writer.writeheader()
        for row in blinded_rows:
            for member in row["members"]:
                writer.writerow(
                    {
                        "review_id": row["review_id"],
                        "utterance_id": row["group_id"],
                        "row_id": row["row_id"],
                        "source": member["source"],
                        "original_category": member["category"],
                    }
                )

    summary = {
        "predictions": str(pred_path),
        "model_label": args.model_label,
        "disagreement_items": group_counter,
        "blinded_review_rows": len(blinded_rows),
        "review_sheet": str(review_path),
        "answer_key": str(key_path),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
