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
    rows = []
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
        description="Build a blinded manual-review sheet for direct-vs-curriculum disagreements."
    )
    parser.add_argument(
        "--direct-predictions",
        default="results/recon_nonword_only_one-run_seed3407/eval_outputs/predictions_test_real.jsonl",
        help="Direct stage-3 predictions JSONL.",
    )
    parser.add_argument(
        "--curriculum-predictions",
        default="results/recon_full_curr_nonwordonly_seed3407/eval_outputs/predictions_test_real.jsonl",
        help="Curriculum predictions JSONL.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/manual_review_test_real_disagreements",
        help="Output directory for blinded review files.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Shuffle seed for blinded row order.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direct_path = resolve_path(args.direct_predictions)
    curriculum_path = resolve_path(args.curriculum_predictions)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    direct_rows = load_jsonl(direct_path)
    curriculum_rows = load_jsonl(curriculum_path)
    curriculum_by_row = {row["row_id"]: row for row in curriculum_rows}

    blinded_rows = []
    key_rows = []
    group_counter = 0

    for direct in direct_rows:
        curriculum = curriculum_by_row[direct["row_id"]]
        if direct["model_prediction"] == curriculum["model_prediction"]:
            continue

        group_counter += 1
        group_id = f"U{group_counter:03d}"
        candidate_map: Dict[str, Dict] = {}
        candidates = [
            ("gold", direct["human_gold"]),
            ("direct_model", direct["model_prediction"]),
            ("curriculum_model", curriculum["model_prediction"]),
        ]
        for source, candidate in candidates:
            entry = candidate_map.setdefault(
                candidate,
                {
                    "group_id": group_id,
                    "row_id": direct["row_id"],
                    "input": direct["input"],
                    "candidate_annotation": candidate,
                    "members": [],
                },
            )
            entry["members"].append(
                {
                    "source": source,
                    "category": classify_candidate(direct["human_gold"], candidate, source),
                }
            )

        for entry in candidate_map.values():
            blinded_rows.append(entry)

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
            fieldnames=[
                "review_id",
                "utterance_id",
                "row_id",
                "source",
                "original_category",
            ],
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
        "direct_predictions": str(direct_path),
        "curriculum_predictions": str(curriculum_path),
        "disagreement_items": group_counter,
        "blinded_review_rows": len(blinded_rows),
        "review_sheet": str(review_path),
        "answer_key": str(key_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
