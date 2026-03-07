import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from common import resolve_path


TAG_PATTERN = re.compile(r"\[\*\s*[ms](?::[^\]]+)?\]")
DEFAULT_HOLDOUT_LABELS = ["[* m:++er]", "[* m:++est]", "[* m:0er]", "[* m:0est]"]


def canonical_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    return f"[* {body}]"


def extract_tags(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [canonical_tag(x) for x in TAG_PATTERN.findall(text)]


def operator_family(tag: str) -> str:
    if not tag.startswith("[* m:"):
        return "other"
    code = tag[len("[* m:") : -1]
    if code.startswith("++"):
        return "++"
    if code.startswith("0"):
        return "0"
    if code.startswith("+"):
        return "+"
    if code.startswith("="):
        return "="
    if code.startswith("base"):
        return "base"
    if code.startswith("irr"):
        return "irr"
    if code.startswith("sub"):
        return "sub"
    if code.startswith("vsg"):
        return "vsg"
    if code.startswith("vun"):
        return "vun"
    if code.startswith("allo"):
        return "allo"
    return "other"


def get_legal_label_set(experiment_dir: Path) -> Set[str]:
    labels: Set[str] = set()
    for split in ["train", "eval", "test", "eval_coverage", "test_coverage", "holdout"]:
        path = experiment_dir / f"stage3_{split}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for tag in extract_tags(row.get("output", "")):
                    labels.add(tag)
    return labels


def first_holdout_label(tags: List[str], holdout_set: Set[str]) -> Optional[str]:
    for tag in tags:
        if tag in holdout_set:
            return tag
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Experiment 4 unseen-tag generalization metrics.")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Prediction JSONL (input, human_gold, model_prediction).",
    )
    parser.add_argument(
        "--experiment-dir",
        default="FT-3/experiments/exp4_unseen_tags",
        help="Experiment directory to derive legal label inventory.",
    )
    parser.add_argument(
        "--holdout-label",
        action="append",
        default=None,
        help="Held-out label set used in Experiment 4 (repeat flag).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path for JSON metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = resolve_path(args.predictions)
    experiment_dir = resolve_path(args.experiment_dir)
    holdout_labels = args.holdout_label if args.holdout_label else DEFAULT_HOLDOUT_LABELS
    holdout_set = set(holdout_labels)
    legal_labels = get_legal_label_set(experiment_dir)

    n_rows = 0
    valid_rows = 0
    exact_label_correct = 0
    operator_correct = 0
    invalid_tag_total = 0
    predicted_tag_total = 0
    rows_with_invalid = 0

    support = Counter()
    exact_per_label = Counter()
    operator_per_label = Counter()

    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            row = json.loads(line)
            gold_tags = extract_tags(row.get("human_gold", ""))
            pred_tags = extract_tags(row.get("model_prediction", ""))
            gold = first_holdout_label(gold_tags, holdout_set)
            pred = first_holdout_label(pred_tags, holdout_set)

            if gold is None:
                continue
            valid_rows += 1
            support[gold] += 1

            if pred == gold:
                exact_label_correct += 1
                exact_per_label[gold] += 1

            if pred is not None and operator_family(pred) == operator_family(gold):
                operator_correct += 1
                operator_per_label[gold] += 1

            invalid_pred_tags = [t for t in pred_tags if t not in legal_labels]
            invalid_tag_total += len(invalid_pred_tags)
            predicted_tag_total += len(pred_tags)
            if invalid_pred_tags:
                rows_with_invalid += 1

    def pct(n: int, d: int) -> float:
        return round((n / d), 6) if d else 0.0

    per_label = {}
    for label in holdout_labels:
        s = support.get(label, 0)
        per_label[label] = {
            "support": s,
            "exact_accuracy": pct(exact_per_label.get(label, 0), s),
            "operator_accuracy": pct(operator_per_label.get(label, 0), s),
        }

    metrics: Dict[str, object] = {
        "predictions_file": str(predictions_path),
        "experiment_dir": str(experiment_dir),
        "holdout_labels": holdout_labels,
        "rows_total": n_rows,
        "rows_with_holdout_gold": valid_rows,
        "exact_label_accuracy": pct(exact_label_correct, valid_rows),
        "operator_accuracy": pct(operator_correct, valid_rows),
        "invalid_label_rate_tag_level": pct(invalid_tag_total, predicted_tag_total),
        "invalid_label_rate_row_level": pct(rows_with_invalid, max(1, valid_rows)),
        "per_label": per_label,
    }

    out_path = resolve_path(args.out) if args.out else predictions_path.with_name("metrics_exp4_generalization.json")
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Experiment 4 metrics: {out_path}")
    print(json.dumps({k: metrics[k] for k in ['rows_with_holdout_gold', 'exact_label_accuracy', 'operator_accuracy']}, indent=2))


if __name__ == "__main__":
    main()
