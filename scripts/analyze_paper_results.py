import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from common import resolve_path


VALID_TAG_RE = re.compile(r"\[\*\s*[ms](?::[^\]]+)?\]")
VALID_TAG_FULL_RE = re.compile(r"^\[\*\s*[ms](?::[^\]]+)?\]$")
CANDIDATE_ERROR_TAG_RE = re.compile(r"\[\*[^\]]*\]")
RECON_RE = re.compile(r"\[(::?)\s+[^\]]+\]")
SEED_SUFFIX_RE = re.compile(r"_seed\d+$")
DEFAULT_HOLDOUT_LABELS = ["[* m:++er]", "[* m:++est]", "[* m:0er]", "[* m:0est]"]
PREDICTION_TO_STAGE3_SPLIT = {
    "eval_real": "eval",
    "test_real": "test",
    "eval_coverage": "eval_coverage",
    "test_coverage": "test_coverage",
    "holdout_generalization": "holdout",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready analysis bundles from results/* run exports.")
    parser.add_argument("--results-root", default="results", help="Root folder with run bundles.")
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Optional run folder name under results root (repeat flag). If omitted, analyze all runs.",
    )
    parser.add_argument("--out-dir", default="results/paper_analysis", help="Output directory for analysis tables.")
    parser.add_argument(
        "--focus-tag-a",
        default="[* m:03s:a]",
        help="First tag in the targeted confusion diagnostic.",
    )
    parser.add_argument(
        "--focus-tag-b",
        default="[* m:0ed]",
        help="Regular-past target tag in the targeted confusion diagnostic.",
    )
    parser.add_argument(
        "--focus-tag-c",
        default="[* m:base:ed]",
        help="Irregular-past target tag in the targeted confusion diagnostic.",
    )
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        default=False,
        help="Include holdout_generalization in the main analysis bundle.",
    )
    return parser.parse_args()


def canonical_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    return f"[* {body}]"


def extract_valid_tags(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [canonical_tag(x) for x in VALID_TAG_RE.findall(text)]


def extract_candidate_error_tags(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [x.strip() for x in CANDIDATE_ERROR_TAG_RE.findall(text)]


def marker_signature(text: str) -> Tuple[str, ...]:
    if not isinstance(text, str):
        return tuple()
    return tuple(RECON_RE.findall(text))


def list_run_dirs(results_root: Path, selected_runs: Optional[Sequence[str]]) -> List[Path]:
    if selected_runs:
        dirs = [results_root / run_name for run_name in selected_runs]
    else:
        dirs = sorted(p for p in results_root.iterdir() if p.is_dir())
    out = []
    for run_dir in dirs:
        if (run_dir / "eval_outputs" / "run_summary.json").exists():
            out.append(run_dir)
    return out


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(n: int, d: int) -> float:
    return round((n / d), 6) if d else 0.0


def safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def safe_std(values: List[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def infer_experiment(run_name: str, split_dir: str) -> str:
    split_dir = (split_dir or "").lower()
    run_name_l = run_name.lower()
    if "exp3_abs" in split_dir:
        return "exp3"
    if "exp4_unseen_tags" in split_dir:
        return "exp4"
    if "acl_rr_v1" in split_dir:
        if "exp1" in run_name_l:
            return "exp1"
        if "exp2" in run_name_l:
            return "exp2"
        return "exp1_or_exp2"
    if "exp6" in split_dir or "stability" in run_name_l:
        return "exp6"
    return "unknown"


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


def first_holdout_label(tags: List[str], holdout_set: Set[str]) -> Optional[str]:
    for tag in tags:
        if tag in holdout_set:
            return tag
    return None


def legal_label_set_for_split_dir(split_dir: str) -> Set[str]:
    if not split_dir:
        return set()
    split_root = resolve_path(split_dir)
    labels: Set[str] = set()
    for split_name in ["train", "eval", "test", "eval_coverage", "test_coverage", "holdout"]:
        path = split_root / f"stage3_{split_name}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                labels.update(extract_valid_tags(row.get("output", "")))
    return labels


def holdout_labels_for_split_dir(split_dir: str) -> List[str]:
    if not split_dir:
        return list(DEFAULT_HOLDOUT_LABELS)
    summary_path = resolve_path(split_dir) / "summary.json"
    if summary_path.exists():
        data = load_json(summary_path)
        labels = data.get("holdout_labels")
        if isinstance(labels, list) and labels:
            return [str(x) for x in labels]
    return list(DEFAULT_HOLDOUT_LABELS)


def stage3_split_name(prediction_split_name: str) -> Optional[str]:
    return PREDICTION_TO_STAGE3_SPLIT.get(prediction_split_name)


def split_metadata_lookup(split_dir: str, prediction_split_name: str) -> Dict[Tuple[object, str], Dict]:
    split_name = stage3_split_name(prediction_split_name)
    if not split_dir or not split_name:
        return {}
    path = resolve_path(split_dir) / f"stage3_{split_name}.jsonl"
    if not path.exists():
        return {}
    lookup: Dict[Tuple[object, str], Dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row.get("row_id"), row.get("input", ""))
            lookup[key] = row
    return lookup


def prediction_file_map(eval_dir: Path) -> Dict[str, Path]:
    out = {}
    for path in sorted(eval_dir.glob("predictions_*.jsonl")):
        split_name = path.name[len("predictions_") : -len(".jsonl")]
        out[split_name] = path
    return out


def item_level_prediction_rows(
    pred_path: Path,
    split_name: str,
    split_lookup: Dict[Tuple[object, str], Dict],
    legal_labels: Set[str],
    run_name: str,
    experiment: str,
    system_name: str,
    seed: int,
) -> List[Dict]:
    rows: List[Dict] = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pred_row = json.loads(line)
            key = (pred_row.get("row_id"), pred_row.get("input", ""))
            meta = split_lookup.get(key, {})
            gold_text = pred_row.get("human_gold", "")
            pred_text = pred_row.get("model_prediction", "")
            gold_tags = extract_valid_tags(gold_text)
            pred_tags = extract_valid_tags(pred_text)
            candidate_tags = extract_candidate_error_tags(pred_text)
            syntax_valid = int(all(VALID_TAG_FULL_RE.match(t) for t in candidate_tags))
            invalid_label_count = sum(1 for t in pred_tags if legal_labels and t not in legal_labels)
            gold_primary = gold_tags[0] if gold_tags else ""
            pred_primary = pred_tags[0] if pred_tags else ""
            rows.append(
                {
                    "run_name": run_name,
                    "experiment": experiment,
                    "system_key": system_name,
                    "seed": seed,
                    "split": split_name,
                    "row_id": pred_row.get("row_id"),
                    "input": pred_row.get("input", ""),
                    "provenance_label": meta.get("provenance_label", pred_row.get("provenance_label", "")),
                    "trace_method": meta.get("trace_method", ""),
                    "trace_ambiguous": int(bool(meta.get("trace_ambiguous", False))),
                    "source_file_count": int(meta.get("source_file_count", 0) or 0),
                    "error_count": int(meta.get("error_count", pred_row.get("error_count", 0)) or 0),
                    "gold_has_error": int(bool(gold_tags)),
                    "pred_has_error": int(bool(pred_tags)),
                    "exact_text_match": int(gold_text.strip() == pred_text.strip()),
                    "exact_tag_match": int(gold_tags == pred_tags),
                    "primary_tag_correct": int(gold_primary == pred_primary and bool(gold_primary or pred_primary)),
                    "gold_primary_tag": gold_primary,
                    "pred_primary_tag": pred_primary,
                    "gold_tag_count": len(gold_tags),
                    "pred_tag_count": len(pred_tags),
                    "syntax_valid": syntax_valid,
                    "invalid_label_count": invalid_label_count,
                    "reconstruction_presence_match": int(bool(marker_signature(gold_text)) == bool(marker_signature(pred_text))),
                    "reconstruction_signature_match": int(marker_signature(gold_text) == marker_signature(pred_text)),
                }
            )
    return rows


def summarize_item_rows(rows: List[Dict], group_fields: Sequence[str]) -> List[Dict]:
    grouped: Dict[Tuple[object, ...], List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)
    out: List[Dict] = []
    for key, bucket in sorted(grouped.items()):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary.update(
            {
                "n": len(bucket),
                "exact_text_accuracy": round(safe_mean([r["exact_text_match"] for r in bucket]), 6),
                "exact_tag_accuracy": round(safe_mean([r["exact_tag_match"] for r in bucket]), 6),
                "primary_tag_accuracy": round(safe_mean([r["primary_tag_correct"] for r in bucket]), 6),
                "pred_error_rate": round(safe_mean([r["pred_has_error"] for r in bucket]), 6),
                "gold_error_rate": round(safe_mean([r["gold_has_error"] for r in bucket]), 6),
                "syntax_validity_rate": round(safe_mean([r["syntax_valid"] for r in bucket]), 6),
            }
        )
        out.append(summary)
    return out


def analyze_predictions(
    pred_path: Path,
    legal_labels: Set[str],
    focus_a: str,
    focus_b: str,
    focus_c: str,
    holdout_labels: Sequence[str],
) -> Dict:
    rows_total = 0
    rows_with_gold_error = 0

    syntax_candidate_total = 0
    syntax_valid_total = 0
    syntax_invalid_total = 0

    valid_pred_tag_total = 0
    invalid_legal_label_total = 0

    recon_presence_correct = 0
    recon_signature_exact = 0

    confusion = Counter()
    gold_support = Counter()

    holdout_set = set(holdout_labels)
    holdout_rows = 0
    holdout_exact = 0
    holdout_operator = 0
    holdout_support = Counter()
    holdout_exact_per_label = Counter()
    holdout_operator_per_label = Counter()

    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_total += 1

            gold_text = row.get("human_gold", "")
            pred_text = row.get("model_prediction", "")

            candidate_tags = extract_candidate_error_tags(pred_text)
            syntax_candidate_total += len(candidate_tags)
            valid_candidates = [t for t in candidate_tags if VALID_TAG_FULL_RE.match(t)]
            syntax_valid_total += len(valid_candidates)
            syntax_invalid_total += len(candidate_tags) - len(valid_candidates)

            gold_tags = extract_valid_tags(gold_text)
            pred_tags = extract_valid_tags(pred_text)

            valid_pred_tag_total += len(pred_tags)
            if legal_labels:
                invalid_legal_label_total += sum(1 for t in pred_tags if t not in legal_labels)

            gold_sig = marker_signature(gold_text)
            pred_sig = marker_signature(pred_text)
            recon_presence_correct += int(bool(gold_sig) == bool(pred_sig))
            recon_signature_exact += int(gold_sig == pred_sig)

            if gold_tags:
                rows_with_gold_error += 1
                primary_gold = gold_tags[0]
                primary_pred = pred_tags[0] if pred_tags else "<none>"
                confusion[(primary_gold, primary_pred)] += 1
                gold_support[primary_gold] += 1

            holdout_gold = first_holdout_label(gold_tags, holdout_set)
            if holdout_gold is not None:
                holdout_rows += 1
                holdout_support[holdout_gold] += 1
                holdout_pred = first_holdout_label(pred_tags, holdout_set)
                if holdout_pred == holdout_gold:
                    holdout_exact += 1
                    holdout_exact_per_label[holdout_gold] += 1
                if holdout_pred is not None and operator_family(holdout_pred) == operator_family(holdout_gold):
                    holdout_operator += 1
                    holdout_operator_per_label[holdout_gold] += 1

    top_confusions = []
    for (gold, pred), count in confusion.most_common(25):
        support = gold_support[gold]
        top_confusions.append(
            {
                "gold": gold,
                "pred": pred,
                "count": count,
                "rate_within_gold": pct(count, support),
                "gold_support": support,
            }
        )

    holdout_per_label = {}
    for label in holdout_labels:
        support = holdout_support.get(label, 0)
        holdout_per_label[label] = {
            "support": support,
            "exact_accuracy": pct(holdout_exact_per_label.get(label, 0), support),
            "operator_accuracy": pct(holdout_operator_per_label.get(label, 0), support),
        }

    focus_pairs = []
    for left, right in [(focus_a, focus_b), (focus_a, focus_c)]:
        left_support = gold_support.get(left, 0)
        right_support = gold_support.get(right, 0)
        left_pred_right = confusion.get((left, right), 0)
        right_pred_left = confusion.get((right, left), 0)
        focus_pairs.append(
            {
                "focus_left": left,
                "focus_right": right,
                "focus_left_support": left_support,
                "focus_right_support": right_support,
                "focus_left_pred_right_count": left_pred_right,
                "focus_right_pred_left_count": right_pred_left,
                "focus_left_pred_right_rate": pct(left_pred_right, left_support),
                "focus_right_pred_left_rate": pct(right_pred_left, right_support),
            }
        )

    primary_pair = focus_pairs[0]

    return {
        "rows_total": rows_total,
        "rows_with_gold_error": rows_with_gold_error,
        "syntax_candidate_total": syntax_candidate_total,
        "syntax_valid_total": syntax_valid_total,
        "syntax_invalid_total": syntax_invalid_total,
        "syntax_validity_rate": pct(syntax_valid_total, syntax_candidate_total),
        "invalid_label_rate": pct(invalid_legal_label_total, valid_pred_tag_total),
        "reconstruction_presence_accuracy": pct(recon_presence_correct, rows_total),
        "reconstruction_signature_accuracy": pct(recon_signature_exact, rows_total),
        "focus_a": primary_pair["focus_left"],
        "focus_b": primary_pair["focus_right"],
        "focus_a_support": primary_pair["focus_left_support"],
        "focus_b_support": primary_pair["focus_right_support"],
        "focus_a_pred_b_count": primary_pair["focus_left_pred_right_count"],
        "focus_b_pred_a_count": primary_pair["focus_right_pred_left_count"],
        "focus_a_pred_b_rate": primary_pair["focus_left_pred_right_rate"],
        "focus_b_pred_a_rate": primary_pair["focus_right_pred_left_rate"],
        "focus_pairs": focus_pairs,
        "top_confusions": top_confusions,
        "holdout_rows": holdout_rows,
        "holdout_exact_label_accuracy": pct(holdout_exact, holdout_rows),
        "holdout_operator_accuracy": pct(holdout_operator, holdout_rows),
        "holdout_per_label": holdout_per_label,
    }


def system_key(run_name: str) -> str:
    return SEED_SUFFIX_RE.sub("", run_name)


def write_csv(path: Path, rows: Iterable[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    results_root = resolve_path(args.results_root)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = list_run_dirs(results_root, args.run)
    if not run_dirs:
        raise SystemExit("No runs with eval_outputs/run_summary.json found.")

    run_metrics_rows: List[Dict] = []
    syntax_rows: List[Dict] = []
    focus_rows: List[Dict] = []
    focus_3sg_past_rows: List[Dict] = []
    reconstruction_rows: List[Dict] = []
    holdout_rows: List[Dict] = []
    confusion_rows: List[Dict] = []
    overfitting_rows: List[Dict] = []
    item_level_rows: List[Dict] = []

    all_summary = {"runs": []}
    seed_agg = defaultdict(lambda: defaultdict(lambda: {"micro_f1": [], "exact_match": []}))

    split_order = ["eval_real", "test_real", "eval_coverage", "test_coverage"]
    if args.include_holdout:
        split_order.append("holdout_generalization")

    for run_dir in run_dirs:
        run_name = run_dir.name
        eval_dir = run_dir / "eval_outputs"
        run_summary_path = eval_dir / "run_summary.json"
        run_summary = load_json(run_summary_path)

        split_dir = str(run_summary.get("split_dir", ""))
        seed = int(run_summary.get("seed", 0))
        experiment = infer_experiment(run_name, split_dir)

        legal_labels = legal_label_set_for_split_dir(split_dir)
        holdout_labels = holdout_labels_for_split_dir(split_dir)
        prediction_files = prediction_file_map(eval_dir)

        run_out = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "seed": seed,
            "split_dir": split_dir,
            "experiment": experiment,
            "splits": {},
            "prediction_diagnostics": {},
        }

        for split_name in split_order:
            metrics_path = eval_dir / f"metrics_{split_name}.json"
            if not metrics_path.exists():
                continue
            metrics = load_json(metrics_path)
            per_label = metrics.get("per_label", [])
            macro_f1 = safe_mean([float(x.get("f1", 0.0)) for x in per_label]) if per_label else 0.0
            ci = metrics.get("micro_f1_bootstrap", {}) or {}

            row = {
                "run_name": run_name,
                "experiment": experiment,
                "system_key": system_key(run_name),
                "seed": seed,
                "split_dir": split_dir,
                "split": split_name,
                "n": int(metrics.get("n", 0)),
                "exact_match": float(metrics.get("exact_match", 0.0)),
                "micro_f1": float(metrics.get("micro_f1", 0.0)),
                "macro_f1": float(macro_f1),
                "ci_low": float(ci.get("ci_low", 0.0)),
                "ci_high": float(ci.get("ci_high", 0.0)),
                "confirmatory_min_support": int(metrics.get("confirmatory_min_support", 0)),
                "confirmatory_labels": len(metrics.get("per_label_confirmatory", []) or []),
                "exploratory_labels": len(metrics.get("per_label_exploratory", []) or []),
            }
            run_metrics_rows.append(row)
            run_out["splits"][split_name] = row

            seed_agg[system_key(run_name)][split_name]["micro_f1"].append(row["micro_f1"])
            seed_agg[system_key(run_name)][split_name]["exact_match"].append(row["exact_match"])

        prediction_diagnostic_splits = ["test_real", "test_coverage"]
        if args.include_holdout:
            prediction_diagnostic_splits.append("holdout_generalization")
        for pred_split in prediction_diagnostic_splits:
            pred_path = eval_dir / f"predictions_{pred_split}.jsonl"
            if not pred_path.exists():
                continue
            pred_diag = analyze_predictions(
                pred_path=pred_path,
                legal_labels=legal_labels,
                focus_a=args.focus_tag_a,
                focus_b=args.focus_tag_b,
                focus_c=args.focus_tag_c,
                holdout_labels=holdout_labels,
            )
            run_out["prediction_diagnostics"][pred_split] = pred_diag

            syntax_rows.append(
                {
                    "run_name": run_name,
                    "experiment": experiment,
                    "split": pred_split,
                    "syntax_candidate_total": pred_diag["syntax_candidate_total"],
                    "syntax_valid_total": pred_diag["syntax_valid_total"],
                    "syntax_invalid_total": pred_diag["syntax_invalid_total"],
                    "syntax_validity_rate": pred_diag["syntax_validity_rate"],
                    "invalid_label_rate": pred_diag["invalid_label_rate"],
                }
            )

            focus_rows.append(
                {
                    "run_name": run_name,
                    "experiment": experiment,
                    "split": pred_split,
                    "focus_a": pred_diag["focus_a"],
                    "focus_b": pred_diag["focus_b"],
                    "focus_a_support": pred_diag["focus_a_support"],
                    "focus_b_support": pred_diag["focus_b_support"],
                    "focus_a_pred_b_count": pred_diag["focus_a_pred_b_count"],
                    "focus_b_pred_a_count": pred_diag["focus_b_pred_a_count"],
                    "focus_a_pred_b_rate": pred_diag["focus_a_pred_b_rate"],
                    "focus_b_pred_a_rate": pred_diag["focus_b_pred_a_rate"],
                }
            )
            for pair in pred_diag.get("focus_pairs", []):
                focus_3sg_past_rows.append(
                    {
                        "run_name": run_name,
                        "experiment": experiment,
                        "split": pred_split,
                        "focus_left": pair["focus_left"],
                        "focus_right": pair["focus_right"],
                        "focus_left_support": pair["focus_left_support"],
                        "focus_right_support": pair["focus_right_support"],
                        "focus_left_pred_right_count": pair["focus_left_pred_right_count"],
                        "focus_right_pred_left_count": pair["focus_right_pred_left_count"],
                        "focus_left_pred_right_rate": pair["focus_left_pred_right_rate"],
                        "focus_right_pred_left_rate": pair["focus_right_pred_left_rate"],
                    }
                )

            reconstruction_rows.append(
                {
                    "run_name": run_name,
                    "experiment": experiment,
                    "split": pred_split,
                    "rows_total": pred_diag["rows_total"],
                    "reconstruction_presence_accuracy": pred_diag["reconstruction_presence_accuracy"],
                    "reconstruction_signature_accuracy": pred_diag["reconstruction_signature_accuracy"],
                }
            )

            if args.include_holdout:
                holdout_rows.append(
                    {
                        "run_name": run_name,
                        "experiment": experiment,
                        "split": pred_split,
                        "holdout_rows": pred_diag["holdout_rows"],
                        "holdout_exact_label_accuracy": pred_diag["holdout_exact_label_accuracy"],
                        "holdout_operator_accuracy": pred_diag["holdout_operator_accuracy"],
                    }
                )

            for conf in pred_diag["top_confusions"]:
                confusion_rows.append(
                    {
                        "run_name": run_name,
                        "experiment": experiment,
                        "split": pred_split,
                        "gold": conf["gold"],
                        "pred": conf["pred"],
                        "count": conf["count"],
                        "rate_within_gold": conf["rate_within_gold"],
                        "gold_support": conf["gold_support"],
                    }
                )

        for pred_split, pred_path in prediction_files.items():
            split_lookup = split_metadata_lookup(split_dir, pred_split)
            item_level_rows.extend(
                item_level_prediction_rows(
                    pred_path=pred_path,
                    split_name=pred_split,
                    split_lookup=split_lookup,
                    legal_labels=legal_labels,
                    run_name=run_name,
                    experiment=experiment,
                    system_name=system_key(run_name),
                    seed=seed,
                )
            )

        all_summary["runs"].append(run_out)

        # Generalization-gap diagnostics (overfitting proxy).
        eval_real = run_out["splits"].get("eval_real")
        test_real = run_out["splits"].get("test_real")
        eval_cov = run_out["splits"].get("eval_coverage")
        test_cov = run_out["splits"].get("test_coverage")
        if eval_real and test_real:
            micro_gap = float(eval_real["micro_f1"]) - float(test_real["micro_f1"])
            exact_gap = float(eval_real["exact_match"]) - float(test_real["exact_match"])
            overfitting_rows.append(
                {
                    "run_name": run_name,
                    "experiment": experiment,
                    "system_key": system_key(run_name),
                    "seed": seed,
                    "split_pair": "real_eval_to_real_test",
                    "eval_split": "eval_real",
                    "test_split": "test_real",
                    "eval_micro_f1": round(float(eval_real["micro_f1"]), 6),
                    "test_micro_f1": round(float(test_real["micro_f1"]), 6),
                    "micro_f1_gap": round(micro_gap, 6),
                    "eval_exact_match": round(float(eval_real["exact_match"]), 6),
                    "test_exact_match": round(float(test_real["exact_match"]), 6),
                    "exact_match_gap": round(exact_gap, 6),
                    "micro_gap_flag_ge_0_05": int(micro_gap >= 0.05),
                    "micro_gap_flag_ge_0_10": int(micro_gap >= 0.10),
                }
            )
        if eval_cov and test_cov:
            micro_gap = float(eval_cov["micro_f1"]) - float(test_cov["micro_f1"])
            exact_gap = float(eval_cov["exact_match"]) - float(test_cov["exact_match"])
            overfitting_rows.append(
                {
                    "run_name": run_name,
                    "experiment": experiment,
                    "system_key": system_key(run_name),
                    "seed": seed,
                    "split_pair": "coverage_eval_to_coverage_test",
                    "eval_split": "eval_coverage",
                    "test_split": "test_coverage",
                    "eval_micro_f1": round(float(eval_cov["micro_f1"]), 6),
                    "test_micro_f1": round(float(test_cov["micro_f1"]), 6),
                    "micro_f1_gap": round(micro_gap, 6),
                    "eval_exact_match": round(float(eval_cov["exact_match"]), 6),
                    "test_exact_match": round(float(test_cov["exact_match"]), 6),
                    "exact_match_gap": round(exact_gap, 6),
                    "micro_gap_flag_ge_0_05": int(micro_gap >= 0.05),
                    "micro_gap_flag_ge_0_10": int(micro_gap >= 0.10),
                }
            )

    experiment6_rows = []
    for sys_key, split_map in sorted(seed_agg.items()):
        for split_name, vals in sorted(split_map.items()):
            micro_vals = vals["micro_f1"]
            exact_vals = vals["exact_match"]
            experiment6_rows.append(
                {
                    "system_key": sys_key,
                    "split": split_name,
                    "n_runs": len(micro_vals),
                    "micro_f1_mean": round(safe_mean(micro_vals), 6),
                    "micro_f1_std": round(safe_std(micro_vals), 6),
                    "exact_match_mean": round(safe_mean(exact_vals), 6),
                    "exact_match_std": round(safe_std(exact_vals), 6),
                }
            )

    write_csv(
        out_dir / "run_metrics.csv",
        run_metrics_rows,
        [
            "run_name",
            "experiment",
            "system_key",
            "seed",
            "split_dir",
            "split",
            "n",
            "exact_match",
            "micro_f1",
            "macro_f1",
            "ci_low",
            "ci_high",
            "confirmatory_min_support",
            "confirmatory_labels",
            "exploratory_labels",
        ],
    )

    if syntax_rows:
        write_csv(
            out_dir / "syntax_validity.csv",
            syntax_rows,
            [
                "run_name",
                "experiment",
                "split",
                "syntax_candidate_total",
                "syntax_valid_total",
                "syntax_invalid_total",
                "syntax_validity_rate",
                "invalid_label_rate",
            ],
        )

    if focus_rows:
        write_csv(
            out_dir / "focus_confusion_03s_vs_0ed.csv",
            focus_rows,
            [
                "run_name",
                "experiment",
                "split",
                "focus_a",
                "focus_b",
                "focus_a_support",
                "focus_b_support",
                "focus_a_pred_b_count",
                "focus_b_pred_a_count",
                "focus_a_pred_b_rate",
                "focus_b_pred_a_rate",
            ],
        )

    if focus_3sg_past_rows:
        write_csv(
            out_dir / "focus_confusion_3sg_vs_past.csv",
            focus_3sg_past_rows,
            [
                "run_name",
                "experiment",
                "split",
                "focus_left",
                "focus_right",
                "focus_left_support",
                "focus_right_support",
                "focus_left_pred_right_count",
                "focus_right_pred_left_count",
                "focus_left_pred_right_rate",
                "focus_right_pred_left_rate",
            ],
        )

    if reconstruction_rows:
        write_csv(
            out_dir / "reconstruction_diagnostics.csv",
            reconstruction_rows,
            [
                "run_name",
                "experiment",
                "split",
                "rows_total",
                "reconstruction_presence_accuracy",
                "reconstruction_signature_accuracy",
            ],
        )

    if args.include_holdout and holdout_rows:
        write_csv(
            out_dir / "holdout_generalization_diagnostics.csv",
            holdout_rows,
            [
                "run_name",
                "experiment",
                "split",
                "holdout_rows",
                "holdout_exact_label_accuracy",
                "holdout_operator_accuracy",
            ],
        )

    if confusion_rows:
        write_csv(
            out_dir / "top_confusions.csv",
            confusion_rows,
            [
                "run_name",
                "experiment",
                "split",
                "gold",
                "pred",
                "count",
                "rate_within_gold",
                "gold_support",
            ],
        )

    write_csv(
        out_dir / "seed_stability_summary.csv",
        experiment6_rows,
        ["system_key", "split", "n_runs", "micro_f1_mean", "micro_f1_std", "exact_match_mean", "exact_match_std"],
    )

    if overfitting_rows:
        write_csv(
            out_dir / "overfitting_report.csv",
            overfitting_rows,
            [
                "run_name",
                "experiment",
                "system_key",
                "seed",
                "split_pair",
                "eval_split",
                "test_split",
                "eval_micro_f1",
                "test_micro_f1",
                "micro_f1_gap",
                "eval_exact_match",
                "test_exact_match",
                "exact_match_gap",
                "micro_gap_flag_ge_0_05",
                "micro_gap_flag_ge_0_10",
            ],
        )

    if item_level_rows:
        write_csv(
            out_dir / "mixed_model_ready_items.csv",
            item_level_rows,
            [
                "run_name",
                "experiment",
                "system_key",
                "seed",
                "split",
                "row_id",
                "input",
                "provenance_label",
                "trace_method",
                "trace_ambiguous",
                "source_file_count",
                "error_count",
                "gold_has_error",
                "pred_has_error",
                "exact_text_match",
                "exact_tag_match",
                "primary_tag_correct",
                "gold_primary_tag",
                "pred_primary_tag",
                "gold_tag_count",
                "pred_tag_count",
                "syntax_valid",
                "invalid_label_count",
                "reconstruction_presence_match",
                "reconstruction_signature_match",
            ],
        )

        synthetic_rows = [row for row in item_level_rows if row.get("provenance_label") == "synthetic"]
        write_csv(
            out_dir / "synthetic_artifact_summary.csv",
            summarize_item_rows(
                synthetic_rows,
                ["run_name", "experiment", "system_key", "split", "trace_method", "error_count"],
            ),
            [
                "run_name",
                "experiment",
                "system_key",
                "split",
                "trace_method",
                "error_count",
                "n",
                "exact_text_accuracy",
                "exact_tag_accuracy",
                "primary_tag_accuracy",
                "pred_error_rate",
                "gold_error_rate",
                "syntax_validity_rate",
            ],
        )
        write_csv(
            out_dir / "provenance_summary.csv",
            summarize_item_rows(
                item_level_rows,
                ["run_name", "experiment", "system_key", "split", "provenance_label"],
            ),
            [
                "run_name",
                "experiment",
                "system_key",
                "split",
                "provenance_label",
                "n",
                "exact_text_accuracy",
                "exact_tag_accuracy",
                "primary_tag_accuracy",
                "pred_error_rate",
                "gold_error_rate",
                "syntax_validity_rate",
            ],
        )

    (out_dir / "paper_analysis_summary.json").write_text(
        json.dumps(all_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Lightweight markdown overview for quick paper drafting.
    by_exp = Counter(row["experiment"] for row in run_metrics_rows if row["split"] == "test_real")
    lines = [
        "# Paper Analysis Overview",
        "",
        "## Runs detected (test_real)",
    ]
    for exp_name, n in sorted(by_exp.items()):
        lines.append(f"- {exp_name}: {n} run(s)")
    lines.extend(
        [
            "",
            "## Shared analyses across experiments",
            "- `run_metrics.csv`: exact match, micro-F1, macro-F1, CI, confirmatory/exploratory label counts",
            "- `syntax_validity.csv`: syntax-valid error-tag rate + invalid-label rate",
            "- `top_confusions.csv`: top gold->pred confusions from prediction files",
            f"- `focus_confusion_03s_vs_0ed.csv`: targeted confusion between `{args.focus_tag_a}` and `{args.focus_tag_b}`",
            f"- `focus_confusion_3sg_vs_past.csv`: third-person singular vs regular/irregular past diagnostics (`{args.focus_tag_b}`, `{args.focus_tag_c}`)",
            "- `overfitting_report.csv`: eval->test generalization gaps (overfitting proxy)",
            "- `mixed_model_ready_items.csv`: item-level merged prediction table for R mixed models",
            "- `synthetic_artifact_summary.csv`: grouped synthetic-only performance by trace method and error count",
            "- `provenance_summary.csv`: grouped performance by provenance label",
            "",
            "## Experiment-specific analyses",
            "- Exp1/Exp2: use shared analyses; for Exp2 compare deltas vs matched Exp1 baseline",
            "- Exp3: add `reconstruction_diagnostics.csv` (marker presence/signature agreement)",
            "- Exp6: use `seed_stability_summary.csv` (mean/std across seed-matched systems)",
        ]
    )
    if args.include_holdout:
        lines.insert(
            lines.index("- Exp6: use `seed_stability_summary.csv` (mean/std across seed-matched systems)"),
            "- Exp4: add `holdout_generalization_diagnostics.csv` (exact held-out label + operator-family accuracy)",
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Analyzed {len(run_dirs)} run(s). Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
