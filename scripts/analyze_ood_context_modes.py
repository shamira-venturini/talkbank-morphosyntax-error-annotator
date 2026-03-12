from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List

from common import resolve_path


TAG_RE = re.compile(r"\[\*\s*[ms](?::[^\]]+)?\]")
UNCERTAINTY_FIELDS = [
    "uncertainty_seq_logprob",
    "uncertainty_mean_token_logprob",
    "uncertainty_min_token_logprob",
    "uncertainty_mean_token_margin",
    "uncertainty_min_token_margin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OOD predictions across context modes.")
    parser.add_argument(
        "--predictions-dir",
        default="results/ood_vercellotti",
        help="Directory containing predictions_<mode>.jsonl files.",
    )
    parser.add_argument(
        "--baseline-mode",
        default="utterance_only",
        help="Baseline context mode for pairwise comparisons.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/ood_vercellotti/context_analysis",
        help="Output directory.",
    )
    parser.add_argument(
        "--changed-only",
        action="store_true",
        default=False,
        help="Restrict statistical comparisons to rows whose decoded outputs differ from baseline.",
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


def write_csv(path: Path, rows: Iterable[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def canonical_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    return f"[* {body}]"


def tag_set(text: str) -> set[str]:
    if not isinstance(text, str):
        return set()
    return {canonical_tag(tag) for tag in TAG_RE.findall(text)}


def prediction_files(pred_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in sorted(pred_dir.glob("predictions_*.jsonl")):
        mode = path.name[len("predictions_") : -len(".jsonl")]
        out[mode] = path
    return out


def safe_mean(values: List[float]) -> float:
    return mean(values) if values else float("nan")


def maybe_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_finite(x: float) -> bool:
    return not (x != x)


def exact_two_sided_sign_test(pos_count: int, neg_count: int) -> float:
    n = pos_count + neg_count
    if n == 0:
        return float("nan")
    try:
        from scipy.stats import binomtest

        return float(binomtest(pos_count, n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        # Normal approximation fallback when scipy is unavailable.
        import math

        expected = n / 2.0
        variance = n / 4.0
        if variance == 0:
            return float("nan")
        z = (abs(pos_count - expected) - 0.5) / math.sqrt(variance)
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        return max(0.0, min(1.0, 2.0 * (1.0 - cdf)))


def paired_stats(baseline_values: List[float], mode_values: List[float]) -> Dict[str, float]:
    paired = [
        (base, alt)
        for base, alt in zip(baseline_values, mode_values)
        if is_finite(base) and is_finite(alt)
    ]
    if not paired:
        return {
            "n_pairs": 0,
            "baseline_mean": float("nan"),
            "mode_mean": float("nan"),
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
            "positive_delta_count": 0,
            "negative_delta_count": 0,
            "zero_delta_count": 0,
            "sign_test_pvalue": float("nan"),
        }

    deltas = [alt - base for base, alt in paired]
    pos_count = sum(1 for d in deltas if d > 0)
    neg_count = sum(1 for d in deltas if d < 0)
    zero_count = len(deltas) - pos_count - neg_count
    return {
        "n_pairs": len(deltas),
        "baseline_mean": mean(base for base, _ in paired),
        "mode_mean": mean(alt for _, alt in paired),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "positive_delta_count": pos_count,
        "negative_delta_count": neg_count,
        "zero_delta_count": zero_count,
        "sign_test_pvalue": exact_two_sided_sign_test(pos_count, neg_count),
    }


def main() -> None:
    args = parse_args()
    pred_dir = resolve_path(args.predictions_dir)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = prediction_files(pred_dir)
    if not files:
        raise SystemExit(f"No prediction files found under: {pred_dir}")
    if args.baseline_mode not in files:
        raise SystemExit(f"Missing baseline file predictions_{args.baseline_mode}.jsonl in {pred_dir}")

    by_mode_rows: Dict[str, List[Dict]] = {mode: load_jsonl(path) for mode, path in files.items()}
    by_mode_lookup: Dict[str, Dict[int, Dict]] = {}
    for mode, rows in by_mode_rows.items():
        by_mode_lookup[mode] = {int(row["row_id"]): row for row in rows}

    shared_row_ids = sorted(set.intersection(*(set(lookup.keys()) for lookup in by_mode_lookup.values())))
    if not shared_row_ids:
        raise SystemExit("No shared row_id set across modes.")

    # Mode-level summary
    mode_summary_rows: List[Dict] = []
    for mode, lookup in sorted(by_mode_lookup.items()):
        rows = [lookup[rid] for rid in shared_row_ids]
        summary = {
            "mode": mode,
            "n_rows": len(rows),
            "mean_pred_tag_count": safe_mean([float(r.get("pred_tag_count", 0)) for r in rows]),
            "mean_context_utterance_count": safe_mean([float(r.get("context_utterance_count", 0)) for r in rows]),
            "mean_context_char_count": safe_mean([float(r.get("context_char_count", 0)) for r in rows]),
        }
        for field in UNCERTAINTY_FIELDS:
            vals = [maybe_float(r.get(field)) for r in rows]
            vals = [v for v in vals if is_finite(v)]
            summary[f"mean_{field}"] = safe_mean(vals) if vals else float("nan")
        mode_summary_rows.append(summary)

    write_csv(
        out_dir / "context_mode_summary.csv",
        mode_summary_rows,
        [
            "mode",
            "n_rows",
            "mean_pred_tag_count",
            "mean_context_utterance_count",
            "mean_context_char_count",
            *[f"mean_{f}" for f in UNCERTAINTY_FIELDS],
        ],
    )

    # Pairwise vs baseline
    baseline = by_mode_lookup[args.baseline_mode]
    pairwise_rows: List[Dict] = []
    paired_uncertainty_rows: List[Dict] = []
    changed_items: List[Dict] = []
    review_rows: List[Dict] = []

    for mode, lookup in sorted(by_mode_lookup.items()):
        if mode == args.baseline_mode:
            continue

        exact_changes = 0
        tagset_changes = 0
        uncertainty_pairs = {field: {"baseline": [], "mode": []} for field in UNCERTAINTY_FIELDS}
        uncertainty_pairs_changed = {field: {"baseline": [], "mode": []} for field in UNCERTAINTY_FIELDS}

        for rid in shared_row_ids:
            base = baseline[rid]
            alt = lookup[rid]
            base_pred = (base.get("model_prediction") or "").strip()
            alt_pred = (alt.get("model_prediction") or "").strip()
            same_exact = int(base_pred == alt_pred)
            same_tagset = int(tag_set(base_pred) == tag_set(alt_pred))
            changed = int(not same_exact)
            if changed:
                exact_changes += 1
            if not same_tagset:
                tagset_changes += 1

            for field in UNCERTAINTY_FIELDS:
                base_u = maybe_float(base.get(field))
                alt_u = maybe_float(alt.get(field))
                if is_finite(base_u) and is_finite(alt_u):
                    uncertainty_pairs[field]["baseline"].append(base_u)
                    uncertainty_pairs[field]["mode"].append(alt_u)
                    if changed:
                        uncertainty_pairs_changed[field]["baseline"].append(base_u)
                        uncertainty_pairs_changed[field]["mode"].append(alt_u)

            if changed:
                changed_items.append(
                    {
                        "row_id": rid,
                        "mode": mode,
                        "file_name": base.get("file_name", ""),
                        "speaker": base.get("speaker", ""),
                        "line_no": base.get("line_no", ""),
                        "input": base.get("input", ""),
                        "baseline_prediction": base_pred,
                        "mode_prediction": alt_pred,
                        "baseline_tag_count": base.get("pred_tag_count", 0),
                        "mode_tag_count": alt.get("pred_tag_count", 0),
                        "baseline_mean_token_logprob": base.get("uncertainty_mean_token_logprob"),
                        "mode_mean_token_logprob": alt.get("uncertainty_mean_token_logprob"),
                        "baseline_min_token_margin": base.get("uncertainty_min_token_margin"),
                        "mode_min_token_margin": alt.get("uncertainty_min_token_margin"),
                    }
                )
                review_rows.append(
                    {
                        "row_id": rid,
                        "mode": mode,
                        "file_name": base.get("file_name", ""),
                        "speaker": base.get("speaker", ""),
                        "line_no": base.get("line_no", ""),
                        "input": base.get("input", ""),
                        "baseline_prediction": base_pred,
                        "mode_prediction": alt_pred,
                        "preferred_output": "",
                        "is_ambiguous": "",
                        "notes": "",
                    }
                )

        n = len(shared_row_ids)
        summary_field = "uncertainty_mean_token_logprob"
        source_pairs = uncertainty_pairs_changed if args.changed_only else uncertainty_pairs
        summary_stats = paired_stats(
            source_pairs[summary_field]["baseline"],
            source_pairs[summary_field]["mode"],
        )
        pairwise_rows.append(
            {
                "baseline_mode": args.baseline_mode,
                "mode": mode,
                "n_rows": n,
                "exact_change_count": exact_changes,
                "exact_change_rate": exact_changes / n if n else 0.0,
                "tagset_change_count": tagset_changes,
                "tagset_change_rate": tagset_changes / n if n else 0.0,
                "paired_subset": "changed_only" if args.changed_only else "all_rows",
                "baseline_mean_token_logprob_mean": summary_stats["baseline_mean"],
                "mode_mean_token_logprob_mean": summary_stats["mode_mean"],
                "delta_mean_token_logprob": summary_stats["mean_delta"],
                "delta_mean_token_logprob_median": summary_stats["median_delta"],
                "delta_mean_token_logprob_sign_test_pvalue": summary_stats["sign_test_pvalue"],
                "delta_mean_token_logprob_positive_count": summary_stats["positive_delta_count"],
                "delta_mean_token_logprob_negative_count": summary_stats["negative_delta_count"],
                "delta_mean_token_logprob_zero_count": summary_stats["zero_delta_count"],
                "delta_mean_token_logprob_n_pairs": summary_stats["n_pairs"],
            }
        )

        for field in UNCERTAINTY_FIELDS:
            stats = paired_stats(
                source_pairs[field]["baseline"],
                source_pairs[field]["mode"],
            )
            paired_uncertainty_rows.append(
                {
                    "baseline_mode": args.baseline_mode,
                    "mode": mode,
                    "paired_subset": "changed_only" if args.changed_only else "all_rows",
                    "uncertainty_field": field,
                    "n_pairs": stats["n_pairs"],
                    "baseline_mean": stats["baseline_mean"],
                    "mode_mean": stats["mode_mean"],
                    "mean_delta": stats["mean_delta"],
                    "median_delta": stats["median_delta"],
                    "positive_delta_count": stats["positive_delta_count"],
                    "negative_delta_count": stats["negative_delta_count"],
                    "zero_delta_count": stats["zero_delta_count"],
                    "sign_test_pvalue": stats["sign_test_pvalue"],
                }
            )

    write_csv(
        out_dir / "pairwise_vs_baseline.csv",
        pairwise_rows,
        [
            "baseline_mode",
            "mode",
            "n_rows",
            "exact_change_count",
            "exact_change_rate",
            "tagset_change_count",
            "tagset_change_rate",
            "paired_subset",
            "baseline_mean_token_logprob_mean",
            "mode_mean_token_logprob_mean",
            "delta_mean_token_logprob",
            "delta_mean_token_logprob_median",
            "delta_mean_token_logprob_sign_test_pvalue",
            "delta_mean_token_logprob_positive_count",
            "delta_mean_token_logprob_negative_count",
            "delta_mean_token_logprob_zero_count",
            "delta_mean_token_logprob_n_pairs",
        ],
    )
    write_csv(
        out_dir / "uncertainty_paired_tests.csv",
        paired_uncertainty_rows,
        [
            "baseline_mode",
            "mode",
            "paired_subset",
            "uncertainty_field",
            "n_pairs",
            "baseline_mean",
            "mode_mean",
            "mean_delta",
            "median_delta",
            "positive_delta_count",
            "negative_delta_count",
            "zero_delta_count",
            "sign_test_pvalue",
        ],
    )

    write_csv(
        out_dir / "changed_items.csv",
        changed_items,
        [
            "row_id",
            "mode",
            "file_name",
            "speaker",
            "line_no",
            "input",
            "baseline_prediction",
            "mode_prediction",
            "baseline_tag_count",
            "mode_tag_count",
            "baseline_mean_token_logprob",
            "mode_mean_token_logprob",
            "baseline_min_token_margin",
            "mode_min_token_margin",
        ],
    )
    write_csv(
        out_dir / "manual_review_changed_outputs.csv",
        review_rows,
        [
            "row_id",
            "mode",
            "file_name",
            "speaker",
            "line_no",
            "input",
            "baseline_prediction",
            "mode_prediction",
            "preferred_output",
            "is_ambiguous",
            "notes",
        ],
    )

    summary = {
        "predictions_dir": str(pred_dir),
        "baseline_mode": args.baseline_mode,
        "modes": sorted(files.keys()),
        "shared_rows": len(shared_row_ids),
        "pairwise_rows": len(pairwise_rows),
        "changed_item_rows": len(changed_items),
        "outputs": {
            "context_mode_summary_csv": str(out_dir / "context_mode_summary.csv"),
            "pairwise_vs_baseline_csv": str(out_dir / "pairwise_vs_baseline.csv"),
            "uncertainty_paired_tests_csv": str(out_dir / "uncertainty_paired_tests.csv"),
            "changed_items_csv": str(out_dir / "changed_items.csv"),
            "manual_review_changed_outputs_csv": str(out_dir / "manual_review_changed_outputs.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
