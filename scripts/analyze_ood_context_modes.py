from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
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
    changed_items: List[Dict] = []
    review_rows: List[Dict] = []

    for mode, lookup in sorted(by_mode_lookup.items()):
        if mode == args.baseline_mode:
            continue

        exact_changes = 0
        tagset_changes = 0
        baseline_unc = []
        mode_unc = []
        baseline_unc_changed = []
        mode_unc_changed = []

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

            base_u = maybe_float(base.get("uncertainty_mean_token_logprob"))
            alt_u = maybe_float(alt.get("uncertainty_mean_token_logprob"))
            if is_finite(base_u):
                baseline_unc.append(base_u)
            if is_finite(alt_u):
                mode_unc.append(alt_u)
            if changed and is_finite(base_u):
                baseline_unc_changed.append(base_u)
            if changed and is_finite(alt_u):
                mode_unc_changed.append(alt_u)

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
        pairwise_rows.append(
            {
                "baseline_mode": args.baseline_mode,
                "mode": mode,
                "n_rows": n,
                "exact_change_count": exact_changes,
                "exact_change_rate": exact_changes / n if n else 0.0,
                "tagset_change_count": tagset_changes,
                "tagset_change_rate": tagset_changes / n if n else 0.0,
                "baseline_mean_token_logprob_mean": safe_mean(baseline_unc),
                "mode_mean_token_logprob_mean": safe_mean(mode_unc),
                "delta_mean_token_logprob": safe_mean(mode_unc) - safe_mean(baseline_unc)
                if baseline_unc and mode_unc
                else float("nan"),
                "changed_only_baseline_mean_token_logprob_mean": safe_mean(baseline_unc_changed),
                "changed_only_mode_mean_token_logprob_mean": safe_mean(mode_unc_changed),
                "changed_only_delta_mean_token_logprob": safe_mean(mode_unc_changed) - safe_mean(baseline_unc_changed)
                if baseline_unc_changed and mode_unc_changed
                else float("nan"),
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
            "baseline_mean_token_logprob_mean",
            "mode_mean_token_logprob_mean",
            "delta_mean_token_logprob",
            "changed_only_baseline_mean_token_logprob_mean",
            "changed_only_mode_mean_token_logprob_mean",
            "changed_only_delta_mean_token_logprob",
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
            "changed_items_csv": str(out_dir / "changed_items.csv"),
            "manual_review_changed_outputs_csv": str(out_dir / "manual_review_changed_outputs.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
