from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from common import resolve_path


TAG_RE = re.compile(r"\[\*\s*[ms](?::[^\]]+)?\]")


def canonical_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    return f"[* {body}]"


def extract_tags(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [canonical_tag(tag) for tag in TAG_RE.findall(text)]


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_group(row: Dict) -> str:
    return str(row.get("provenance_label", "synthetic"))


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether real/synthetic training support is associated with performance on real or synthetic evaluation splits."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run directory under results/. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Per-label split to analyze. Defaults to eval_real, test_real, eval_coverage, test_coverage.",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing run outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/synthetic_data_diagnostic",
        help="Output directory for diagnostic CSV/JSON artifacts.",
    )
    return parser.parse_args()


def read_per_label_metrics(run_dir: Path, split_name: str) -> Dict[str, Dict]:
    rows: Dict[str, Dict] = {}
    for bucket in ["confirmatory", "exploratory"]:
        path = run_dir / "eval_outputs" / f"per_label_{bucket}_{split_name}.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                tag = row.get("tag", "").strip()
                if not tag:
                    continue
                rows[tag] = {
                    "support": int(float(row.get("support", 0) or 0)),
                    "precision": float(row.get("precision", 0.0) or 0.0),
                    "recall": float(row.get("recall", 0.0) or 0.0),
                    "f1": float(row.get("f1", 0.0) or 0.0),
                }
    return rows


def train_label_support(split_dir: Path) -> Dict[str, Dict[str, int]]:
    path = split_dir / "stage3_train.jsonl"
    support: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "train_rows_total": 0,
            "train_rows_real": 0,
            "train_rows_synthetic": 0,
            "train_rows_td": 0,
            "train_rows_dld": 0,
            "train_occ_total": 0,
            "train_occ_real": 0,
            "train_occ_synthetic": 0,
            "train_occ_td": 0,
            "train_occ_dld": 0,
        }
    )
    for row in load_jsonl(path):
        tags = extract_tags(row.get("output", ""))
        unique_tags = set(tags)
        prov = source_group(row)
        is_real = prov in {"TD", "DLD"}
        for tag in unique_tags:
            support[tag]["train_rows_total"] += 1
            if is_real:
                support[tag]["train_rows_real"] += 1
            if prov == "synthetic":
                support[tag]["train_rows_synthetic"] += 1
            if prov == "TD":
                support[tag]["train_rows_td"] += 1
            if prov == "DLD":
                support[tag]["train_rows_dld"] += 1
        for tag in tags:
            support[tag]["train_occ_total"] += 1
            if is_real:
                support[tag]["train_occ_real"] += 1
            if prov == "synthetic":
                support[tag]["train_occ_synthetic"] += 1
            if prov == "TD":
                support[tag]["train_occ_td"] += 1
            if prov == "DLD":
                support[tag]["train_occ_dld"] += 1
    return support


def support_bucket(real_rows: int, synth_rows: int) -> str:
    if real_rows > 0 and synth_rows > 0:
        return "mixed"
    if real_rows > 0:
        return "real_only"
    if synth_rows > 0:
        return "synthetic_only"
    return "unseen"


def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def bucket_summary(rows: List[Dict]) -> List[Dict]:
    buckets = defaultdict(list)
    for row in rows:
        if int(row.get("eval_support", 0)) <= 0:
            continue
        buckets[row["support_bucket"]].append(row)

    out: List[Dict] = []
    for bucket_name in ["mixed", "real_only", "synthetic_only", "unseen"]:
        items = buckets.get(bucket_name, [])
        total_support = sum(int(r["eval_support"]) for r in items)
        weighted_f1 = (
            sum(float(r["f1"]) * int(r["eval_support"]) for r in items) / total_support if total_support else 0.0
        )
        out.append(
            {
                "support_bucket": bucket_name,
                "labels_in_bucket": len(items),
                "labels_with_eval_support": len(items),
                "total_eval_support": total_support,
                "mean_f1_unweighted": round(sum(float(r["f1"]) for r in items) / len(items), 6) if items else 0.0,
                "mean_f1_weighted_by_eval_support": round(weighted_f1, 6),
                "mean_train_synthetic_share": round(
                    sum(float(r["train_synthetic_share_rows"]) for r in items) / len(items), 6
                )
                if items
                else 0.0,
            }
        )
    return out


def compare_rows_for_split(run_tables: Dict[str, List[Dict]], split_name: str) -> Tuple[List[Dict], List[Dict]]:
    if len(run_tables) != 2:
        return [], []
    run_names = list(run_tables.keys())
    left_name, right_name = run_names[0], run_names[1]
    left_index = {row["tag"]: row for row in run_tables[left_name]}
    right_index = {row["tag"]: row for row in run_tables[right_name]}
    merged_rows: List[Dict] = []
    for tag in sorted(set(left_index) | set(right_index)):
        left = left_index.get(tag)
        right = right_index.get(tag)
        if left is None or right is None:
            continue
        merged_rows.append(
            {
                "tag": tag,
                "support_bucket": left["support_bucket"],
                "eval_support": left["eval_support"],
                "train_rows_real": left["train_rows_real"],
                "train_rows_synthetic": left["train_rows_synthetic"],
                "train_synthetic_share_rows": left["train_synthetic_share_rows"],
                f"{left_name}_f1": left["f1"],
                f"{right_name}_f1": right["f1"],
                f"f1_delta_{right_name}_minus_{left_name}": round(float(right["f1"]) - float(left["f1"]), 6),
            }
        )

    bucket_rows: List[Dict] = []
    buckets = defaultdict(list)
    delta_key = f"f1_delta_{right_name}_minus_{left_name}"
    for row in merged_rows:
        if int(row["eval_support"]) <= 0:
            continue
        buckets[row["support_bucket"]].append(row)
    for bucket_name in ["mixed", "real_only", "synthetic_only", "unseen"]:
        items = buckets.get(bucket_name, [])
        total_support = sum(int(r["eval_support"]) for r in items)
        weighted = (
            sum(float(r[delta_key]) * int(r["eval_support"]) for r in items) / total_support if total_support else 0.0
        )
        bucket_rows.append(
            {
                "support_bucket": bucket_name,
                "labels_with_eval_support": len(items),
                "total_eval_support": total_support,
                f"mean_delta_{right_name}_minus_{left_name}": round(
                    sum(float(r[delta_key]) for r in items) / len(items), 6
                )
                if items
                else 0.0,
                f"weighted_delta_{right_name}_minus_{left_name}": round(weighted, 6),
            }
        )
    return merged_rows, bucket_rows


def main() -> None:
    args = parse_args()
    results_root = resolve_path(args.results_root)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_names = args.split if args.split else ["eval_real", "test_real", "eval_coverage", "test_coverage"]

    summary = {"runs": {}, "comparisons": {}}
    run_tables_by_split: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)

    for run_name in args.run:
        run_dir = results_root / run_name
        run_summary = load_json(run_dir / "eval_outputs" / "run_summary.json")
        split_dir = resolve_path(str(run_summary.get("split_dir", "")))
        support = train_label_support(split_dir)
        summary["runs"][run_name] = {"split_dir": str(split_dir), "splits": {}}

        for split_name in split_names:
            metrics = read_per_label_metrics(run_dir, split_name)
            rows: List[Dict] = []
            xs: List[float] = []
            ys: List[float] = []
            for tag in sorted(set(support) | set(metrics)):
                sup = support.get(tag, {})
                met = metrics.get(tag, {})
                real_rows = int(sup.get("train_rows_real", 0))
                synth_rows = int(sup.get("train_rows_synthetic", 0))
                total_rows = int(sup.get("train_rows_total", 0))
                synth_share = (synth_rows / total_rows) if total_rows else 0.0
                row = {
                    "tag": tag,
                    "support_bucket": support_bucket(real_rows, synth_rows),
                    "train_rows_total": total_rows,
                    "train_rows_real": real_rows,
                    "train_rows_synthetic": synth_rows,
                    "train_rows_td": int(sup.get("train_rows_td", 0)),
                    "train_rows_dld": int(sup.get("train_rows_dld", 0)),
                    "train_occ_total": int(sup.get("train_occ_total", 0)),
                    "train_occ_real": int(sup.get("train_occ_real", 0)),
                    "train_occ_synthetic": int(sup.get("train_occ_synthetic", 0)),
                    "train_synthetic_share_rows": round(synth_share, 6),
                    "eval_support": int(met.get("support", 0)),
                    "precision": round(float(met.get("precision", 0.0)), 6),
                    "recall": round(float(met.get("recall", 0.0)), 6),
                    "f1": round(float(met.get("f1", 0.0)), 6),
                }
                rows.append(row)
                if row["eval_support"] > 0 and row["train_rows_total"] > 0:
                    xs.append(synth_share)
                    ys.append(float(met.get("f1", 0.0)))

            run_tables_by_split[split_name][run_name] = rows
            support_rows = bucket_summary(rows)
            write_csv(out_dir / f"{run_name}_{split_name}_label_support.csv", rows)
            write_csv(out_dir / f"{run_name}_{split_name}_bucket_summary.csv", support_rows)

            summary["runs"][run_name]["splits"][split_name] = {
                "labels_total": len(rows),
                "labels_with_eval_support": sum(1 for r in rows if int(r["eval_support"]) > 0),
                "pearson_synth_share_vs_f1": round(pearson(xs, ys), 6) if xs else 0.0,
                "bucket_summary": support_rows,
            }

    if len(args.run) == 2:
        left_name, right_name = args.run[0], args.run[1]
        compare_key = f"{right_name}_minus_{left_name}"
        for split_name in split_names:
            merged_rows, bucket_rows = compare_rows_for_split(run_tables_by_split[split_name], split_name)
            write_csv(out_dir / f"{split_name}_{compare_key}_label_delta.csv", merged_rows)
            write_csv(out_dir / f"{split_name}_{compare_key}_bucket_delta.csv", bucket_rows)
            summary["comparisons"][split_name] = {
                "systems": [left_name, right_name],
                "bucket_delta_summary": bucket_rows,
            }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote synthetic-data diagnostic to: {out_dir}")


if __name__ == "__main__":
    main()
