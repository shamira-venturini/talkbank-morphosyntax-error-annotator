import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

from common import resolve_path


TAG_RE = re.compile(r"\[\*\s*[ms](?::[^\]]+)?\]")
SEED_RE = re.compile(r"_seed(\d+)$")


def parse_run_group(value: str) -> Tuple[str, List[str]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--system must look like name=run1,run2,run3")
    name, raw_runs = value.split("=", 1)
    runs = [item.strip() for item in raw_runs.split(",") if item.strip()]
    if not name.strip() or not runs:
        raise argparse.ArgumentTypeError("--system must include a name and at least one run")
    return name.strip(), runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the reconstruction mini-ablation across paired seeds.")
    parser.add_argument("--results-root", default="results", help="Root directory containing run outputs.")
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        type=parse_run_group,
        help="System name and comma-separated run directories. Example: --system preserve=run_a,run_b,run_c",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Prediction split to analyze. Defaults to test_real, test_coverage, holdout_generalization.",
    )
    parser.add_argument(
        "--qualitative-limit",
        type=int,
        default=40,
        help="Maximum number of qualitative comparison rows per split and direction.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/reconstruction_ablation_analysis",
        help="Output directory for reconstruction-ablation analysis artifacts.",
    )
    return parser.parse_args()


def canonical_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    return f"[* {body}]"


def extract_tags(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [canonical_tag(tag) for tag in TAG_RE.findall(text)]


def safe_std(values: List[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def metric_file(run_dir: Path, split_name: str) -> Path:
    return run_dir / "eval_outputs" / f"metrics_{split_name}.json"


def prediction_file(run_dir: Path, split_name: str) -> Path:
    return run_dir / "eval_outputs" / f"predictions_{split_name}.jsonl"


def infer_seed(run_name: str, run_summary: Dict) -> int:
    summary_seed = run_summary.get("seed")
    if isinstance(summary_seed, int):
        return summary_seed
    match = SEED_RE.search(run_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not infer seed for run: {run_name}")


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> Dict[Tuple[object, str], Dict]:
    rows: Dict[Tuple[object, str], Dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[(row.get("row_id"), row.get("input", ""))] = row
    return rows


def paired_runs(results_root: Path, system_map: Dict[str, List[str]]) -> Dict[int, Dict[str, Path]]:
    by_seed: Dict[int, Dict[str, Path]] = {}
    for system_name, runs in system_map.items():
        for run_name in runs:
            run_dir = results_root / run_name
            run_summary = load_json(run_dir / "eval_outputs" / "run_summary.json")
            seed = infer_seed(run_name, run_summary)
            by_seed.setdefault(seed, {})[system_name] = run_dir
    return by_seed


def qualitative_rows(
    baseline_rows: Dict[Tuple[object, str], Dict],
    compare_rows: Dict[Tuple[object, str], Dict],
    baseline_name: str,
    compare_name: str,
    split_name: str,
    seed: int,
    direction: str,
    limit: int,
) -> List[Dict]:
    rows: List[Dict] = []
    for key in sorted(compare_rows):
        if key not in baseline_rows:
            continue
        base = baseline_rows[key]
        comp = compare_rows[key]
        gold_tags = extract_tags(comp.get("human_gold", ""))
        base_tags = extract_tags(base.get("model_prediction", ""))
        comp_tags = extract_tags(comp.get("model_prediction", ""))
        base_ok = int(base_tags == gold_tags)
        comp_ok = int(comp_tags == gold_tags)
        if direction == "compare_wins" and not (comp_ok and not base_ok):
            continue
        if direction == "baseline_wins" and not (base_ok and not comp_ok):
            continue
        rows.append(
            {
                "split": split_name,
                "seed": seed,
                "row_id": comp.get("row_id"),
                "input": comp.get("input", ""),
                "gold": comp.get("human_gold", ""),
                baseline_name: base.get("model_prediction", ""),
                compare_name: comp.get("model_prediction", ""),
                "gold_tags": "|".join(gold_tags),
                f"{baseline_name}_tags": "|".join(base_tags),
                f"{compare_name}_tags": "|".join(comp_tags),
            }
        )
        if len(rows) >= limit:
            break
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


def main() -> None:
    args = parse_args()
    results_root = resolve_path(args.results_root)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system_map = dict(args.system)
    if len(system_map) != 2:
        raise ValueError("Expected exactly two --system groups: baseline and reconstruction comparison.")
    system_names = list(system_map.keys())
    baseline_name, compare_name = system_names[0], system_names[1]
    splits = args.split if args.split else ["test_real", "test_coverage", "holdout_generalization"]

    paired = paired_runs(results_root, system_map)
    aggregate_rows: List[Dict] = []
    delta_rows: List[Dict] = []
    compare_win_rows: List[Dict] = []
    baseline_win_rows: List[Dict] = []

    system_split_metrics: Dict[str, Dict[str, List[float]]] = {
        name: {split: [] for split in splits} for name in system_names
    }

    for seed, systems in sorted(paired.items()):
        missing = [name for name in system_names if name not in systems]
        if missing:
            raise ValueError(f"Missing paired runs for seed {seed}: {missing}")

        for split_name in splits:
            split_metrics = {}
            for system_name in system_names:
                metrics = load_json(metric_file(systems[system_name], split_name))
                split_metrics[system_name] = metrics
                system_split_metrics[system_name][split_name].append(float(metrics.get("micro_f1", 0.0)))
                aggregate_rows.append(
                    {
                        "seed": seed,
                        "system": system_name,
                        "split": split_name,
                        "n": int(metrics.get("n", 0)),
                        "exact_match": float(metrics.get("exact_match", 0.0)),
                        "micro_f1": float(metrics.get("micro_f1", 0.0)),
                    }
                )

            delta_rows.append(
                {
                    "seed": seed,
                    "split": split_name,
                    "baseline_system": baseline_name,
                    "compare_system": compare_name,
                    "baseline_micro_f1": float(split_metrics[baseline_name].get("micro_f1", 0.0)),
                    "compare_micro_f1": float(split_metrics[compare_name].get("micro_f1", 0.0)),
                    "micro_f1_delta_compare_minus_baseline": float(split_metrics[compare_name].get("micro_f1", 0.0))
                    - float(split_metrics[baseline_name].get("micro_f1", 0.0)),
                    "baseline_exact_match": float(split_metrics[baseline_name].get("exact_match", 0.0)),
                    "compare_exact_match": float(split_metrics[compare_name].get("exact_match", 0.0)),
                    "exact_match_delta_compare_minus_baseline": float(split_metrics[compare_name].get("exact_match", 0.0))
                    - float(split_metrics[baseline_name].get("exact_match", 0.0)),
                }
            )

            base_pred = prediction_file(systems[baseline_name], split_name)
            comp_pred = prediction_file(systems[compare_name], split_name)
            if base_pred.exists() and comp_pred.exists():
                baseline_rows = load_predictions(base_pred)
                compare_rows = load_predictions(comp_pred)
                compare_win_rows.extend(
                    qualitative_rows(
                        baseline_rows,
                        compare_rows,
                        baseline_name=baseline_name,
                        compare_name=compare_name,
                        split_name=split_name,
                        seed=seed,
                        direction="compare_wins",
                        limit=args.qualitative_limit,
                    )
                )
                baseline_win_rows.extend(
                    qualitative_rows(
                        baseline_rows,
                        compare_rows,
                        baseline_name=baseline_name,
                        compare_name=compare_name,
                        split_name=split_name,
                        seed=seed,
                        direction="baseline_wins",
                        limit=args.qualitative_limit,
                    )
                )

    summary = {
        "systems": system_names,
        "splits": splits,
        "paired_seeds": sorted(paired),
        "aggregate": {},
    }
    for system_name in system_names:
        summary["aggregate"][system_name] = {}
        for split_name in splits:
            values = system_split_metrics[system_name][split_name]
            summary["aggregate"][system_name][split_name] = {
                "mean_micro_f1": mean(values) if values else 0.0,
                "std_micro_f1": safe_std(values),
                "n_runs": len(values),
            }

    write_csv(out_dir / "per_run_metrics.csv", aggregate_rows)
    write_csv(out_dir / "paired_deltas.csv", delta_rows)
    write_csv(out_dir / "qualitative_compare_wins.csv", compare_win_rows)
    write_csv(out_dir / "qualitative_baseline_wins.csv", baseline_win_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote reconstruction-ablation analysis: {out_dir}")


if __name__ == "__main__":
    main()
