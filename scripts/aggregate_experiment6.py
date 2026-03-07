import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

from common import resolve_path


def parse_system_arg(value: str) -> Tuple[str, List[str]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--system must look like name=path1,path2,path3")
    name, raw_paths = value.split("=", 1)
    paths = [p.strip() for p in raw_paths.split(",") if p.strip()]
    if not name.strip() or not paths:
        raise argparse.ArgumentTypeError("--system must include a name and at least one path")
    return name.strip(), paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed stability results for Experiment 6.")
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        type=parse_system_arg,
        help=(
            "System name and comma-separated run directories or metric files. "
            "Example: --system baseline=/runs/base_seed3407,/runs/base_seed3408,/runs/base_seed3409"
        ),
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Split name to aggregate (repeat flag). Defaults to test_real.",
    )
    parser.add_argument(
        "--metric",
        default="micro_f1",
        help="Metric key inside metrics JSON. Defaults to micro_f1.",
    )
    parser.add_argument(
        "--out",
        default="FT-3/experiments/experiment6_stability_summary.json",
        help="Output JSON summary path.",
    )
    return parser.parse_args()


def resolve_metric_file(path_str: str, split_name: str) -> Path:
    path = resolve_path(path_str)
    if path.is_file():
        return path
    candidate = path / "eval_outputs" / f"metrics_{split_name}.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(candidate)


def safe_std(values: List[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def load_metric(metric_file: Path, metric_key: str) -> Dict:
    data = json.loads(metric_file.read_text(encoding="utf-8"))
    if metric_key not in data:
        raise KeyError(f"{metric_key} not found in {metric_file}")
    return data


def summarize_values(values: List[float]) -> Dict[str, float]:
    return {
        "mean": mean(values) if values else 0.0,
        "std": safe_std(values),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "n_runs": len(values),
    }


def main() -> None:
    args = parse_args()
    splits = args.split if args.split else ["test_real"]
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    systems = dict(args.system)
    summary: Dict[str, object] = {
        "metric": args.metric,
        "splits": splits,
        "systems": {},
        "rankings": {},
    }

    for split_name in splits:
        ranking_rows = []
        for system_name, run_paths in systems.items():
            metric_values: List[float] = []
            ci_lows: List[float] = []
            ci_highs: List[float] = []
            exact_values: List[float] = []
            metric_files: List[str] = []

            for run_path in run_paths:
                metric_file = resolve_metric_file(run_path, split_name)
                data = load_metric(metric_file, args.metric)
                metric_values.append(float(data[args.metric]))
                exact_values.append(float(data.get("exact_match", 0.0)))
                metric_files.append(str(metric_file))
                ci = data.get("micro_f1_bootstrap")
                if isinstance(ci, dict):
                    ci_lows.append(float(ci.get("ci_low", 0.0)))
                    ci_highs.append(float(ci.get("ci_high", 0.0)))

            metric_summary = summarize_values(metric_values)
            exact_summary = summarize_values(exact_values)
            system_split_summary = {
                "metric_files": metric_files,
                "metric_summary": metric_summary,
                "exact_match_summary": exact_summary,
                "seed_values": metric_values,
            }
            if ci_lows and ci_highs:
                system_split_summary["bootstrap_ci_mean"] = {
                    "ci_low_mean": mean(ci_lows),
                    "ci_high_mean": mean(ci_highs),
                }

            summary["systems"].setdefault(system_name, {})[split_name] = system_split_summary
            ranking_rows.append((system_name, metric_summary["mean"], metric_summary["std"]))

        ranking_rows.sort(key=lambda x: x[1], reverse=True)
        summary["rankings"][split_name] = [
            {"system": name, "mean": avg, "std": std, "rank": idx + 1}
            for idx, (name, avg, std) in enumerate(ranking_rows)
        ]

    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Experiment 6 summary: {out_path}")


if __name__ == "__main__":
    main()
