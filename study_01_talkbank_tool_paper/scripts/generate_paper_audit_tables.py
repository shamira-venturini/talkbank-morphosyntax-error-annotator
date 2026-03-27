from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List

from common import resolve_path


TAG_RE = re.compile(r"\[\*\s*([ms](?::[^\]]+)?)\]")
PROVENANCE_ORDER = ["synthetic", "TD", "DLD"]


def load_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_tags(text: str) -> List[str]:
    return [f"[* {m.group(1)}]" for m in TAG_RE.finditer(text or "")]


def to_level2(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    parts = body.split(":")
    if not parts:
        return body

    if parts[0] == "m":
        if len(parts) >= 2:
            p1 = parts[1]
            if p1 in {"base", "irr", "sub"}:
                return f"m:{p1}"
            if p1.startswith("++"):
                return "m:++"
            if p1.startswith("+"):
                return "m:+"
            if p1.startswith("0"):
                return "m:0"
            if p1.startswith("="):
                return "m:="
            if p1 == "allo":
                return "m:allo"
            if p1 in {"vsg", "vun"}:
                if len(parts) >= 3 and parts[2] == "a":
                    return f"m:{p1}:a"
                return f"m:{p1}"
        return "m"

    if parts[0] == "s":
        if len(parts) >= 3 and parts[1] == "r" and parts[2] == "gc":
            return "s:r:gc"
        if len(parts) >= 2 and parts[1] == "r":
            return "s:r"
        if len(parts) >= 2:
            return f"s:{parts[1]}"
    return body


def pct(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return round((num * 100.0) / den, 2)


def write_csv(path: Path, headers: Iterable[str], rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(rows)


def generate_tables(rows: List[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(rows)
    composition = Counter(r.get("provenance_label", "synthetic") for r in rows)
    method_counts = Counter(r.get("trace_method", "") for r in rows)
    trace_ambiguous_rows = sum(1 for r in rows if r.get("trace_ambiguous"))
    error_counts = [int(r.get("error_count", 0) or 0) for r in rows]
    error_dist = Counter(error_counts)

    detailed_counts = Counter()
    detailed_by_prov = defaultdict(Counter)
    level2_counts = Counter()
    level2_by_prov = defaultdict(Counter)

    for row in rows:
        provenance = row.get("provenance_label", "synthetic")
        for tag in extract_tags(row.get("output", "")):
            detailed_counts[tag] += 1
            detailed_by_prov[tag][provenance] += 1
            level2 = to_level2(tag)
            level2_counts[level2] += 1
            level2_by_prov[level2][provenance] += 1

    total_error_tags = sum(detailed_counts.values())
    unique_detailed_labels = len(detailed_counts)
    top5_share = pct(sum(v for _, v in detailed_counts.most_common(5)), total_error_tags)
    top10_share = pct(sum(v for _, v in detailed_counts.most_common(10)), total_error_tags)
    rows_with_any_error = sum(1 for c in error_counts if c >= 1)
    rows_with_multi_error = sum(1 for c in error_counts if c >= 2)

    table_01_rows = []
    for subset in PROVENANCE_ORDER:
        count = composition.get(subset, 0)
        table_01_rows.append({"subset": subset, "rows": count, "percent": pct(count, n_rows)})
    table_01_rows.append({"subset": "TOTAL", "rows": n_rows, "percent": 100.0})
    write_csv(out_dir / "table_01_composition.csv", ["subset", "rows", "percent"], table_01_rows)

    table_02_rows = [
        {"metric": "rows_total", "value": n_rows},
        {"metric": "trace_ambiguous_rows", "value": trace_ambiguous_rows},
        {"metric": "trace_ambiguous_percent", "value": pct(trace_ambiguous_rows, n_rows)},
    ]
    for method in [
        "curated_synthetic_input",
        "curated_synthetic_exact",
        "synthetic_no_real_match",
        "real_input_clean",
        "real_output_exact",
    ]:
        table_02_rows.append({"metric": f"method::{method}", "value": method_counts.get(method, 0)})
    write_csv(out_dir / "table_02_provenance_quality.csv", ["metric", "value"], table_02_rows)

    table_03_rows = [
        {"error_count": 0, "rows": error_dist.get(0, 0), "percent": pct(error_dist.get(0, 0), n_rows)},
        {"error_count": 1, "rows": error_dist.get(1, 0), "percent": pct(error_dist.get(1, 0), n_rows)},
        {"error_count": 2, "rows": error_dist.get(2, 0), "percent": pct(error_dist.get(2, 0), n_rows)},
        {"error_count": 3, "rows": error_dist.get(3, 0), "percent": pct(error_dist.get(3, 0), n_rows)},
        {"error_count": "mean", "rows": round(sum(error_counts) / n_rows, 6) if n_rows else 0.0, "percent": ""},
        {"error_count": "median", "rows": int(median(error_counts)) if error_counts else 0, "percent": ""},
        {"error_count": "rows_with_>=1_error", "rows": rows_with_any_error, "percent": pct(rows_with_any_error, n_rows)},
        {"error_count": "rows_with_>=2_errors", "rows": rows_with_multi_error, "percent": pct(rows_with_multi_error, n_rows)},
    ]
    write_csv(out_dir / "table_03_error_density.csv", ["error_count", "rows", "percent"], table_03_rows)

    table_04_rows = []
    for tag, count in sorted(detailed_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        table_04_rows.append(
            {"label": tag, "count": count, "percent_of_all_error_tags": pct(count, total_error_tags)}
        )
    write_csv(out_dir / "table_04_detailed_label_counts.csv", ["label", "count", "percent_of_all_error_tags"], table_04_rows)

    table_05_rows = []
    for tag in sorted(detailed_counts):
        row = {"label": tag}
        total = 0
        for provenance in PROVENANCE_ORDER:
            value = detailed_by_prov[tag].get(provenance, 0)
            row[provenance] = value
            total += value
        row["total"] = total
        table_05_rows.append(row)
    write_csv(out_dir / "table_05_detailed_label_by_provenance.csv", ["label", *PROVENANCE_ORDER, "total"], table_05_rows)

    table_06_rows = []
    for level2, count in sorted(level2_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        table_06_rows.append(
            {"level2": level2, "count": count, "percent_of_all_error_tags": pct(count, total_error_tags)}
        )
    write_csv(out_dir / "table_06_level2_counts.csv", ["level2", "count", "percent_of_all_error_tags"], table_06_rows)

    table_07_rows = []
    for level2 in sorted(level2_counts):
        row = {"level2": level2}
        total = 0
        for provenance in PROVENANCE_ORDER:
            value = level2_by_prov[level2].get(provenance, 0)
            row[provenance] = value
            total += value
        row["total"] = total
        table_07_rows.append(row)
    write_csv(out_dir / "table_07_level2_by_provenance.csv", ["level2", *PROVENANCE_ORDER, "total"], table_07_rows)

    table_08_rows = []
    for provenance in PROVENANCE_ORDER:
        subset_rows = [int(r.get("error_count", 0) or 0) for r in rows if r.get("provenance_label", "synthetic") == provenance]
        total = len(subset_rows)
        if total == 0:
            table_08_rows.append(
                {
                    "provenance_label": provenance,
                    "rows_total": 0,
                    "err0": 0,
                    "err1": 0,
                    "err2": 0,
                    "err3plus": 0,
                    "err>=1_percent": 0.0,
                }
            )
            continue
        err0 = sum(1 for c in subset_rows if c == 0)
        err1 = sum(1 for c in subset_rows if c == 1)
        err2 = sum(1 for c in subset_rows if c == 2)
        err3plus = sum(1 for c in subset_rows if c >= 3)
        table_08_rows.append(
            {
                "provenance_label": provenance,
                "rows_total": total,
                "err0": err0,
                "err1": err1,
                "err2": err2,
                "err3plus": err3plus,
                "err>=1_percent": pct(total - err0, total),
            }
        )
    write_csv(
        out_dir / "table_08_provenance_x_errorcount.csv",
        ["provenance_label", "rows_total", "err0", "err1", "err2", "err3plus", "err>=1_percent"],
        table_08_rows,
    )

    summary_payload = {
        "rows_total": n_rows,
        "composition": {k: composition.get(k, 0) for k in PROVENANCE_ORDER},
        "trace_ambiguous_rows": trace_ambiguous_rows,
        "trace_ambiguous_percent": pct(trace_ambiguous_rows, n_rows),
        "method_counts": dict(method_counts),
        "error_count_distribution": dict(sorted(error_dist.items())),
        "mean_error_count": round(sum(error_counts) / n_rows, 6) if n_rows else 0.0,
        "median_error_count": int(median(error_counts)) if error_counts else 0,
        "rows_with_any_error_tag": rows_with_any_error,
        "rows_with_multi_error_tags": rows_with_multi_error,
        "total_error_tags": total_error_tags,
        "unique_detailed_labels": unique_detailed_labels,
        "top5_error_tag_share_percent": top5_share,
        "top10_error_tag_share_percent": top10_share,
    }
    (out_dir / "paper_stats_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = (
        "# Data Statistics (Paper-Ready)\n\n"
        f"- Rows: **{n_rows}**\n"
        f"- Composition: synthetic **{composition.get('synthetic', 0)}** ({pct(composition.get('synthetic', 0), n_rows)}%), "
        f"TD **{composition.get('TD', 0)}** ({pct(composition.get('TD', 0), n_rows)}%), "
        f"DLD **{composition.get('DLD', 0)}** ({pct(composition.get('DLD', 0), n_rows)}%)\n"
        f"- Ambiguous provenance rows: **{trace_ambiguous_rows}** ({pct(trace_ambiguous_rows, n_rows)}%)\n"
        f"- Total error tags (`[* ...]`): **{total_error_tags}** across **{unique_detailed_labels}** unique detailed labels\n"
        f"- Error-count mean/median per sentence: **{round(sum(error_counts) / n_rows, 3) if n_rows else 0.0} / {int(median(error_counts)) if error_counts else 0}**\n"
        f"- Error density: 0 errors **{error_dist.get(0, 0)}** ({pct(error_dist.get(0, 0), n_rows)}%), "
        f"1 error **{error_dist.get(1, 0)}** ({pct(error_dist.get(1, 0), n_rows)}%), "
        f"2+ errors **{rows_with_multi_error}** ({pct(rows_with_multi_error, n_rows)}%)\n"
        f"- Concentration: top-5 labels = **{top5_share}%** of all error tags; top-10 = **{top10_share}%**\n\n"
        "## Generated Tables\n"
        "- `table_01_composition.csv`\n"
        "- `table_02_provenance_quality.csv`\n"
        "- `table_03_error_density.csv`\n"
        "- `table_04_detailed_label_counts.csv`\n"
        "- `table_05_detailed_label_by_provenance.csv`\n"
        "- `table_06_level2_counts.csv`\n"
        "- `table_07_level2_by_provenance.csv`\n"
        "- `paper_stats_summary.json`\n"
    )
    (out_dir / "paper_stats_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate paper-facing audit tables from master_training.jsonl.")
    parser.add_argument("--input", default="data/processed/master_training.jsonl")
    parser.add_argument("--out-dir", default="data/audits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = resolve_path(args.input)
    out_dir = resolve_path(args.out_dir)
    rows = load_rows(in_path)
    generate_tables(rows, out_dir)
    print(f"Generated paper audit tables at: {out_dir}")
    print(f"Rows processed: {len(rows)}")


if __name__ == "__main__":
    main()
