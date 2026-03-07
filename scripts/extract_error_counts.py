from __future__ import annotations

#!/usr/bin/env python3

import argparse
import json
import re
from collections import Counter

from common import resolve_path


def get_regex_patterns() -> tuple[dict[str, str], str]:
    patterns = {
        "error_code": r"\[\*\s*([^\]]+)\]",
        "omission": r"\b0\w+",
        "retracing": r"\[//\]",
        "repetition": r"\[/\]",
        "overlap": r"[<>]",
        "unintel": r"xxx",
        "pause": r"\(\.\+\)",
        "post_gram": r"\[\+\s*gram\]",
        "bracketed": r"\[::?\s*[^\]]+\]",
    }
    combined_filter = "|".join(p.replace(r"([^\]]+)", r"[^\]]+") for p in patterns.values())
    return patterns, combined_filter


def process_data(input_file: str, clean_output_file: str, stats_output_file: str, append_clean: bool = False) -> tuple[int, int]:
    patterns, filter_regex = get_regex_patterns()
    error_counts = Counter()
    clean_data = []

    src = resolve_path(input_file)
    clean_out = resolve_path(clean_output_file)
    stats_out = resolve_path(stats_output_file)

    with src.open("r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping line {line_num}: Invalid JSON")
                continue

            text = data.get("output", "")

            for code in re.findall(patterns["error_code"], text):
                error_counts[code.strip()] += 1

            error_counts["[//]"] += text.count("[//]")
            error_counts["[/]"] += max(0, text.count("[/]") - text.count("[//]"))
            error_counts["xxx"] += text.count("xxx")
            if "(..)" in text or "(.+)" in text:
                error_counts["(.+)"] += 1
            error_counts["<"] += text.count("<")
            error_counts[">"] += text.count(">")

            for om in re.findall(patterns["omission"], text):
                error_counts[om] += 1
            error_counts["[+ gram]"] += len(re.findall(patterns["post_gram"], text))

            if not re.search(filter_regex, text):
                clean_data.append(data)

    clean_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append_clean else "w"
    with clean_out.open(mode, encoding="utf-8") as f:
        for entry in clean_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with stats_out.open("w", encoding="utf-8") as f:
        f.write(f"{'Pattern/Code':<20} | Count\n")
        f.write("-" * 30 + "\n")
        for code, count in sorted(error_counts.items()):
            if count:
                f.write(f"{code:<20} | {count}\n")

    return len(error_counts), len(clean_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract error marker counts and clean subset from JSONL.")
    parser.add_argument("input", nargs="?", default="data/intermediate/df_master_training_v3.jsonl")
    parser.add_argument("clean_output", nargs="?", default="data/intermediate/clean_output.jsonl")
    parser.add_argument("stats_output", nargs="?", default="data/intermediate/error_summary_std.txt")
    parser.add_argument("--append-clean", action="store_true", help="Append clean records instead of overwriting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    marker_count, clean_count = process_data(
        args.input,
        args.clean_output,
        args.stats_output,
        append_clean=args.append_clean,
    )
    print("--- Summary ---")
    print(f"Total Unique Markers Found: {marker_count}")
    print(f"Total Clean Lines Found:    {clean_count}")


if __name__ == "__main__":
    main()
