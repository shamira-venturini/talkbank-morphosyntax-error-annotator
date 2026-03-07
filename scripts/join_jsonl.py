from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from common import resolve_path


def join_jsonl(input_glob: str, output_path: str) -> tuple[int, int, int, int]:
    pattern = str(resolve_path(input_glob))
    files = sorted(glob.glob(pattern))
    corrupted_lines_count = 0
    valid_records = []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    valid_records.append(json.loads(line))
                except json.JSONDecodeError:
                    corrupted_lines_count += 1
                    print(f"Invalid JSON: {Path(file_path).name}:{line_number}")

    if not valid_records:
        raise ValueError("No valid JSON records found in input files.")

    df_master_training = pd.DataFrame(valid_records)
    initial_len = len(df_master_training)
    if "input" in df_master_training.columns:
        df_master_training.drop_duplicates(subset=["input"], keep="last", inplace=True)
    final_len = len(df_master_training)

    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_master_training.to_json(out, orient="records", lines=True, force_ascii=False)

    return len(files), corrupted_lines_count, initial_len - final_len, final_len


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join JSONL files into one deduplicated JSONL.")
    parser.add_argument("input_glob", nargs="?", default="data/curated/synthetic/*.jsonl")
    parser.add_argument("output", nargs="?", default="data/curated/synthetic/synthetic_sentences.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_count, corrupted, dropped, saved = join_jsonl(args.input_glob, args.output)
    print("--- Summary ---")
    print(f"Successfully combined {file_count} files.")
    print(f"Skipped {corrupted} corrupted lines.")
    print(f"Dropped {dropped} duplicate sentences.")
    print(f"Saved {saved} unique, valid sentences to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
