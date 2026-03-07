from __future__ import annotations

import argparse
import json

from common import resolve_path


def update_records(file1_path: str, file2_path: str, output_path: str) -> tuple[int, int]:
    file1 = resolve_path(file1_path)
    file2 = resolve_path(file2_path)
    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    file2_lookup = {}
    with file2.open("r", encoding="utf-8") as f2:
        for line in f2:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = record.get("input")
            if key is not None:
                file2_lookup[key] = record.get("output", "")

    total_lines = 0
    updated_lines = 0

    with file1.open("r", encoding="utf-8") as f1, out.open("w", encoding="utf-8") as out_file:
        for line in f1:
            line = line.strip()
            if not line:
                continue

            total_lines += 1
            record = json.loads(line)
            current_input = record.get("input")
            if current_input in file2_lookup:
                record["output"] = file2_lookup[current_input]
                updated_lines += 1
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return total_lines, updated_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update outputs in file1 using matching input/output pairs from file2.")
    parser.add_argument("file1", nargs="?", default="norming/jsonl/df_children_sentences.jsonl")
    parser.add_argument("file2", nargs="?", default="norming/jsonl/training_realAB_errors_updated.jsonl")
    parser.add_argument("output", nargs="?", default="FT-3/training_realAB_errors_updated.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total, updated = update_records(args.file1, args.file2, args.output)
    print("--- Update Complete ---")
    print(f"Total records in file_1: {total}")
    print(f"Records successfully matched and updated: {updated}")
    print(f"New file saved to: {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
