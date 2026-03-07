from __future__ import annotations

import argparse
import json
import re

from common import resolve_path


def process_repetition_tags(input_filepath: str, output_filepath: str) -> tuple[int, int]:
    pattern_error = re.compile(r"\[\*\s*[^\]]*\]")
    pattern_spec = re.compile(r"\[::?\s*[^\]]*\]")

    src = resolve_path(input_filepath)
    dst = resolve_path(output_filepath)
    dst.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    updated_lines = 0

    with src.open("r", encoding="utf-8") as infile, dst.open("w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue

            total_lines += 1
            record = json.loads(line)
            output_text = record.get("output", "")

            if any(marker in output_text for marker in ("[/]", "[//]", "<", ">")):
                new_input = pattern_error.sub("", output_text)
                new_input = pattern_spec.sub("", new_input)
                new_input = re.sub(r"\s+", " ", new_input).strip()
                record["input"] = new_input
                updated_lines += 1

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    return total_lines, updated_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace input with cleaned output for repetition/retracing records.")
    parser.add_argument("input", nargs="?", default="data/intermediate/df_master_training_v3_FINAL.jsonl")
    parser.add_argument("output", nargs="?", default="data/intermediate/df_master_training_v3.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total, updated = process_repetition_tags(args.input, args.output)
    print("--- Processing Complete ---")
    print(f"Total sentences evaluated: {total}")
    print(f"Sentences successfully updated: {updated}")
    print(f"Cleaned dataset saved to: {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
