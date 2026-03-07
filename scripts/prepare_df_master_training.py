from __future__ import annotations

import argparse
import json
import re

from common import resolve_path


def merge_and_clean_annotations(input_filepath: str, output_filepath: str) -> tuple[int, int]:
    pattern_error = re.compile(r"\[\*\s*[^\]]*\]")

    src = resolve_path(input_filepath)
    dst = resolve_path(output_filepath)
    dst.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    substituted_lines = 0

    with src.open("r", encoding="utf-8") as infile, dst.open("w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue

            total_lines += 1
            record = json.loads(line)

            output_text = record.get("output", "")
            model_text = record.get("model_annotation", "")

            output_tags = pattern_error.findall(output_text)
            model_tags = pattern_error.findall(model_text)

            if output_tags == model_tags and model_text:
                new_output = model_text.strip()
                new_output = re.sub(r"(?<!\s)([\.\?!]+)$", r" \1", new_output)
                new_output = new_output.replace("].", "] .")
                new_output = re.sub(r"\s+", " ", new_output).strip()
                output_text = new_output
                substituted_lines += 1

            clean_record = {
                "instruction": record.get("instruction", ""),
                "input": record.get("input", ""),
                "output": output_text,
            }
            outfile.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

    return total_lines, substituted_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge model annotations while preserving canonical keys.")
    parser.add_argument("input", nargs="?", default="FT-3/preparation/df_master_training_v3_CLEAN_INPUT_corrected.jsonl")
    parser.add_argument("output", nargs="?", default="FT-3/preparation/df_master_training_v3_FINAL.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total, substituted = merge_and_clean_annotations(args.input, args.output)
    print("--- Processing Complete ---")
    print(f"Total lines evaluated: {total}")
    print(f"Lines updated with model annotations: {substituted}")
    print(f"Cleaned file saved to: {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
