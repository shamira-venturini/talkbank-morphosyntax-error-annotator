from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import resolve_path


def append_unique_jsonl(source_path: str, target_path: str) -> tuple[int, int]:
    source = resolve_path(source_path)
    target = resolve_path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    seen_inputs = set()
    if target.exists():
        with target.open("r", encoding="utf-8") as target_file:
            for line in target_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = record.get("input")
                if value:
                    seen_inputs.add(value)

    lines_added = 0
    lines_skipped = 0

    with source.open("r", encoding="utf-8") as source_file, target.open("a", encoding="utf-8") as target_file:
        for line in source_file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                lines_skipped += 1
                continue

            current_input = record.get("input", "")
            if current_input and current_input not in seen_inputs:
                target_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen_inputs.add(current_input)
                lines_added += 1
            else:
                lines_skipped += 1

    return lines_added, lines_skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append unique JSONL records by `input` key.")
    parser.add_argument("source", nargs="?", default="data/curated/synthetic/synthetic_sentences.jsonl")
    parser.add_argument("target", nargs="?", default="data/intermediate/df_master_training_v3.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    added, skipped = append_unique_jsonl(args.source, args.target)
    print("--- Process Complete ---")
    print(f"Lines successfully added: {added}")
    print(f"Duplicate/invalid lines skipped: {skipped}")


if __name__ == "__main__":
    main()
