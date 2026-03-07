import argparse
import json
import re

from common import resolve_path


def stage_1_transformation(text: str) -> str:
    if not isinstance(text, str):
        return text

    text = text.lower()
    text = re.sub(r"\bi\b", "I", text)
    text = re.sub(r"\[\*\s+([ms]):[^\]]+\]", r"[* \1]", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def process_jsonl(input_file: str, output_file: str) -> int:
    in_path = resolve_path(input_file)
    out_path = resolve_path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    with in_path.open("r", encoding="utf-8") as infile, out_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data["input"] = stage_1_transformation(data.get("input", ""))
            data["output"] = stage_1_transformation(data.get("output", ""))
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            processed_count += 1
    return processed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Stage 1 transformed training JSONL.")
    parser.add_argument("input", nargs="?", default="FT-3/df_master_training_v3.jsonl")
    parser.add_argument("output", nargs="?", default="FT-3/df_master_training_v1.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = process_jsonl(args.input, args.output)
    print(f"Finished! Processed {count} sentences into Stage 1 format.")


if __name__ == "__main__":
    main()
