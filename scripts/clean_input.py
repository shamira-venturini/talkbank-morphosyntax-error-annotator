import argparse
import json
import re

from common import resolve_path


def clean_input_string(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\[::?\s+[^\]]+\]", "", text)
    text = re.sub(r"\[\*\s+[^\]]+\]", "", text)
    text = re.sub(r"\[\/+\]", "", text)
    text = re.sub(r"\[\+\s+[^\]]+\]", "", text)
    text = text.replace("<", "").replace(">", "")
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([a-zA-Z0-9])([\.!\?])", r"\1 \2", text)
    return text


def purify_dataset(input_path: str, output_path: str) -> int:
    src = resolve_path(input_path)
    dst = resolve_path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with src.open("r", encoding="utf-8") as infile, dst.open("w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data["input"] = clean_input_string(data.get("input", ""))
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean CHAT markers from JSONL `input` field.")
    parser.add_argument("input", nargs="?", default="data/intermediate/df_master_training_v3.jsonl")
    parser.add_argument("output", nargs="?", default="data/intermediate/df_master_training_v3_CLEAN_INPUT.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = purify_dataset(args.input, args.output)
    print(f"Done! {count} lines purified.")
    print(f"Saved as: {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
