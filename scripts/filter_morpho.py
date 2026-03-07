import argparse
import json
import re

from common import resolve_path


def filter_to_morphology_only(text: str) -> str:
    text = re.sub(r"\b0(det|aux|v|prep|obj|subj|conj)\b", "", text)
    text = re.sub(r"\[\+ gram\]", "", text)
    text = re.sub(r"\[\* p:[^\]]+\]|\[\* p\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([\.\?!])", r" \1", text)
    return text


def process_file(file_path: str) -> int:
    path = resolve_path(file_path)

    processed_data = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data["output"] = filter_to_morphology_only(data.get("output", ""))
            processed_data.append(data)

    with path.open("w", encoding="utf-8") as outfile:
        for data in processed_data:
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

    return len(processed_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter non-morphology markers from `output` field in-place.")
    parser.add_argument("file", nargs="?", default="FT-3/df_master_training_v3.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = process_file(args.file)
    print(f"Processed {count} records in {resolve_path(args.file)}")


if __name__ == "__main__":
    main()
