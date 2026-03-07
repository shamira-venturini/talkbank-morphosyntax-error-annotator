import argparse
import json
import re

from common import resolve_path


def stage_2_transformation(text: str) -> str:
    if not isinstance(text, str):
        return text

    text = text.lower()
    text = re.sub(r"\bi\b", "I", text)

    text = re.sub(r"\[\*\s+m:0[a-z0-9:']+\]", r"[* m:0]", text)
    text = re.sub(r"\[\*\s+m:\+[a-z0-9:']+\]", r"[* m:+]", text)
    text = re.sub(r"\[\*\s+m:=[a-z0-9:']+\]", r"[* m:=]", text)
    text = re.sub(r"\[\*\s+m:\+\+[a-z0-9:']+\]", r"[* m:++]", text)
    text = re.sub(r"\[\*\s+m:base:[a-z0-9:']+\]", r"[* m:base]", text)
    text = re.sub(r"\[\*\s+m:vsg:a\]", r"[* m:vsg]", text)
    text = re.sub(r"\[\*\s+m:vun:a\]", r"[* m:vun]", text)
    text = re.sub(r"\[\*\s+m:irr:[a-z0-9:']+\]", r"[* m:irr]", text)
    text = re.sub(r"\[\*\s+m:sub:[a-z0-9:']+\]", r"[* m:sub]", text)

    text = re.sub(r"\[\*\s+s:r:gc:[a-z]+\]", r"[* s:r:gc]", text)
    text = re.sub(r"\[\*\s+s:r:(?![gc])[a-z:]+\]", r"[* s:r]", text)

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
            data["input"] = stage_2_transformation(data.get("input", ""))
            data["output"] = stage_2_transformation(data.get("output", ""))
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            processed_count += 1
    return processed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Stage 2 transformed training JSONL.")
    parser.add_argument("input", nargs="?", default="data/intermediate/df_master_training_v3.jsonl")
    parser.add_argument("output", nargs="?", default="data/intermediate/df_master_training_v2.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = process_jsonl(args.input, args.output)
    print(f"Stage 2 complete! Processed {count} sentences.")


if __name__ == "__main__":
    main()
