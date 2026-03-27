import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def create_csv(jsonl_path: str, csv_path: str) -> int:
    src = resolve_path(jsonl_path)
    dst = resolve_path(csv_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(src, lines=True)
    df_csv = pd.DataFrame({"index": df.index, "sentence": df["input"]})
    df_csv.to_csv(dst, index=False, encoding="utf-8")
    return len(df_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSONL input field to CSV index+sentence format.")
    parser.add_argument("input", nargs="?", default="data/norming/jsonl/df_children_sentences.jsonl")
    parser.add_argument("output", nargs="?", default="data/norming/jsonl/df_children_sentences.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = create_csv(args.input, args.output)
    print(f"Successfully created CSV with {count} rows at: {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
