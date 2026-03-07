from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ERROR_TAG_RE = re.compile(r"\[\*\s*[^\]]+\]")


def count_errors(output_text: str) -> int:
    return len(ERROR_TAG_RE.findall(output_text or ""))


def add_error_count(input_path: Path, output_path: Path, summary_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_zero = 0
    total_errors = 0
    by_count = Counter()
    by_provenance = Counter()

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            n_rows += 1
            error_count = count_errors(record.get("output", ""))
            record["error_count"] = error_count
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

            by_count[error_count] += 1
            by_provenance[record.get("provenance_label", "unknown")] += 1
            total_errors += error_count
            if error_count == 0:
                n_zero += 1

    payload = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "rows": n_rows,
        "rows_with_zero_errors": n_zero,
        "rows_with_one_or_more_errors": n_rows - n_zero,
        "total_error_tags": total_errors,
        "mean_errors_per_sentence": round(total_errors / n_rows, 6) if n_rows else 0.0,
        "distribution_error_count": dict(sorted(by_count.items())),
        "distribution_provenance_label": dict(sorted(by_provenance.items())),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add `error_count` (number of [* ...] tags in output) to each JSONL row.")
    parser.add_argument("--input", default="FT-3/df_master_training_v3_with_provenance.jsonl")
    parser.add_argument("--output", default="FT-3/df_master_training_v3_with_provenance_errorcount.jsonl")
    parser.add_argument("--summary", default="FT-3/df_master_training_v3_with_provenance_errorcount_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_error_count(Path(args.input), Path(args.output), Path(args.summary))
    print("Error-count enrichment completed.")
    print(f"Output: {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
