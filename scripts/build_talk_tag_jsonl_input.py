from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from common import iter_jsonl, resolve_path, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an audit/input JSONL into a talk-tag folder-ready JSONL file."
    )
    parser.add_argument(
        "--input-jsonl",
        default=(
            "studies/02_uncertainty_and_feedback/audits/"
            "training_missing_error_audit_exact_stage3_train_zero_error_only_v1/"
            "input_utterance_only.jsonl"
        ),
        help="Source JSONL containing an `input` field.",
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "studies/02_uncertainty_and_feedback/audits/"
            "training_missing_error_audit_exact_stage3_train_zero_error_only_v1/"
            "talk_tag_input"
        ),
        help="Directory to use as talk-tag --input-dir.",
    )
    parser.add_argument(
        "--out-filename",
        default="training_zero_error_only.jsonl",
        help="JSONL filename to place inside --out-dir.",
    )
    parser.add_argument(
        "--speaker-value",
        default="*AUD",
        help="Speaker value to assign so talk-tag will annotate all rows.",
    )
    parser.add_argument(
        "--speaker-field",
        default="speaker",
        help="Speaker field name for the output JSONL.",
    )
    parser.add_argument(
        "--text-field",
        default="utterance",
        help="Text field name for the output JSONL.",
    )
    return parser.parse_args()


def convert_row(row: Dict, speaker_field: str, text_field: str, speaker_value: str) -> Dict:
    converted = dict(row)
    original_speaker = converted.get("speaker", "")
    converted["original_speaker"] = original_speaker
    converted[speaker_field] = speaker_value
    converted[text_field] = converted.pop("input", "")
    return converted


def main() -> None:
    args = parse_args()
    input_jsonl = resolve_path(args.input_jsonl)
    out_dir = resolve_path(args.out_dir)
    out_path = out_dir / args.out_filename

    source_rows = list(iter_jsonl(input_jsonl))
    converted_rows: List[Dict] = [
        convert_row(
            row,
            speaker_field=args.speaker_field,
            text_field=args.text_field,
            speaker_value=args.speaker_value,
        )
        for row in source_rows
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, converted_rows)

    summary = {
        "input_jsonl": str(input_jsonl),
        "out_dir": str(out_dir),
        "output_jsonl": str(out_path),
        "rows_written": len(converted_rows),
        "speaker_field": args.speaker_field,
        "text_field": args.text_field,
        "speaker_value": args.speaker_value,
        "run_example": (
            f"talk-tag annotate --input-dir {out_dir} --output-dir {out_dir.parent / 'talk_tag_output'} "
            f"--target-speaker {args.speaker_value} --speaker-field {args.speaker_field} "
            f"--text-field {args.text_field}"
        ),
    }
    (out_dir / "_talk_tag_input_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
