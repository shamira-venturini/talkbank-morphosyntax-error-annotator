from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from common import resolve_path
from ood_chat_utils import parse_chat_file, select_speakers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare utterance-level OOD inputs from Vercellotti CHAT files.")
    parser.add_argument(
        "--corpus-dir",
        default="data/OOD_data/Vercellotti",
        help="Directory containing Vercellotti .cha files.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/ood_vercellotti",
        help="Output directory for prepared JSONL/CSV files.",
    )
    parser.add_argument(
        "--speaker-policy",
        choices=["dominant", "first_participant", "all"],
        default="dominant",
        help="How to select target speaker tiers from each file.",
    )
    parser.add_argument(
        "--include-speaker",
        action="append",
        default=None,
        help="Optional explicit speaker id(s) to include; overrides --speaker-policy.",
    )
    parser.add_argument(
        "--min-word-count",
        type=int,
        default=1,
        help="Drop utterances with fewer than this many whitespace-separated words.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of files to process (0 = all).",
    )
    return parser.parse_args()


def iter_chat_files(corpus_dir: Path, max_files: int) -> Iterable[Path]:
    files = sorted(corpus_dir.glob("*.cha"))
    if max_files > 0:
        files = files[:max_files]
    return files


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    corpus_dir = resolve_path(args.corpus_dir)
    out_dir = resolve_path(args.out_dir)
    include_speakers = args.include_speaker or []

    if not corpus_dir.exists():
        raise SystemExit(f"Missing corpus directory: {corpus_dir}")

    utterance_rows: List[Dict] = []
    file_rows: List[Dict] = []
    speaker_counts = Counter()
    dropped_short = 0
    next_row_id = 1

    for path in iter_chat_files(corpus_dir, args.max_files):
        parsed = parse_chat_file(path)
        selected = select_speakers(
            utterances=parsed["utterances"],
            participants=parsed["participants"],
            policy=args.speaker_policy,
            include_speakers=include_speakers,
        )
        selected_set = set(selected)
        kept_in_file = 0

        for utt in parsed["utterances"]:
            if utt["speaker"] not in selected_set:
                continue
            word_count = len([w for w in utt["text"].split(" ") if w])
            if word_count < args.min_word_count:
                dropped_short += 1
                continue

            row = {
                "row_id": next_row_id,
                "corpus": "Vercellotti",
                "file_name": parsed["file_name"],
                "file_path": parsed["file_path"],
                "speaker": utt["speaker"],
                "line_no": utt["line_no"],
                "utterance_index_raw": utt["raw_index"],
                "input": utt["text"],
                "word_count": word_count,
            }
            utterance_rows.append(row)
            speaker_counts[utt["speaker"]] += 1
            kept_in_file += 1
            next_row_id += 1

        file_rows.append(
            {
                "file_name": parsed["file_name"],
                "participants": "|".join(parsed["participants"]),
                "selected_speakers": "|".join(selected),
                "utterances_total_main_tier": len(parsed["utterances"]),
                "utterances_kept": kept_in_file,
            }
        )

    out_jsonl = out_dir / "vercellotti_utterances.jsonl"
    out_file_summary = out_dir / "vercellotti_file_summary.csv"
    out_speaker_summary = out_dir / "vercellotti_speaker_summary.csv"
    out_summary_json = out_dir / "summary.json"

    n_rows = write_jsonl(out_jsonl, utterance_rows)
    write_csv(
        out_file_summary,
        file_rows,
        [
            "file_name",
            "participants",
            "selected_speakers",
            "utterances_total_main_tier",
            "utterances_kept",
        ],
    )
    write_csv(
        out_speaker_summary,
        [{"speaker": speaker, "utterances_kept": n} for speaker, n in sorted(speaker_counts.items())],
        ["speaker", "utterances_kept"],
    )

    summary = {
        "corpus_dir": str(corpus_dir),
        "out_dir": str(out_dir),
        "speaker_policy": args.speaker_policy,
        "include_speakers_override": include_speakers,
        "min_word_count": args.min_word_count,
        "files_processed": len(file_rows),
        "utterances_kept": n_rows,
        "utterances_dropped_short": dropped_short,
        "outputs": {
            "utterances_jsonl": str(out_jsonl),
            "file_summary_csv": str(out_file_summary),
            "speaker_summary_csv": str(out_speaker_summary),
        },
    }
    out_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
