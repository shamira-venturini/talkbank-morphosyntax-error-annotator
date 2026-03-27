from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from common import iter_jsonl, resolve_path, write_jsonl
from ood_chat_utils import parse_chat_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ENNI reviewed rows for utterance_only vs prev_same_speaker context ablation."
    )
    parser.add_argument(
        "--input-jsonl",
        default="study_02_hitl_adaptation/data/stage3_corrected_plus_enni_review_utterance_only_merged.jsonl",
        help="Merged ENNI review JSONL.",
    )
    parser.add_argument(
        "--enni-dir",
        default="study_02_hitl_adaptation/ENNI_patched_from_merged_metadata",
        help="Patched ENNI transcript tree used to recover previous same-speaker context.",
    )
    parser.add_argument(
        "--out-jsonl",
        default="study_04_context_windows/data/context_eval/enni_reviewed_prev_same_speaker_eval.jsonl",
        help="Output JSONL with ENNI ablation inputs.",
    )
    parser.add_argument(
        "--out-summary",
        default="study_04_context_windows/data/context_eval/enni_reviewed_prev_same_speaker_eval_summary.json",
        help="Output summary JSON.",
    )
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        default=True,
        help="Keep only rows with review_is_reviewed=true (default: true).",
    )
    parser.add_argument(
        "--all-enni-rows",
        action="store_true",
        default=False,
        help="Override --reviewed-only and export all ENNI rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on exported rows for quick checks.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def iter_enni_rows(path: Path, reviewed_only: bool) -> Iterable[Dict]:
    for row in iter_jsonl(path):
        if row.get("source_dataset") != "ENNI_OOD":
            continue
        if reviewed_only and not bool(row.get("review_is_reviewed")):
            continue
        yield row


def index_transcripts(enni_dir: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(enni_dir.rglob("*.cha")):
        index[path.name].append(path)
    return dict(index)


def previous_same_speaker(
    utterances: List[Dict],
    target_index: int,
    speaker: str,
) -> Optional[Dict]:
    for idx in range(target_index - 1, -1, -1):
        if utterances[idx]["speaker"] == speaker:
            return utterances[idx]
    return None


def find_target_match(
    row: Dict,
    transcript_candidates: List[Path],
    parsed_cache: Dict[Path, Dict],
) -> Tuple[Path, Dict, int]:
    speaker = str(row.get("speaker", "") or "")
    line_no = int(row.get("line_no", 0) or 0)
    raw_index = int(row.get("utterance_index_raw", 0) or 0)
    input_text = normalize_text(row.get("input", ""))

    exact_matches: List[Tuple[Path, Dict, int]] = []
    line_matches: List[Tuple[Path, Dict, int]] = []
    raw_index_matches: List[Tuple[Path, Dict, int]] = []

    for path in transcript_candidates:
        parsed = parsed_cache.setdefault(path, parse_chat_file(path))
        for idx, utt in enumerate(parsed["utterances"]):
            if utt["speaker"] != speaker:
                continue
            utt_text = normalize_text(utt["text"])
            if utt["line_no"] == line_no and utt["raw_index"] == raw_index:
                exact_matches.append((path, parsed, idx))
            elif utt["line_no"] == line_no and utt_text == input_text:
                line_matches.append((path, parsed, idx))
            elif utt["raw_index"] == raw_index and utt_text == input_text:
                raw_index_matches.append((path, parsed, idx))

    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(line_matches) == 1:
        return line_matches[0]
    if len(raw_index_matches) == 1:
        return raw_index_matches[0]

    raise LookupError(
        f"Could not resolve unique transcript match for {row.get('file_name')} "
        f"(line_no={line_no}, utterance_index_raw={raw_index}, speaker={speaker})."
    )


def build_prev_same_speaker_input(prev_text: str, target_text: str) -> str:
    prev_text = normalize_text(prev_text)
    target_text = normalize_text(target_text)
    if not prev_text:
        return target_text
    return (
        "Context (for disambiguation only, do NOT annotate context lines):\n"
        f"[PREV_SAME_SPEAKER] {prev_text}\n\n"
        "Target utterance to annotate:\n"
        f"{target_text}"
    )


def build_output_row(
    row: Dict,
    transcript_path: Path,
    prev_utt: Optional[Dict],
) -> Dict:
    prev_text = normalize_text(prev_utt["text"]) if prev_utt else ""
    target_text = normalize_text(row.get("input", ""))
    output_row = {
        "row_id": row.get("row_id"),
        "source_dataset": row.get("source_dataset", ""),
        "file_name": row.get("file_name", ""),
        "transcript_path": str(transcript_path),
        "speaker": row.get("speaker", ""),
        "line_no": row.get("line_no"),
        "utterance_index_raw": row.get("utterance_index_raw"),
        "input": target_text,
        "output": normalize_text(row.get("output", "")),
        "review_is_reviewed": bool(row.get("review_is_reviewed")),
        "review_review_decision": row.get("review_review_decision", ""),
        "review_prediction_status": row.get("review_prediction_status", ""),
        "utterance_only_input": target_text,
        "prev_same_speaker_text": prev_text,
        "prev_same_speaker_line_no": prev_utt["line_no"] if prev_utt else None,
        "prev_same_speaker_utterance_index_raw": prev_utt["raw_index"] if prev_utt else None,
        "prev_same_speaker_input": build_prev_same_speaker_input(prev_text, target_text),
        "has_prev_same_speaker": bool(prev_text),
    }
    return output_row


def main() -> None:
    args = parse_args()
    input_jsonl = resolve_path(args.input_jsonl)
    enni_dir = resolve_path(args.enni_dir)
    out_jsonl = resolve_path(args.out_jsonl)
    out_summary = resolve_path(args.out_summary)

    reviewed_only = args.reviewed_only and not args.all_enni_rows
    transcript_index = index_transcripts(enni_dir)
    parsed_cache: Dict[Path, Dict] = {}

    output_rows: List[Dict] = []
    unresolved_rows: List[Dict] = []

    for row in iter_enni_rows(input_jsonl, reviewed_only=reviewed_only):
        if args.limit > 0 and len(output_rows) >= args.limit:
            break
        candidates = transcript_index.get(str(row.get("file_name", "")), [])
        if not candidates:
            unresolved_rows.append(
                {
                    "row_id": row.get("row_id"),
                    "file_name": row.get("file_name", ""),
                    "reason": "missing transcript candidate",
                }
            )
            continue

        try:
            transcript_path, parsed, target_idx = find_target_match(row, candidates, parsed_cache)
        except LookupError as exc:
            unresolved_rows.append(
                {
                    "row_id": row.get("row_id"),
                    "file_name": row.get("file_name", ""),
                    "reason": str(exc),
                }
            )
            continue

        prev_utt = previous_same_speaker(parsed["utterances"], target_idx, str(row.get("speaker", "")))
        output_rows.append(build_output_row(row, transcript_path, prev_utt))

    write_jsonl(out_jsonl, output_rows)
    summary = {
        "input_jsonl": str(input_jsonl),
        "enni_dir": str(enni_dir),
        "out_jsonl": str(out_jsonl),
        "rows_exported": len(output_rows),
        "reviewed_only": reviewed_only,
        "rows_with_prev_same_speaker": sum(1 for row in output_rows if row["has_prev_same_speaker"]),
        "rows_without_prev_same_speaker": sum(1 for row in output_rows if not row["has_prev_same_speaker"]),
        "unresolved_rows": len(unresolved_rows),
        "unresolved_examples": unresolved_rows[:10],
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
