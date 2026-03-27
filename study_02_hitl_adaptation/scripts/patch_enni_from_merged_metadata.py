from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from common import resolve_path


SPEAKER_LINE_RE = re.compile(
    r"^(?P<prefix>\*(?P<token>[A-Z0-9]{1,8}):[ \t]+)(?P<body>.*?)(?P<ending>\r\n|\n|\r)?$"
)
LINE_ENDING_RE = re.compile(r"(\r\n|\n|\r)$")
CHAT_PUNCTUATION = frozenset({".", "!", ",", "?", ":"})
OLD_STORY_RE = re.compile(r"^(?P<base>\d+)_(?P<story>[AB]\d)\.cha$")


@dataclass
class StoryLine:
    line_index: int
    line_no: int
    speaker: str
    prefix: str
    visible_body: str
    separator: str
    trailing_suffix: str
    ending: str
    canonical_body: str
    clean_body: str


@dataclass
class StorySection:
    code: str
    lines: List[StoryLine]
    start_line_no: int
    end_line_no: int


@dataclass
class DirectPatchRow:
    metadata_row_id: str
    file_name: str
    line_no: int | None
    utterance_index_raw: int | None
    speaker: str
    input_text: str
    output_text: str


@dataclass
class Study1StoryRow:
    metadata_row_id: str
    old_story_name: str
    base_name: str
    story_code: str
    input_text: str
    output_text: str
    source_files: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch the current ENNI transcript tree using direct ENNI review rows and "
            "story-aligned Study 1 rows from the merged metadata CSV."
        )
    )
    parser.add_argument(
        "--metadata-csv",
        default=(
            "study_02_hitl_adaptation/data/2026-03-23_training-data-auto-annotated/"
            "stage3_corrected_plus_enni_review_utterance_only_merged_metadata.csv"
        ),
        help="Merged metadata CSV used as the master reference.",
    )
    parser.add_argument(
        "--enni-root",
        default="study_02_hitl_adaptation/ENNI",
        help="Current ENNI source tree.",
    )
    parser.add_argument(
        "--out-dir",
        default="study_02_hitl_adaptation/ENNI_patched_from_merged_metadata",
        help="Output tree for the patched ENNI corpus.",
    )
    parser.add_argument(
        "--copy-all-files",
        action="store_true",
        help="Mirror the full ENNI tree into the output directory before patching.",
    )
    parser.add_argument(
        "--direct-enni-only",
        action="store_true",
        help="Apply only direct ENNI review rows and skip Study 1 story recovery.",
    )
    return parser.parse_args()


def split_line_ending(line: str) -> tuple[str, str]:
    match = LINE_ENDING_RE.search(line)
    if not match:
        return line, ""
    return line[: match.start()], match.group(1)


def split_body_suffix(body: str) -> tuple[str, str, str]:
    marker_idx = body.find("\x15")
    if marker_idx < 0:
        return body, "", ""
    visible_with_sep = body[:marker_idx]
    visible = visible_with_sep.rstrip()
    separator = visible_with_sep[len(visible) :]
    return visible, separator, body[marker_idx:]


def canonical_ws(text: str) -> str:
    text = (text or "").replace("\u2019", "'").replace("\u2018", "'").replace("\u00a0", " ")
    text = re.sub(r"\x15.*?\x15", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_chat_input(text: str) -> str:
    text = canonical_ws(text)
    text = re.sub(r"\[::?\s+[^\]]+\]", "", text)
    text = re.sub(r"\[\*\s+[^\]]+\]", "", text)
    text = re.sub(r"\[\/+\]", "", text)
    text = re.sub(r"\[\+\s+[^\]]+\]", "", text)
    text = text.replace("<", "").replace(">", "")
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([a-zA-Z0-9])([\.!\?])", r"\1 \2", text)
    return text


def normalize_chat_punctuation(text: str) -> str:
    chars: list[str] = []
    idx = 0
    bracket_depth = 0
    while idx < len(text):
        char = text[idx]
        if char == "[":
            bracket_depth += 1
            chars.append(char)
            idx += 1
            continue

        if bracket_depth > 0:
            chars.append(char)
            if char == "]":
                bracket_depth -= 1
            idx += 1
            continue

        if char not in CHAT_PUNCTUATION:
            chars.append(char)
            idx += 1
            continue

        while chars and chars[-1] == " ":
            chars.pop()
        if chars and chars[-1] not in {"\t", "\n", "\r"}:
            chars.append(" ")
        chars.append(char)

        scan = idx + 1
        while scan < len(text) and text[scan] == " ":
            scan += 1
        if scan < len(text) and text[scan] not in {"\t", "\n", "\r"}:
            chars.append(" ")
        idx = scan
    return "".join(chars)


def canonical_speaker(token: str) -> str:
    token = (token or "").strip().upper()
    if not token:
        return token
    return token if token.startswith("*") else f"*{token}"


def parse_json_list(text: str) -> List[str]:
    if not text:
        return []
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def iter_metadata_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield row


def parse_cha_file(path: Path) -> tuple[List[str], Dict[str, StorySection]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        raw_lines = handle.readlines()

    sections: Dict[str, StorySection] = {}
    current_code = "PRE0"
    current_lines: List[StoryLine] = []
    section_start = 1

    def flush_section(end_line_no: int) -> None:
        if current_lines:
            sections[current_code] = StorySection(
                code=current_code,
                lines=list(current_lines),
                start_line_no=section_start,
                end_line_no=end_line_no,
            )

    for idx, raw_line in enumerate(raw_lines):
        content, ending = split_line_ending(raw_line)
        if content.startswith("@G:"):
            flush_section(idx)
            current_lines = []
            current_code = content.split(":", 1)[1].strip()
            section_start = idx + 1
            continue

        match = SPEAKER_LINE_RE.match(content)
        if not match:
            continue
        speaker = canonical_speaker(match.group("token"))
        body = match.group("body")
        visible_body, separator, trailing_suffix = split_body_suffix(body)
        current_lines.append(
            StoryLine(
                line_index=idx,
                line_no=idx + 1,
                speaker=speaker,
                prefix=match.group("prefix"),
                visible_body=visible_body,
                separator=separator,
                trailing_suffix=trailing_suffix,
                ending=ending,
                canonical_body=canonical_ws(visible_body),
                clean_body=clean_chat_input(visible_body),
            )
        )

    flush_section(len(raw_lines))
    return raw_lines, sections


def build_current_enni_index(enni_root: Path) -> Dict[str, List[Path]]:
    by_name: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(enni_root.rglob("*.cha")):
        by_name[path.name].append(path)
    return by_name


def safe_int(text: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def output_source_text(row: Dict[str, str]) -> str:
    corrected = (row.get("review_corrected_prediction") or row.get("corrected_prediction") or "").strip()
    if corrected:
        return corrected
    if row.get("output"):
        return row["output"]
    return row.get("model_prediction", "")


def collect_direct_enni_rows(rows: Iterable[Dict[str, str]]) -> List[DirectPatchRow]:
    direct_rows: List[DirectPatchRow] = []
    for row in rows:
        if row.get("source_dataset") != "ENNI_OOD":
            continue
        direct_rows.append(
            DirectPatchRow(
                metadata_row_id=row.get("row_id", ""),
                file_name=row.get("file_name", "") or row.get("review_file_name", ""),
                line_no=safe_int(row.get("line_no") or row.get("review_line_no", "")),
                utterance_index_raw=safe_int(
                    row.get("utterance_index_raw") or row.get("review_utterance_index_raw", "")
                ),
                speaker=row.get("speaker", "") or row.get("review_speaker", ""),
                input_text=row.get("input", "") or row.get("review_input", ""),
                output_text=output_source_text(row),
            )
        )
    return direct_rows


def collect_study1_story_rows(rows: Iterable[Dict[str, str]]) -> List[Study1StoryRow]:
    story_rows: List[Study1StoryRow] = []
    for row in rows:
        if row.get("source_dataset") != "study1_train":
            continue
        if not row.get("trace_method", "").startswith("real_"):
            continue
        source_files = parse_json_list(row.get("source_files", ""))
        if not source_files:
            continue
        story_name = Path(source_files[0]).name
        match = OLD_STORY_RE.match(story_name)
        if not match:
            continue
        story_rows.append(
            Study1StoryRow(
                metadata_row_id=row.get("row_id", ""),
                old_story_name=story_name,
                base_name=f"{match.group('base')}.cha",
                story_code=match.group("story"),
                input_text=row.get("input", ""),
                output_text=row.get("output", ""),
                source_files=source_files,
            )
        )
    return story_rows


def select_direct_candidate(
    row: DirectPatchRow,
    candidates: Sequence[Path],
) -> tuple[Path | None, str]:
    target_speaker = canonical_speaker(row.speaker)
    matches: List[Path] = []
    for path in candidates:
        raw_lines, _ = parse_cha_file(path)
        if row.line_no is None or row.line_no < 1 or row.line_no > len(raw_lines):
            continue
        content, _ = split_line_ending(raw_lines[row.line_no - 1])
        match = SPEAKER_LINE_RE.match(content)
        if not match:
            continue
        if target_speaker and canonical_speaker(match.group("token")) != target_speaker:
            continue
        visible_body, _, _ = split_body_suffix(match.group("body"))
        if canonical_ws(visible_body) == canonical_ws(row.input_text):
            matches.append(path)
    if len(matches) == 1:
        return matches[0], "matched_by_line_no"
    if len(matches) > 1:
        return None, "ambiguous_duplicate_filename"
    return None, "no_direct_match"


def patch_direct_rows(
    direct_rows: Sequence[DirectPatchRow],
    current_index: Dict[str, List[Path]],
    file_cache: Dict[Path, tuple[List[str], Dict[str, StorySection]]],
    patched_lines: Dict[Path, List[str]],
    manifest: List[Dict[str, object]],
) -> None:
    for row in direct_rows:
        candidates = current_index.get(row.file_name, [])
        if not candidates:
            manifest.append(
                {
                    "patch_source": "ENNI_OOD",
                    "status": "unresolved_file",
                    "metadata_row_id": row.metadata_row_id,
                    "file_name": row.file_name,
                    "story_code": "",
                    "resolved_file": "",
                    "line_no": row.line_no,
                    "input_text": row.input_text,
                    "output_text": row.output_text,
                    "notes": "no current ENNI file with this basename",
                }
            )
            continue

        selected, note = select_direct_candidate(row, candidates)
        if selected is None:
            manifest.append(
                {
                    "patch_source": "ENNI_OOD",
                    "status": "unmatched",
                    "metadata_row_id": row.metadata_row_id,
                    "file_name": row.file_name,
                    "story_code": "",
                    "resolved_file": "",
                    "line_no": row.line_no,
                    "input_text": row.input_text,
                    "output_text": row.output_text,
                    "notes": note,
                }
            )
            continue

        if selected not in file_cache:
            file_cache[selected] = parse_cha_file(selected)
        raw_lines, _ = file_cache[selected]
        if selected not in patched_lines:
            patched_lines[selected] = list(raw_lines)

        content, ending = split_line_ending(patched_lines[selected][row.line_no - 1])
        match = SPEAKER_LINE_RE.match(content)
        if not match:
            manifest.append(
                {
                    "patch_source": "ENNI_OOD",
                    "status": "unmatched",
                    "metadata_row_id": row.metadata_row_id,
                    "file_name": row.file_name,
                    "story_code": "",
                    "resolved_file": str(selected),
                    "line_no": row.line_no,
                    "input_text": row.input_text,
                    "output_text": row.output_text,
                    "notes": "target line is no longer a speaker line",
                }
            )
            continue

        visible_body, separator, trailing_suffix = split_body_suffix(match.group("body"))
        replacement = normalize_chat_punctuation(row.output_text)
        patched_lines[selected][row.line_no - 1] = (
            f"{match.group('prefix')}{replacement}{separator}{trailing_suffix}{ending}"
        )
        manifest.append(
            {
                "patch_source": "ENNI_OOD",
                "status": "patched",
                "metadata_row_id": row.metadata_row_id,
                "file_name": row.file_name,
                "story_code": "",
                "resolved_file": str(selected),
                "line_no": row.line_no,
                "input_text": row.input_text,
                "source_body": visible_body,
                "output_text": replacement,
                "notes": note,
            }
        )


def story_match_positions(section: StorySection, input_text: str) -> List[int]:
    input_canonical = canonical_ws(input_text)
    matches: List[int] = []
    for idx, line in enumerate(section.lines):
        if line.speaker != "*CHI":
            continue
        if line.canonical_body == input_canonical or line.clean_body == input_canonical:
            matches.append(idx)
    return matches


def score_story_candidate(section: StorySection, rows: Sequence[Study1StoryRow]) -> tuple[int, int]:
    score = 0
    changed = 0
    for row in rows:
        positions = story_match_positions(section, row.input_text)
        if positions:
            score += 1
            if canonical_ws(row.output_text) != canonical_ws(row.input_text):
                changed += 1
    return score, changed


def choose_story_candidate(
    rows: Sequence[Study1StoryRow],
    candidates: Sequence[Path],
    file_cache: Dict[Path, tuple[List[str], Dict[str, StorySection]]],
) -> tuple[Path | None, StorySection | None, str]:
    story_code = rows[0].story_code
    best: tuple[Path, StorySection, int, int] | None = None
    tied = False

    for path in candidates:
        if path not in file_cache:
            file_cache[path] = parse_cha_file(path)
        _, sections = file_cache[path]
        section = sections.get("PRE0") if story_code.endswith("0") else sections.get(story_code)
        if section is None:
            continue
        score, changed = score_story_candidate(section, rows)
        if score == 0:
            continue
        if best is None or (score, changed) > (best[2], best[3]):
            best = (path, section, score, changed)
            tied = False
        elif best is not None and (score, changed) == (best[2], best[3]):
            tied = True

    if best is None:
        return None, None, "no_story_candidate"
    if tied:
        return None, None, "ambiguous_story_candidate"
    return best[0], best[1], "matched_story_candidate"


def patch_study1_story_rows(
    story_rows: Sequence[Study1StoryRow],
    current_index: Dict[str, List[Path]],
    file_cache: Dict[Path, tuple[List[str], Dict[str, StorySection]]],
    patched_lines: Dict[Path, List[str]],
    manifest: List[Dict[str, object]],
) -> None:
    rows_by_story: Dict[str, List[Study1StoryRow]] = defaultdict(list)
    for row in story_rows:
        rows_by_story[row.old_story_name].append(row)

    for story_name, rows in sorted(rows_by_story.items()):
        base_name = rows[0].base_name
        story_code = rows[0].story_code
        candidates = current_index.get(base_name, [])
        if not candidates:
            for row in rows:
                manifest.append(
                    {
                        "patch_source": "study1_train",
                        "status": "unresolved_file",
                        "metadata_row_id": row.metadata_row_id,
                        "file_name": base_name,
                        "story_code": story_code,
                        "resolved_file": "",
                        "line_no": "",
                        "input_text": row.input_text,
                        "output_text": row.output_text,
                        "notes": f"no current ENNI file named {base_name}",
                    }
                )
            continue

        selected_file, section, note = choose_story_candidate(rows, candidates, file_cache)
        if selected_file is None or section is None:
            for row in rows:
                manifest.append(
                    {
                        "patch_source": "study1_train",
                        "status": "unmatched_story",
                        "metadata_row_id": row.metadata_row_id,
                        "file_name": base_name,
                        "story_code": story_code,
                        "resolved_file": "",
                        "line_no": "",
                        "input_text": row.input_text,
                        "output_text": row.output_text,
                        "notes": note,
                    }
                )
            continue

        if selected_file not in patched_lines:
            patched_lines[selected_file] = list(file_cache[selected_file][0])

        grouped: Dict[Tuple[str, str], List[Study1StoryRow]] = defaultdict(list)
        for row in rows:
            grouped[(canonical_ws(row.input_text), canonical_ws(row.output_text))].append(row)

        for (_, _), group_rows in grouped.items():
            sample = group_rows[0]
            positions = story_match_positions(section, sample.input_text)
            changed_output = canonical_ws(sample.output_text) != canonical_ws(sample.input_text)

            if len(positions) < len(group_rows):
                for row in group_rows:
                    manifest.append(
                        {
                            "patch_source": "study1_train",
                            "status": "insufficient_story_matches",
                            "metadata_row_id": row.metadata_row_id,
                            "file_name": base_name,
                            "story_code": story_code,
                            "resolved_file": str(selected_file),
                            "line_no": "",
                            "input_text": row.input_text,
                            "output_text": row.output_text,
                            "notes": f"matched {len(positions)} positions for {len(group_rows)} rows",
                        }
                    )
                continue

            if len(positions) > len(group_rows) and changed_output:
                for row in group_rows:
                    manifest.append(
                        {
                            "patch_source": "study1_train",
                            "status": "ambiguous_repeated_input",
                            "metadata_row_id": row.metadata_row_id,
                            "file_name": base_name,
                            "story_code": story_code,
                            "resolved_file": str(selected_file),
                            "line_no": "",
                            "input_text": row.input_text,
                            "output_text": row.output_text,
                            "notes": f"{len(positions)} candidate lines for {len(group_rows)} rows",
                        }
                    )
                continue

            positions_to_use = positions[: len(group_rows)]
            for idx, row in enumerate(group_rows):
                if not changed_output:
                    manifest.append(
                        {
                            "patch_source": "study1_train",
                            "status": "matched_no_change",
                            "metadata_row_id": row.metadata_row_id,
                            "file_name": base_name,
                            "story_code": story_code,
                            "resolved_file": str(selected_file),
                            "line_no": section.lines[positions_to_use[min(idx, len(positions_to_use) - 1)]].line_no
                            if positions_to_use
                            else "",
                            "input_text": row.input_text,
                            "output_text": row.output_text,
                            "notes": note,
                        }
                    )
                    continue

                story_line = section.lines[positions_to_use[idx]]
                content, ending = split_line_ending(patched_lines[selected_file][story_line.line_index])
                match = SPEAKER_LINE_RE.match(content)
                if not match:
                    manifest.append(
                        {
                            "patch_source": "study1_train",
                            "status": "unmatched_after_select",
                            "metadata_row_id": row.metadata_row_id,
                            "file_name": base_name,
                            "story_code": story_code,
                            "resolved_file": str(selected_file),
                            "line_no": story_line.line_no,
                            "input_text": row.input_text,
                            "output_text": row.output_text,
                            "notes": "target line is no longer a speaker line",
                        }
                    )
                    continue

                replacement = normalize_chat_punctuation(row.output_text)
                patched_lines[selected_file][story_line.line_index] = (
                    f"{match.group('prefix')}{replacement}{story_line.separator}"
                    f"{story_line.trailing_suffix}{ending}"
                )
                manifest.append(
                    {
                        "patch_source": "study1_train",
                        "status": "patched",
                        "metadata_row_id": row.metadata_row_id,
                        "file_name": base_name,
                        "story_code": story_code,
                        "resolved_file": str(selected_file),
                        "line_no": story_line.line_no,
                        "input_text": row.input_text,
                        "source_body": story_line.visible_body,
                        "output_text": replacement,
                        "notes": note,
                    }
                )


def mirror_tree(src_root: Path, dst_root: Path) -> None:
    for src in src_root.rglob("*"):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_manifest(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_patched_files(
    patched_lines: Dict[Path, List[str]],
    enni_root: Path,
    out_dir: Path,
) -> None:
    for src_path, lines in patched_lines.items():
        rel = src_path.relative_to(enni_root)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("".join(lines))


def main() -> None:
    args = parse_args()
    metadata_csv = resolve_path(args.metadata_csv)
    enni_root = resolve_path(args.enni_root)
    out_dir = resolve_path(args.out_dir)

    rows = list(iter_metadata_rows(metadata_csv))
    current_index = build_current_enni_index(enni_root)
    file_cache: Dict[Path, tuple[List[str], Dict[str, StorySection]]] = {}
    patched_lines: Dict[Path, List[str]] = {}
    manifest: List[Dict[str, object]] = []

    direct_rows = collect_direct_enni_rows(rows)
    patch_direct_rows(direct_rows, current_index, file_cache, patched_lines, manifest)

    if not args.direct_enni_only:
        story_rows = collect_study1_story_rows(rows)
        patch_study1_story_rows(story_rows, current_index, file_cache, patched_lines, manifest)

    if args.copy_all_files:
        out_dir.mkdir(parents=True, exist_ok=True)
        mirror_tree(enni_root, out_dir)
    write_patched_files(patched_lines, enni_root, out_dir)

    manifest_path = out_dir / "_patch_manifest.csv"
    summary_path = out_dir / "_patch_summary.json"
    write_manifest(manifest_path, manifest)

    status_counts = Counter(row["status"] for row in manifest)
    patch_source_counts = Counter(row["patch_source"] for row in manifest)
    summary = {
        "metadata_csv": str(metadata_csv),
        "enni_root": str(enni_root),
        "out_dir": str(out_dir),
        "copy_all_files": bool(args.copy_all_files),
        "current_enni_files": sum(len(paths) for paths in current_index.values()),
        "direct_enni_rows": len(direct_rows),
        "study1_story_rows": 0 if args.direct_enni_only else len(collect_study1_story_rows(rows)),
        "patched_files": len(patched_lines),
        "status_counts": dict(status_counts),
        "patch_source_counts": dict(patch_source_counts),
        "manifest_csv": str(manifest_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
