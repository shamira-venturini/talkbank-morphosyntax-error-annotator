from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from common import iter_jsonl, resolve_path


SPEAKER_LINE_RE = re.compile(
    r"^(?P<prefix>\*(?P<token>[A-Z0-9]{1,8}):[ \t]+)(?P<body>.*?)(?P<ending>\r\n|\n|\r)?$"
)
LINE_ENDING_RE = re.compile(r"(\r\n|\n|\r)$")
CHAT_PUNCTUATION = frozenset({".", "!", ",", "?", ":"})


@dataclass
class AnnotationRow:
    annotation_row_id: str
    source_path_raw: str
    file_name: str
    speaker: str
    line_no: int | None
    utterance_index_raw: int | None
    input_text: str
    output_text: str
    metadata: Dict[str, object]


@dataclass
class MatchResult:
    status: str
    resolved_source_path: str
    output_path: str
    line_no: int | None
    utterance_index_raw: int | None
    speaker: str
    input_text: str
    output_text: str
    source_body: str
    annotation_row_id: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch source .cha transcripts from annotated CSV/JSONL rows and write patched copies."
        )
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="CSV or JSONL containing source provenance and annotated outputs.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where patched .cha copies and reports will be written.",
    )
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        help=(
            "Root directory to search for .cha files when stored paths are stale. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--path-prefix-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help=(
            "Path prefix rewrite to try before search, e.g. "
            "--path-prefix-map /content/talkbank-morphosyntax-error-annotator=/Users/me/PycharmProjects/talkbank-morphosyntax-error-annotator"
        ),
    )
    parser.add_argument(
        "--output-field",
        default="auto",
        help=(
            "Field to use as the patched utterance text. Default auto prefers "
            "corrected_output/corrected_prediction/correct_utterance/output/model_prediction/tt_annotated_text."
        ),
    )
    parser.add_argument(
        "--input-field",
        default="auto",
        help="Field to use as the source utterance text. Default auto prefers input/utterance.",
    )
    parser.add_argument(
        "--source-path-field",
        default="auto",
        help="Field containing the source .cha path. Default auto prefers file_path/source_file/source_path_raw.",
    )
    parser.add_argument(
        "--file-name-field",
        default="auto",
        help="Field containing the source filename. Default auto prefers file_name.",
    )
    parser.add_argument(
        "--speaker-field",
        default="auto",
        help="Field containing the speaker code. Default auto prefers speaker.",
    )
    parser.add_argument(
        "--line-no-field",
        default="auto",
        help="Field containing the original .cha line number. Default auto prefers line_no.",
    )
    parser.add_argument(
        "--utterance-index-field",
        default="auto",
        help="Field containing the utterance index within the source file. Default auto prefers utterance_index_raw.",
    )
    parser.add_argument(
        "--annotation-row-id-field",
        default="auto",
        help="Field used only for reporting. Default auto prefers row_id/original_row_id/review_id.",
    )
    parser.add_argument(
        "--only-filled-output",
        action="store_true",
        help="Skip rows where the chosen output field is empty.",
    )
    parser.add_argument(
        "--normalize-output-punctuation",
        action="store_true",
        help="Normalize output to CHAT spacing before writing into .cha files.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, object]]:
    if path.suffix.lower() == ".jsonl":
        return list(iter_jsonl(path))
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported annotation file type: {path}")


def first_present(row: Dict[str, object], candidates: Sequence[str]) -> tuple[str, object] | None:
    for name in candidates:
        if name in row:
            value = row[name]
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return name, value
    return None


def parse_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_json_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def choose_field(
    row: Dict[str, object],
    requested: str,
    auto_candidates: Sequence[str],
    *,
    allow_blank: bool = False,
) -> tuple[str, object] | None:
    if requested != "auto":
        if requested not in row:
            return None
        value = row[requested]
        if allow_blank:
            return requested, value
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return requested, value
    if allow_blank:
        for name in auto_candidates:
            if name in row:
                return name, row[name]
        return None
    return first_present(row, auto_candidates)


def choose_output_value(
    row: Dict[str, object],
    requested: str,
    auto_candidates: Sequence[str],
) -> tuple[str, object] | None:
    if requested != "auto":
        return choose_field(row, requested, auto_candidates, allow_blank=True)

    nonblank = first_present(row, auto_candidates)
    if nonblank is not None:
        return nonblank

    for name in auto_candidates:
        if name in row:
            return name, row[name]
    return None


def build_annotation_rows(args: argparse.Namespace, rows: Iterable[Dict[str, object]]) -> List[AnnotationRow]:
    annotation_rows: List[AnnotationRow] = []
    for row in rows:
        row_dict = dict(row)

        output_match = choose_output_value(
            row_dict,
            args.output_field,
            [
                "corrected_output",
                "corrected_prediction",
                "correct_utterance",
                "output",
                "model_prediction",
                "tt_annotated_text",
            ],
        )
        if output_match is None:
            continue
        _, output_value = output_match
        output_text = "" if output_value is None else str(output_value)
        if args.only_filled_output and not output_text.strip():
            continue

        input_match = choose_field(
            row_dict,
            args.input_field,
            ["input", "utterance", "review_input"],
            allow_blank=True,
        )
        input_text = "" if input_match is None or input_match[1] is None else str(input_match[1])

        source_path_match = choose_field(
            row_dict,
            args.source_path_field,
            ["file_path", "source_file", "source_path_raw"],
            allow_blank=True,
        )
        source_path_raw = ""
        if source_path_match is not None and source_path_match[1] is not None:
            source_path_raw = str(source_path_match[1]).strip()
        if not source_path_raw:
            source_files = parse_json_list(row_dict.get("source_files"))
            if source_files:
                source_path_raw = source_files[0]

        file_name_match = choose_field(
            row_dict,
            args.file_name_field,
            ["file_name", "review_file_name"],
            allow_blank=True,
        )
        file_name = "" if file_name_match is None or file_name_match[1] is None else str(file_name_match[1]).strip()

        speaker_match = choose_field(
            row_dict,
            args.speaker_field,
            ["speaker", "review_speaker", "original_speaker"],
            allow_blank=True,
        )
        speaker = "" if speaker_match is None or speaker_match[1] is None else str(speaker_match[1]).strip()

        line_no_match = choose_field(
            row_dict,
            args.line_no_field,
            ["line_no", "review_line_no"],
            allow_blank=True,
        )
        utterance_index_match = choose_field(
            row_dict,
            args.utterance_index_field,
            ["utterance_index_raw", "review_utterance_index_raw"],
            allow_blank=True,
        )
        row_id_match = choose_field(
            row_dict,
            args.annotation_row_id_field,
            ["row_id", "original_row_id", "review_id"],
            allow_blank=True,
        )
        annotation_row_id = (
            "" if row_id_match is None or row_id_match[1] is None else str(row_id_match[1]).strip()
        )

        annotation_rows.append(
            AnnotationRow(
                annotation_row_id=annotation_row_id,
                source_path_raw=source_path_raw,
                file_name=file_name,
                speaker=speaker,
                line_no=parse_int(None if line_no_match is None else line_no_match[1]),
                utterance_index_raw=parse_int(
                    None if utterance_index_match is None else utterance_index_match[1]
                ),
                input_text=input_text,
                output_text=output_text,
                metadata=row_dict,
            )
        )
    return annotation_rows


def split_line_ending(line: str) -> tuple[str, str]:
    match = LINE_ENDING_RE.search(line)
    if not match:
        return line, ""
    return line[: match.start()], match.group(1)


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


def canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def canonical_speaker(token: str) -> str:
    token = token.strip().upper()
    if not token:
        return token
    return token if token.startswith("*") else f"*{token}"


class PathResolver:
    def __init__(self, search_roots: Sequence[Path], prefix_maps: Sequence[tuple[str, str]]) -> None:
        self.search_roots = [root.resolve() for root in search_roots]
        self.prefix_maps = prefix_maps
        self._all_files: List[Path] | None = None
        self._by_name: Dict[str, List[Path]] | None = None

    def _ensure_index(self) -> None:
        if self._all_files is not None:
            return
        all_files: List[Path] = []
        by_name: Dict[str, List[Path]] = {}
        for root in self.search_roots:
            if not root.exists():
                continue
            for path in root.rglob("*.cha"):
                resolved = path.resolve()
                all_files.append(resolved)
                by_name.setdefault(resolved.name, []).append(resolved)
        self._all_files = all_files
        self._by_name = by_name

    def resolve(self, raw_path: str, file_name: str) -> Path | None:
        candidates: List[Path] = []
        if raw_path:
            direct = Path(raw_path)
            if direct.exists():
                return direct.resolve()
            for old, new in self.prefix_maps:
                if raw_path.startswith(old):
                    rewritten = Path(new + raw_path[len(old) :])
                    if rewritten.exists():
                        return rewritten.resolve()
            self._ensure_index()
            assert self._all_files is not None
            raw_parts = [part for part in Path(raw_path).parts if part not in {"/", ""}]
            for size in range(len(raw_parts), 1, -1):
                suffix = raw_parts[-size:]
                matches = [
                    path
                    for path in self._all_files
                    if len(path.parts) >= size and list(path.parts[-size:]) == suffix
                ]
                if len(matches) == 1:
                    return matches[0]
                if matches:
                    candidates = matches
                    break

        if file_name:
            self._ensure_index()
            assert self._by_name is not None
            name_matches = self._by_name.get(file_name, [])
            if len(name_matches) == 1:
                return name_matches[0]
            if not candidates and name_matches:
                candidates = name_matches

        if len(candidates) == 1:
            return candidates[0]
        return None

    def relative_output_path(self, resolved_source: Path) -> Path:
        for root in self.search_roots:
            try:
                return resolved_source.relative_to(root)
            except ValueError:
                continue
        return Path(resolved_source.name)


def parse_prefix(line: str) -> tuple[str, str, str] | None:
    match = SPEAKER_LINE_RE.match(line)
    if not match:
        return None
    return match.group("prefix"), canonical_speaker(match.group("token")), match.group("body")


def split_body_suffix(body: str) -> tuple[str, str, str]:
    marker_idx = body.find("\x15")
    if marker_idx < 0:
        return body, "", ""
    visible_with_sep = body[:marker_idx]
    visible = visible_with_sep.rstrip()
    separator = visible_with_sep[len(visible) :]
    return visible, separator, body[marker_idx:]


def candidate_line_indexes(lines: Sequence[str], speaker: str) -> List[int]:
    indexes: List[int] = []
    target_speaker = canonical_speaker(speaker)
    for idx, raw_line in enumerate(lines):
        content, _ = split_line_ending(raw_line)
        parsed = parse_prefix(content)
        if parsed is None:
            continue
        _, token, _ = parsed
        if target_speaker and token != target_speaker:
            continue
        indexes.append(idx)
    return indexes


def choose_line_index(row: AnnotationRow, lines: Sequence[str]) -> tuple[int | None, str]:
    target_speaker = canonical_speaker(row.speaker)
    fallback_note: str | None = None

    if row.line_no is not None and 1 <= row.line_no <= len(lines):
        idx = row.line_no - 1
        content, _ = split_line_ending(lines[idx])
        parsed = parse_prefix(content)
        if parsed is not None:
            _, token, body = parsed
            visible_body, _, _ = split_body_suffix(body)
            if not target_speaker or token == target_speaker:
                if not row.input_text or canonical_text(visible_body) == canonical_text(row.input_text):
                    return idx, "matched_by_line_no"
                if "\ufffd" in row.input_text:
                    return idx, "matched_by_line_no_with_corrupt_input"
                fallback_note = "line_no_body_mismatch"

    if row.utterance_index_raw is not None:
        speaker_lines = candidate_line_indexes(lines, row.speaker)
        ordinal = row.utterance_index_raw - 1
        if 0 <= ordinal < len(speaker_lines):
            idx = speaker_lines[ordinal]
            content, _ = split_line_ending(lines[idx])
            parsed = parse_prefix(content)
            if parsed is not None:
                _, _, body = parsed
                visible_body, _, _ = split_body_suffix(body)
                if not row.input_text or canonical_text(visible_body) == canonical_text(row.input_text):
                    return idx, "matched_by_utterance_index"
                if fallback_note is None:
                    fallback_note = "utterance_index_body_mismatch"

    exact_matches: List[int] = []
    for idx in candidate_line_indexes(lines, row.speaker):
        content, _ = split_line_ending(lines[idx])
        parsed = parse_prefix(content)
        if parsed is None:
            continue
        _, _, body = parsed
        visible_body, _, _ = split_body_suffix(body)
        if canonical_text(visible_body) == canonical_text(row.input_text):
            exact_matches.append(idx)
    if len(exact_matches) == 1:
        return exact_matches[0], "matched_by_body"
    if len(exact_matches) > 1:
        return None, "ambiguous_body_match"

    return None, fallback_note or "no_match"


def patch_file_rows(
    rows: Sequence[AnnotationRow],
    source_path: Path,
    *,
    normalize_output: bool,
) -> tuple[List[str], List[MatchResult]]:
    with source_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        original_lines = handle.readlines()
    patched_lines = list(original_lines)
    results: List[MatchResult] = []

    for row in sorted(
        rows,
        key=lambda item: (
            item.line_no if item.line_no is not None else 10**9,
            item.utterance_index_raw if item.utterance_index_raw is not None else 10**9,
            item.annotation_row_id,
        ),
    ):
        idx, match_note = choose_line_index(row, patched_lines)
        if idx is None:
            results.append(
                MatchResult(
                    status="unmatched",
                    resolved_source_path=str(source_path),
                    output_path="",
                    line_no=row.line_no,
                    utterance_index_raw=row.utterance_index_raw,
                    speaker=row.speaker,
                    input_text=row.input_text,
                    output_text=row.output_text,
                    source_body="",
                    annotation_row_id=row.annotation_row_id,
                    notes=match_note,
                )
            )
            continue

        content, ending = split_line_ending(patched_lines[idx])
        parsed = parse_prefix(content)
        if parsed is None:
            results.append(
                MatchResult(
                    status="unmatched",
                    resolved_source_path=str(source_path),
                    output_path="",
                    line_no=row.line_no,
                    utterance_index_raw=row.utterance_index_raw,
                    speaker=row.speaker,
                    input_text=row.input_text,
                    output_text=row.output_text,
                    source_body="",
                    annotation_row_id=row.annotation_row_id,
                    notes=f"{match_note}; target line is not a speaker line",
                )
            )
            continue

        prefix, _, current_body = parsed
        visible_body, separator, trailing_suffix = split_body_suffix(current_body)
        replacement = row.output_text
        if normalize_output:
            replacement = normalize_chat_punctuation(replacement)

        if canonical_text(visible_body) != canonical_text(row.input_text) and match_note.endswith("mismatch"):
            results.append(
                MatchResult(
                    status="mismatch",
                    resolved_source_path=str(source_path),
                    output_path="",
                    line_no=row.line_no,
                    utterance_index_raw=row.utterance_index_raw,
                    speaker=row.speaker,
                    input_text=row.input_text,
                    output_text=row.output_text,
                    source_body=visible_body,
                    annotation_row_id=row.annotation_row_id,
                    notes=match_note,
                )
            )
            continue

        patched_lines[idx] = f"{prefix}{replacement}{separator}{trailing_suffix}{ending}"
        results.append(
            MatchResult(
                status="patched",
                resolved_source_path=str(source_path),
                output_path="",
                line_no=idx + 1,
                utterance_index_raw=row.utterance_index_raw,
                speaker=row.speaker,
                input_text=row.input_text,
                output_text=replacement,
                source_body=visible_body,
                annotation_row_id=row.annotation_row_id,
                notes=match_note,
            )
        )

    return patched_lines, results


def parse_prefix_maps(values: Sequence[str]) -> List[tuple[str, str]]:
    mappings: List[tuple[str, str]] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --path-prefix-map value: {item}")
        old, new = item.split("=", 1)
        mappings.append((old.rstrip("/"), new.rstrip("/")))
    return mappings


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    annotations_path = resolve_path(args.annotations)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    search_roots = [resolve_path(path) for path in args.source_root]
    prefix_maps = parse_prefix_maps(args.path_prefix_map)
    resolver = PathResolver(search_roots=search_roots, prefix_maps=prefix_maps)

    raw_rows = load_rows(annotations_path)
    annotation_rows = build_annotation_rows(args, raw_rows)

    rows_by_source: Dict[str, List[AnnotationRow]] = {}
    unresolved_rows: List[Dict[str, object]] = []
    for row in annotation_rows:
        resolved = resolver.resolve(row.source_path_raw, row.file_name)
        if resolved is None:
            unresolved_rows.append(
                {
                    "status": "unresolved_source_path",
                    "annotation_row_id": row.annotation_row_id,
                    "source_path_raw": row.source_path_raw,
                    "file_name": row.file_name,
                    "speaker": row.speaker,
                    "line_no": row.line_no,
                    "utterance_index_raw": row.utterance_index_raw,
                    "input_text": row.input_text,
                    "output_text": row.output_text,
                }
            )
            continue
        rows_by_source.setdefault(str(resolved), []).append(row)

    manifest_rows: List[Dict[str, object]] = list(unresolved_rows)
    patched_files = 0
    patched_rows = 0

    for source_path_str, source_rows in sorted(rows_by_source.items()):
        source_path = Path(source_path_str)
        patched_lines, results = patch_file_rows(
            source_rows,
            source_path,
            normalize_output=args.normalize_output_punctuation,
        )
        if any(result.status == "patched" for result in results):
            relative_path = resolver.relative_output_path(source_path)
            output_path = out_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                handle.write("".join(patched_lines))
            patched_files += 1
        else:
            output_path = Path()

        for result in results:
            if output_path:
                result.output_path = str(output_path)
            manifest_rows.append(
                {
                    "status": result.status,
                    "annotation_row_id": result.annotation_row_id,
                    "resolved_source_path": result.resolved_source_path,
                    "output_path": result.output_path,
                    "line_no": result.line_no,
                    "utterance_index_raw": result.utterance_index_raw,
                    "speaker": result.speaker,
                    "input_text": result.input_text,
                    "source_body": result.source_body,
                    "output_text": result.output_text,
                    "notes": result.notes,
                }
            )
            if result.status == "patched":
                patched_rows += 1

    manifest_csv = out_dir / "_patch_manifest.csv"
    write_csv(manifest_csv, manifest_rows)

    summary = {
        "annotations": str(annotations_path),
        "out_dir": str(out_dir),
        "source_roots": [str(path) for path in search_roots],
        "path_prefix_maps": [{"old": old, "new": new} for old, new in prefix_maps],
        "rows_loaded": len(annotation_rows),
        "source_files_resolved": len(rows_by_source),
        "patched_files": patched_files,
        "patched_rows": patched_rows,
        "unresolved_rows": sum(row["status"] == "unresolved_source_path" for row in manifest_rows),
        "mismatch_rows": sum(row["status"] == "mismatch" for row in manifest_rows),
        "unmatched_rows": sum(row["status"] == "unmatched" for row in manifest_rows),
        "manifest_csv": str(manifest_csv),
    }
    summary_path = out_dir / "_patch_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
