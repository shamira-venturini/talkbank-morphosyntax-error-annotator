from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ERROR_TAG_RE = re.compile(r"\[\*\s*[^\]]+\]")
BRACKET_CHUNK_RE = re.compile(r"(\[[^\]]*\])")
OPTIONAL_G_TOKEN_RE = re.compile(r"\b([A-Za-z]+)\(g\)")

# Transcription variants that should be consistently represented in both
# input/output surface text so the model does not learn them as correction tasks.
SURFACE_VARIANT_RULES = (
    # Only normalize standalone "nt", not substrings like "we:nt".
    ("token_nt", re.compile(r"(?<![:\w])nt(?![\w:])"), "n(o)t"),
    # Avoid double-normalizing already parenthesized "(a)nother".
    # Do not touch parenthesized or lengthened variants like "(a)nother" / "a::nother".
    ("token_nother", re.compile(r"(?<!\(a\))(?<!:)\bnother\b"), "(a)nother"),
    ("token_a_other", re.compile(r"\ba other\b"), "a(n)other"),
    # Avoid double-normalizing already parenthesized "(be)cause".
    ("token_cause", re.compile(r"(?<!\(be\))\bcause\b"), "(be)cause"),
)


def count_errors(output_text: str) -> int:
    return len(ERROR_TAG_RE.findall(output_text or ""))


def replace_outside_brackets(text: str, pattern: re.Pattern, replacement: str) -> tuple[str, int]:
    if not text:
        return "", 0
    total = 0
    chunks = BRACKET_CHUNK_RE.split(text)
    for idx, chunk in enumerate(chunks):
        if not chunk or (chunk.startswith("[") and chunk.endswith("]")):
            continue
        updated, n_sub = pattern.subn(replacement, chunk)
        if n_sub:
            total += n_sub
            chunks[idx] = updated
    return "".join(chunks), total


def strip_bracket_chunks(text: str) -> str:
    if not text:
        return ""
    chunks = BRACKET_CHUNK_RE.split(text)
    kept = []
    for chunk in chunks:
        if not chunk or (chunk.startswith("[") and chunk.endswith("]")):
            continue
        kept.append(chunk)
    return " ".join("".join(kept).split())


def normalize_surface_variants(text: str) -> tuple[str, Counter]:
    if not text:
        return "", Counter()

    counts: Counter = Counter()
    updated = text
    for name, pattern, replacement in SURFACE_VARIANT_RULES:
        updated, n_sub = replace_outside_brackets(updated, pattern, replacement)
        if n_sub:
            counts[name] += n_sub
    return updated, counts


def align_input_surface_variants(input_text: str, output_text: str) -> tuple[str, Counter]:
    if not input_text:
        return "", Counter()

    counts: Counter = Counter()
    output_surface = strip_bracket_chunks(output_text)
    aligned = input_text

    # Align known optional/transcription variants in input to match output surface.
    if "n(o)t" in output_surface and " nt " not in f" {output_surface} ":
        aligned, n_sub = replace_outside_brackets(aligned, re.compile(r"(?<![:\w])nt(?![\w:])"), "n(o)t")
        if n_sub:
            counts["align_nt_to_norm"] += n_sub
    elif re.search(r"(?<![:\w])nt(?![\w:])", output_surface):
        aligned, n_sub = replace_outside_brackets(aligned, re.compile(r"\bn\(o\)t\b"), "nt")
        if n_sub:
            counts["align_nt_to_raw"] += n_sub

    if "(a)nother" in output_surface and not re.search(r"(?<!\(a\))(?<!:)\bnother\b", output_surface):
        aligned, n_sub = replace_outside_brackets(aligned, re.compile(r"(?<!\(a\))(?<!:)\bnother\b"), "(a)nother")
        if n_sub:
            counts["align_nother_to_norm"] += n_sub
    elif re.search(r"(?<!\(a\))(?<!:)\bnother\b", output_surface):
        aligned, n_sub = replace_outside_brackets(aligned, re.compile(r"\(a\)nother"), "nother")
        if n_sub:
            counts["align_nother_to_raw"] += n_sub

    if "a(n)other" in output_surface and "a other" not in output_surface:
        aligned, n_sub = replace_outside_brackets(aligned, re.compile(r"\ba other\b"), "a(n)other")
        if n_sub:
            counts["align_a_other_to_norm"] += n_sub
    elif re.search(r"\ba other\b", output_surface):
        aligned, n_sub = replace_outside_brackets(aligned, re.compile(r"\ba\(n\)other\b"), "a other")
        if n_sub:
            counts["align_a_other_to_raw"] += n_sub

    if "(be)cause" in output_surface and not re.search(r"(?<!\(be\))\bcause\b", output_surface):
        aligned, n_sub = replace_outside_brackets(aligned, re.compile(r"(?<!\(be\))\bcause\b"), "(be)cause")
        if n_sub:
            counts["align_cause_to_norm"] += n_sub
    elif re.search(r"(?<!\(be\))\bcause\b", output_surface):
        aligned, n_sub = replace_outside_brackets(aligned, re.compile(r"\(be\)cause"), "cause")
        if n_sub:
            counts["align_cause_to_raw"] += n_sub

    # Optional consonant in CHAT transcription, e.g. lookin -> lookin(g), but only
    # when the same optional form appears in the output surface.
    for match in sorted(set(OPTIONAL_G_TOKEN_RE.findall(output_surface))):
        base = match
        target = f"{base}(g)"
        pattern = re.compile(rf"(?<![\w:]){re.escape(base)}(?![\w:(])")
        aligned, n_sub = replace_outside_brackets(aligned, pattern, target)
        if n_sub:
            counts["align_optional_g"] += n_sub

    return aligned, counts


def add_error_count(input_path: Path, output_path: Path, summary_path: Path, normalize_variants: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_zero = 0
    total_errors = 0
    by_count = Counter()
    by_provenance = Counter()
    normalization_counts_input = Counter()
    normalization_counts_output = Counter()
    alignment_counts_input = Counter()
    rows_changed_input = 0
    rows_changed_output = 0

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
            original_input = record.get("input", "")
            original_output = record.get("output", "")
            if normalize_variants:
                normalized_input, c_input = normalize_surface_variants(original_input)
                normalized_output, c_output = normalize_surface_variants(original_output)
                aligned_input, c_align = align_input_surface_variants(normalized_input, normalized_output)
                record["input"] = aligned_input
                record["output"] = normalized_output
                normalization_counts_input.update(c_input)
                normalization_counts_output.update(c_output)
                alignment_counts_input.update(c_align)

            if record.get("input", "") != original_input:
                rows_changed_input += 1
            if record.get("output", "") != original_output:
                rows_changed_output += 1

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
        "surface_variant_normalization": {
            "enabled": normalize_variants,
            "rows_changed_input": rows_changed_input,
            "rows_changed_output": rows_changed_output,
            "counts_input": dict(sorted(normalization_counts_input.items())),
            "counts_output": dict(sorted(normalization_counts_output.items())),
            "input_aligned_from_output": dict(sorted(alignment_counts_input.items())),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add `error_count` (number of [* ...] tags in output) to each JSONL row.")
    parser.add_argument("--input", default="data/processed/master_training_with_provenance.jsonl")
    parser.add_argument("--output", default="data/processed/master_training.jsonl")
    parser.add_argument("--summary", default="data/processed/master_training.summary.json")
    parser.add_argument(
        "--normalize-surface-variants",
        dest="normalize_surface_variants",
        action="store_true",
        help="Normalize selected transcription variants on both input and output surface text.",
    )
    parser.add_argument(
        "--no-normalize-surface-variants",
        dest="normalize_surface_variants",
        action="store_false",
        help="Disable transcription variant normalization.",
    )
    parser.set_defaults(normalize_surface_variants=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_error_count(
        Path(args.input),
        Path(args.output),
        Path(args.summary),
        normalize_variants=args.normalize_surface_variants,
    )
    print("Error-count enrichment completed.")
    print(f"Output: {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
