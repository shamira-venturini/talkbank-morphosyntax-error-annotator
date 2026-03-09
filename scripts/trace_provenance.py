from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

SYNTHETIC_METHODS = {"curated_synthetic_exact", "curated_synthetic_input", "synthetic_no_real_match"}
REAL_METHODS = {"real_output_exact", "real_input_clean"}


def canonical_ws(text: str) -> str:
    text = (text or "").replace("\u2019", "'").replace("\u2018", "'")
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


def normalize_group(raw_group: str) -> str:
    value = (raw_group or "").strip().upper()
    if value == "SLI":
        return "DLD"
    if value == "TD":
        return "TD"
    return "UNKNOWN"


def safe_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_synthetic_indexes(curated_synthetic_dir: Path) -> Tuple[DefaultDict[Tuple[str, str], set], DefaultDict[str, set]]:
    by_exact: DefaultDict[Tuple[str, str], set] = defaultdict(set)
    by_input: DefaultDict[str, set] = defaultdict(set)

    for path in sorted(curated_synthetic_dir.glob("*.jsonl")):
        rel = str(path)
        for record in safe_jsonl(path):
            inp = canonical_ws(record.get("input", ""))
            out = canonical_ws(record.get("output", ""))
            by_exact[(inp, out)].add(rel)
            by_input[inp].add(rel)

    return by_exact, by_input


def build_original_indexes(data_original_root: Path) -> Tuple[DefaultDict[str, list], DefaultDict[str, list]]:
    raw_idx: DefaultDict[str, list] = defaultdict(list)
    clean_idx: DefaultDict[str, list] = defaultdict(list)

    cha_files = sorted((data_original_root / "A_original").glob("*.cha")) + sorted((data_original_root / "B_original").glob("*.cha"))

    for path in cha_files:
        current_group = "UNKNOWN"

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("@ID:"):
                    value = line.split("\t", 1)[1].strip() if "\t" in line else line.split(":", 1)[1].strip()
                    parts = value.split("|")
                    if len(parts) >= 6:
                        current_group = normalize_group(parts[5])
                elif line.startswith("@Types:") and current_group == "UNKNOWN":
                    lowered = line.lower()
                    if "sli" in lowered:
                        current_group = "DLD"
                    elif "td" in lowered:
                        current_group = "TD"
                elif line.startswith("*CHI:"):
                    utt = line.split("\t", 1)[1] if "\t" in line else line.split(":", 1)[1]
                    raw = canonical_ws(utt)
                    clean = clean_chat_input(utt)
                    entry = {"file": str(path), "group": current_group}
                    raw_idx[raw].append(entry)
                    clean_idx[clean].append(entry)

    return raw_idx, clean_idx


def choose_real_label(candidates: List[dict]) -> Tuple[str, bool, Dict[str, int]]:
    counts = Counter(c["group"] for c in candidates if c["group"] in {"TD", "DLD"})
    if not counts:
        return "TD", True, {"TD": 0, "DLD": 0}
    if len(counts) == 1:
        return next(iter(counts.keys())), False, {"TD": counts.get("TD", 0), "DLD": counts.get("DLD", 0)}

    # Deterministic tie-break to force one of required labels: TD on ties.
    label = "TD" if counts.get("TD", 0) >= counts.get("DLD", 0) else "DLD"
    return label, True, {"TD": counts.get("TD", 0), "DLD": counts.get("DLD", 0)}


def enforce_provenance_label(trace_method: str, provenance_label: str) -> str:
    """Ensure only real-matched rows can be TD/DLD."""
    if trace_method in SYNTHETIC_METHODS:
        return "synthetic"
    if trace_method in REAL_METHODS and provenance_label in {"TD", "DLD"}:
        return provenance_label
    if trace_method in REAL_METHODS:
        return "TD"
    return provenance_label


def trace_provenance(
    training_file: Path,
    curated_synthetic_dir: Path,
    data_original_root: Path,
    out_manifest: Path,
    out_augmented: Path,
    out_summary: Path,
) -> None:
    synth_exact, synth_input = build_synthetic_indexes(curated_synthetic_dir)
    real_raw, real_clean = build_original_indexes(data_original_root)

    summary = Counter()

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_augmented.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    with training_file.open("r", encoding="utf-8") as src, \
        out_manifest.open("w", encoding="utf-8") as manifest_f, \
        out_augmented.open("w", encoding="utf-8") as augmented_f:

        row_id = 0
        for line in src:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            row_id += 1
            inp = canonical_ws(record.get("input", ""))
            out = canonical_ws(record.get("output", ""))
            key = (inp, out)

            provenance_label = "synthetic"
            trace_method = "synthetic_no_real_match"
            trace_ambiguous = False
            group_counts = {"TD": 0, "DLD": 0}
            source_files: List[str] = []

            if key in synth_exact:
                provenance_label = "synthetic"
                trace_method = "curated_synthetic_exact"
                source_files = sorted(synth_exact[key])
            elif inp in synth_input:
                provenance_label = "synthetic"
                trace_method = "curated_synthetic_input"
                source_files = sorted(synth_input[inp])
            else:
                candidates: List[dict] = []
                if out in real_raw:
                    candidates = real_raw[out]
                    trace_method = "real_output_exact"
                elif inp in real_clean:
                    candidates = real_clean[inp]
                    trace_method = "real_input_clean"

                if candidates:
                    provenance_label, trace_ambiguous, group_counts = choose_real_label(candidates)
                    source_files = sorted({c["file"] for c in candidates})
                else:
                    provenance_label = "synthetic"
                    trace_method = "synthetic_no_real_match"
                    source_files = []

            normalized_label = enforce_provenance_label(trace_method, provenance_label)
            if normalized_label != provenance_label:
                summary["provenance_label_overrides"] += 1
                provenance_label = normalized_label

            if trace_ambiguous:
                summary["ambiguous_rows"] += 1
            summary[f"label_{provenance_label}"] += 1
            summary[f"method_{trace_method}"] += 1

            provenance_record = {
                "row_id": row_id,
                "provenance_label": provenance_label,
                "trace_method": trace_method,
                "trace_ambiguous": trace_ambiguous,
                "group_candidate_counts": group_counts,
                "source_file_count": len(source_files),
                "source_files": source_files,
            }

            augmented = dict(record)
            augmented.update(provenance_record)

            manifest_f.write(json.dumps(provenance_record, ensure_ascii=False) + "\n")
            augmented_f.write(json.dumps(augmented, ensure_ascii=False) + "\n")

    summary_payload = {
        "training_file": str(training_file),
        "curated_synthetic_dir": str(curated_synthetic_dir),
        "data_original_root": str(data_original_root),
        "summary": dict(summary),
    }
    out_summary.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace sentence provenance for training JSONL.")
    parser.add_argument("--training-file", default="data/intermediate/df_master_training_v3.jsonl")
    parser.add_argument("--curated-synthetic-dir", default="data/curated/synthetic")
    parser.add_argument("--data-original-root", default="/Users/shamiraventurini/PycharmProjects/ICL-PILOT/data_original")
    parser.add_argument("--out-manifest", default="data/processed/provenance_manifest.jsonl")
    parser.add_argument("--out-augmented", default="data/processed/master_training_with_provenance.jsonl")
    parser.add_argument("--out-summary", default="data/processed/provenance_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_provenance(
        training_file=Path(args.training_file),
        curated_synthetic_dir=Path(args.curated_synthetic_dir),
        data_original_root=Path(args.data_original_root),
        out_manifest=Path(args.out_manifest),
        out_augmented=Path(args.out_augmented),
        out_summary=Path(args.out_summary),
    )
    print("Provenance tracing completed.")
    print(f"Manifest: {args.out_manifest}")
    print(f"Augmented training file: {args.out_augmented}")
    print(f"Summary: {args.out_summary}")


if __name__ == "__main__":
    main()
