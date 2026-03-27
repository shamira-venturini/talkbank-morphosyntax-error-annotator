from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from common import resolve_path
from normalize_ood_review_csv import classify_hallucination


MODES = ["utterance_only", "local_prev", "full_prev", "full_document"]
TAG_RE = re.compile(r"\[\*\s*[ms](?::[^\]]+)?\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag suspicious unannotated OOD utterances with likely missed errors.")
    parser.add_argument(
        "--predictions-dir",
        default="results/ood_vercellotti 4",
        help="Directory containing predictions_<mode>.jsonl files.",
    )
    parser.add_argument(
        "--out-csv",
        default="results/ood_vercellotti/suspicious_missed_errors_utterance_only.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--summary-json",
        default="results/ood_vercellotti/suspicious_missed_errors_utterance_only_summary.json",
        help="Summary JSON path.",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=2,
        help="Minimum heuristic score to keep a row.",
    )
    return parser.parse_args()


def load_jsonl_lookup(path: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["row_id"])] = row
    return out


def write_csv(path: Path, rows: Iterable[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def extract_tags(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return sorted(dict.fromkeys(TAG_RE.findall(text)))


def normalize_text(text: str) -> str:
    text = (text or "").replace("\xa0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def heuristic_flags(text: str) -> List[str]:
    t = f" {normalize_text(text).lower()} "
    flags: List[str] = []

    patterns = [
        ("3SG_BARE_VERB_HAVE", r"\b(he|she|it|that|this)\s+have\b"),
        ("3SG_BARE_VERB_LIKE", r"\b(he|she)\s+like\b"),
        ("3SG_BARE_VERB_LOOK", r"\b(he|she|it|that|this)\s+look\b"),
        ("3SG_BARE_VERB_LIVE", r"\b(he|she|it|that|this)\s+live\b"),
        ("3SG_BARE_VERB_WANT", r"\b(he|she|it|that|this)\s+want\b"),
        ("3SG_BARE_VERB_MATCH", r"\b(he|she|it|that|this)\s+match\b"),
        ("3SG_BARE_VERB_CHANGE", r"\b(he|she|it|that|this)\s+change\b"),
        ("BE_PLUS_BARE_VERB", r"\b(am|is|are|was|were|be|been)\s+(talk|compare|describe|challenge|add|ask|remember|visited|study|try)\b"),
        ("GOING_BARE_VERB", r"\b(go|going)\s+to\s+talk\b"),
        ("WANT_BARE_BE", r"\bwant\s+be\b"),
        ("WITHOUT_PAST_PARTICIPLE", r"\bwithout\s+(studied|finish|finished)\b"),
        ("TO_GONE", r"\bto\s+gone\b"),
        ("NUMBER_SINGULAR_TIME", r"\b(two|three|four|five|six|seven|eight|nine|ten)\s+(month|year|day|generation|person)\b"),
        ("MANY_SINGULAR_NOUN", r"\bmany\s+(store|advantage|generation)\b"),
        ("ENJOY_IN", r"\benjoy\s+in\b"),
        ("ENTER_TO_HOME", r"\benter\s+to\s+home\b"),
        ("TO_GETS", r"\bto\s+gets\b"),
        ("PEOPLES", r"\bpeoples\b"),
        ("REGULAR_JOB_AMOUNT_OF_SALARY", r"\bamount of salary\b"),
        ("THERE_HAS_PLURAL", r"\bthere\s+has\s+many\b"),
        ("THERE_IS_PLURAL", r"\bpeople\s+there\s+is\b"),
        ("WAS_HAVE", r"\bwas\s+have\b"),
        ("MUST_TO", r"\bmust\s+to\b"),
        ("IT_WILL_AUX", r"\bit'?s\s+will\b"),
    ]

    for name, pattern in patterns:
        if re.search(pattern, t):
            flags.append(name)
    return flags


def build_rows(lookups: Dict[str, Dict[str, Dict]], min_score: int) -> List[Dict]:
    base = lookups["utterance_only"]
    out_rows: List[Dict] = []

    for row_id, row in base.items():
        base_pred = normalize_text(row.get("model_prediction", ""))
        if "[*" in base_pred:
            continue

        input_text = row.get("input", "")
        base_hall, base_hall_type = classify_hallucination(base_pred)
        flags = heuristic_flags(input_text)

        other_mode_tags = {}
        other_mode_hall = {}
        modes_with_tags: List[str] = []
        for mode in MODES[1:]:
            pred = normalize_text(lookups[mode][row_id].get("model_prediction", ""))
            tags = extract_tags(pred)
            if tags:
                modes_with_tags.append(mode)
            other_mode_tags[mode] = " ; ".join(tags)
            hall, hall_type = classify_hallucination(pred)
            other_mode_hall[mode] = hall_type if hall == "1" else ""

        if not flags:
            continue

        score = len(flags)
        if len(modes_with_tags) >= 2:
            score += 1
        if len(modes_with_tags) == 3:
            score += 1
        if base_hall == "1":
            score -= 1

        if score < min_score:
            continue

        out_rows.append(
            {
                "row_id": row_id,
                "file_name": row.get("file_name", ""),
                "speaker": row.get("speaker", ""),
                "line_no": row.get("line_no", ""),
                "suspicion_score": score,
                "heuristic_flags": " ; ".join(flags),
                "utterance_only_hallucination": base_hall_type,
                "modes_with_tags": " ; ".join(modes_with_tags),
                "local_prev_labels": other_mode_tags["local_prev"],
                "full_prev_labels": other_mode_tags["full_prev"],
                "full_document_labels": other_mode_tags["full_document"],
                "local_prev_hallucination": other_mode_hall["local_prev"],
                "full_prev_hallucination": other_mode_hall["full_prev"],
                "full_document_hallucination": other_mode_hall["full_document"],
                "input": input_text,
                "utterance_only_prediction": row.get("model_prediction", ""),
                "review_decision": "",
                "corrected_annotation": "",
                "notes": "",
            }
        )

    out_rows.sort(key=lambda r: (-int(r["suspicion_score"]), r["file_name"], int(r["line_no"])))
    return out_rows


def build_summary(rows: List[Dict]) -> Dict:
    flag_counts = Counter()
    for row in rows:
        for flag in [f.strip() for f in row["heuristic_flags"].split(";") if f.strip()]:
            flag_counts[flag] += 1
    return {
        "rows": len(rows),
        "flag_counts": dict(flag_counts),
        "rows_with_other_mode_tags": sum(1 for row in rows if row["modes_with_tags"]),
    }


def main() -> None:
    args = parse_args()
    predictions_dir = resolve_path(args.predictions_dir)
    out_csv = resolve_path(args.out_csv)
    summary_json = resolve_path(args.summary_json)

    lookups = {
        mode: load_jsonl_lookup(predictions_dir / f"predictions_{mode}.jsonl") for mode in MODES
    }
    rows = build_rows(lookups, min_score=args.min_score)
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(out_csv, rows, fieldnames)

    summary = build_summary(rows)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out_csv": str(out_csv), "summary_json": str(summary_json), **summary}, indent=2))


if __name__ == "__main__":
    main()
