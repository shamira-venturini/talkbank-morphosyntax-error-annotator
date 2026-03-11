import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from common import iter_jsonl, resolve_path, write_jsonl


TAG_PATTERN = re.compile(r"\[\*\s*[ms](?::[^\]]+)?\]")
RECON_ANY_PATTERN = re.compile(r"\[(?::|::)\s+[^\]]+\]")
RECON_DOUBLE_PATTERN = re.compile(r"\[::\s+[^\]]+\]")
M03_NO_RECON_PATTERN = re.compile(r"(?P<lemma>[^\s\[\]]+)\s+(?P<tag>\[\*\s*m:03s:a\])")
MBASEED_NO_RECON_PATTERN = re.compile(r"(?P<lemma>[^\s\[\]]+)\s+(?P<tag>\[\*\s*m:base:ed\])")

# Kept in the project (user decision): [* s:r:gc:det]
DEFAULT_HOLDOUT_LABELS = [
    "[* m:++er]",
    "[* m:++est]",
    "[* m:0er]",
    "[* m:0est]",
]

LEVEL1_LABELS = [
    "[* m]",
    "[* s]",
]

LEVEL2_LABELS = [
    "[* m:++]",
    "[* m:+]",
    "[* m:=]",
    "[* m:0]",
    "[* m:allo]",
    "[* m:base]",
    "[* m:irr]",
    "[* m:sub]",
    "[* m:vsg]",
    "[* m:vun]",
    "[* s:r]",
    "[* s:r:gc]",
]

CHAT_TOKEN_CANONICAL_ORDER = [
    "[* m]",
    "[* s]",
    "[* m:++]",
    "[* m:+]",
    "[* m:=]",
    "[* m:0]",
    "[* m:allo]",
    "[* m:base]",
    "[* m:irr]",
    "[* m:sub]",
    "[* m:vsg]",
    "[* m:vun]",
    "[* s:r]",
    "[* s:r:gc]",
    "[* m:03s:a]",
    "[* m:irr:en]",
    "[* m:=ed]",
    "[* s:r:gc:pro]",
    "[* s:r:gc:det]",
    "[* m:0ing]",
    "[* m:base:ed]",
    "[* s:r:prep]",
    "[* m:++en:i]",
    "[* s:r:der]",
    "[* m:0's]",
    "[* m:+ed]",
    "[* m:++s:i]",
    "[* m:irr:s]",
    "[* m:+3s]",
    "[* m:++ed:i]",
    "[* m:++s]",
    "[* m:0s:a]",
    "[* m:++ed]",
    "[* m:+ing]",
    "[* m:+s:a]",
    "[* m:0ed]",
    "[* m:irr:ed]",
    "[* m:sub:en]",
    "[* m:=s]",
    "[* m:+en]",
    "[* m:base:en]",
    "[* m:base:s]",
    "[* m:sub:ed]",
    "[* m:+3s:a]",
    "[* m:=en]",
    "[* m:base:er]",
    "[* m:base:est]",
    "[* m:++er]",
    "[* m:++est]",
    "[* m:0er]",
    "[* m:0est]",
    "[/]",
    "[//]",
    "xxx",
    "(.)",
    "(..)",
    "(...)",
    "+...",
    "[!]",
    "&-",
    "&+",
]

CHAT_COMPONENT_TOKEN_CANONICAL_ORDER = [
    "[* m]",
    "[* s]",
    "[* m:++]",
    "[* m:+]",
    "[* m:=]",
    "[* m:0]",
    "[* m:allo]",
    "[* m:base]",
    "[* m:irr]",
    "[* m:sub]",
    "[* m:vsg]",
    "[* m:vun]",
    "[* s:r]",
    "[* s:r:gc]",
    "[* m:++",
    "[* m:+",
    "[* m:=",
    "[* m:0",
    "[* m:base",
    "[* m:irr",
    "[* m:sub",
    "[* m:vsg",
    "[* m:vun",
    "[* m:03s",
    "[* m:+3s",
    "[* m:+s",
    "[* m:0s",
    "[* s:r",
    "[* s:r:gc",
    ":ed]",
    ":en]",
    ":er]",
    ":est]",
    ":s]",
    ":3s]",
    ":ing]",
    ":prep]",
    ":der]",
    ":pro]",
    ":det]",
    ":a]",
    ":i]",
    ":allo]",
    "[/]",
    "[//]",
    "xxx",
    "(.)",
    "(..)",
    "(...)",
    "+...",
    "[!]",
    "&-",
    "&+",
]

# Known off-schema outliers in current data (2 rows).
DEFAULT_DROP_LABELS = [
    "[* m:+s]",
]


THIRD_PERSON_IRREGULAR = {
    "do": "does",
    "say": "says",
}

M03_AUTOFILL_EXCLUDED_LEMMAS = {
    "be",
    "am",
    "is",
    "are",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
}


IRREGULAR_PAST_MAP = {
    "be": "was",
    "begin": "began",
    "break": "broke",
    "bring": "brought",
    "buy": "bought",
    "catch": "caught",
    "come": "came",
    "do": "did",
    "drink": "drank",
    "drive": "drove",
    "eat": "ate",
    "fall": "fell",
    "feel": "felt",
    "find": "found",
    "fly": "flew",
    "forget": "forgot",
    "get": "got",
    "give": "gave",
    "go": "went",
    "grow": "grew",
    "have": "had",
    "hear": "heard",
    "hold": "held",
    "keep": "kept",
    "know": "knew",
    "leave": "left",
    "lose": "lost",
    "make": "made",
    "meet": "met",
    "pay": "paid",
    "read": "read",
    "run": "ran",
    "say": "said",
    "see": "saw",
    "sell": "sold",
    "send": "sent",
    "sit": "sat",
    "speak": "spoke",
    "stand": "stood",
    "swim": "swam",
    "take": "took",
    "teach": "taught",
    "tell": "told",
    "think": "thought",
    "write": "wrote",
}


def canonical_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()  # remove leading "[*" and trailing "]"
    return f"[* {body}]"


def extract_tags(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    tags = [canonical_tag(t) for t in TAG_PATTERN.findall(text)]
    return tags


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def source_group(record: Dict) -> str:
    value = (record.get("provenance_label") or "").strip()
    if value in {"TD", "DLD", "synthetic"}:
        return value
    return "unknown"


def primary_tag(tags: Sequence[str]) -> str:
    return tags[0] if tags else "CLEAN"


def sample_counts_by_split(rows: Sequence[Dict]) -> Dict[str, int]:
    counts = Counter()
    for row in rows:
        counts[source_group(row)] += 1
    return dict(sorted(counts.items()))


def downsample_synthetic_train(
    synth_rows: Sequence[Dict],
    target_count: int,
    seed: int,
) -> List[Dict]:
    if target_count <= 0:
        return []
    if target_count >= len(synth_rows):
        return list(synth_rows)

    rng = random.Random(seed)
    by_tag = defaultdict(list)
    for row in synth_rows:
        tag = primary_tag(row["_meta"]["tags_unique"])
        by_tag[tag].append(row)

    tags = sorted(by_tag.keys())
    total = sum(len(v) for v in by_tag.values())
    raw = {t: (len(by_tag[t]) * target_count / total) for t in tags}
    alloc = {t: int(raw[t]) for t in tags}

    # Largest-remainder to hit exact target_count.
    assigned = sum(alloc.values())
    remainders = sorted(((raw[t] - alloc[t], t) for t in tags), reverse=True)
    for _, t in remainders:
        if assigned >= target_count:
            break
        alloc[t] += 1
        assigned += 1

    sampled = []
    for t in tags:
        bucket = list(by_tag[t])
        rng.shuffle(bucket)
        sampled.extend(bucket[: alloc[t]])

    # Safety trim if any numerical edge case.
    rng.shuffle(sampled)
    return sampled[:target_count]


def stratified_real_split(
    real_rows: Sequence[Dict],
    eval_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    strata = defaultdict(list)
    for row in real_rows:
        tags = row["_meta"]["tags_unique"]
        key = f"{source_group(row)}__{primary_tag(tags)}"
        strata[key].append(row)

    train, eval_rows, test = [], [], []
    for key in sorted(strata.keys()):
        bucket = list(strata[key])
        rng.shuffle(bucket)
        n = len(bucket)
        if n == 1:
            train.extend(bucket)
            continue
        if n == 2:
            eval_rows.append(bucket[0])
            train.append(bucket[1])
            continue
        if n == 3:
            eval_rows.append(bucket[0])
            test.append(bucket[1])
            train.append(bucket[2])
            continue

        n_eval = max(1, int(round(n * eval_ratio)))
        n_test = max(1, int(round(n * test_ratio)))

        # Keep at least 1 item in train.
        if n_eval + n_test >= n:
            overflow = (n_eval + n_test) - (n - 1)
            # Reduce test first, then eval.
            take_from_test = min(overflow, n_test - 1)
            n_test -= take_from_test
            overflow -= take_from_test
            if overflow > 0:
                n_eval = max(1, n_eval - overflow)

        eval_rows.extend(bucket[:n_eval])
        test.extend(bucket[n_eval : n_eval + n_test])
        train.extend(bucket[n_eval + n_test :])

    return {"train": train, "eval": eval_rows, "test": test}


def enforce_no_input_overlap(train_rows: List[Dict], heldout_rows: Sequence[Dict]) -> List[Dict]:
    blocked = {row["input"] for row in heldout_rows}
    return [row for row in train_rows if row["input"] not in blocked]


def strip_meta(rows: Sequence[Dict]) -> List[Dict]:
    out = []
    for row in rows:
        cleaned = dict(row)
        cleaned.pop("_meta", None)
        out.append(cleaned)
    return out


def transform_stage1_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    def repl(match: re.Match) -> str:
        tag = canonical_tag(match.group(0))
        if tag.startswith("[* m:"):
            return "[* m]"
        if tag.startswith("[* s:"):
            return "[* s]"
        return tag

    return TAG_PATTERN.sub(repl, text)


def map_tag_stage2(tag: str) -> str:
    if tag.startswith("[* m:"):
        code = tag[len("[* m:") : -1]
        if code.startswith("++"):
            return "[* m:++]"
        if code.startswith("+"):
            return "[* m:+]"
        if code.startswith("="):
            return "[* m:=]"
        if code.startswith("0"):
            return "[* m:0]"
        if code.startswith("base"):
            return "[* m:base]"
        if code.startswith("irr"):
            return "[* m:irr]"
        if code.startswith("sub"):
            return "[* m:sub]"
        if code.startswith("vsg"):
            return "[* m:vsg]"
        if code.startswith("vun"):
            return "[* m:vun]"
        if code.startswith("allo"):
            return "[* m:allo]"
        return "[* m]"

    if tag.startswith("[* s:"):
        code = tag[len("[* s:") : -1]
        if code.startswith("r:gc"):
            return "[* s:r:gc]"
        if code.startswith("r"):
            return "[* s:r]"
        return "[* s]"

    return tag


def transform_stage2_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    def repl(match: re.Match) -> str:
        return map_tag_stage2(canonical_tag(match.group(0)))

    return TAG_PATTERN.sub(repl, text)


def transform_rows(rows: Sequence[Dict], stage: int) -> List[Dict]:
    out = []
    for row in rows:
        new_row = dict(row)
        if stage == 1:
            new_row["input"] = transform_stage1_text(new_row.get("input", ""))
            new_row["output"] = transform_stage1_text(new_row.get("output", ""))
        elif stage == 2:
            new_row["input"] = transform_stage2_text(new_row.get("input", ""))
            new_row["output"] = transform_stage2_text(new_row.get("output", ""))
        out.append(new_row)
    return out


def normalize_reconstruction_single_colon(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Normalize CHAT reconstruction to single-colon style for experiments 1/2.
    return re.sub(r"\[::(\s+[^\]]+)\]", r"[:\1]", text)


def drop_all_reconstructions(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Remove both [: target] and [:: target], then normalize spacing.
    text = re.sub(r"\s*\[(?::|::)\s+[^\]]+\]", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def keep_nonword_only_reconstructions(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Heuristic: treat [:: target] as explicit target hint to drop, keep [: target].
    text = re.sub(r"\s*\[::\s+[^\]]+\]", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def match_case(surface: str, target: str) -> str:
    if surface.isupper():
        return target.upper()
    if surface[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def to_third_person_singular(lemma: str) -> str:
    lower = lemma.lower()
    if lower in THIRD_PERSON_IRREGULAR:
        return match_case(lemma, THIRD_PERSON_IRREGULAR[lower])
    if len(lower) > 1 and lower.endswith("y") and lower[-2] not in "aeiou":
        return match_case(lemma, lower[:-1] + "ies")
    if lower.endswith(("s", "sh", "ch", "x", "z", "o")):
        return match_case(lemma, lower + "es")
    return match_case(lemma, lower + "s")


def to_irregular_past(lemma: str) -> str:
    lower = lemma.lower()
    if lower not in IRREGULAR_PAST_MAP:
        return ""
    return match_case(lemma, IRREGULAR_PAST_MAP[lower])


def autofill_selected_reconstructions(
    split_to_rows: Dict[str, Sequence[Dict]],
) -> Tuple[Dict[str, List[Dict]], Dict[str, int], List[Dict]]:
    stats = Counter()
    unresolved = []
    updated: Dict[str, List[Dict]] = {}

    for split_name, rows in split_to_rows.items():
        out_rows: List[Dict] = []
        for row in rows:
            new_row = dict(row)
            output_text = new_row.get("output", "")
            original_text = output_text

            def repl_m03(match: re.Match) -> str:
                lemma = match.group("lemma")
                if not lemma.isalpha():
                    stats["m03_unresolved_nonalpha"] += 1
                    unresolved.append(
                        {
                            "split": split_name,
                            "row_id": new_row.get("row_id"),
                            "label": "[* m:03s:a]",
                            "token": lemma,
                            "reason": "non_alpha_token",
                            "input": new_row.get("input", ""),
                            "output_before": original_text,
                        }
                    )
                    return match.group(0)
                if lemma.lower() in M03_AUTOFILL_EXCLUDED_LEMMAS:
                    stats["m03_unresolved_label_specific_exclusion"] += 1
                    unresolved.append(
                        {
                            "split": split_name,
                            "row_id": new_row.get("row_id"),
                            "label": "[* m:03s:a]",
                            "token": lemma,
                            "reason": "label_specific_exclusion_be_have_family",
                            "input": new_row.get("input", ""),
                            "output_before": original_text,
                        }
                    )
                    return match.group(0)
                target = to_third_person_singular(lemma)
                stats["m03_filled_occurrences"] += 1
                return f"{lemma} [:: {target}] {match.group('tag')}"

            output_text = M03_NO_RECON_PATTERN.sub(repl_m03, output_text)

            def repl_mbaseed(match: re.Match) -> str:
                lemma = match.group("lemma")
                if not lemma.isalpha():
                    stats["mbaseed_unresolved_nonalpha"] += 1
                    unresolved.append(
                        {
                            "split": split_name,
                            "row_id": new_row.get("row_id"),
                            "label": "[* m:base:ed]",
                            "token": lemma,
                            "reason": "non_alpha_token",
                            "input": new_row.get("input", ""),
                            "output_before": original_text,
                        }
                    )
                    return match.group(0)
                target = to_irregular_past(lemma)
                if not target:
                    stats["mbaseed_unresolved_missing_irregular_map"] += 1
                    unresolved.append(
                        {
                            "split": split_name,
                            "row_id": new_row.get("row_id"),
                            "label": "[* m:base:ed]",
                            "token": lemma,
                            "reason": "missing_irregular_mapping",
                            "input": new_row.get("input", ""),
                            "output_before": original_text,
                        }
                    )
                    return match.group(0)
                stats["mbaseed_filled_occurrences"] += 1
                return f"{lemma} [:: {target}] {match.group('tag')}"

            output_text = MBASEED_NO_RECON_PATTERN.sub(repl_mbaseed, output_text)

            if output_text != original_text:
                stats["rows_changed"] += 1
                new_row["output"] = output_text
            out_rows.append(new_row)

        updated[split_name] = out_rows

    return updated, dict(stats), unresolved


def collect_missing_reconstruction_rows(split_to_rows: Dict[str, Sequence[Dict]]) -> List[Dict]:
    missing = []
    for split_name, rows in split_to_rows.items():
        for row in rows:
            output_text = row.get("output", "")
            if int(row.get("error_count", 0) or 0) <= 0:
                continue
            if RECON_ANY_PATTERN.search(output_text):
                continue
            missing.append(
                {
                    "split": split_name,
                    "row_id": row.get("row_id"),
                    "provenance_label": row.get("provenance_label", ""),
                    "error_count": int(row.get("error_count", 0) or 0),
                    "input": row.get("input", ""),
                    "output": output_text,
                    "manual_output": "",
                    "notes": "",
                }
            )
    return missing


def merge_unresolved_into_manual_review(
    manual_rows: Sequence[Dict],
    unresolved_rows: Sequence[Dict],
    split_to_rows: Dict[str, Sequence[Dict]],
) -> List[Dict]:
    by_key = {(str(r.get("split")), str(r.get("row_id"))): dict(r) for r in manual_rows}
    row_lookup = {}
    for split_name, rows in split_to_rows.items():
        for row in rows:
            row_lookup[(split_name, str(row.get("row_id")))] = row

    for item in unresolved_rows:
        key = (str(item.get("split")), str(item.get("row_id")))
        if key in by_key:
            continue
        src = row_lookup.get(key, {})
        by_key[key] = {
            "split": key[0],
            "row_id": key[1],
            "provenance_label": src.get("provenance_label", ""),
            "error_count": int(src.get("error_count", 0) or 0),
            "input": src.get("input", item.get("input", "")),
            "output": src.get("output", item.get("output_before", "")),
            "manual_output": "",
            "notes": f"autofill_unresolved:{item.get('label','')}:{item.get('reason','')}",
        }
    return list(by_key.values())


def write_csv(path: Path, rows: Sequence[Dict], headers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def sanitize_zero_error_reconstruction_rows(rows: Sequence[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    out = []
    stats = Counter()
    for row in rows:
        new_row = dict(row)
        output_text = new_row.get("output", "")
        has_recon = bool(RECON_ANY_PATTERN.search(output_text))
        has_error_tag = bool(TAG_PATTERN.search(output_text))
        if has_recon and not has_error_tag and int(new_row.get("error_count", 0) or 0) == 0:
            stats["rows_sanitized"] += 1
            stats["reconstruction_markers_removed"] += len(RECON_ANY_PATTERN.findall(output_text))
            if RECON_DOUBLE_PATTERN.search(output_text):
                stats["rows_with_double_colon_removed"] += 1
            new_row["output"] = drop_all_reconstructions(output_text)
        out.append(new_row)
    return out, dict(stats)


def apply_reconstruction_mode(rows: Sequence[Dict], mode: str) -> List[Dict]:
    if mode == "preserve":
        return list(rows)
    if mode not in {"single_colon", "drop_all", "nonword_only"}:
        raise ValueError(f"Unknown reconstruction mode: {mode}")
    out = []
    for row in rows:
        new_row = dict(row)
        output_text = new_row.get("output", "")
        if mode == "single_colon":
            output_text = normalize_reconstruction_single_colon(output_text)
        elif mode == "drop_all":
            output_text = drop_all_reconstructions(output_text)
        elif mode == "nonword_only":
            output_text = keep_nonword_only_reconstructions(output_text)
        new_row["output"] = output_text
        out.append(new_row)
    return out


def write_manifest(path: Path, split_to_rows: Dict[str, Sequence[Dict]]) -> None:
    headers = [
        "row_id",
        "split",
        "source_group",
        "error_count",
        "trace_ambiguous",
        "tags",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for split_name in ["train", "eval", "test", "eval_coverage", "test_coverage", "holdout"]:
            for row in split_to_rows.get(split_name, []):
                tags = row.get("_meta", {}).get("tags_unique", [])
                writer.writerow(
                    {
                        "row_id": row.get("row_id"),
                        "split": split_name,
                        "source_group": source_group(row),
                        "error_count": row.get("error_count", 0),
                        "trace_ambiguous": row.get("trace_ambiguous", False),
                        "tags": "|".join(tags),
                    }
                )


def build_chat_tokens(records: Sequence[Dict], extra_labels: Sequence[str] = (), strategy: str = "hybrid") -> List[str]:
    observed = set()
    for row in records:
        for tag in extract_tags(row.get("output", "")):
            observed.add(tag)
    observed.update(extra_labels)

    if strategy == "hybrid":
        tokens = list(CHAT_TOKEN_CANONICAL_ORDER)
        # Preserve canonical order first, then append any previously unseen tags to stay robust.
        for tag in sorted(observed):
            if tag not in tokens:
                tokens.append(tag)
        return tokens

    if strategy == "components":
        # Exploratory mode: keep scheme scaffolding and reusable fragments, not full detailed labels.
        return list(dict.fromkeys(CHAT_COMPONENT_TOKEN_CANONICAL_ORDER))

    raise ValueError(f"Unknown chat token strategy: {strategy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze ACL-ready splits from enriched v3 JSONL.")
    parser.add_argument(
        "--input",
        default="data/processed/master_training.jsonl",
        help="Input JSONL with provenance + error_count fields.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/acl_rr_v1",
        help="Output directory for frozen split files.",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.15, help="Eval ratio for real-data split.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio for real-data split.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument(
        "--synth-eval-per-label",
        type=int,
        default=10,
        help="Synthetic eval-coverage samples per detailed label (outside heldout labels).",
    )
    parser.add_argument(
        "--synth-test-per-label",
        type=int,
        default=10,
        help="Synthetic test-coverage samples per detailed label (outside heldout labels).",
    )
    parser.add_argument(
        "--train-synthetic-ratio",
        type=float,
        default=0.5,
        help=(
            "Target synthetic proportion in train set after split and coverage extraction. "
            "Set <0 to disable downsampling and keep all synthetic."
        ),
    )
    parser.add_argument(
        "--holdout-synthetic-only",
        action="store_true",
        default=True,
        help="Keep only synthetic rows in holdout split.",
    )
    parser.add_argument(
        "--allow-real-in-holdout",
        action="store_true",
        default=False,
        help="If set, do not filter non-synthetic rows out of holdout.",
    )
    parser.add_argument(
        "--holdout-label",
        action="append",
        default=None,
        help="Optional override list for held-out generalization labels (repeat flag).",
    )
    parser.add_argument(
        "--drop-label",
        action="append",
        default=None,
        help="Optional override list for labels to drop from all outputs (repeat flag).",
    )
    parser.add_argument(
        "--reconstruction-mode",
        choices=["preserve", "single_colon", "drop_all", "nonword_only"],
        default="preserve",
        help="How to write reconstruction tokens in outputs.",
    )
    parser.add_argument(
        "--chat-token-strategy",
        choices=["hybrid", "components"],
        default="hybrid",
        help="Tokenizer augmentation strategy: current hybrid label inventory or exploratory scheme components.",
    )
    parser.add_argument(
        "--autofill-recon-03s-baseed",
        action="store_true",
        default=False,
        help=(
            "For preserve mode only: auto-add [:: target] for missing [* m:03s:a] "
            "and [* m:base:ed] corrections, then export manual review CSV for remaining no-reconstruction error rows."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = resolve_path(args.input)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout_labels = args.holdout_label if args.holdout_label else DEFAULT_HOLDOUT_LABELS
    drop_labels = args.drop_label if args.drop_label else DEFAULT_DROP_LABELS

    all_rows_raw = list(iter_jsonl(in_path))
    all_rows, zero_error_recon_stats = sanitize_zero_error_reconstruction_rows(all_rows_raw)
    for row in all_rows:
        tags = extract_tags(row.get("output", ""))
        row["_meta"] = {
            "tags_all": tags,
            "tags_unique": unique_preserve_order(tags),
            "has_holdout": any(tag in holdout_labels for tag in tags),
            "has_drop_label": any(tag in drop_labels for tag in tags),
        }

    dropped_rows = [r for r in all_rows if r["_meta"]["has_drop_label"]]
    filtered_rows = [r for r in all_rows if not r["_meta"]["has_drop_label"]]

    holdout_rows = [r for r in filtered_rows if r["_meta"]["has_holdout"]]
    main_rows = [r for r in filtered_rows if not r["_meta"]["has_holdout"]]

    real_rows = [r for r in main_rows if source_group(r) in {"TD", "DLD"}]
    synth_rows = [r for r in main_rows if source_group(r) == "synthetic"]

    split_real = stratified_real_split(
        real_rows=real_rows,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    synth_by_tag = defaultdict(list)
    for row in synth_rows:
        tag = primary_tag(row["_meta"]["tags_unique"])
        synth_by_tag[tag].append(row)

    rng = random.Random(args.seed)
    synth_eval_coverage = []
    synth_test_coverage = []
    synth_train = []
    for tag, rows in sorted(synth_by_tag.items()):
        bucket = list(rows)
        rng.shuffle(bucket)
        n = len(bucket)
        target_eval = max(0, args.synth_eval_per_label)
        target_test = max(0, args.synth_test_per_label)

        # Keep at least one example in synthetic train when possible.
        if n <= 2:
            synth_train.extend(bucket)
            continue

        if target_eval + target_test + 1 <= n:
            n_eval_cov = target_eval
            n_test_cov = target_test
        else:
            # Backoff for small buckets, still preserving train support.
            max_for_cov = n - 1
            n_eval_cov = min(target_eval, max_for_cov // 2)
            n_test_cov = min(target_test, max_for_cov - n_eval_cov)
            if n_eval_cov == 0 and max_for_cov >= 1:
                n_eval_cov = 1
                n_test_cov = max(0, max_for_cov - 1)
            if n_test_cov == 0 and max_for_cov >= 2:
                n_test_cov = 1
                n_eval_cov = max(1, max_for_cov - 1)

        synth_eval_coverage.extend(bucket[:n_eval_cov])
        synth_test_coverage.extend(bucket[n_eval_cov : n_eval_cov + n_test_cov])
        synth_train.extend(bucket[n_eval_cov + n_test_cov :])

    real_train_rows = list(split_real["train"])
    if args.train_synthetic_ratio is not None and args.train_synthetic_ratio >= 0:
        ratio = args.train_synthetic_ratio
        if not (0 <= ratio < 1):
            raise ValueError("--train-synthetic-ratio must be in [0, 1).")
        target_synth = int(round(len(real_train_rows) * ratio / max(1e-12, (1 - ratio))))
        synth_train = downsample_synthetic_train(
            synth_rows=synth_train,
            target_count=target_synth,
            seed=args.seed,
        )
    train_rows = real_train_rows + synth_train
    eval_rows = list(split_real["eval"])
    test_rows = list(split_real["test"])
    eval_coverage_rows = list(synth_eval_coverage)
    test_coverage_rows = list(synth_test_coverage)

    train_rows = enforce_no_input_overlap(
        train_rows,
        eval_rows + test_rows + eval_coverage_rows + test_coverage_rows + holdout_rows,
    )

    # Exclude ambiguous provenance from non-train splits for cleaner reporting.
    eval_rows = [r for r in eval_rows if not r.get("trace_ambiguous", False)]
    test_rows = [r for r in test_rows if not r.get("trace_ambiguous", False)]
    eval_coverage_rows = [r for r in eval_coverage_rows if not r.get("trace_ambiguous", False)]
    test_coverage_rows = [r for r in test_coverage_rows if not r.get("trace_ambiguous", False)]
    holdout_rows = [r for r in holdout_rows if not r.get("trace_ambiguous", False)]
    if args.holdout_synthetic_only and not args.allow_real_in_holdout:
        holdout_rows = [r for r in holdout_rows if source_group(r) == "synthetic"]

    # Stage 3 is detailed labels (no transform)
    split_to_stage3 = {
        "train": strip_meta(train_rows),
        "eval": strip_meta(eval_rows),
        "test": strip_meta(test_rows),
        "eval_coverage": strip_meta(eval_coverage_rows),
        "test_coverage": strip_meta(test_coverage_rows),
        "holdout": strip_meta(holdout_rows),
    }
    split_to_stage3 = {k: apply_reconstruction_mode(v, args.reconstruction_mode) for k, v in split_to_stage3.items()}

    autofill_stats: Dict[str, int] = {}
    autofill_unresolved_rows: List[Dict] = []
    if args.autofill_recon_03s_baseed and args.reconstruction_mode == "preserve":
        split_to_stage3, autofill_stats, autofill_unresolved_rows = autofill_selected_reconstructions(split_to_stage3)

    manual_review_rows = collect_missing_reconstruction_rows(split_to_stage3)
    manual_review_rows = merge_unresolved_into_manual_review(
        manual_review_rows,
        autofill_unresolved_rows,
        split_to_stage3,
    )
    manual_review_path = out_dir / "manual_reconstruction_review_stage3.csv"
    write_csv(
        manual_review_path,
        manual_review_rows,
        headers=[
            "split",
            "row_id",
            "provenance_label",
            "error_count",
            "input",
            "output",
            "manual_output",
            "notes",
        ],
    )
    unresolved_path = out_dir / "manual_reconstruction_unresolved_autofill.csv"
    write_csv(
        unresolved_path,
        autofill_unresolved_rows,
        headers=[
            "split",
            "row_id",
            "label",
            "token",
            "reason",
            "input",
            "output_before",
        ],
    )

    # Stage 2 and Stage 1 transforms use exact same row membership.
    split_to_stage2 = {k: transform_rows(v, stage=2) for k, v in split_to_stage3.items()}
    split_to_stage1 = {k: transform_rows(v, stage=1) for k, v in split_to_stage3.items()}

    for stage, split_rows in [(1, split_to_stage1), (2, split_to_stage2), (3, split_to_stage3)]:
        for split_name, rows in split_rows.items():
            out_path = out_dir / f"stage{stage}_{split_name}.jsonl"
            write_jsonl(out_path, rows)

    # Manifest + summary + tokenizer token list
    write_manifest(
        out_dir / "split_manifest.csv",
        {
            "train": train_rows,
            "eval": eval_rows,
            "test": test_rows,
            "eval_coverage": eval_coverage_rows,
            "test_coverage": test_coverage_rows,
            "holdout": holdout_rows,
        },
    )

    # Keep withheld labels tokenizable under the default hybrid strategy; the components strategy is exploratory.
    chat_tokens = build_chat_tokens(
        split_to_stage3["train"],
        extra_labels=holdout_labels,
        strategy=args.chat_token_strategy,
    )
    (out_dir / "chat_tokens.json").write_text(json.dumps(chat_tokens, ensure_ascii=False, indent=2) + "\n")

    summary = {
        "input_file": str(in_path),
        "output_dir": str(out_dir),
        "seed": args.seed,
        "holdout_labels": holdout_labels,
        "dropped_labels": drop_labels,
        "rows_total_input": len(all_rows),
        "rows_dropped_off_schema": len(dropped_rows),
        "rows_holdout": len(holdout_rows),
        "rows_main_after_filters": len(train_rows) + len(eval_rows) + len(test_rows),
        "zero_error_reconstruction_sanitation": zero_error_recon_stats,
        "split_sizes_stage3": {
            "train": len(split_to_stage3["train"]),
            "eval": len(split_to_stage3["eval"]),
            "test": len(split_to_stage3["test"]),
            "eval_coverage": len(split_to_stage3["eval_coverage"]),
            "test_coverage": len(split_to_stage3["test_coverage"]),
            "holdout": len(split_to_stage3["holdout"]),
        },
        "source_by_split_stage3": {
            "train": sample_counts_by_split(split_to_stage3["train"]),
            "eval": sample_counts_by_split(split_to_stage3["eval"]),
            "test": sample_counts_by_split(split_to_stage3["test"]),
            "eval_coverage": sample_counts_by_split(split_to_stage3["eval_coverage"]),
            "test_coverage": sample_counts_by_split(split_to_stage3["test_coverage"]),
            "holdout": sample_counts_by_split(split_to_stage3["holdout"]),
        },
        "notes": [
            "Eval and test are real-data only (TD/DLD).",
            "Eval_coverage and test_coverage are synthetic-only label-coverage splits.",
            "Train mixes real-train with synthetic for data augmentation.",
            "Holdout set is excluded from training and intended for generalization diagnostics.",
            "Ambiguous-provenance rows are excluded from non-train splits.",
            "No input overlap is allowed between train and all non-train splits.",
            "Rows with reconstruction markers but no in-scope error tags are sanitized to unchanged outputs.",
            "The project extension label [* s:r:gc:det] is retained (det, not art).",
        ],
        "train_mixture": {
            "train_synthetic_ratio_target": args.train_synthetic_ratio,
            "train_real_rows": len(real_train_rows),
            "train_synthetic_rows": len(synth_train),
            "train_synthetic_ratio_actual": round(
                (len(synth_train) / max(1, len(train_rows))), 6
            ),
        },
        "reconstruction_mode": args.reconstruction_mode,
        "autofill_recon_03s_baseed": args.autofill_recon_03s_baseed,
        "autofill_recon_03s_baseed_stats": autofill_stats,
        "manual_reconstruction_review_stage3_file": str(manual_review_path),
        "manual_reconstruction_review_stage3_rows": len(manual_review_rows),
        "manual_reconstruction_unresolved_autofill_file": str(unresolved_path),
        "manual_reconstruction_unresolved_autofill_rows": len(autofill_unresolved_rows),
        "chat_token_strategy": args.chat_token_strategy,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print(f"Wrote frozen ACL split package to: {out_dir}")
    print(json.dumps(summary["split_sizes_stage3"], indent=2))


if __name__ == "__main__":
    main()
