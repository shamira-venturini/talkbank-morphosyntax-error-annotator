import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from common import resolve_path


TAG_RE = re.compile(r"\[\*\s*[ms](?::[^\]]+)?\]")


def canonical_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    return f"[* {body}]"


def collect_labels(path: Path) -> List[str]:
    labels = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out = row.get("output", "")
            for tag in TAG_RE.findall(out):
                labels.add(canonical_tag(tag))
    return sorted(labels)


def reconstruction_rule_text(mode: str) -> str:
    if mode == "preserve":
        return (
            "7. When reconstruction is needed, use CHAT reconstruction markers according to the manual: "
            "[: target] or [:: target], preserving their intended distinction.\n"
        )
    if mode == "drop_all":
        return "7. Do not output reconstruction markers (no [: target] and no [:: target]).\n"
    if mode == "nonword_only":
        return (
            "7. Use reconstruction markers only for nonword corrections with [: target]; "
            "do not use [:: target].\n"
        )
    if mode == "single_colon":
        return "7. When reconstruction is needed, use only [: target] (never [:: target]).\n"
    raise ValueError(f"Unknown reconstruction instruction mode: {mode}")


def build_prompt(allowed_labels: List[str], reconstruction_mode: str) -> str:
    return (
        "You are a TalkBank CHAT annotator for morphosyntactic error coding.\n\n"
        "Task:\n"
        "Annotate the input utterance by inserting valid CHAT error tags inline.\n\n"
        "Rules:\n"
        "1. Preserve original token order, spelling, casing, punctuation, disfluencies, and CHAT symbols.\n"
        "2. Do NOT rewrite, paraphrase, or correct the utterance.\n"
        "3. Insert only error tags (and reconstruction tokens when required by CHAT).\n"
        "4. If no target error is present, return the utterance unchanged.\n"
        f"{reconstruction_rule_text(reconstruction_mode).replace('7.', '5.', 1)}"
        "6. Output exactly one annotated utterance line and nothing else."
    )


def update_file(path: Path, prompt: str) -> int:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["instruction"] = prompt
            rows.append(row)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set strict experiment prompts on frozen split JSONL files.")
    parser.add_argument(
        "--experiment-dir",
        default="experiments/acl_rr_v1",
        help="Directory containing stage{1,2,3}_*.jsonl files.",
    )
    parser.add_argument(
        "--label-source-splits",
        default="train",
        help=(
            "Comma-separated split names used to derive allowed labels for the prompt. "
            "Defaults to train to avoid holdout leakage."
        ),
    )
    parser.add_argument(
        "--holdout-label-source-splits",
        default=None,
        help=(
            "Optional comma-separated split names used only for holdout prompt labels. "
            "If omitted, uses --label-source-splits."
        ),
    )
    parser.add_argument(
        "--reconstruction-mode",
        choices=["preserve", "single_colon", "drop_all", "nonword_only"],
        default="preserve",
        help="Instruction rule for reconstruction markers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = resolve_path(args.experiment_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(exp_dir)
    label_splits = {x.strip() for x in args.label_source_splits.split(",") if x.strip()}
    if not label_splits:
        raise ValueError("--label-source-splits produced an empty set.")
    holdout_label_splits = (
        {x.strip() for x in args.holdout_label_source_splits.split(",") if x.strip()}
        if args.holdout_label_source_splits
        else set(label_splits)
    )
    if not holdout_label_splits:
        raise ValueError("--holdout-label-source-splits produced an empty set.")

    stage_files: Dict[int, List[Path]] = {1: [], 2: [], 3: []}
    for path in sorted(exp_dir.glob("stage*_*.jsonl")):
        name = path.name
        if name.startswith("stage1_"):
            stage_files[1].append(path)
        elif name.startswith("stage2_"):
            stage_files[2].append(path)
        elif name.startswith("stage3_"):
            stage_files[3].append(path)

    prompts = {}
    holdout_prompts = {}
    for stage in [1, 2, 3]:
        label_set = set()
        holdout_label_set = set()
        for file_path in stage_files[stage]:
            split_name = file_path.stem.split("_", 1)[1]
            for label in collect_labels(file_path):
                if label == "[* m:+s]":
                    # Excluded off-schema label.
                    continue
                if split_name in label_splits:
                    label_set.add(label)
                if split_name in holdout_label_splits:
                    holdout_label_set.add(label)
        if not label_set:
            raise ValueError(f"No labels found for stage {stage} with --label-source-splits={sorted(label_splits)}")
        if not holdout_label_set:
            raise ValueError(
                f"No labels found for stage {stage} with --holdout-label-source-splits={sorted(holdout_label_splits)}"
            )
        prompts[stage] = build_prompt(sorted(label_set), args.reconstruction_mode)
        holdout_prompts[stage] = build_prompt(sorted(holdout_label_set), args.reconstruction_mode)

    counts = {}
    for stage in [1, 2, 3]:
        for file_path in stage_files[stage]:
            split_name = file_path.stem.split("_", 1)[1]
            prompt = holdout_prompts[stage] if split_name == "holdout" else prompts[stage]
            n = update_file(file_path, prompt)
            counts[str(file_path)] = n

    summary = {
        "experiment_dir": str(exp_dir),
        "stage_file_counts": {str(k): len(v) for k, v in stage_files.items()},
        "label_source_splits": sorted(label_splits),
        "holdout_label_source_splits": sorted(holdout_label_splits),
        "reconstruction_mode": args.reconstruction_mode,
        "rows_updated_by_file": counts,
        "prompt_preview_stage1": prompts[1][:300],
        "prompt_preview_stage2": prompts[2][:300],
        "prompt_preview_stage3": prompts[3][:300],
        "holdout_prompt_preview_stage1": holdout_prompts[1][:300],
        "holdout_prompt_preview_stage2": holdout_prompts[2][:300],
        "holdout_prompt_preview_stage3": holdout_prompts[3][:300],
    }
    out = exp_dir / "prompt_update_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated prompts in {sum(len(v) for v in stage_files.values())} files.")
    print(f"Summary: {out}")


if __name__ == "__main__":
    main()
