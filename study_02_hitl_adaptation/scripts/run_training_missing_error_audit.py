from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import resolve_path


DEFAULT_AUDIT_ROOT = (
    "studies/02_uncertainty_and_feedback/audits/training_missing_error_audit_real_only_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible final-model audit for possible missed errors in the frozen training split."
    )
    parser.add_argument(
        "--audit-root",
        default=DEFAULT_AUDIT_ROOT,
        help="Directory holding the built input, inference outputs, and analysis outputs.",
    )
    parser.add_argument(
        "--train-split",
        default="experiments/recon_full_comp_preserve/stage3_train.jsonl",
        help="Frozen stage3_train JSONL.",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        default=False,
        help="Include synthetic rows in the audit.",
    )
    parser.add_argument(
        "--include-ambiguous",
        action="store_true",
        default=False,
        help="Include trace_ambiguous rows in the audit.",
    )
    parser.add_argument(
        "--zero-error-only",
        action="store_true",
        default=False,
        help="Audit only gold-clean rows.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
    parser.add_argument("--limit", type=int, default=0, help="Optional inference row cap for quick dry runs.")
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="HF token env var to forward to inference.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    audit_root = resolve_path(args.audit_root)
    input_jsonl = audit_root / "input_utterance_only.jsonl"
    inference_dir = audit_root / "inference"
    analysis_dir = audit_root / "analysis"

    build_cmd = [
        sys.executable,
        "scripts/build_training_missing_error_audit_input.py",
        "--train-split",
        args.train_split,
        "--out-jsonl",
        str(input_jsonl),
    ]
    if args.include_synthetic:
        build_cmd.append("--include-synthetic")
    if args.include_ambiguous:
        build_cmd.append("--include-ambiguous")
    if args.zero_error_only:
        build_cmd.append("--zero-error-only")
    run(build_cmd)

    inference_cmd = [
        sys.executable,
        "scripts/run_ood_context_inference.py",
        "--input-jsonl",
        str(input_jsonl),
        "--context-mode",
        "utterance_only",
        "--context-scope",
        "same_speaker",
        "--stage3-split",
        args.train_split,
        "--chat-tokens",
        "experiments/recon_full_comp_preserve/chat_tokens.json",
        "--out-dir",
        str(inference_dir),
        "--batch-size",
        str(args.batch_size),
        "--hf-token-env",
        args.hf_token_env,
    ]
    if args.limit > 0:
        inference_cmd.extend(["--limit", str(args.limit)])
    run(inference_cmd)

    analysis_cmd = [
        sys.executable,
        "scripts/analyze_training_missing_error_audit.py",
        "--train-split",
        args.train_split,
        "--predictions",
        str(inference_dir / "predictions_utterance_only.jsonl"),
        "--out-dir",
        str(analysis_dir),
    ]
    if args.include_synthetic:
        analysis_cmd.append("--include-synthetic")
    if args.include_ambiguous:
        analysis_cmd.append("--include-ambiguous")
    run(analysis_cmd)


if __name__ == "__main__":
    main()
