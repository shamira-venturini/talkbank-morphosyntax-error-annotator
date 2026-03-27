from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List

from common import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command OOD Vercellotti evaluation runner.")
    parser.add_argument("--corpus-dir", default="data/OOD_data/Vercellotti")
    parser.add_argument("--prepared-dir", default="data/processed/ood_vercellotti")
    parser.add_argument("--results-dir", default="results/ood_vercellotti")
    parser.add_argument("--speaker-policy", choices=["dominant", "first_participant", "all"], default="dominant")
    parser.add_argument("--min-word-count", type=int, default=1)
    parser.add_argument("--local-prev-k", type=int, default=2)
    parser.add_argument("--context-scope", choices=["same_speaker", "file_selected"], default="same_speaker")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--full-document-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-prepare", action="store_true", default=False)
    parser.add_argument("--skip-inference", action="store_true", default=False)
    parser.add_argument("--skip-analysis", action="store_true", default=False)
    parser.add_argument("--base-model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    parser.add_argument(
        "--adapter-repo",
        default="mash-mash/talkbank-morphosyntax-annotator-final-recon_full_comp_preserve_final_seed3407",
    )
    parser.add_argument("--chat-tokens", default="experiments/recon_full_comp_preserve/chat_tokens.json")
    parser.add_argument("--stage3-split", default="experiments/recon_full_comp_preserve/stage3_train.jsonl")
    return parser.parse_args()


def run(cmd: List[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    prepared_dir = resolve_path(args.prepared_dir)
    results_dir = resolve_path(args.results_dir)
    prepared_jsonl = prepared_dir / "vercellotti_utterances.jsonl"

    if not args.skip_prepare:
        run(
            [
                "python3",
                "scripts/prepare_ood_vercellotti_dataset.py",
                "--corpus-dir",
                str(resolve_path(args.corpus_dir)),
                "--out-dir",
                str(prepared_dir),
                "--speaker-policy",
                args.speaker_policy,
                "--min-word-count",
                str(args.min_word_count),
            ]
        )

    if not prepared_jsonl.exists():
        raise SystemExit(f"Missing prepared input JSONL: {prepared_jsonl}")

    if not args.skip_inference:
        modes = [
            ("utterance_only", args.batch_size),
            ("local_prev", args.batch_size),
            ("full_prev", args.batch_size),
            ("full_document", args.full_document_batch_size),
        ]
        for mode, batch_size in modes:
            cmd = [
                "python3",
                "scripts/run_ood_context_inference.py",
                "--input-jsonl",
                str(prepared_jsonl),
                "--out-dir",
                str(results_dir),
                "--context-mode",
                mode,
                "--context-scope",
                args.context_scope,
                "--local-prev-k",
                str(args.local_prev_k),
                "--batch-size",
                str(batch_size),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--max-seq-length",
                str(args.max_seq_length),
                "--max-context-chars",
                str(args.max_context_chars),
                "--base-model",
                args.base_model,
                "--adapter-repo",
                args.adapter_repo,
                "--chat-tokens",
                args.chat_tokens,
                "--stage3-split",
                args.stage3_split,
            ]
            if args.limit > 0:
                cmd.extend(["--limit", str(args.limit)])
            run(cmd)

    if not args.skip_analysis:
        run(
            [
                "python3",
                "scripts/analyze_ood_context_modes.py",
                "--predictions-dir",
                str(results_dir),
                "--baseline-mode",
                "utterance_only",
                "--out-dir",
                str(results_dir / "context_analysis"),
            ]
        )

    print(f"OOD Vercellotti pipeline complete. Outputs under: {results_dir}")


if __name__ == "__main__":
    main()
