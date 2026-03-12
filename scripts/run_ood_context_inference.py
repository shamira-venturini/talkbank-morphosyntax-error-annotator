from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from common import resolve_path


DEFAULT_BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
DEFAULT_ADAPTER = "mash-mash/talkbank-morphosyntax-annotator-final-recon_full_comp_preserve_final_seed3407"
DEFAULT_INPUT_JSONL = "data/processed/ood_vercellotti/vercellotti_utterances.jsonl"
DEFAULT_CHAT_TOKENS = "experiments/recon_full_comp_preserve/chat_tokens.json"
DEFAULT_STAGE3_SPLIT = "experiments/recon_full_comp_preserve/stage3_train.jsonl"
DEFAULT_OUT_DIR = "results/ood_vercellotti"
SYSTEM_PROMPT_WRAPPER = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
TAG_RE = re.compile(r"\[\*\s*[ms](?::[^\]]+)?\]")
RECON_RE = re.compile(r"\[(::?)\s+[^\]]+\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OOD CHAT annotation with configurable context windows.")
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL, help="Prepared OOD utterance JSONL.")
    parser.add_argument(
        "--context-mode",
        choices=["utterance_only", "local_prev", "full_prev", "full_document"],
        default="utterance_only",
        help="Context strategy for inference.",
    )
    parser.add_argument("--local-prev-k", type=int, default=2, help="Previous utterances in local_prev mode.")
    parser.add_argument(
        "--context-scope",
        choices=["same_speaker", "file_selected"],
        default="same_speaker",
        help="Context pool in local/full modes.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=4000,
        help="Hard cap for serialized context text length.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model repo id.")
    parser.add_argument("--adapter-repo", default=DEFAULT_ADAPTER, help="LoRA adapter HF repo id.")
    parser.add_argument("--chat-tokens", default=DEFAULT_CHAT_TOKENS, help="JSON file with extra CHAT tokens.")
    parser.add_argument(
        "--stage3-split",
        default=DEFAULT_STAGE3_SPLIT,
        help="Stage3 split JSONL to read the canonical instruction from.",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=8, help="Generation batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="max_new_tokens for generation.")
    parser.add_argument("--max-seq-length", type=int, default=384, help="Tokenizer truncation max length.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on utterances for quick dry runs.")
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="Env var name for HF token.")
    parser.add_argument(
        "--save-token-level-uncertainty",
        action="store_true",
        default=False,
        help="Save per-token logprobs and margins (large output).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def canonical_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    return f"[* {body}]"


def extract_tag_set(text: str) -> set[str]:
    if not isinstance(text, str):
        return set()
    return {canonical_tag(tag) for tag in TAG_RE.findall(text)}


def marker_signature(text: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        return tuple()
    return tuple(RECON_RE.findall(text))


def load_instruction(stage3_split_path: Path) -> str:
    with stage3_split_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        raise RuntimeError(f"No rows found in {stage3_split_path}")
    row = json.loads(first_line)
    instruction = row.get("instruction", "").strip()
    if not instruction:
        raise RuntimeError(f"Missing instruction in first row of {stage3_split_path}")
    return instruction


def clip_context(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[-max_chars:]


def build_context_maps(rows: List[Dict], scope: str) -> Dict[int, Dict]:
    by_file: Dict[str, List[Dict]] = {}
    for row in rows:
        by_file.setdefault(row["file_name"], []).append(row)

    context_map: Dict[int, Dict] = {}
    for _, bucket in by_file.items():
        bucket_sorted = sorted(bucket, key=lambda r: int(r.get("utterance_index_raw", 0)))
        for idx, row in enumerate(bucket_sorted):
            if scope == "same_speaker":
                prev = [x for x in bucket_sorted[:idx] if x["speaker"] == row["speaker"]]
                pool = [x for x in bucket_sorted if x["speaker"] == row["speaker"]]
            else:
                prev = bucket_sorted[:idx]
                pool = bucket_sorted
            context_map[int(row["row_id"])] = {
                "prev_rows": prev,
                "pool_rows": pool,
            }
    return context_map


def serialize_rows(rows: Iterable[Dict], target_row_id: Optional[int] = None) -> str:
    parts: List[str] = []
    for row in rows:
        rid = int(row["row_id"])
        prefix = f"[{row['speaker']}#{row.get('utterance_index_raw', 0)}]"
        text = row["input"]
        if target_row_id is not None and rid == target_row_id:
            parts.append(f"{prefix} <TARGET> {text}")
        else:
            parts.append(f"{prefix} {text}")
    return "\n".join(parts)


def build_augmented_input(
    row: Dict,
    context_mode: str,
    context_map: Dict[int, Dict],
    local_prev_k: int,
    max_context_chars: int,
) -> Dict:
    row_id = int(row["row_id"])
    info = context_map[row_id]
    prev_rows = info["prev_rows"]
    pool_rows = info["pool_rows"]

    if context_mode == "utterance_only":
        context_text = ""
        context_count = 0
    elif context_mode == "local_prev":
        ctx_rows = prev_rows[-local_prev_k:] if local_prev_k > 0 else []
        context_text = serialize_rows(ctx_rows)
        context_count = len(ctx_rows)
    elif context_mode == "full_prev":
        context_text = serialize_rows(prev_rows)
        context_count = len(prev_rows)
    elif context_mode == "full_document":
        context_text = serialize_rows(pool_rows, target_row_id=row_id)
        context_count = len(pool_rows)
    else:
        raise ValueError(f"Unknown context_mode: {context_mode}")

    context_text = clip_context(context_text, max_context_chars=max_context_chars)

    if context_mode == "utterance_only":
        input_text = row["input"]
    else:
        input_text = (
            "Context (for disambiguation only, do NOT annotate context lines):\n"
            f"{context_text}\n\n"
            "Target utterance to annotate:\n"
            f"{row['input']}"
        )

    return {
        "input_text": input_text,
        "context_text": context_text,
        "context_utterance_count": context_count,
        "context_char_count": len(context_text),
    }


def prepare_model_and_tokenizer(
    base_model: str,
    adapter_repo: str,
    chat_tokens_path: Path,
    hf_token: Optional[str],
) -> Tuple[object, object]:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing inference dependency. Install required packages first, e.g. "
            "`pip install transformers peft bitsandbytes accelerate`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token, use_fast=True)
    model_kwargs = {
        "token": hf_token,
        "device_map": "auto",
    }
    try:
        import torch

        compute_dtype = torch.float16
        if torch.cuda.is_available():
            major_cc, _ = torch.cuda.get_device_capability()
            if major_cc >= 8:
                compute_dtype = torch.bfloat16

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except TypeError:
        # Compatibility fallback for older transformers that still expect load_in_4bit.
        model_kwargs.pop("quantization_config", None)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs,
            load_in_4bit=True,
        )

    extra_tokens = json.loads(chat_tokens_path.read_text(encoding="utf-8"))
    extra_tokens = list(dict.fromkeys(extra_tokens))
    if extra_tokens:
        added = tokenizer.add_tokens(extra_tokens, special_tokens=False)
        if added > 0:
            model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, adapter_repo, token=hf_token)
    model.eval()
    return tokenizer, model


def decode_first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def summarize_generation_uncertainty(
    sequences,
    prompt_lengths: List[int],
    scores,
    tokenizer,
    save_token_level: bool,
) -> List[Dict]:
    import torch

    batch_size = sequences.shape[0]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    out: List[Dict] = []

    for i in range(batch_size):
        start = prompt_lengths[i]
        gen_tokens = sequences[i, start:].tolist()
        trimmed = []
        for tok in gen_tokens:
            if pad_id is not None and tok == pad_id:
                break
            if eos_id is not None and tok == eos_id:
                break
            trimmed.append(tok)

        token_logprobs: List[float] = []
        token_margins: List[float] = []
        for step, tok in enumerate(trimmed):
            if step >= len(scores):
                break
            step_logits = scores[step][i]
            step_logprobs = torch.log_softmax(step_logits, dim=-1)
            token_logprobs.append(float(step_logprobs[tok].item()))
            top2 = torch.topk(step_logits, k=2, dim=-1).values
            token_margins.append(float((top2[0] - top2[1]).item()))

        if token_logprobs:
            payload = {
                "uncertainty_num_tokens": len(token_logprobs),
                "uncertainty_seq_logprob": float(sum(token_logprobs)),
                "uncertainty_mean_token_logprob": float(sum(token_logprobs) / len(token_logprobs)),
                "uncertainty_min_token_logprob": float(min(token_logprobs)),
                "uncertainty_mean_token_margin": float(sum(token_margins) / len(token_margins)),
                "uncertainty_min_token_margin": float(min(token_margins)),
            }
        else:
            payload = {
                "uncertainty_num_tokens": 0,
                "uncertainty_seq_logprob": None,
                "uncertainty_mean_token_logprob": None,
                "uncertainty_min_token_logprob": None,
                "uncertainty_mean_token_margin": None,
                "uncertainty_min_token_margin": None,
            }

        if save_token_level:
            payload["uncertainty_generated_tokens"] = tokenizer.convert_ids_to_tokens(trimmed)
            payload["uncertainty_token_logprobs"] = token_logprobs
            payload["uncertainty_token_margins"] = token_margins

        out.append(payload)
    return out


def batched(seq: List[Dict], n: int) -> Iterable[List[Dict]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input_jsonl)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_token = __import__("os").environ.get(args.hf_token_env)

    rows = load_jsonl(input_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No rows in input JSONL: {input_path}")

    instruction = load_instruction(resolve_path(args.stage3_split))
    tokenizer, model = prepare_model_and_tokenizer(
        base_model=args.base_model,
        adapter_repo=args.adapter_repo,
        chat_tokens_path=resolve_path(args.chat_tokens),
        hf_token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    context_map = build_context_maps(rows, scope=args.context_scope)

    output_rows: List[Dict] = []
    for batch_rows in batched(rows, max(1, args.batch_size)):
        prepared = [
            build_augmented_input(
                row=row,
                context_mode=args.context_mode,
                context_map=context_map,
                local_prev_k=args.local_prev_k,
                max_context_chars=args.max_context_chars,
            )
            for row in batch_rows
        ]
        prompts = [
            SYSTEM_PROMPT_WRAPPER.format(
                instruction=instruction,
                input=prep["input_text"],
            )
            for prep in prepared
        ]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lengths = enc["attention_mask"].sum(dim=1).tolist()

        import torch

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        sequences = gen.sequences
        uncertainty = summarize_generation_uncertainty(
            sequences=sequences,
            prompt_lengths=prompt_lengths,
            scores=gen.scores,
            tokenizer=tokenizer,
            save_token_level=args.save_token_level_uncertainty,
        )

        for j, row in enumerate(batch_rows):
            start = prompt_lengths[j]
            gen_ids = sequences[j, start:]
            raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            first_line = decode_first_nonempty_line(raw_text)
            out_row = {
                "row_id": row["row_id"],
                "file_name": row["file_name"],
                "file_path": row["file_path"],
                "speaker": row["speaker"],
                "line_no": row["line_no"],
                "utterance_index_raw": row["utterance_index_raw"],
                "context_mode": args.context_mode,
                "context_scope": args.context_scope,
                "context_utterance_count": prepared[j]["context_utterance_count"],
                "context_char_count": prepared[j]["context_char_count"],
                "instruction": instruction,
                "input": row["input"],
                "model_prediction": first_line,
                "model_prediction_raw": raw_text,
                "pred_tag_count": len(extract_tag_set(first_line)),
                "pred_reconstruction_marker_count": len(marker_signature(first_line)),
                "adapter_repo": args.adapter_repo,
                "base_model": args.base_model,
            }
            out_row.update(uncertainty[j])
            output_rows.append(out_row)

    pred_path = out_dir / f"predictions_{args.context_mode}.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(input_path),
        "context_mode": args.context_mode,
        "context_scope": args.context_scope,
        "rows": len(output_rows),
        "prediction_file": str(pred_path),
        "base_model": args.base_model,
        "adapter_repo": args.adapter_repo,
        "local_prev_k": args.local_prev_k,
        "max_context_chars": args.max_context_chars,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    (out_dir / f"summary_{args.context_mode}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
