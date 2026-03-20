from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from common import resolve_path

DEFAULT_BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
DEFAULT_ADAPTER_REPO = "mash-mash/talkbank-morphosyntax-annotator-final-recon_full_comp_preserve_final_seed3407"
DEFAULT_CHAT_TOKENS = "experiments/recon_full_comp_preserve/chat_tokens.json"
DEFAULT_OUT_DIR = "artifacts/merged_llama_talktag_chat_error_annotator"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a standalone model checkpoint.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model repo id.")
    parser.add_argument("--adapter-repo", default=DEFAULT_ADAPTER_REPO, help="Adapter repo id.")
    parser.add_argument("--chat-tokens", default=DEFAULT_CHAT_TOKENS, help="JSON file with CHAT tokens.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to save the merged model.")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Torch dtype for loading the base model before merge.",
    )
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable name holding the Hugging Face token.",
    )
    parser.add_argument(
        "--hub-repo-id",
        default="",
        help="Optional Hugging Face repo id to push the merged model after saving locally.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create the Hub repo as private when --hub-repo-id is provided.",
    )
    parser.add_argument(
        "--safe-merge",
        action="store_true",
        default=False,
        help="Enable PEFT safe_merge during merge_and_unload().",
    )
    return parser.parse_args()


def load_token(env_name: str) -> Optional[str]:
    token = os.environ.get(env_name, "").strip()
    return token or None


def resolve_dtype(name: str):
    import torch

    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available():
        major_cc, _ = torch.cuda.get_device_capability()
        if major_cc >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def main() -> None:
    args = parse_args()
    hf_token = load_token(args.hf_token_env)
    chat_tokens_path = resolve_path(args.chat_tokens)
    out_dir = resolve_path(args.out_dir)

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency. Install required packages first, e.g. "
            "`pip install transformers peft accelerate sentencepiece`."
        ) from exc

    def load_with_auth(loader, model_id: str, **kwargs):
        if hf_token:
            try:
                return loader.from_pretrained(model_id, token=hf_token, **kwargs)
            except TypeError:
                return loader.from_pretrained(model_id, use_auth_token=hf_token, **kwargs)
        return loader.from_pretrained(model_id, **kwargs)

    if not chat_tokens_path.exists():
        raise SystemExit(f"Missing chat tokens file: {chat_tokens_path}")

    print(f"Loading tokenizer from: {args.base_model}")
    tokenizer = load_with_auth(AutoTokenizer, args.base_model, use_fast=True)

    torch_dtype = resolve_dtype(args.torch_dtype)
    print(f"Loading base model from: {args.base_model}")
    print(f"Using dtype: {torch_dtype}")
    model = load_with_auth(
        AutoModelForCausalLM,
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
    )

    chat_tokens = json.loads(chat_tokens_path.read_text(encoding="utf-8"))
    chat_tokens = list(dict.fromkeys(chat_tokens))
    added = tokenizer.add_tokens(chat_tokens, special_tokens=False)
    if added > 0:
        print(f"Added {added} CHAT tokens; resizing embeddings.")
        model.resize_token_embeddings(len(tokenizer))
    else:
        print("No new CHAT tokens were added; tokenizer already contains them.")

    print(f"Loading adapter from: {args.adapter_repo}")
    if hf_token:
        try:
            model = PeftModel.from_pretrained(model, args.adapter_repo, token=hf_token)
        except TypeError:
            model = PeftModel.from_pretrained(model, args.adapter_repo, use_auth_token=hf_token)
    else:
        model = PeftModel.from_pretrained(model, args.adapter_repo)

    print("Merging adapter into base model.")
    model = model.merge_and_unload(safe_merge=args.safe_merge)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to: {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    merge_manifest = {
        "base_model": args.base_model,
        "adapter_repo": args.adapter_repo,
        "chat_tokens_path": str(chat_tokens_path),
        "chat_tokens_count": len(chat_tokens),
        "chat_tokens_added_at_merge_time": added,
        "torch_dtype": str(torch_dtype).replace("torch.", ""),
        "device_map": args.device_map,
        "safe_merge": args.safe_merge,
        "output_dir": str(out_dir),
    }
    (out_dir / "merge_manifest.json").write_text(json.dumps(merge_manifest, indent=2) + "\n", encoding="utf-8")

    if args.hub_repo_id:
        try:
            from huggingface_hub import create_repo
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing dependency for Hub upload. Install with `pip install huggingface_hub`."
            ) from exc

        print(f"Pushing merged model to Hub: {args.hub_repo_id}")
        create_repo(args.hub_repo_id, private=args.private, exist_ok=True, token=hf_token)
        if hf_token:
            model.push_to_hub(args.hub_repo_id, token=hf_token)
            tokenizer.push_to_hub(args.hub_repo_id, token=hf_token)
        else:
            model.push_to_hub(args.hub_repo_id)
            tokenizer.push_to_hub(args.hub_repo_id)

    print("Merge complete.")


if __name__ == "__main__":
    main()
