from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional


DEFAULT_BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
DEFAULT_ADAPTER_REPO = "mash-mash/talkbank-morphosyntax-annotator-final-recon_full_comp_preserve_final_seed3407"
DEFAULT_CHAT_TOKENS = "chat_tokens.json"
DEFAULT_PROMPT = "FINAL_PROMPT.txt"
DEFAULT_PROMPT_WRAPPER = "prompt_wrapper.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference adapter-based inference loader for the TalkBank handoff.")
    parser.add_argument("--utterance", required=True, help="Utterance text to annotate.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model repo id.")
    parser.add_argument("--adapter-repo", default=DEFAULT_ADAPTER_REPO, help="Adapter repo id.")
    parser.add_argument("--chat-tokens", default=DEFAULT_CHAT_TOKENS, help="Path to chat_tokens.json.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Path to FINAL_PROMPT.txt.")
    parser.add_argument("--prompt-wrapper", default=DEFAULT_PROMPT_WRAPPER, help="Path to prompt_wrapper.txt.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Generation limit.")
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="Environment variable holding the HF token.")
    return parser.parse_args()


def load_token(env_name: str) -> Optional[str]:
    token = os.environ.get(env_name, "").strip()
    return token or None


def load_with_auth(loader, model_id: str, hf_token: Optional[str], **kwargs):
    if hf_token:
        try:
            return loader.from_pretrained(model_id, token=hf_token, **kwargs)
        except TypeError:
            return loader.from_pretrained(model_id, use_auth_token=hf_token, **kwargs)
    return loader.from_pretrained(model_id, **kwargs)


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    chat_tokens_path = (here / args.chat_tokens).resolve()
    prompt_path = (here / args.prompt).resolve()
    prompt_wrapper_path = (here / args.prompt_wrapper).resolve()
    hf_token = load_token(args.hf_token_env)

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency. Install required packages first, e.g. "
            "`pip install transformers peft accelerate bitsandbytes sentencepiece`."
        ) from exc

    instruction = prompt_path.read_text(encoding="utf-8").strip()
    prompt_wrapper = prompt_wrapper_path.read_text(encoding="utf-8")
    chat_tokens = json.loads(chat_tokens_path.read_text(encoding="utf-8"))
    chat_tokens = list(dict.fromkeys(chat_tokens))

    tokenizer = load_with_auth(AutoTokenizer, args.base_model, hf_token, use_fast=True)
    model = load_with_auth(
        AutoModelForCausalLM,
        args.base_model,
        hf_token,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # The adapter was trained with expanded CHAT token embeddings. Add tokens and resize
    # before loading PEFT weights or the embedding / lm_head shapes will not match.
    added = tokenizer.add_tokens(chat_tokens, special_tokens=False)
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    if hf_token:
        try:
            model = PeftModel.from_pretrained(model, args.adapter_repo, token=hf_token)
        except TypeError:
            model = PeftModel.from_pretrained(model, args.adapter_repo, use_auth_token=hf_token)
    else:
        model = PeftModel.from_pretrained(model, args.adapter_repo)

    prompt = prompt_wrapper.format(instruction=instruction, input=args.utterance)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_ids = generated[0][encoded["input_ids"].shape[1] :]
    print(tokenizer.decode(response_ids, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
