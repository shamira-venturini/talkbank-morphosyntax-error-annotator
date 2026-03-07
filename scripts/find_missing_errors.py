from __future__ import annotations

import argparse
import json

from common import resolve_path


def find_hidden_agreement_errors(jsonl_filepath: str, model: str = "en_core_web_sm") -> list[dict]:
    import spacy

    nlp = spacy.load(model)
    hidden_errors = []
    target_pronouns = {"he", "she", "it"}

    path = resolve_path(jsonl_filepath)
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i, line in enumerate(lines):
        example = json.loads(line)
        text = example.get("output", "")
        if "[* m:03s:a]" in text or not text:
            continue

        doc = nlp(text)
        for token in doc:
            is_singular_subj = (token.text.lower() in target_pronouns) or (
                token.tag_ == "NN" and token.dep_ == "nsubj"
            )
            if not is_singular_subj:
                continue

            verb = token.head
            if verb.tag_ in {"VB", "VBP"} and verb.pos_ == "VERB":
                has_modal = any(t.dep_ == "aux" for t in verb.children)
                if not has_modal:
                    hidden_errors.append(
                        {
                            "index": i,
                            "original": text,
                            "missing_at": verb.text,
                            "suggestion": f"{verb.text} [* m:03s:a]",
                        }
                    )
                    break

    return hidden_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find likely missing 3sg agreement annotations.")
    parser.add_argument("input", nargs="?", default="data/intermediate/df_master_training_v3.jsonl")
    parser.add_argument("output", nargs="?", default="missing_annotations_test.json")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    potential_fixes = find_hidden_agreement_errors(args.input, model=args.spacy_model)
    out = resolve_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(potential_fixes, f, indent=2)
    print(f"Found {len(potential_fixes)} potential missing annotations.")


if __name__ == "__main__":
    main()
