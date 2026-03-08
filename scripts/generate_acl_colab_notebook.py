import json
import subprocess

from common import resolve_path


DEFAULT_REPO_URL = "https://github.com/shamira-venturini/talkbank-morphosyntax-error-annotator.git"


def infer_repo_url() -> str:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return DEFAULT_REPO_URL

    remote_url = result.stdout.strip()
    if remote_url.startswith("git@github.com:"):
        remote_url = "https://github.com/" + remote_url[len("git@github.com:") :]
    if remote_url.startswith("https://github.com/") and not remote_url.endswith(".git"):
        remote_url += ".git"
    return remote_url or DEFAULT_REPO_URL


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build_notebook(repo_url: str) -> dict:
    cells = []

    cells.append(
        md_cell(
            """# ACL-Ready CLAN SFT (Llama-3.1-8B, Unsloth, LoRA)

This notebook is execution-only. Data composition and splitting are frozen in-repo.

## Evaluation design
- `eval` / `test`: real-only (TD + DLD), primary confirmatory evaluation
- `eval_coverage` / `test_coverage`: synthetic label-coverage splits, secondary diagnostics
- `holdout`: synthetic-only held-out labels for generalization (`++er`, `++est`, `0er`, `0est`)
- minimal paper-ready results are exported back into the repo under `results/<run_name>`

Default configuration in this notebook is **Experiment 1**:
- direct Stage-3 (no curriculum)
- 50/50 real-synthetic train split (prepared in frozen data)
- support threshold = 20 on confirmatory real splits

## Statistical reporting policy
- Confirmatory per-label reporting uses minimum support threshold (`min_label_support_confirmatory`)
- Low-support labels are reported separately as exploratory
- Micro-F1 includes bootstrap confidence interval
"""
        )
    )

    cells.append(
        code_cell(
            """#@title 1) Environment setup
from pathlib import Path

REPO_URL = "https://github.com/shamira-venturini/talkbank-morphosyntax-error-annotator.git"
REPO_BRANCH = "master"
REPO_DIR = Path("/content/talkbank-morphosyntax-error-annotator")

if not REPO_DIR.exists():
    !git clone -b {REPO_BRANCH} {REPO_URL} {REPO_DIR}
else:
    print("Repo already present:", REPO_DIR)

%cd /content/talkbank-morphosyntax-error-annotator

!pip -q install unsloth
!pip -q install --no-deps bitsandbytes accelerate peft trl
!pip -q install datasets evaluate scikit-learn pandas numpy matplotlib
"""
        )
    )

    cells.append(
        code_cell(
            """#@title 2) Config (single source of truth)
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    split_dir: str = "experiments/acl_rr_v1"
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    seed: int = 3407
    load_in_4bit: bool = True
    max_seq_length: int = 384

    # Curriculum
    run_full_curriculum: bool = False
    single_stage: int = 3
    stage_plan: dict = field(default_factory=lambda: {
        1: {"lr": 2e-4, "epochs": 2},
        2: {"lr": 1e-4, "epochs": 2},
        3: {"lr": 5e-5, "epochs": 3},
    })

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Training
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 20
    output_root: str = "/content/outputs/acl_rr_exp1_direct_stage3"
    save_predictions_splits: tuple = ("test_real", "holdout_generalization")
    save_eval_coverage_predictions: bool = False
    eval_only_from_hub: bool = False
    hub_eval_repo_id: str = ""
    save_checkpoints_locally: bool = True
    load_best_model_at_end: bool = True
    use_early_stopping: bool = True
    save_adapter_locally: bool = True
    save_optimizer_state: bool = False
    cleanup_local_output_after_run: bool = False

    # Evaluation policy
    min_label_support_confirmatory: int = 20
    min_label_support_coverage: int = 10
    min_label_support_holdout: int = 10
    bootstrap_iterations: int = 500
    bootstrap_seed: int = 3407

    # Persistence
    save_to_drive: bool = True
    drive_root: str = "/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/CLAN_annotator_runs"
    save_to_repo_results: bool = True
    repo_results_root: str = "results"
    push_to_hub: bool = False
    push_all_stages: bool = False
    hf_repo_prefix: str = "mash-mash/clan-annotator-exp1"
    hf_repo_visibility: str = "private"
    hf_repo_include_run_name: bool = True
    hf_append_stage_suffix: bool = False
    hf_repo_final_alias: str = ""
    git_commit_repo_results: bool = False
    git_push_repo_results: bool = False
    git_branch: str = "master"
    git_user_name: str = "shamira-venturini"
    git_user_email: str = "venturinishamira@gmail.com"

cfg = Config()
split_root = Path(cfg.split_dir)
assert split_root.exists(), f"Missing split directory: {split_root}"
cfg.output_root = f"{cfg.output_root}_seed{cfg.seed}"
print(cfg)
"""
        )
    )

    cells.append(
        code_cell(
            """#@title 2b) Persistence setup (Drive + Hub auth)
import os
from pathlib import Path
from huggingface_hub import create_repo

if cfg.save_to_drive:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

if cfg.push_to_hub or cfg.eval_only_from_hub:
    from huggingface_hub import login
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in Colab Secrets while Hub access is enabled")
    login(token=hf_token)

def normalize_hf_repo_id(repo_id: str) -> str:
    repo_id = repo_id.strip().strip("/")
    if not repo_id:
        raise ValueError("Empty Hugging Face repo id")
    if "/" not in repo_id:
        raise ValueError(f"Hugging Face repo id must include namespace/repo_name: {repo_id}")
    return repo_id

def build_hf_repo_id(stage: int) -> str:
    base_repo_id = normalize_hf_repo_id(cfg.hf_repo_prefix)
    suffixes = []
    if cfg.hf_repo_include_run_name:
        suffixes.append(run_name)
    if cfg.hf_append_stage_suffix or cfg.push_all_stages:
        suffixes.append(f"stage{stage}")
    if not suffixes:
        return base_repo_id
    namespace, repo_name = base_repo_id.split("/", 1)
    return f"{namespace}/{repo_name}-{'-'.join(suffixes)}"

def create_hf_model_repo(repo_id: str) -> None:
    create_repo(
        repo_id=normalize_hf_repo_id(repo_id),
        repo_type="model",
        private=(cfg.hf_repo_visibility == "private"),
        exist_ok=True,
    )

run_root = Path(cfg.output_root)
run_root.mkdir(parents=True, exist_ok=True)
run_name = run_root.name
repo_results_dir = Path(cfg.repo_results_root) / run_name
if cfg.save_to_repo_results:
    repo_results_dir.mkdir(parents=True, exist_ok=True)
"""
        )
    )

    cells.append(
        code_cell(
            """#@title 3) Load model/tokenizer + seed + token extension
import json
import random
import numpy as np
import torch
from peft import PeftModel
from unsloth import FastLanguageModel

assert torch.cuda.is_available(), "CUDA GPU not available. Switch Colab runtime to GPU."
print("GPU:", torch.cuda.get_device_name(0))

random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg.base_model,
    max_seq_length=cfg.max_seq_length,
    load_in_4bit=cfg.load_in_4bit,
    dtype=None,
)

chat_tokens = json.loads((split_root / "chat_tokens.json").read_text())
chat_tokens = list(dict.fromkeys(chat_tokens))
added = tokenizer.add_tokens(chat_tokens, special_tokens=False)
if added > 0:
    model.resize_token_embeddings(len(tokenizer))

if cfg.eval_only_from_hub:
    hub_repo_id = normalize_hf_repo_id(cfg.hub_eval_repo_id)
    model = PeftModel.from_pretrained(model, hub_repo_id, is_trainable=False)
    model.eval()
    print("Loaded adapter from Hub:", hub_repo_id)

print("Added tokens:", added)
print("Tokenizer size:", len(tokenizer))
"""
        )
    )

    cells.append(
        code_cell(
            """#@title 4) Attach LoRA
if not cfg.eval_only_from_hub:
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
else:
    print("Skipping LoRA attachment because cfg.eval_only_from_hub=True")
"""
        )
    )

    cells.append(
        code_cell(
            """#@title 5) Dataset helpers
from datasets import load_dataset

def load_split(stage: int, split_name: str):
    path = split_root / f"stage{stage}_{split_name}.jsonl"
    if not path.exists():
        return None
    return load_dataset("json", data_files=str(path), split="train")

def assert_no_train_overlap(stage: int):
    ds_train = load_split(stage, "train")
    non_train_names = ["eval", "test", "eval_coverage", "test_coverage", "holdout"]
    train_inputs = set(ds_train["input"])
    for name in non_train_names:
        ds = load_split(stage, name)
        if ds is None:
            continue
        overlap = train_inputs.intersection(set(ds["input"]))
        assert not overlap, f"Input leakage: stage{stage} train vs {name} ({len(overlap)})"
    return ds_train

for s in [1, 2, 3]:
    _ = assert_no_train_overlap(s)
print("Leakage checks passed for stages 1/2/3.")
"""
        )
    )

    cells.append(
        code_cell(
            """#@title 6) Prompt formatting
def formatting_prompts_func(examples):
    texts = []
    for instruction, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        text = (
            f"### Instruction:\\n{instruction}\\n\\n"
            f"### Input:\\n{inp}\\n\\n"
            f"### Response:\\n{out}{tokenizer.eos_token}"
        )
        texts.append(text)
    return texts
"""
        )
    )

    cells.append(
        code_cell(
            """#@title 7) Training loop (single stage or full 3-stage curriculum)
import os
import shutil
import json
from pathlib import Path
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
from unsloth.chat_templates import train_on_responses_only

def build_sft_args(stage: int, lr: float, epochs: int):
    common = dict(
        output_dir=f"{cfg.output_root}/stage{stage}",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=cfg.seed + stage,
        data_seed=cfg.seed + stage,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        max_grad_norm=1.0,
    )
    if cfg.save_checkpoints_locally:
        common.update(
            save_steps=cfg.save_steps,
            save_total_limit=2,
            load_best_model_at_end=cfg.load_best_model_at_end,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    else:
        common.update(load_best_model_at_end=False)
    try:
        return SFTConfig(
            evaluation_strategy="steps",
            save_strategy=("steps" if cfg.save_checkpoints_locally else "no"),
            **common,
        )
    except TypeError:
        return SFTConfig(
            eval_strategy="steps",
            save_strategy=("steps" if cfg.save_checkpoints_locally else "no"),
            **common,
        )

def train_stage(stage: int, lr: float, epochs: int):
    ds_train = load_split(stage, "train")
    ds_eval = load_split(stage, "eval")
    args = build_sft_args(stage, lr, epochs)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        max_seq_length=cfg.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        formatting_func=formatting_prompts_func,
        args=args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)] if cfg.use_early_stopping and cfg.save_checkpoints_locally else [],
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="### Instruction:\\n",
        response_part="### Response:\\n",
    )
    metrics = trainer.train().metrics
    if cfg.save_adapter_locally:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    if cfg.save_optimizer_state:
        trainer.save_state()

    # Save compact stage metrics for reproducibility.
    stage_metrics_path = Path(args.output_dir) / f"stage{stage}_train_metrics.json"
    stage_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Optional Hub push: adapters/tokenizer only (storage-efficient).
    if cfg.push_to_hub and (cfg.push_all_stages or stage == run_order[-1]):
        repo_ids = [build_hf_repo_id(stage)]
        if stage == run_order[-1] and cfg.hf_repo_final_alias.strip():
            repo_ids.append(normalize_hf_repo_id(cfg.hf_repo_final_alias))
        for repo_id in dict.fromkeys(repo_ids):
            create_hf_model_repo(repo_id)
            model.push_to_hub(repo_id, save_embedding_layers=True)
            tokenizer.push_to_hub(repo_id)

    # Optional Drive sync: copy only minimal artifacts.
    if cfg.save_to_drive:
        dst = Path(cfg.drive_root) / run_name / f"stage{stage}"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(stage_metrics_path, dst / stage_metrics_path.name)
        trainer_state = Path(args.output_dir) / "trainer_state.json"
        if trainer_state.exists():
            shutil.copy(trainer_state, dst / "trainer_state.json")
        if cfg.save_adapter_locally:
            for name in ["adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", "special_tokens_map.json"]:
                src = Path(args.output_dir) / name
                if src.exists():
                    shutil.copy(src, dst / name)

    print(f"Stage {stage} done:", metrics)
    return metrics

if cfg.run_full_curriculum:
    run_order = [1, 2, 3]
else:
    run_order = [cfg.single_stage]

stage_metrics = {}
if cfg.eval_only_from_hub:
    print("Skipping training because cfg.eval_only_from_hub=True")
else:
    for stage in run_order:
        p = cfg.stage_plan[stage]
        stage_metrics[stage] = train_stage(stage, lr=p["lr"], epochs=p["epochs"])
    print("Training finished for stages:", run_order)
"""
        )
    )

    cells.append(
        code_cell(
            """#@title 8) Inference helpers
import re
from tqdm import tqdm

FastLanguageModel.for_inference(model)
tokenizer.padding_side = "left"

def build_prompt(instruction: str, inp: str) -> str:
    return f"### Instruction:\\n{instruction}\\n\\n### Input:\\n{inp}\\n\\n### Response:\\n"

def batch_predict(dataset, batch_size=16, max_new_tokens=96):
    out_rows = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Predict"):
        batch = dataset[i:i+batch_size]
        prompts = [build_prompt(ins, inp) for ins, inp in zip(batch["instruction"], batch["input"])]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        dec = tokenizer.batch_decode(gen, skip_special_tokens=False)
        for j, txt in enumerate(dec):
            pred = txt.split("### Response:\\n")[-1]
            pred = pred.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
            if tokenizer.pad_token:
                pred = pred.replace(tokenizer.pad_token, "").strip()
            pred = re.sub(r"([A-Za-z\\]])([\\.!\\?])", r"\\1 \\2", pred)
            out_rows.append(
                {
                    "input": batch["input"][j],
                    "instruction": batch["instruction"][j],
                    "human_gold": batch["output"][j],
                    "model_prediction": pred,
                    "provenance_label": batch["provenance_label"][j] if "provenance_label" in batch else None,
                    "error_count": batch["error_count"][j] if "error_count" in batch else None,
                    "row_id": batch["row_id"][j] if "row_id" in batch else None,
                }
            )
    return out_rows
"""
        )
    )

    cells.append(
        code_cell(
            """#@title 9) Metrics with support control + bootstrap CI
import json
from collections import defaultdict
from pathlib import Path
import random
import numpy as np

TAG_RE = re.compile(r"\\[\\*\\s*[ms](?::[^\\]]+)?\\]")

def norm_tag(tag: str) -> str:
    body = tag.strip()[2:-1].strip()
    return f"[* {body}]"

def tags_set(text: str):
    return set(norm_tag(t) for t in TAG_RE.findall(text or ""))

def score_rows(rows):
    exact = 0
    tp = fp = fn = 0
    per = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    for r in rows:
        g = tags_set(r["human_gold"])
        p = tags_set(r["model_prediction"])
        if r["human_gold"].strip() == r["model_prediction"].strip():
            exact += 1
        for t in g:
            per[t]["support"] += 1
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        for t in (g & p):
            per[t]["tp"] += 1
        for t in (p - g):
            per[t]["fp"] += 1
        for t in (g - p):
            per[t]["fn"] += 1

    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    per_rows = []
    for tag, c in sorted(per.items(), key=lambda kv: (-kv[1]["support"], kv[0])):
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f = (2 * p * r / (p + r)) if (p + r) else 0.0
        per_rows.append({"tag": tag, "support": c["support"], "precision": p, "recall": r, "f1": f})

    return {
        "n": len(rows),
        "exact_match": exact / len(rows) if rows else 0.0,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "per_label": per_rows,
    }

def bootstrap_micro_f1(rows, n_iter=500, seed=3407):
    if not rows:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    vals = []
    n = len(rows)
    for _ in range(n_iter):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        vals.append(score_rows(sample)["micro_f1"])
    vals = np.array(vals)
    return {
        "mean": float(vals.mean()),
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
    }

def split_confirmatory(per_label, min_support):
    confirmatory = [x for x in per_label if x["support"] >= min_support]
    exploratory = [x for x in per_label if x["support"] < min_support]
    return confirmatory, exploratory

def support_threshold_for_split(split_name: str) -> int:
    if "coverage" in split_name:
        return cfg.min_label_support_coverage
    if "holdout" in split_name:
        return cfg.min_label_support_holdout
    return cfg.min_label_support_confirmatory

def run_eval(dataset, split_name):
    rows = batch_predict(dataset)
    metrics = score_rows(rows)
    ci = bootstrap_micro_f1(rows, n_iter=cfg.bootstrap_iterations, seed=cfg.bootstrap_seed)
    support_min = support_threshold_for_split(split_name)
    confirm, explor = split_confirmatory(metrics["per_label"], support_min)
    metrics["micro_f1_bootstrap"] = ci
    metrics["confirmatory_min_support"] = support_min
    metrics["per_label_confirmatory"] = confirm
    metrics["per_label_exploratory"] = explor
    provenance_values = sorted({(r.get("provenance_label") or "") for r in rows if r.get("provenance_label")})
    by_source = {}
    for src in provenance_values:
        src_rows = [r for r in rows if r.get("provenance_label") == src]
        src_metrics = score_rows(src_rows)
        by_source[src] = {
            "n": src_metrics["n"],
            "exact_match": src_metrics["exact_match"],
            "micro_f1": src_metrics["micro_f1"],
        }
    metrics["by_source"] = by_source

    out_dir = Path(cfg.output_root) / "eval_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    should_save_predictions = split_name in set(cfg.save_predictions_splits) or (
        cfg.save_eval_coverage_predictions and "coverage" in split_name
    )
    if should_save_predictions:
        (out_dir / f"predictions_{split_name}.jsonl").write_text(
            "\\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\\n",
            encoding="utf-8",
        )
    with (out_dir / f"metrics_{split_name}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    for bucket_name, bucket_rows in [
        ("confirmatory", metrics["per_label_confirmatory"]),
        ("exploratory", metrics["per_label_exploratory"]),
    ]:
        with (out_dir / f"per_label_{bucket_name}_{split_name}.csv").open("w", encoding="utf-8") as f:
            f.write("tag,support,precision,recall,f1\\n")
            for row in bucket_rows:
                f.write(
                    f"{row['tag']},{row['support']},{row['precision']:.6f},{row['recall']:.6f},{row['f1']:.6f}\\n"
                )

    print(split_name)
    print(json.dumps({
        "n": metrics["n"],
        "exact_match": metrics["exact_match"],
        "micro_f1": metrics["micro_f1"],
        "micro_f1_ci95": [metrics["micro_f1_bootstrap"]["ci_low"], metrics["micro_f1_bootstrap"]["ci_high"]],
        "confirmatory_labels": len(confirm),
        "exploratory_labels": len(explor),
    }, indent=2))
    return rows, metrics
"""
        )
    )

    cells.append(
        code_cell(
            """#@title 10) Evaluate final model on all relevant splits
# Final evaluation always uses Stage-3 targets.
ds_test_real = load_split(3, "test")
ds_holdout = load_split(3, "holdout")
ds_test_cov = load_split(3, "test_coverage")
ds_eval_real = load_split(3, "eval")
ds_eval_cov = load_split(3, "eval_coverage")

_ = run_eval(ds_eval_real, "eval_real")
_ = run_eval(ds_test_real, "test_real")
if ds_eval_cov is not None:
    _ = run_eval(ds_eval_cov, "eval_coverage")
if ds_test_cov is not None:
    _ = run_eval(ds_test_cov, "test_coverage")
_ = run_eval(ds_holdout, "holdout_generalization")

# Save a compact run-level metrics summary and optional Drive / repo sync.
from pathlib import Path
import shutil
import json
import subprocess

eval_dir = Path(cfg.output_root) / "eval_outputs"
run_summary = {
    "seed": cfg.seed,
    "split_dir": cfg.split_dir,
    "output_root": cfg.output_root,
}
for split_name in ["eval_real", "test_real", "eval_coverage", "test_coverage", "holdout_generalization"]:
    p = eval_dir / f"metrics_{split_name}.json"
    if p.exists():
        run_summary[split_name] = json.loads(p.read_text())

summary_path = eval_dir / "run_summary.json"
summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

paper_csv_path = eval_dir / "paper_main_metrics.csv"
with paper_csv_path.open("w", encoding="utf-8") as f:
    f.write("split,n,exact_match,micro_f1,ci_low,ci_high,confirmatory_labels,exploratory_labels\\n")
    for split_name in ["eval_real", "test_real", "eval_coverage", "test_coverage", "holdout_generalization"]:
        data = run_summary.get(split_name)
        if not isinstance(data, dict):
            continue
        ci = data.get("micro_f1_bootstrap", {})
        f.write(
            f"{split_name},{data.get('n', 0)},{data.get('exact_match', 0.0):.6f},{data.get('micro_f1', 0.0):.6f},"
            f"{ci.get('ci_low', 0.0):.6f},{ci.get('ci_high', 0.0):.6f},"
            f"{len(data.get('per_label_confirmatory', []))},{len(data.get('per_label_exploratory', []))}\\n"
        )

if cfg.save_to_repo_results:
    repo_results_dir.mkdir(parents=True, exist_ok=True)
    repo_manifest = {
        "run_name": run_name,
        "seed": cfg.seed,
        "split_dir": cfg.split_dir,
        "output_root": cfg.output_root,
        "run_order": run_order,
        "push_to_hub": cfg.push_to_hub,
        "hf_repo_prefix": cfg.hf_repo_prefix,
        "hf_repo_visibility": cfg.hf_repo_visibility,
        "hf_repo_include_run_name": cfg.hf_repo_include_run_name,
        "hf_append_stage_suffix": cfg.hf_append_stage_suffix,
        "hf_repo_final_alias": cfg.hf_repo_final_alias,
        "saved_predictions_splits": list(cfg.save_predictions_splits),
    }
    for split_name in ["eval_real", "test_real", "eval_coverage", "test_coverage", "holdout_generalization"]:
        src = eval_dir / f"metrics_{split_name}.json"
        if src.exists():
            shutil.copy(src, repo_results_dir / src.name)
        for bucket_name in ["confirmatory", "exploratory"]:
            csv_src = eval_dir / f"per_label_{bucket_name}_{split_name}.csv"
            if csv_src.exists():
                shutil.copy(csv_src, repo_results_dir / csv_src.name)
    shutil.copy(summary_path, repo_results_dir / "run_summary.json")
    shutil.copy(paper_csv_path, repo_results_dir / "paper_main_metrics.csv")
    for split_name in cfg.save_predictions_splits:
        src = eval_dir / f"predictions_{split_name}.jsonl"
        if src.exists():
            shutil.copy(src, repo_results_dir / src.name)
    for stage in run_order:
        stage_metrics = Path(cfg.output_root) / f"stage{stage}" / f"stage{stage}_train_metrics.json"
        if stage_metrics.exists():
            shutil.copy(stage_metrics, repo_results_dir / stage_metrics.name)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    if git_head.returncode == 0:
        repo_manifest["repo_head"] = git_head.stdout.strip()
    manifest_path = repo_results_dir / "results_bundle_manifest.json"
    manifest_path.write_text(json.dumps(repo_manifest, indent=2), encoding="utf-8")

    if cfg.git_commit_repo_results or cfg.git_push_repo_results:
        subprocess.run(["git", "config", "user.name", cfg.git_user_name], check=True)
        subprocess.run(["git", "config", "user.email", cfg.git_user_email], check=True)
        subprocess.run(["git", "add", str(repo_results_dir)], check=True)
        has_changes = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if has_changes.returncode != 0:
            commit_msg = f"Add results bundle {run_name}"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            if cfg.git_push_repo_results:
                from google.colab import userdata
                gh_token = userdata.get("GITHUB_TOKEN")
                if not gh_token:
                    raise ValueError("GITHUB_TOKEN not found in Colab Secrets while cfg.git_push_repo_results=True")
                remote = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True)
                remote_url = remote.stdout.strip()
                if remote_url.startswith("git@github.com:"):
                    remote_url = "https://github.com/" + remote_url[len("git@github.com:") :]
                if not remote_url.startswith("https://github.com/"):
                    raise ValueError(f"Unsupported GitHub remote URL: {remote_url}")
                push_url = remote_url.replace("https://", f"https://x-access-token:{gh_token}@")
                subprocess.run(["git", "push", push_url, f"HEAD:{cfg.git_branch}"], check=True)

if cfg.save_to_drive:
    dst = Path(cfg.drive_root) / run_name / "eval_outputs"
    dst.mkdir(parents=True, exist_ok=True)
    for split_name in ["eval_real", "test_real", "eval_coverage", "test_coverage", "holdout_generalization"]:
        src = eval_dir / f"metrics_{split_name}.json"
        if src.exists():
            shutil.copy(src, dst / src.name)
        for bucket_name in ["confirmatory", "exploratory"]:
            csv_src = eval_dir / f"per_label_{bucket_name}_{split_name}.csv"
            if csv_src.exists():
                shutil.copy(csv_src, dst / csv_src.name)
    shutil.copy(summary_path, dst / "run_summary.json")
    shutil.copy(paper_csv_path, dst / "paper_main_metrics.csv")
    for split_name in cfg.save_predictions_splits:
        src = eval_dir / f"predictions_{split_name}.jsonl"
        if src.exists():
            shutil.copy(src, dst / src.name)

if cfg.cleanup_local_output_after_run:
    shutil.rmtree(Path(cfg.output_root), ignore_errors=True)
"""
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out_path = resolve_path("experiments/acl_rr_v1/ACL_SFT_CLAN_Llama3_1_8B_ACL.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook(infer_repo_url())
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
