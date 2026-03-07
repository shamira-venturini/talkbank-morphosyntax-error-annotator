# %% [markdown] cell 1
# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth your local device, follow [our guide](https://docs.unsloth.ai/get-started/install-and-update). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)
# %% [code] cell 2
from google.colab import drive
drive.mount('/content/drive')
# %% [markdown] cell 3
# ### Installation
# %% [code] cell 4
%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    import torch; v = re.match(r"[0-9]{1,}\.[0-9]{1,}", str(torch.__version__)).group(0)
    xformers = "xformers==" + ("0.0.33.post1" if v=="2.9" else "0.0.32.post2" if v=="2.8" else "0.0.29.post3")
    !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
# %% [markdown] cell 5
# ### Unsloth
# 
# The Unsloth library optimises backpropagation processes, resulting in a 2-5x increase in training speed and 70% reduction in VRAM usage compared to standard HF implementations. This optimisation is achieved without loss of accuracy.
# 
# #### Model loading
# %% [code] cell 6
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

    "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" # NEW! Llama 3.3 70B!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = 512,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
# %% [markdown] cell 7
# #### Embeddings Resizing
# 
# We resized the model's embedding layer to incorporate the CHAT tags as 'Special Tokens' so the model assigns a vector to each tag, preventing fragmentation, reducing computational noise during training and treating error codes as atomic units.
# %% [code] cell 8
# Create a list of all your unique tags from the table
chat_tags = ["0", "0det", "0aux", "0subj", "0obj", "0prep", "[/]", "[//]", "[+ gram]", "[* m:allo]", "[* m:03s:a]", "[* m:0]", "[* m:03s]", "[* m:++]", "[* m:++ed]", "[* m:++ed:i]", "[* m:++en:i]",
             "[* m:++er]", "[* m:++est]", "[* m:++s]",  "[* m:++s:i]", "[* m:+]", "[* m:+3s]", "[* m:+3s:a]","[* m:+ed]", "[* m:+ing]", "[* m:+s:a]", "[* m:0]", "[* m:0's]", "[* m:0ed]",
             "[* m:0er]", "[* m:0est]", "[* m:0ing]", "[* m:0s:a]", "[* m:=ed]", "[* m:=en]", "[* m:=s]",  "[* m:base]", "[* m:base:ed]", "[* m:base:en]", "[* m:base:er]", "[* m:base:est]",
             "[* m:base:s]", "[* m:irr]", "[* m:irr:ed]", "[* m:irr:en]", "[* m:irr:s]", "[* m:sub:en]", "[* m:irr:en]", "[* m:irr:s]", "[* m:=]", "[* m:sub]", "[* m:sub:ed]", "[* m:sub:en]",
             "[* m:vsg]", "[* m:vsg:a]", "[* m:vun]", "[* m:vun:a]", "[* s:r]", "[* s:r:der]", "[* s:r:prep]", "[* s:r:pro]", "[* s:r:gc]", "[* s:r:gc:det]", "[* s:r:gc:pro]",
             "[* s:r]", "[* s:r:gc]", "[* s]", "[* m]", "[* p]", "xxx", "(.)", "(..)", "(...)", "+...", "[!]", "&-", "&+"]

# Add them as special tokens
tokenizer.add_tokens(chat_tags)
model.resize_token_embeddings(len(tokenizer))
# %% [markdown] cell 9
# #### LoRA configuration
# 
# * $r=64$ the complex task necessitates a higher-dimentional 'update space' to capture the nuances of the mapping.
# 
# * $\alpha$ = 32 (scaling factor) to ensure stable gradients during the early stages of the curriculum.
# 
# * `{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}`: we update the attention heads and the MLP essential for knowledge-heavy reasoning tasks.
# %% [code] cell 10
from unsloth import FastLanguageModel

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0.05,
    bias = "none",
    modules_to_save = ["embed_tokens", "lm_head"],
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
# %% [markdown] cell 11
# #### Stage 2 and 3 Set-Up
# %% [code] cell 12
#model_s1 = "CHAT-Annotator-stage_1"
model_s2 = "CHAT-Annotator-stage_2"
#model_s3 = "CHAT-Annotator-stage_3"
# %% [markdown] cell 13
# #### Load HF-saved model
# %% [code] cell 14
from google.colab import userdata
hf_token = userdata.get('HF_TOKEN')

from huggingface_hub import login
login(token=hf_token)

# Define your repo name with the version
hf_username = "mash-mash"

model.load_adapter(f"{hf_username}/{model_s2}", adapter_name='default')
model.set_adapter("default")
# %% [markdown] cell 15
# <a name="Data"></a>
# ### Data Prep
# We utilize a custom dataset comprising approximately 7,600 sentences from a subset of the CHILDES ENNI corpus, supplemented with synthetic examples generated via Gemini 2.5 Pro and using direction from the CHAT Transcription Guidelines (ADD REF).
# 
# The dataset is formatted to support the Llama-3.1 Instruct conversation style, using a structured System Prompt to enforce strict adherence to TalkBank/CHAT guidelines.
# We use the get_chat_template function to wrap our transcriptions into a multi-turn conversation format. In this project, the "System" role defines the annotation rules, the "User" provides the raw transcription, and the "Assistant" provides the gold-standard CHAT annotation.
# The conversation structure rendered by Llama-3.1:
# 
# 
# 
# ```
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# Annotate the following sentence with CLAN error codes. Preserve spelling and order.<|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# he run to the park .<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# he run [* m:base:ed] to the park .<|eot_id|>
# ```
# 
# 
# %% [code] cell 16
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # We simplify to a basic prompt to avoid header hallucinations
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}" + tokenizer.eos_token
        texts.append(text)
    return texts # Changed from { "text" : texts, } to texts
# %% [code] cell 17
from datasets import load_dataset

master_path = "/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/train/df_master_training_v3.jsonl"
v1_path = "/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/train/df_master_training_v1.jsonl"
v2_path = "/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/train/df_master_training_v2.jsonl"

master_ds = load_dataset("json", data_files=master_path, split="train")
v1_ds = load_dataset("json", data_files=v1_path, split="train")
v2_ds = load_dataset("json", data_files=v2_path, split="train")

print(f"Master Dataset rows: {len(master_ds)}")
print(f"V1 Dataset rows:     {len(v1_ds)}")
print(f"V2 Dataset rows:     {len(v2_ds)}")

if len(master_ds) != len(v1_ds) or len(master_ds) != len(v2_ds):
    print("\nSTOP: Your files are not the same length! You must re-generate V1 and V2 from the same Master V4 file.")
# %% [code] cell 18
import re
import random
from collections import defaultdict


def get_stratified_indices_fixed(dataset, n_test=15, n_eval=15):
    tag_to_indices = defaultdict(list)
    clean_indices = []
    tag_regex = r'\[\* ([ms]:[^\]]+)\]'

    # 1. Group every row index by its tag
    for idx, item in enumerate(dataset):
        match = re.search(tag_regex, item['output'])
        if match:
            tag = match.group(1)
            tag_to_indices[tag].append(idx)
        else:
            clean_indices.append(idx)

    train_idx, eval_idx, test_idx = [], [], []
    random.seed(3407)

    # 2. Stratify Error Tags
    for tag, indices in tag_to_indices.items():
        random.shuffle(indices)

        # Ensure we have enough for the requested split
        # We take n_test, n_eval, and the rest goes to train
        current_test = indices[:n_test]
        current_eval = indices[n_test : n_test + n_eval]
        current_train = indices[n_test + n_eval :]

        test_idx.extend(current_test)
        eval_idx.extend(current_eval)
        train_idx.extend(current_train)

    # 3. Split Clean Sentences (70/15/15)
    random.shuffle(clean_indices)
    c_total = len(clean_indices)
    c_test_end = int(c_total * 0.15)
    c_eval_end = int(c_total * 0.30)

    test_idx.extend(clean_indices[:c_test_end])
    eval_idx.extend(clean_indices[c_test_end:c_eval_end])
    train_idx.extend(clean_indices[c_eval_end:])

    return train_idx, eval_idx, test_idx

# Define apply_split function (previously missing)
def apply_split(ds, train_idx, eval_idx, test_idx):
    return {
        "train": ds.select(train_idx),
        "eval":  ds.select(eval_idx),
        "test":  ds.select(test_idx)
    }

# # EXECUTION
train_idx, eval_idx, test_idx = get_stratified_indices_fixed(master_ds)

# # Resulting Sets
dataset_s1 = apply_split(v1_ds, train_idx, eval_idx, test_idx)
dataset_s2 = apply_split(v2_ds, train_idx, eval_idx, test_idx)
dataset_s3 = apply_split(master_ds, train_idx, eval_idx, test_idx)

# print(f"14k Dataset Split Successfully (70/15/15):")
print(f"   Train: {len(train_idx)} | Eval: {len(eval_idx)} | Test: {len(test_idx)}")
# %% [markdown] cell 19
# ### Train the model
# 
# **Stage 1**: `learning_rate`: `2e-4`; `epochs`: 2; `weight_decay`:
# 
# **Stage 2**: `learning_rate`: `1e-4`; `epochs`: 2; `weight_decay`:
# 
# **Stage 3**:`learning_rate`: `5e-5`; `epochs`: 3; `weight_decay`:
# %% [code] cell 20
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
from unsloth.chat_templates import train_on_responses_only
import torch

STAGE_NAME = "stage_3"

print(f"Starting Guaranteed Run for: {STAGE_NAME}")
local_output_dir = f"outputs_{STAGE_NAME.lower()}"
drive_stat_path = f"/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/stats/{STAGE_NAME}"
os.makedirs(drive_stat_path, exist_ok=True)

# A. Setup the Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_s3["train"],
    eval_dataset = dataset_s3["eval"],
    max_seq_length = 512,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    formatting_func = formatting_prompts_func,
    args = SFTConfig(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 2,
        warmup_steps = 10,
        num_train_epochs = 2,
        learning_rate = 5e-5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        optim = "adamw_8bit",
        weight_decay = 0.1,
        eval_strategy = "steps",
        eval_steps = 50,
        logging_dir = "logs",
        logging_steps = 1,           # Log every single step for a smooth curve
        save_strategy = "epoch",     # Save a copy of the model every epoch as a backup
        report_to = "tensorboard",
        output_dir = local_output_dir,
    )
)

# B. Apply the response masking
# If the error persists here, it means the dataset dictionary is 'dirty'.
# Restarting the session and loading the S1 model again is the only fix.
trainer = train_on_responses_only(
    trainer,
    instruction_part = "### Instruction:\n",
    response_part = "### Response:\n",
)
# %% [code] cell 21
import shutil
import os

trainer_stats = trainer.train()
trainer.save_state()
# %% [code] cell 22
drive_stat_path = f"/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/stats/{STAGE_NAME}"
os.makedirs(drive_stat_path, exist_ok=True)
trainer.save_state() # Force write to disk
if os.path.exists(f"{local_output_dir}/trainer_state.json"):
    shutil.copy(f"{local_output_dir}/trainer_state.json", os.path.join(drive_stat_path, "trainer_state.json"))
    print(f"Science logs for {STAGE_NAME} saved to Drive.")

# Save a text summary of the training for the appendix
with open(f"{drive_stat_path}/training_summary.txt", "w") as f:
    f.write(f"Model: Llama-3.1-8B-Instruct\nStage: {STAGE_NAME}\n")
    f.write(f"Trainable Params: {trainer.get_num_trainable_parameters()}\n")
    f.write(f"Total FLOPs: {trainer_stats.metrics.get('train_total_flos', 'N/A')}\n") # Corrected access to metrics dictionary
    f.write(f"Final Train Loss: {trainer_stats.training_loss}\n")

print(f"Scientific Archive complete for {STAGE_NAME} at: {drive_stat_path}")
# %% [code] cell 23
from google.colab import userdata
hf_token = userdata.get('HF_TOKEN')

from huggingface_hub import login
login(token=hf_token)

# Define your repo name with the version
hf_username = "mash-mash" # Change this
model_repo_name = f"{hf_username}/CHAT-Annotator-{STAGE_NAME}" # CHANGE NAME

model.push_to_hub(
    model_repo_name,
    tokenizer,
    save_embedding_layers = True
)
tokenizer.push_to_hub(model_repo_name)


print(f"Model successfully pushed to: https://huggingface.co/{model_repo_name}")
# %% [markdown] cell 24
# ## EVALUATION
# %% [code] cell 25
import json
import re
import os
import torch
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# CHANGE THIS FOR EVERY STAGE: "stage_1", "stage_2", or "stage_3"
TEST_DATASET = dataset_s3['test'] # Ensure this matches the stage!

# Define output path dynamically
output_dir = f"/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/evaluation/{STAGE_NAME}"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"inference_results_{STAGE_NAME}.jsonl")

# --- 2. PREPARE MODEL ---
tokenizer.padding_side = "left"
FastLanguageModel.for_inference(model)
eos_token_id = tokenizer.eos_token_id

# Use the EXACT instruction from our training cell
instruction = "You are a CHAT annotator. Identify morphological and semantic errors based on the full obligatory context. Preserve input spelling and word order exactly."

final_evaluation_results = []
batch_size = 16

# --- 3. BATCH INFERENCE LOOP ---
for i in tqdm(range(0, len(TEST_DATASET), batch_size), desc=f"Evaluating {STAGE_NAME}"):
    batch_dict = TEST_DATASET[i : i + batch_size]

    # Convert to list of dicts for safety
    batch_list = [
        {"input": batch_dict['input'][j], "output": batch_dict['output'][j]}
        for j in range(len(batch_dict['input']))
    ]

    # Build prompts using the standard "Linguistic Auditor" format
    prompts = [
        f"### Instruction:\n{instruction}\n\n### Input:\n{item['input']}\n\n### Response:\n"
        for item in batch_list
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

    # OPTIMIZED GENERATION
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,   # Increased to 128 to accommodate [: reconstruction] NEEDS TO BE CHANGED LATER!!!
            use_cache=True,
            do_sample=False,      # Mandatory for scientific reproducibility
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode everything including special tokens to see [ ] and < >
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)

    for j, decoded in enumerate(decoded_outputs):
        # 1. Extract the prediction after the header
        if "### Response:\n" in decoded:
            prediction = decoded.split("### Response:\n")[-1]
        else:
            prediction = decoded

        # 2. Clean up Llama-specific artifacts
        prediction = prediction.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
        prediction = prediction.replace(tokenizer.pad_token, "").strip()

        # 3. Standardize spacing for the Evaluation Regex match
        # This ensures "word ." matches "word ." even if the model missed a space
        prediction = re.sub(r'([a-zA-Z\]])([\.!\?])', r'\1 \2', prediction)

        final_evaluation_results.append({
            "input": batch_list[j]["input"],
            "human_gold": batch_list[j]["output"],
            "model_prediction": prediction
        })

# --- 4. SAVE & VERIFY ---
with open(output_path, "w", encoding="utf-8") as f:
    for res in final_evaluation_results:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")

print(f"\nEvaluation of {STAGE_NAME} complete.")
print(f"Results saved to: {output_path}")
# %% [code] cell 26
import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_scientific_convergence(log_path, STAGE_NAME):
    with open(log_path, 'r') as f:
        data = json.load(f)['log_history']

    df = pd.DataFrame(data)

    # Separate Training and Evaluation logs
    train_df = df[df['loss'].notna()].copy()
    eval_df = df[df['eval_loss'].notna()].copy()

    plt.figure(figsize=(10, 6))

    # Plotting Training Loss (smoothed)
    plt.plot(train_df['step'], train_df['loss'].rolling(window=5).mean(),
             label='Training Loss (Moving Avg)', color='#1f77b4', linewidth=2)

    # Plotting Validation Loss
    if not eval_df.empty:
        plt.plot(eval_df['step'], eval_df['eval_loss'],
                 label='Validation Loss (Generalization)', color='#d62728',
                 marker='o', linestyle='--')

    plt.title(f'Convergence Analysis: {STAGE_NAME}', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # Save for LaTeX
    plt.savefig(f"{STAGE_NAME}_convergence_plot.pdf")
    plt.show()
# %% [code] cell 27
def calculate_stage_metrics(log_path, learning_rate, STAGE_NAME):
    with open(log_path, 'r') as f:
        data = json.load(f)['log_history']

    # Extract only logs that contain 'eval_loss'
    eval_logs = [log for log in data if 'eval_loss' in log]

    if not eval_logs:
        print(f"No evaluation logs found in {log_path}")
        return

    final_eval = eval_logs[-1]
    eval_loss = final_eval['eval_loss']
    perplexity = np.exp(eval_loss)

    print(f"\n🔬 --- SCIENTIFIC METRICS FOR {STAGE_NAME.upper()} ---")
    print(f"Final Validation Loss: {eval_loss:.4f}")
    print(f"Model Perplexity:      {perplexity:.4f}")
    # Logic fix: show the LR found in the file, or fallback to the provided one
    reported_lr = final_eval.get('learning_rate', learning_rate)
    print(f"Learning Rate:         {reported_lr}")
    print("-" * 40)
# %% [code] cell 28
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. PUBLICATION STYLE SETTINGS ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "figure.dpi": 300
})

def plot_professional_accuracy(json_path, stage_name="Stage 3"):
    with open(json_path, 'r') as f:
        data = json.load(f)['log_history']

    df = pd.DataFrame(data)

    # Filter for evaluation steps where accuracy proxy can be calculated
    eval_df = df[df['eval_loss'].notna()].copy()

    # Mathematical transformation to Accuracy Proxy (Predictive Confidence)
    eval_df['accuracy'] = np.exp(-eval_df['eval_loss'])

    # Professional Palette: "Forest Green" for success/accuracy
    color_main = '#2ca02c'
    color_fill = '#dbf0db'

    fig, ax = plt.subplots(figsize=(8, 5))

    # 1. Plot the Accuracy Line
    ax.plot(eval_df['epoch'], eval_df['accuracy'],
            color=color_main, linewidth=2.5, marker='s',
            markersize=7, label='Predictive Accuracy (Validation)')

    # 2. Fill area under the curve for visual weight (common in high-end papers)
    ax.fill_between(eval_df['epoch'], eval_df['accuracy'], 0,
                    color=color_fill, alpha=0.3)

    # 3. Formatting
    ax.set_title(f'Linguistic Proficiency: {stage_name.replace("_", " ").title()}')
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Accuracy Proxy (Token Likelihood)')

    # Set Y-axis limits (0 to 1.0)
    ax.set_ylim(0, 1.05)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, loc='lower right')

    plt.tight_layout()
    # Save as PDF for the dissertation
    plt.savefig(f"{stage_name}_accuracy.pdf", bbox_inches='tight')
    plt.show()
# %% [markdown] cell 29
# #### Stage 1
# %% [code] cell 30
import json
import re
import pandas as pd
from sklearn.metrics import classification_report
import os

# --- 1. CONFIGURATION ---
INPUT_FILE = f"/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/evaluation/stage_1/inference_results_stage_1.jsonl"
OUTPUT_FOLDER = f"/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/evaluation/stage_1/reports"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 2. THE LINGUISTIC REGEX ---
def get_tags_v1(text):
    """
    Surgical extraction for Stage 1.
    Targets: [* m], [* s], and [: target]
    """
    if not text or not isinstance(text, str): return []
    # This regex handles potential spacing issues (e.g. [*m] vs [* m])
    pattern = r'\[\*?\s*[ms]\]|\[::?\s+[^\]]+\]'
    raw_tags = re.findall(pattern, text)

    normalized = []
    for t in raw_tags:
        if "[:" in t: normalized.append("[:]") # Aggregate all corrections
        elif "m" in t: normalized.append("[* m]")
        elif "s" in t: normalized.append("[* s]")
    return normalized

# --- 3. PROCESSING ---
results = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        results.append(json.loads(line))

y_true = []
y_pred = []
audit_data = []

# For 'Detection' accuracy (Binary: is there a tag or not?)
y_true_detect = []
y_pred_detect = []

for res in results:
    gold = get_tags_v1(res["human_gold"])
    pred = get_tags_v1(res["model_prediction"])

    # Binary detection check
    y_true_detect.append(1 if gold else 0)
    y_pred_detect.append(1 if pred else 0)

    # Categorical alignment
    all_tags = set(gold + pred)
    for tag in all_tags:
        if tag in gold and tag in pred:
            y_true.append(tag); y_pred.append(tag)
        elif tag in gold:
            y_true.append(tag); y_pred.append("MISSING")
        else:
            y_true.append("NOT_PRESENT"); y_pred.append(tag)

    # Collect data for qualitative audit if they don't match
    if gold != pred:
        audit_data.append(res)

# --- 4. GENERATE REPORTS ---
report_categorical = classification_report(y_true, y_pred, zero_division=0, digits=3)
report_detection = classification_report(y_true_detect, y_pred_detect, zero_division=0, digits=3, target_names=["Clean", "Error"])

print(f"--- STAGE 2 FINAL METRICS ---")
print("\n1. DETECTION PERFORMANCE (Is there an error?)")
print(report_detection)
print("\n2. CATEGORICAL PERFORMANCE (Which error type?)")
print(report_categorical)

# --- 5. SAVE TO DRIVE ---
with open(f"{OUTPUT_FOLDER}/metrics_summary.txt", "w") as f:
    f.write(f"Evaluation for STAGE 2\n\n")
    f.write("DETECTION REPORT\n" + report_detection + "\n\n")
    f.write("CATEGORICAL REPORT\n" + report_categorical)

with open(f"{OUTPUT_FOLDER}/discrepancy_audit.txt", "w") as f:
    for i, item in enumerate(audit_data):
        f.write(f"[{i}] Input: {item['input']}\n")
        f.write(f"    Gold: {item['human_gold']}\n")
        f.write(f"    Pred: {item['model_prediction']}\n\n")

print(f"Reports saved to {OUTPUT_FOLDER}")
# %% [code] cell 31
# Run this for Stage 1

# --- EXECUTION SETTINGS ---
# Change these for each pass of your analysis
STAGE_NAME = "stage_1"  # or "stage_2_operational" etc.
LR_USED = "2e-4"              # The learning rate you set in SFTConfig

# Path to the trainer_state.json you saved to your Drive
# Make sure this points to the CORRECT file for the stage you are analyzing
log_file_path = f"/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/stats/{STAGE_NAME}/trainer_state.json"

# --- RUN PLOT ---
# This will display the graph and save a PDF for your dissertation
plot_scientific_convergence(log_file_path, STAGE_NAME)
# %% [code] cell 32
# --- RUN METRICS ---
# This calculates the Perplexity and reports the final loss
calculate_stage_metrics(log_file_path, LR_USED, STAGE_NAME)
plot_scientific_convergence(f"{drive_stat_path}/trainer_state.json", "stage_1")
# %% [code] cell 33
# Run the plot (Ensure path is correct)
plot_professional_accuracy(f"/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/stats/{STAGE_NAME}/trainer_state.json", "Stage_1")
# %% [markdown] cell 34
# #### Stage 2
# %% [code] cell 35
import json
import re
import os
from sklearn.metrics import classification_report

# --- 1. CONFIGURATION ---
INPUT_FILE = f"/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/evaluation/stage_2/inference_results_stage_2.jsonl"
OUTPUT_FOLDER = f"/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/evaluation/stage_2/reports"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 2. THE REFINED REGEX (Operational Level) ---
def get_tags_v2(text):
    if not text or not isinstance(text, str):
        return []

    # This regex is broader to ensure we don't miss anything,
    # then we normalize it in the loop below.
    pattern = (
        r'\[\* m:[^\]]+\]|'   # Catch any morphological tag
        r'\[\* s:r[^\]]*\]|'  # Catch any semantic/lexical tag
        r'\[::?\s+[^\]]+\]'    # Catch reconstructions
    )

    raw_tags = re.findall(pattern, text)
    normalized_tags = []

    for tag in raw_tags:
        # 1. Normalize Reconstructions to [:]
        if "[:" in tag:
            normalized_tags.append("[:]")

        # 2. Normalize Morphological to Level 2 (Operational)
        elif "[* m:" in tag:
            # We look for the operational prefix (0, +, =, ++, base, allo, vun, vsg)
            if ":0" in tag: normalized_tags.append("[* m:0]")
            elif ":+" in tag and "++" not in tag: normalized_tags.append("[* m:+]")
            elif ":++" in tag: normalized_tags.append("[* m:++]")
            elif ":=" in tag: normalized_tags.append("[* m:=]")
            elif ":base" in tag: normalized_tags.append("[* m:base]")
            elif ":allo" in tag: normalized_tags.append("[* m:allo]")
            elif ":v" in tag: normalized_tags.append("[* m:v]") # Catch vsg/vun
            elif ":sub" in tag or ":irr" in tag: normalized_tags.append("[* m:sub]")
            else: normalized_tags.append("[* m:other]")

        # 3. Normalize Semantic to Level 2
        elif "[* s:r" in tag:
            if ":gc" in tag: normalized_tags.append("[* s:r:gc]")
            else: normalized_tags.append("[* s:r]")

    return normalized_tags

# --- 3. PROCESSING ---
results = []
if not os.path.exists(INPUT_FILE):
    print(f"File not found: {INPUT_FILE}")
else:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))

y_true = []
y_pred = []
audit_data = []

y_true_detect = []
y_pred_detect = []

for res in results:
    # FIX 1: Use v2 function here!
    gold = get_tags_v2(res["human_gold"])
    pred = get_tags_v2(res["model_prediction"])

    # Binary detection check
    y_true_detect.append(1 if gold else 0)
    y_pred_detect.append(1 if pred else 0)

    # Categorical alignment
    all_tags = set(gold + pred)
    for tag in all_tags:
        if tag in gold and tag in pred:
            y_true.append(tag); y_pred.append(tag)
        elif tag in gold:
            y_true.append(tag); y_pred.append("MISSING")
        else:
            y_true.append("NOT_PRESENT"); y_pred.append(tag)

    if gold != pred:
        audit_data.append(res)

# --- 4. GENERATE REPORTS ---
report_categorical = classification_report(y_true, y_pred, zero_division=0, digits=3)
report_detection = classification_report(y_true_detect, y_pred_detect, zero_division=0, digits=3, target_names=["Clean", "Error"])

print(f"🔬 --- STAGE 2 OPERATIONAL METRICS ---")
print("\n1. DETECTION PERFORMANCE (Does the model find the error location?)")
print(report_detection)
print("\n2. OPERATIONAL LOGIC PERFORMANCE (Does it know Omission vs. Addition?)")
print(report_categorical)

# --- 5. SAVE TO DRIVE ---
with open(f"{OUTPUT_FOLDER}/metrics_summary_s2.txt", "w") as f:
    f.write(f"Evaluation for STAGE 2\n\n")
    f.write("DETECTION REPORT\n" + report_detection + "\n\n")
    f.write("OPERATIONAL CATEGORY REPORT\n" + report_categorical)

with open(f"{OUTPUT_FOLDER}/discrepancy_audit_s2.txt", "w") as f:
    for i, item in enumerate(audit_data):
        f.write(f"[{i}] Input: {item['input']}\n")
        f.write(f"    Gold: {item['human_gold']}\n")
        f.write(f"    Pred: {item['model_prediction']}\n\n")

print(f"✅ Reports saved to {OUTPUT_FOLDER}")
# %% [code] cell 36
# Run this for Stage 1

# --- EXECUTION SETTINGS ---
# Change these for each pass of your analysis
STAGE_NAME = "stage_2"  # or "stage_2_operational" etc.
LR_USED = "1e-4"              # The learning rate you set in SFTConfig

# Path to the trainer_state.json you saved to your Drive
# Make sure this points to the CORRECT file for the stage you are analyzing
log_file_path = f"/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/stats/{STAGE_NAME}/trainer_state.json"

# --- RUN PLOT ---
# This will display the graph and save a PDF for your dissertation
plot_scientific_convergence(log_file_path, STAGE_NAME)
# %% [code] cell 37
# Run the plot (Ensure path is correct)
plot_professional_accuracy(f"/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/stats/{STAGE_NAME}/trainer_state.json", "Stage_2")
# %% [code] cell 38
calculate_stage_metrics(f"{drive_stat_path}/trainer_state.json", "1e-4", STAGE_NAME)
# %% [markdown] cell 39
# #### Stage 3
# %% [code] cell 40
import json
import re
import os
from sklearn.metrics import classification_report

# --- 1. CONFIGURATION ---
STAGE_NAME = "stage_3"
# CRITICAL: Ensure this points to the STAGE 3 inference results!
INPUT_FILE = f"/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/evaluation/stage_3/inference_results_stage_3.jsonl"

def get_tags_v3_fixed(text):
    if not text or not isinstance(text, str):
        return []

    # Improved Regex: Capture the full bracket content
    pattern = (
        r'\[\* m:[^\]]+\]|'
        r'\[\* s:r[^\]]*\]|'
        r'\[::?\s+[^\]]+\]'
    )

    raw_tags = re.findall(pattern, text)
    normalized_tags = []

    for tag in raw_tags:
        # 1. Normalize Reconstructions to [:]
        if "[:" in tag:
            normalized_tags.append("[:]")

        # 2. MICRO TAGS: Force lowercase to fix the :I vs :i problem
        else:
            # We remove extra spaces and lowercase everything
            clean_tag = re.sub(r'\s+', ' ', tag).replace("[* ", "[*").replace("[*", "[* ").lower()
            normalized_tags.append(clean_tag)

    return normalized_tags

# --- 2. LOAD & VERIFY DATA ---
results = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        results.append(json.loads(line))

# DIAGNOSTIC CHECK: Print the first sentence to see if Gold actually contains Micro tags
print(f"🔬 Sample Check:")
print(f"Gold: {results[0]['human_gold']}")
print(f"Pred: {results[0]['model_prediction']}")

# --- 3. CALCULATE ---
y_true_s3 = []
y_pred_s3 = []

for res in results:
    gold = get_tags_v3_fixed(res["human_gold"])
    pred = get_tags_v3_fixed(res["model_prediction"])

    all_tags = set(gold + pred)
    for tag in all_tags:
        if tag in gold and tag in pred:
            y_true_s3.append(tag); y_pred_s3.append(tag)
        elif tag in gold:
            y_true_s3.append(tag); y_pred_s3.append("MISSING")
        else:
            y_true_s3.append("NOT_PRESENT"); y_pred_s3.append(tag)

print(f"\n📊 --- STAGE 3 CORRECTED METRICS ---")
print(classification_report(y_true_s3, y_pred_s3, zero_division=0, digits=3))
# %% [code] cell 41
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils import resample
from tqdm import tqdm

# --- 1. CONFIGURATION ---
INPUT_FILE = "/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/evaluation/stage_3/inference_results_stage_3.jsonl"
N_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95

# --- 2. FIXED STAGE 3 TAG EXTRACTOR ---
def get_tags_v3_final(text):
    if not text or not isinstance(text, str): return []
    # Regex to capture the full bracket content
    pattern = r'\[\* m:[^\]]+\]|\[\* s:r[^\]]*\]|\[::?\s+[^\]]+\]'
    raw_tags = re.findall(pattern, text)

    normalized = []
    for tag in raw_tags:
        tag = tag.lower().strip()
        if "[:" in tag:
            normalized.append("[:]") # Aggregate reconstructions
        else:
            # Standardize spacing and lowercase
            clean_tag = re.sub(r'\s+', ' ', tag).replace("[* ", "[*").replace("[*", "[* ")
            normalized.append(clean_tag)
    return normalized

# --- 3. LOAD DATA ---
results = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        results.append(json.loads(line))

# --- 4. BOOTSTRAPPING LOOP ---
all_f1_scores = {}

print(f"Starting Bootstrapping (N={N_ITERATIONS})...")

for i in tqdm(range(N_ITERATIONS)):
    # Resample sentences with replacement
    boot_sample = resample(results, replace=True, n_samples=len(results))

    y_true = []
    y_pred = []

    for res in boot_sample:
        gold_tags = get_tags_v3_final(res["human_gold"])
        pred_tags = get_tags_v3_final(res["model_prediction"])

        all_possible_tags = set(gold_tags + pred_tags)

        for tag in all_possible_tags:
            if tag in gold_tags and tag in pred_tags:
                y_true.append(tag); y_pred.append(tag)
            elif tag in gold_tags:
                y_true.append(tag); y_pred.append("MISSING")
            else:
                y_true.append("NOT_PRESENT"); y_pred.append(tag)

    # Calculate metrics for this sample
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Store F1 score for each tag found in this iteration
    for tag, metrics in report.items():
        if tag not in all_f1_scores:
            all_f1_scores[tag] = []
        if isinstance(metrics, dict): # Skip 'accuracy' key which is a float
            all_f1_scores[tag].append(metrics['f1-score'])

# --- 5. CALCULATE STATS ---
final_stats = []
lower_p = ((1 - CONFIDENCE_LEVEL) / 2) * 100
upper_p = (CONFIDENCE_LEVEL + (1 - CONFIDENCE_LEVEL) / 2) * 100

for tag, scores in all_f1_scores.items():
    # Filter out helper labels and low-support metrics
    if tag in ["accuracy", "macro avg", "weighted avg", "MISSING", "NOT_PRESENT"]:
        continue

    mean_f1 = np.mean(scores)
    ci_low = np.percentile(scores, lower_p)
    ci_high = np.percentile(scores, upper_p)

    final_stats.append({
        "CHAT Tag": tag,
        "Mean F1": mean_f1,
        "95% CI Lower": ci_low,
        "95% CI Upper": ci_high,
        "Stability (Range)": ci_high - ci_low
    })

# --- 6. DISPLAY & SAVE ---
df_boot = pd.DataFrame(final_stats).sort_values("Mean F1", ascending=False)
print("\n--- BOOTSTRAPPED STAGE 3 RESULTS ---")
print(df_boot.to_string(index=False))

# Save for LaTeX table generation
df_boot.to_csv("stage3_bootstrapped_metrics.csv", index=False)
print(f"\nResults saved to stage3_bootstrapped_metrics.csv")
# %% [code] cell 42
# Run the plot (Ensure path is correct)
plot_professional_accuracy(f"/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/stats/{STAGE_NAME}/trainer_state.json", "Stage_3")
# %% [code] cell 43
# Run this for Stage 1

# --- EXECUTION SETTINGS ---
# Change these for each pass of your analysis
STAGE_NAME = "stage_3"  # or "stage_2_operational" etc.
LR_USED = "5e-5"              # The learning rate you set in SFTConfig

# Path to the trainer_state.json you saved to your Drive
# Make sure this points to the CORRECT file for the stage you are analyzing
log_file_path = f"/content/drive/MyDrive/00-09.PhDWORK/Projects/03.02.Year_2/CLAN_annotator/stats/{STAGE_NAME}/trainer_state.json"

# --- RUN PLOT ---
# This will display the graph and save a PDF for your dissertation
plot_scientific_convergence(log_file_path, STAGE_NAME)
# %% [code] cell 44
calculate_stage_metrics(f"{drive_stat_path}/trainer_state.json", "5e-5", STAGE_NAME)
# %% [markdown] cell 45
# #### Lexical Overlap Audit
# %% [code] cell 46
import pandas as pd
import re
from difflib import SequenceMatcher
from tqdm import tqdm

def normalize_for_audit(text):
    # Remove all punctuation and lowercase to find "semantic" duplicates
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return " ".join(text.lower().split())

def run_lexical_audit(train_ds, eval_ds, test_ds):
    print("🔬 --- STARTING SCIENTIFIC DATA AUDIT ---")

    # 1. Extract inputs
    train_inputs = [item['input'] for item in train_ds]
    eval_inputs = [item['input'] for item in eval_ds]
    test_inputs = [item['input'] for item in test_ds]

    # 2. Create Normalized Sets for fast O(1) lookup
    train_norm_set = set(normalize_for_audit(s) for s in train_inputs)

    results = []

    for name, subset in [("Validation", eval_inputs), ("Test", test_inputs)]:
        exact_matches = 0
        norm_matches = 0
        total = len(subset)

        train_set = set(train_inputs)

        for s in subset:
            # Check Exact
            if s in train_set:
                exact_matches += 1

            # Check Normalized (Semantic Duplicates)
            if normalize_for_audit(s) in train_norm_set:
                norm_matches += 1

        results.append({
            "Split": name,
            "Total N": total,
            "Exact Leakage": f"{exact_matches} ({round(exact_matches/total*100, 2)}%)",
            "Normalized Leakage": f"{norm_matches} ({round(norm_matches/total*100, 2)}%)"
        })

    # 3. Report Results
    audit_df = pd.DataFrame(results)
    print("\n📊 --- LEAKAGE SUMMARY ---")
    print(audit_df.to_string(index=False))

    # 4. Qualitative Check: Print 5 examples of Normalized Leakage if they exist
    print("\n🔍 --- QUALITATIVE LEAKAGE PREVIEW (Test vs Train) ---")
    found = 0
    for s in test_inputs:
        norm_s = normalize_for_audit(s)
        if norm_s in train_norm_set:
            # Find the original in train to show the comparison
            # (Finding first match in list for display)
            train_match = next(t for t in train_inputs if normalize_for_audit(t) == norm_s)
            print(f"Test Item:  {repr(s)}")
            print(f"Train Match: {repr(train_match)}")
            print("-" * 20)
            found += 1
        if found >= 5: break

    if found == 0:
        print("✅ No lexical overlaps detected.")

# --- EXECUTION ---
run_lexical_audit(dataset_s3['train'], dataset_s3['eval'], dataset_s3['test'])
# %% [code] cell 47
from collections import Counter

def audit_internal_train(train_ds):
    print("🔬 --- INTERNAL TRAINING SET AUDIT ---")

    train_inputs = [item['input'] for item in train_ds]
    total_n = len(train_inputs)

    # Count occurrences of each unique sentence
    counts = Counter(train_inputs)

    # Find sentences that appear more than once
    duplicates = {k: v for k, v in counts.items() if v > 1}
    unique_count = len(counts)
    redundant_count = total_n - unique_count

    print(f"Total Sentences: {total_n}")
    print(f"Unique Sentences: {unique_count}")
    print(f"Redundant (Duplicate) rows: {redundant_count} ({round(redundant_count/total_n*100, 2)}%)")

    if duplicates:
        print("\n🔥 Top 5 Most Redundant Sentences in Train:")
        sorted_dupes = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)
        for s, count in sorted_dupes:
            print(f"   [{count} times]: {repr(s)}")

    return duplicates

# EXECUTION
train_dupes = audit_internal_train(dataset_s3['train'])
