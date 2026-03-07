# Colab Experiment Guide

This guide is for running the project experiments in the simplest possible way.

You do not need to understand all the code to use it.

The basic idea is:

1. Keep the data and experiment files in this GitHub repo.
2. Open the notebook in Google Colab.
3. Change a few settings in one config cell.
4. Run the notebook.
5. Save the results.

## Before You Start

You need:

- a GitHub account
- a Google account for Colab and Google Drive
- this repo pushed to GitHub
- the notebook file in the repo:
  - `experiments/acl_rr_v1/ACL_SFT_CLAN_Llama3_1_8B_ACL.ipynb`

Optional:

- a Hugging Face account if you want to upload trained model adapters

## Very Short Version

If you want the shortest possible version:

1. Open the notebook in Colab from GitHub.
2. Set runtime to `GPU`.
3. In the config cell:
   - choose `split_dir`
   - choose `seed`
   - choose `output_root`
   - set `push_to_hub = False` unless you are ready to use Hugging Face
4. Run the notebook cells from top to bottom.
5. Your results will be saved to:
   - Google Drive
   - the cloned repo inside Colab under `results/`

## Step 1: Push Your Latest Repo Changes

Colab reads the notebook from GitHub, not from your local PyCharm copy.

That means:

- if you changed the notebook locally, push those changes to GitHub first
- if you do not push, Colab will open the old version from GitHub

## Step 2: Open the Notebook in Colab

Use this URL pattern:

`https://colab.research.google.com/github/<your-user>/<your-repo>/blob/main/experiments/acl_rr_v1/ACL_SFT_CLAN_Llama3_1_8B_ACL.ipynb`

For this repo, the URL is:

`https://colab.research.google.com/github/shamira-venturini/talkbank-morphosyntax-error-annotator/blob/main/experiments/acl_rr_v1/ACL_SFT_CLAN_Llama3_1_8B_ACL.ipynb`

Alternative way:

1. Go to `https://colab.research.google.com`
2. Click `File`
3. Click `Open notebook`
4. Click the `GitHub` tab
5. Search for `shamira-venturini/talkbank-morphosyntax-error-annotator`
6. Open the notebook in `experiments/acl_rr_v1/`

## Step 3: Turn On GPU

In Colab:

1. Click `Runtime`
2. Click `Change runtime type`
3. Set `Hardware accelerator` to `GPU`
4. Click `Save`

If the notebook says CUDA or GPU is missing, this step was not done correctly.

## Step 4: Run Cell 1 - Environment Setup

This cell:

- clones the repo into Colab
- installs packages
- moves into the repo folder

You do not usually need to edit this cell.

## Step 5: Edit Cell 2 - Config

This is the main cell you edit.

If you are unsure, only change these fields:

- `split_dir`
- `seed`
- `output_root`
- `drive_root`
- `push_to_hub`
- `hf_repo_prefix` if you are using Hugging Face

Leave the rest alone unless you know why you are changing them.

## Safe Baseline Config

Use this for a normal first run:

```python
split_dir: str = "experiments/acl_rr_v1"
seed: int = 3407
output_root: str = "/content/outputs/acl_rr_exp1_direct_stage3"
save_to_drive: bool = True
save_to_repo_results: bool = True
push_to_hub: bool = False
```

Notes:

- `push_to_hub = False` is the safest default
- the notebook will automatically append `_seed3407` to `output_root`
- if you change the seed, the output name changes too

## Step 6: Run Cell 2b - Drive and Hugging Face Setup

If `save_to_drive = True`, Colab will ask to mount Google Drive.

Do that.

If `push_to_hub = False`, you do not need to do anything else here.

If `push_to_hub = True`, the notebook expects a Colab secret called `HF_TOKEN`.

If you do not know what that is yet, keep `push_to_hub = False`.

## Step 7: Run the Notebook Top to Bottom

After the config cells, run the rest of the notebook in order.

The notebook does:

1. load model and tokenizer
2. attach LoRA
3. load the dataset splits
4. train
5. run inference
6. compute metrics
7. save results

The main training/evaluation flow is already written in the notebook.

## Step 8: Find Your Results

After the run finishes, look in these places:

### In Colab / cloned repo

The notebook writes a small result bundle into:

- `results/<run_name>/`

This includes:

- `run_summary.json`
- `paper_main_metrics.csv`
- `metrics_*.json`
- `predictions_*.jsonl` for selected splits

### In Google Drive

If `save_to_drive = True`, results also go to:

- `<drive_root>/<run_name>/`

The notebook also saves stage metrics and adapter files there.

## What `run_name` Means

The notebook builds a run name from `output_root`.

Example:

- if `output_root = "/content/outputs/acl_rr_exp1_direct_stage3"`
- and `seed = 3407`

then the real run folder becomes something like:

- `acl_rr_exp1_direct_stage3_seed3407`

This is why different seeds do not overwrite each other.

## Experiment 1: Baseline

Use:

```python
split_dir: str = "experiments/acl_rr_v1"
output_root: str = "/content/outputs/acl_rr_exp1_direct_stage3"
```

Run exactly these seeds:

- `3407`
- `3408`
- `3409`

Do one run per seed.

## Experiment 3: Ablation Study

Before opening Colab, create the experiment packages locally in PyCharm:

```bash
python3 scripts/setup_experiment3.py
```

Then push the repo to GitHub.

In Colab, run the notebook three separate times, once for each variant.

### Variant A: AbS On

```python
split_dir: str = "experiments/exp3_abs_on_manual"
output_root: str = "/content/outputs/exp3_abs_on_manual"
```

### Variant B: AbS Off

```python
split_dir: str = "experiments/exp3_abs_off_no_recon"
output_root: str = "/content/outputs/exp3_abs_off_no_recon"
```

### Variant C: Diagnostic

```python
split_dir: str = "experiments/exp3_abs_diag_nonword_only"
output_root: str = "/content/outputs/exp3_abs_diag_nonword_only"
```

For each variant, run seeds:

- `3407`
- `3408`
- `3409`

## Experiment 4: Unseen-Tag Generalization

Before opening Colab, create the experiment package locally in PyCharm:

```bash
python3 scripts/setup_experiment4.py
```

Then push the repo to GitHub.

In Colab use:

```python
split_dir: str = "experiments/exp4_unseen_tags"
output_root: str = "/content/outputs/exp4_unseen_tags"
```

Run seeds:

- `3407`
- `3408`
- `3409`

Important:

- for this experiment, `holdout_generalization` is the main special split to inspect

Optional post-hoc evaluation from PyCharm:

```bash
python3 scripts/evaluate_experiment4.py \
  --predictions /path/to/predictions_holdout_generalization.jsonl \
  --experiment-dir experiments/exp4_unseen_tags
```

## Hugging Face: Only If You Want To Upload Models

You do not need Hugging Face to run the experiments.

Use it only if you want to upload the trained adapter/tokenizer.

Recommended config:

```python
push_to_hub: bool = True
hf_repo_prefix: str = "shamira-venturini/clan-annotator-exp1"
hf_repo_visibility: str = "private"
hf_repo_include_run_name: bool = True
hf_append_stage_suffix: bool = False
hf_repo_final_alias: str = ""
```

This creates one clean Hugging Face model repo per run.

Example output repo name:

- `shamira-venturini/clan-annotator-exp1-acl_rr_exp1_direct_stage3_seed3407`

If you want one stable alias for your best model, set:

```python
hf_repo_final_alias: str = "shamira-venturini/clan-annotator-best"
```

## Common Mistakes

### Mistake 1: Colab is opening the wrong notebook version

Cause:

- local changes were not pushed to GitHub

Fix:

- commit and push first

### Mistake 2: The notebook cannot find the split directory

Cause:

- `split_dir` is wrong
- experiment setup scripts were not run locally
- setup changes were not pushed to GitHub

Fix:

- check the path in the config cell
- if using Exp3 or Exp4, run the setup script locally first and push

### Mistake 3: Hugging Face error

Cause:

- `push_to_hub = True` but no `HF_TOKEN`
- invalid `hf_repo_prefix`

Fix:

- easiest fix is `push_to_hub = False`

### Mistake 4: Out of memory

Cause:

- Colab GPU memory is too small for the current batch size

Fix:

- reduce:
  - `per_device_train_batch_size`
- increase:
  - `gradient_accumulation_steps`

Simple fallback:

```python
per_device_train_batch_size: int = 4
gradient_accumulation_steps: int = 8
```

### Mistake 5: Runs overwrite each other

Cause:

- same `output_root`
- same `seed`

Fix:

- use a different `output_root` for different experiments
- use different seeds for repeated runs

## Recommended First Run

If you want the least confusing path:

1. Open the notebook in Colab.
2. Turn on GPU.
3. Use the baseline config:
   - `split_dir = "experiments/acl_rr_v1"`
   - `seed = 3407`
   - `output_root = "/content/outputs/acl_rr_exp1_direct_stage3"`
   - `push_to_hub = False`
4. Run all cells.
5. Confirm that:
   - Drive contains a new run folder
   - `results/` contains a new result bundle

## After You Finish a Run

Write down:

- experiment name
- split directory
- seed
- output root
- where the results were saved

This matters because the only difference between many runs is a small config change.

## If You Want a Simple Rule

Use this rule:

- PyCharm is for preparing experiment files and pushing to GitHub
- Colab is for training and evaluation

That mental model is enough for most runs in this project.
