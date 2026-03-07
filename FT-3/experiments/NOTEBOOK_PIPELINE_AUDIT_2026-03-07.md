# Notebook and Pipeline Audit (2026-03-07)

## Scope
Audit target:
- `FT-3/experiments/acl_rr_v1/ACL_SFT_CLAN_Llama3_1_8B_ACL.ipynb`
- split builders and experiment setup scripts for Exp1/3/4/6

Checks performed:
- split integrity, leakage, duplicate inputs, JSONL validity
- prompt-label policy consistency and holdout generalization validity
- metric extractability for paper reporting
- persistence policy (HF + Drive) under constrained storage
- runtime feasibility on NVIDIA A1000-class GPU

## Findings (ordered by severity)

### 1) Critical (fixed): holdout generalization prompt confound
Issue:
- Holdout examples were using the same train-derived allowed-label prompt, which excluded held-out labels and could suppress correct outputs in Exp4.

Fix implemented:
- `scripts/set_experiment_prompts.py` now supports split-specific holdout label sources and writes a separate holdout prompt.
- Exp3/Exp4 setup now uses:
  - `--label-source-splits train`
  - `--holdout-label-source-splits train,holdout`

Code refs:
- `/Users/shamiraventurini/PycharmProjects/CLAN-annotator/scripts/set_experiment_prompts.py:100`
- `/Users/shamiraventurini/PycharmProjects/CLAN-annotator/scripts/set_experiment_prompts.py:125`
- `/Users/shamiraventurini/PycharmProjects/CLAN-annotator/scripts/set_experiment_prompts.py:171`
- `/Users/shamiraventurini/PycharmProjects/CLAN-annotator/scripts/setup_experiment4.py:102`
- `/Users/shamiraventurini/PycharmProjects/CLAN-annotator/scripts/setup_experiment3.py:77`

Validation:
- `stage3_train` prompt excludes held-out labels; `stage3_holdout` prompt includes them.

### 2) High: remaining source-label confound risk (not a bug, design risk)
Observed in `acl_rr_v1` Stage-3 train:
- 37 labels in train.
- 11 labels are synthetic-only in train (0 real support).
- 17 labels are train-only relative to real eval/test label set.

Reviewer risk:
- Performance on many labels can be attributed to synthetic distribution rather than robust transfer to naturalistic data.

Mitigation strategy:
- Keep confirmatory claims on real-only `test_real` labels with adequate support.
- Treat synthetic-only/coverage results as secondary diagnostics.
- Add source-stratified metrics in reporting (TD vs DLD vs synthetic where applicable).

### 3) High: low per-label support in real test/eval for confirmatory claims
Observed supports:
- `eval_real`: 20 labels, 17 labels have support <10, 18 labels <20.
- `test_real`: 17 labels, 13 labels <10, 15 labels <20.

Reviewer risk:
- Per-label comparisons are unstable for many labels.

Mitigation strategy:
- Keep split-aware thresholds (`20` real confirmatory, `10` coverage/holdout).
- Report low-support labels as exploratory only.
- Keep micro-F1 + CI as primary performance statistic.

### 4) Medium: real holdout row drop due synthetic-only holdout policy
Observed:
- 1 TD row has a held-out label but is removed from holdout due `holdout_synthetic_only` policy.

Risk:
- Small but should be disclosed to avoid confusion in row accounting.

Mitigation:
- Document explicitly in methods; optional sensitivity run with `--allow-real-in-holdout`.

### 5) Medium: metric granularity limitation
Current evaluation computes label-set overlap per sentence (presence/absence), not span-level tag placement.

Risk:
- Overestimates when repeated same label occurs multiple times in one sentence.

Mitigation:
- Keep as primary if consistent across experiments; disclose as limitation.
- Optional future extension: occurrence-aware scoring.

## Pipeline integrity status
Passed:
- JSONL parse check across all stage files in `acl_rr_v1`, `exp3_*`, `exp4_unseen_tags`.
- No duplicate inputs within splits.
- No input overlap train vs non-train.
- Off-schema dropped label `[* m:+s]` absent from experiment packages.

## Notebook improvements implemented

### Reporting and extraction
- Added automatic per-split CSV exports for confirmatory and exploratory per-label metrics.
- Added run-level compact CSV: `paper_main_metrics.csv`.
- Added source-stratified summary in metrics (`by_source`).

Code refs:
- `/Users/shamiraventurini/PycharmProjects/CLAN-annotator/scripts/generate_acl_colab_notebook.py:532`
- `/Users/shamiraventurini/PycharmProjects/CLAN-annotator/scripts/generate_acl_colab_notebook.py:594`
- `/Users/shamiraventurini/PycharmProjects/CLAN-annotator/scripts/generate_acl_colab_notebook.py:509`

### Statistical defaults aligned to protocol
- Confirmatory threshold default raised to `20` for real splits.

Code ref:
- `/Users/shamiraventurini/PycharmProjects/CLAN-annotator/scripts/generate_acl_colab_notebook.py:114`

### Persistence and storage safety
- Added `run_name`-scoped Drive paths to prevent overwrite across experiments/seeds.
- Added controlled HF pushing: push final stage by default (`push_all_stages=False`), optional all-stage push.
- Maintained minimal artifact sync to Drive (metrics + adapter essentials + selected predictions).

Code refs:
- `/Users/shamiraventurini/PycharmProjects/CLAN-annotator/scripts/generate_acl_colab_notebook.py:156`
- `/Users/shamiraventurini/PycharmProjects/CLAN-annotator/scripts/generate_acl_colab_notebook.py:337`
- `/Users/shamiraventurini/PycharmProjects/CLAN-annotator/scripts/generate_acl_colab_notebook.py:345`
- `/Users/shamiraventurini/PycharmProjects/CLAN-annotator/scripts/generate_acl_colab_notebook.py:609`

## Runtime estimate (NVIDIA A1000 class)
Assumptions from notebook defaults:
- Train rows: 6406
- Effective batch: `8 x 4 = 32`
- Steps per epoch: ~201
- Stage3 direct (3 epochs): ~603 train steps
- Full curriculum (2+2+3 epochs): ~1407 train steps total

Estimated wall-clock (A1000, depends on VRAM/power limit):
- Direct Stage3 single seed: ~35-70 min
- Full 3-stage curriculum single seed: ~90-180 min
- 3 seeds direct Stage3: ~2-4 hours

If VRAM is 8 GB, reduce batch to avoid OOM:
- `per_device_train_batch_size=4` (or 2), increase grad accumulation to keep effective batch.

## Run strategy recommendation
Use blocks, not full notebook all-at-once:
1. Setup + model load
2. Training block
3. Evaluation block
4. Export/sync block

Why:
- Easier recovery from Colab disconnects.
- Lower risk of losing run artifacts.
- Better GPU utilization monitoring and checkpoint sanity checks.

## HF + Drive reproducibility policy (recommended)
- HF: push final-stage adapter/tokenizer per run (default). Enable `push_all_stages=True` only when needed.
- Drive: keep only
  - `run_summary.json`
  - `paper_main_metrics.csv`
  - `metrics_*.json`
  - `per_label_confirmatory_*.csv`
  - `per_label_exploratory_*.csv`
  - selected `predictions_*.jsonl` (`test_real`, `holdout_generalization`)
  - stage train metrics and adapter essentials

This is sufficient for reproducibility/transparency with minimal storage.

## Final status
- Pipeline is runnable and materially cleaner for ACL-facing review.
- The most serious confound (holdout prompt suppression) is fixed.
- Main remaining reviewer risk is statistical support and source-skew interpretation, which should be handled in claims framing and reporting tables.
