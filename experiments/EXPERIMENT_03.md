# Experiment 03 (Analysis-by-Synthesis Ablation)

## Goal
Test whether reconstruction supervision (analysis-by-synthesis) improves annotation robustness.

## Variants
- `experiments/exp3_abs_on_manual`
  - Reconstruction mode: `preserve`
  - Interpretation: AbS-on (manual `[:]` / `[::]` distinction intact)
- `experiments/exp3_abs_off_no_recon`
  - Reconstruction mode: `drop_all`
  - Interpretation: AbS-off (no reconstruction markers)
- `experiments/exp3_abs_diag_nonword_only`
  - Reconstruction mode: `nonword_only`
  - Interpretation: Mechanism diagnostic (keep `[:]`, drop `[::]`)

All three variants keep:
- same split seed
- same train/eval/test/coverage/holdout partitioning policy
- same 50/50 real-synthetic train ratio

## Setup command
```bash
python3 scripts/setup_experiment3.py
```

## Run in notebook
Use:
`experiments/acl_rr_v1/ACL_SFT_CLAN_Llama3_1_8B_ACL.ipynb`

Change only:
```python
cfg.split_dir = "experiments/exp3_abs_on_manual"  # or exp3_abs_off_no_recon / exp3_abs_diag_nonword_only
cfg.seed = 3407  # then 3408, 3409
```
