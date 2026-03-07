# Experiment 06 (Stability Confirmation)

## Goal
Check whether the main performance differences survive seed variation.

## Design
Use the best 2 or 3 systems from Experiments 1 to 4.

For each selected system:
- run exactly 3 seeds
- keep data split and configuration fixed except for `cfg.seed`
- aggregate `test_real` as the primary stability split
- optionally aggregate `holdout_generalization` for Experiment 4 systems

## Run in notebook
Use:
`FT-3/experiments/acl_rr_v1/ACL_SFT_CLAN_Llama3_1_8B_ACL.ipynb`

For each seed:
```python
cfg.seed = 3407  # then 3408, 3409
```

The notebook already writes seed-specific output directories.

## Aggregate results
Example:
```bash
python3 scripts/aggregate_experiment6.py \
  --system baseline=/runs/exp1_seed3407,/runs/exp1_seed3408,/runs/exp1_seed3409 \
  --system improved=/runs/exp4_seed3407,/runs/exp4_seed3408,/runs/exp4_seed3409 \
  --split test_real \
  --split holdout_generalization \
  --out FT-3/experiments/experiment6_stability_summary.json
```

Accepted run paths:
- a run root containing `eval_outputs/metrics_<split>.json`
- or a direct metrics JSON path

## Report
For each system and split, report:
- mean and standard deviation of `micro_f1`
- mean exact match
- bootstrap CI mean from per-seed runs
- ranking by mean performance
