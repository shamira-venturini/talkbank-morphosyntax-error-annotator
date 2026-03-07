# Experiment 01 (Direct Stage-3 Baseline)

## Goal
Establish the primary baseline with controlled train composition.

## Fixed data package
- Split directory: `FT-3/experiments/acl_rr_v1`
- Train: 50/50 real-synthetic
- Eval/Test: real-only
- Coverage: synthetic-only (`eval_coverage`, `test_coverage`)
- Holdout: synthetic-only (`[* m:++er]`, `[* m:++est]`, `[* m:0er]`, `[* m:0est]`)
- Reconstruction format: preserve manual distinction (`[: target]` vs `[:: target]`)

## Training setup
- Direct Stage-3 (`run_full_curriculum=False`, `single_stage=3`)
- LoRA + Unsloth config from notebook defaults
- Label support threshold: 10 (confirmatory threshold for this baseline)

## Seeds
Run exactly 3 seeds:
- 3407
- 3408
- 3409

In the notebook config cell, change only:
```python
cfg.seed = 3407  # then 3408, then 3409
```

Each seed run should write outputs under:
- `/content/outputs/acl_rr_exp1_direct_stage3_seed<SEED>`

No overwrite occurs across seeds because output path is seed-specific.
