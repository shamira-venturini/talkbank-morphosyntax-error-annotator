# Experiment 01 (Direct Stage-3 Baseline)

## Goal
Establish the primary baseline with controlled train composition.

## Fixed data package
- Split directory: `experiments/acl_rr_v1`
- Train: 50/50 real-synthetic
- Eval/Test: real-only
- Coverage: synthetic-only (`eval_coverage`, `test_coverage`)
- Holdout: synthetic-only (`[* m:++er]`, `[* m:++est]`, `[* m:0er]`, `[* m:0est]`)
- Reconstruction format: preserve manual distinction (`[: target]` vs `[:: target]`)

## Training setup
- Direct Stage-3 (`run_full_curriculum=False`, `single_stage=3`)
- LoRA + Unsloth config from notebook defaults
- Split-aware reporting thresholds from protocol:
  - real `eval` / `test`: >= 20 (confirmatory)
  - `eval_coverage` / `test_coverage`: >= 10 (diagnostic confirmatory)
  - `holdout`: >= 10 (generalization diagnostic)

## Reporting guardrails
- Primary confirmatory interpretation should rely on real-only aggregate metrics (`test_real`) and confidence intervals.
- Per-label conclusions on real-only splits should be limited to labels meeting >= 20 support.
- Low-support labels on real-only splits should be discussed as exploratory.
- Coverage and holdout splits are secondary diagnostics and should not be presented as replacements for real-only confirmatory evidence.

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
