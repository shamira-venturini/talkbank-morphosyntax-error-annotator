# Experiment 04 (Unseen-Tag Compositional Generalization)

## Goal
Test whether the model can compositionally generalize to valid detailed labels that are withheld from training.

## Held-out labels
- `[* m:++er]`
- `[* m:++est]`
- `[* m:0er]`
- `[* m:0est]`

## Setup command
```bash
python3 scripts/setup_experiment4.py
```

This creates:
- `experiments/exp4_unseen_tags`
- `experiments/exp4_unseen_tags/experiment4_summary.json`

## Training/inference
Use:
`experiments/acl_rr_v1/ACL_SFT_CLAN_Llama3_1_8B_ACL.ipynb`

Change:
```python
cfg.split_dir = "experiments/exp4_unseen_tags"
cfg.seed = 3407  # then 3408, 3409
```

Evaluate on:
- `holdout_generalization` (primary for this experiment)

## Post-hoc metrics
After inference, run:
```bash
python3 scripts/evaluate_experiment4.py \
  --predictions /path/to/predictions_holdout_generalization.jsonl \
  --experiment-dir experiments/exp4_unseen_tags
```

Reported metrics include:
- exact held-out label accuracy
- operator-family accuracy
- invalid-label rate
