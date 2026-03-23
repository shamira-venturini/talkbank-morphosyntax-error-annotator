# talkbank-morphosyntax-error-annotator

Utilities, frozen experiment packages, and paper assets for automatic morphosyntactic error annotation in TalkBank CHAT data.

## Canonical Structure

- `data/`: shared canonical data pipeline and audits
- `scripts/`: shared reproducible build, setup, evaluation, and analysis scripts
- `studies/01_confirmatory_annotation/`: current confirmatory paper line and final model assets
- `studies/02_uncertainty_and_feedback/`: uncertainty, review-triage, and future feedback work
- `studies/03_compositional_target_recovery/`: compositionality and latent target-recovery follow-up work
- `docs/`: cross-study framing, repo map, experiment registry, and data lineage notes
- `archive/`: legacy or retired material kept for traceability
- `handoff/`: deployment-facing package material

## Layout Note

The large tracked experiment/result payloads remain at their historical top-level paths under `experiments/`, `results/`, `artifacts/`, and `docs/` so Git can manage them normally. The `studies/` tree is the human-facing navigation layer and links back to those canonical tracked assets.

## Current Canonical Assets

- Confirmatory experiment package:
  `studies/01_confirmatory_annotation/experiment_packages/recon_full_comp_preserve/`
- Confirmatory final run:
  `studies/01_confirmatory_annotation/results/recon_full_comp_preserve_final_seed3407/`
- Uncertainty follow-up assets:
  `studies/02_uncertainty_and_feedback/`
- Compositional / target-recovery follow-up assets:
  `studies/03_compositional_target_recovery/`

## Where To Start

- Repo map: `docs/REPO_MAP.md`
- Experiment status and canonical paths: `docs/EXPERIMENT_REGISTRY.md`
- Shared scientific framing: `docs/PROJECT_AIM_AND_RQS.md`
- Shared data lineage: `docs/DATA_LINEAGE.md`

## Script Usage

All scripts use repo-relative paths by default.

Examples:

```bash
python3 scripts/build_acl_splits.py \
  --input data/processed/master_training.jsonl \
  --out-dir studies/01_confirmatory_annotation/experiment_packages/recon_full_comp_preserve

python3 scripts/analyze_paper_results.py
python3 scripts/analyze_prediction_uncertainty.py
python3 scripts/setup_compositional_reconstruction_probe.py
```
