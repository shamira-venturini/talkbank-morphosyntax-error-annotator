# talkbank-morphosyntax-error-annotator

Utilities, frozen experiment packages, and paper assets for automatic morphosyntactic error annotation in TalkBank CHAT data.

## Recommended Entry Points

- first submitted TalkBank study:
  `study_01_talkbank_tool_paper/`
- second human-in-the-loop adaptation study:
  `study_02_hitl_adaptation/`
- fourth context-window study:
  `study_04_context_windows/`

## Canonical Structure

- `study_01_talkbank_tool_paper/`: top-level entry point for the frozen first study
- `study_02_hitl_adaptation/`: top-level entry point for the current second study
- `study_04_context_windows/`: top-level entry point for context-window and prompt-context experiments
- `data/`: shared canonical data pipeline and audits
- `scripts/`: compatibility copy of study-owned scripts; prefer the study-local `scripts/` folders
- `studies/01_confirmatory_annotation/`: frozen study-specific navigation layer for the first study
- `studies/02_uncertainty_and_feedback/`: study-specific navigation layer for the second study
- `studies/03_compositional_target_recovery/`: compositionality and latent target-recovery follow-up work
- `studies/04_context_windows/`: context-window ablations and runtime-context experiments
- `docs/`: cross-study framing, repo map, experiment registry, and data lineage notes
- `reviews/`: active human-reviewed CSVs and blinded review bundles
- `archive/`: legacy or retired material kept for traceability
- `handoff/`: deployment-facing package material

## Layout Note

Scripts and results are now organized under the study entry-point folders:

- `study_01_talkbank_tool_paper/scripts/`
- `study_01_talkbank_tool_paper/results/`
- `study_02_hitl_adaptation/scripts/`
- `study_02_hitl_adaptation/results/`
- `study_04_context_windows/scripts/`
- `study_04_context_windows/results/`

The legacy top-level `scripts/` and `results/` paths remain available as compatibility copies so older notebooks, manifests, and ad hoc commands still resolve.

## Current Canonical Assets

- Confirmatory experiment package:
  `studies/01_confirmatory_annotation/experiment_packages/recon_full_comp_preserve/`
- Confirmatory final run:
  `studies/01_confirmatory_annotation/results/recon_full_comp_preserve_final_seed3407/`
- Uncertainty follow-up assets:
  `studies/02_uncertainty_and_feedback/`
- Compositional / target-recovery follow-up assets:
  `studies/03_compositional_target_recovery/`
- Context-window ablation assets:
  `studies/04_context_windows/`

## Where To Start

- First study entry point: `study_01_talkbank_tool_paper/`
- Second study entry point: `study_02_hitl_adaptation/`
- Context-window study entry point: `study_04_context_windows/`
- Repo map: `docs/REPO_MAP.md`
- Experiment status and canonical paths: `docs/EXPERIMENT_REGISTRY.md`
- Shared scientific framing: `docs/PROJECT_AIM_AND_RQS.md`
- Shared data lineage: `docs/DATA_LINEAGE.md`
- Study A protocol: `studies/02_uncertainty_and_feedback/study_a_human_in_the_loop_protocol.md`

## Script Usage

All scripts use repo-relative paths by default.

Examples:

```bash
python3 study_01_talkbank_tool_paper/scripts/build_acl_splits.py \
  --input data/processed/master_training.jsonl \
  --out-dir studies/01_confirmatory_annotation/experiment_packages/recon_full_comp_preserve

python3 study_01_talkbank_tool_paper/scripts/analyze_paper_results.py
python3 study_01_talkbank_tool_paper/scripts/analyze_prediction_uncertainty.py
python3 study_01_talkbank_tool_paper/scripts/setup_compositional_reconstruction_probe.py
```
