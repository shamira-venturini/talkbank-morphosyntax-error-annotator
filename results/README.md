This directory is now a compatibility copy of study-owned result bundles.

Canonical locations:
- `study_01_talkbank_tool_paper/results/`
- `study_02_hitl_adaptation/results/`
- `study_04_context_windows/results/`

The legacy top-level subdirectories remain available as compatibility copies so older scripts and manifests keep working.

Status update (2026-03-08):
- Pre-retraining outputs generated with the old prompt constraint rules were moved to
  `results/_obsolete_prompt_whitelist_20260308/`.
- That folder is archival only and excluded from version control.
- Treat those runs as non-canonical; do not use them for paper-facing comparisons.
- Use this `results/` root for reruns produced after prompt cleanup.

Expected contents per run:
- `run_summary.json`
- `paper_main_metrics.csv`
- `metrics_*.json`
- `per_label_confirmatory_*.csv`
- `per_label_exploratory_*.csv`
- selected prediction files for qualitative analysis
- `stage*_train_metrics.json`
- `results_bundle_manifest.json`

Do not store model weights or large checkpoints here.

Paper analysis workflow:
- run `python3 scripts/analyze_paper_results.py`
- output directory: `results/paper_analysis/`

Generated analysis files:
- `run_metrics.csv` (exact-match, micro-F1, macro-F1, CI, confirmatory/exploratory counts)
- `syntax_validity.csv` (error-tag syntax validity and invalid-label rate)
- `top_confusions.csv`
- `focus_confusion_03s_vs_0ed.csv` (targeted ambiguity diagnostic)
- `reconstruction_diagnostics.csv` (reconstruction marker diagnostics)
- `holdout_generalization_diagnostics.csv` (held-out label/operator diagnostics)
- `seed_stability_summary.csv` (mean/std by seed-grouped systems)
- `overfitting_report.csv` (eval->test generalization gap proxy)
- `paper_analysis_summary.json`
