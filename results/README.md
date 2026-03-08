This directory stores small, GitHub-safe result bundles exported from Colab runs.

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
