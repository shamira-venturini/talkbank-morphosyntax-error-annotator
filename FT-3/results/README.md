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
