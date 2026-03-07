# talkbank-morphosyntax-error-annotator

Utilities and experiment packages for morphosyntactic error annotation in TalkBank CHAT data.

## Repository Layout

- `data/curated/`: curated synthetic and real example files
- `data/processed/`: canonical training files with provenance and error counts
- `data/intermediate/`: preparation snapshots kept for traceability
- `data/audits/`: paper tables, label audits, and review sets
- `data/norming/`: norming assets and helper scripts
- `experiments/`: frozen ACL-ready split packages and runbooks
- `results/`: small GitHub-safe result bundles exported from Colab runs
- `docs/`: project structure notes and audit reports
- `archive/`: legacy `FT-*` attempts and old artifacts
- `scripts/`: data-processing and experiment setup scripts

## Script Usage

All scripts now support CLI args and use repo-relative paths by default.

Examples:

```bash
python3 scripts/create_df_v1.py
python3 scripts/create_df_v2.py data/intermediate/df_master_training_v3.jsonl data/intermediate/df_master_training_v2.jsonl
python3 scripts/extract_error_counts.py data/intermediate/df_master_training_v3.jsonl data/intermediate/clean_output.jsonl data/intermediate/error_summary_std.txt
python3 scripts/join_jsonl.py "data/curated/synthetic/*.jsonl" data/curated/synthetic/synthetic_sentences.jsonl
python3 data/norming/scripts/create_csv.py
```

## Notes

- Scripts are safe to import (`if __name__ == "__main__"`).
- Hard-coded machine-specific absolute paths were removed in favor of repo-relative defaults.
- `.gitignore` now excludes Python and IDE noise.

## ACL RR Workflow (Recommended)

Keep the repository as source-of-truth for data composition and splitting. Use Colab only to execute training/evaluation.

1. Build frozen experiment splits locally:
```bash
python3 scripts/build_acl_splits.py \
  --input data/processed/master_training.jsonl \
  --out-dir experiments/acl_rr_v1
```
2. Generate Colab notebook from repo template:
```bash
python3 scripts/generate_acl_colab_notebook.py
```
3. Run the generated notebook in Colab:
`experiments/acl_rr_v1/ACL_SFT_CLAN_Llama3_1_8B_ACL.ipynb`

Split package includes:
- primary confirmatory sets: real-only `eval`, `test`
- synthetic diagnostics: `eval_coverage`, `test_coverage`
- held-out generalization: `holdout`

Protocol details: `experiments/acl_rr_v1/PROTOCOL.md`

Experiment 1 runbook: `experiments/acl_rr_v1/EXPERIMENT_01.md`

Experiment 3 (AbS ablation) runbook: `experiments/EXPERIMENT_03.md`

Experiment 4 (unseen-tag generalization) runbook: `experiments/EXPERIMENT_04.md`

Experiment 6 (stability confirmation) runbook: `experiments/EXPERIMENT_06.md`
