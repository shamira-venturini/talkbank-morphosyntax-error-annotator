# CLAN Annotator

Utilities for preparing, transforming, and auditing CLAN-style JSONL datasets.

## Repository Layout

- `scripts/`: core data-processing scripts
- `norming/scripts/`: norming-specific helpers
- `FT-1`, `FT-2`, `FT-3`: training and inference artifacts
- `curated_examples/`: curated synthetic/real examples

## Script Usage

All scripts now support CLI args and use repo-relative paths by default.

Examples:

```bash
python3 scripts/create_df_v1.py
python3 scripts/create_df_v2.py FT-3/df_master_training_v3.jsonl FT-3/df_master_training_v2.jsonl
python3 scripts/extract_error_counts.py FT-3/df_master_training_v3.jsonl FT-3/clean.jsonl FT-3/error_summary_std.txt
python3 scripts/join_jsonl.py "curated_examples/synthetic/*.jsonl" curated_examples/synthetic/synthetic_sentences.jsonl
python3 norming/scripts/create_csv.py
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
  --input FT-3/df_master_training_v3_with_provenance_errorcount.jsonl \
  --out-dir FT-3/experiments/acl_rr_v1
```
2. Generate Colab notebook from repo template:
```bash
python3 scripts/generate_acl_colab_notebook.py
```
3. Run the generated notebook in Colab:
`FT-3/experiments/acl_rr_v1/ACL_SFT_CLAN_Llama3_1_8B_ACL.ipynb`

Split package includes:
- primary confirmatory sets: real-only `eval`, `test`
- synthetic diagnostics: `eval_coverage`, `test_coverage`
- held-out generalization: `holdout`

Protocol details:
`FT-3/experiments/acl_rr_v1/PROTOCOL.md`

Experiment 1 runbook:
`FT-3/experiments/acl_rr_v1/EXPERIMENT_01.md`

Experiment 3 (AbS ablation) runbook:
`FT-3/experiments/EXPERIMENT_03.md`

Experiment 4 (unseen-tag generalization) runbook:
`FT-3/experiments/EXPERIMENT_04.md`

Experiment 6 (stability confirmation) runbook:
`FT-3/experiments/EXPERIMENT_06.md`
