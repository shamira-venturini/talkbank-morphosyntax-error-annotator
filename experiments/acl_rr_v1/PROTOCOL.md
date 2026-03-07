# ACL RR Protocol (v1)

This folder is the frozen experiment package used for ACL-style reporting.

## Label policy
- Keep `[* s:r:gc:det]` as project extension (`det`, not `art`).
- Hold out synthetic-only comparative/superlative detailed labels for generalization:
  - `[* m:++er]`
  - `[* m:++est]`
  - `[* m:0er]`
  - `[* m:0est]`
- Drop known off-schema outlier label from training/eval artifacts:
  - `[* m:+s]` (2 rows)
- Reconstruction policy for this package (Exp 1/2):
  - Preserve existing CHAT reconstruction distinction between `[: target]` and `[:: target]`.

## Split policy
- Split building is done locally in repository (PyCharm), not in Colab.
- Real-only eval/test (`TD` + `DLD`), synthetic data used for train augmentation.
- Train mix is controlled at 50/50 real/synthetic for Experiment 1.
- Synthetic label-coverage diagnostics are generated as separate splits:
  - `eval_coverage`
  - `test_coverage`
- Holdout set is never used for training.
- `trace_ambiguous=true` rows are excluded from all non-train splits.
- Exact input overlap is disallowed between train and all non-train splits.

## Statistical policy
- Report confirmatory per-label metrics with split-aware support thresholds:
  - real `eval` / `test`: >= 20
  - `eval_coverage` / `test_coverage`: >= 10
  - `holdout`: >= 10
- Report labels below each split's threshold as exploratory.
- Report micro-F1 with bootstrap 95% CI.

## Reviewer-facing claim scope
- This is a low-resource, long-tail label setting; per-label support is uneven on real-only `eval` / `test`.
- Confirmatory claims are restricted to:
  - aggregate metrics on real-only splits (especially micro-F1 + 95% bootstrap CI)
  - per-label results that meet the split-aware support thresholds above
- Labels below threshold are reported as exploratory evidence only.
- `eval_coverage` / `test_coverage` and `holdout` are diagnostics for coverage/generalization, not substitutes for real-only confirmatory evidence.
- Any paper or report should state this distinction explicitly to prevent over-interpretation of rare-label results.

## Rebuild commands
```bash
python3 scripts/build_acl_splits.py \
  --input data/processed/master_training.jsonl \
  --out-dir experiments/acl_rr_v1

python3 scripts/generate_acl_colab_notebook.py
```

## Main artifacts
- `stage{1,2,3}_{train,eval,test,eval_coverage,test_coverage,holdout}.jsonl`
- `split_manifest.csv`
- `chat_tokens.json`
- `summary.json`
- `ACL_SFT_CLAN_Llama3_1_8B_ACL.ipynb`
