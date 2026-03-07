# Reviewer-Risk Diagnostics

## High-Priority Risks
1. **Source imbalance**: synthetic 56.42%, TD 36.99%, DLD 6.59%.
2. **Error-density confound by source**: rows with >=1 `[* ...]` tag are synthetic 84.10%, TD 10.65%, DLD 23.05%.
3. **Coverage gap in real data**: 15 detailed labels appear only in synthetic; 25 labels have zero DLD instances.
4. **Partial provenance uncertainty**: 1,713 rows are `synthetic_no_real_match`; 25 rows are flagged `trace_ambiguous`.
5. **Schema inconsistency**: rare off-schema label `[* m:+s]` appears 2 times (expected family appears as `[* m:+s:a]`).

## Suggested Fix Strategy (Pre-Submission)
1. **Freeze two training variants**: `RealOnly` and `Real+Synthetic` (keep the current dataset as `Augmented`).
2. **Create strict eval sets**: TD-only and DLD-only test sets from real data, split by child ID (no child overlap).
3. **Report per-source metrics**: macro/micro F1 by label for TD and DLD separately, not only pooled scores.
4. **Cap synthetic influence**: sample/weight so synthetic does not dominate mini-batches; tune target synthetic ratio (e.g., 30–45%).
5. **Patch label schema**: normalize `[* m:+s]` to your approved taxonomy (after manual audit of those 2 rows).
6. **Provenance policy**: keep `trace_ambiguous` rows for training only; exclude from evaluation and error-analysis tables.
7. **Tail-label plan**: for labels with <10 real instances, either merge to Level-2 in main results or annotate more real examples.
8. **Ablation table for reviewers**: no-curriculum vs curriculum, no-AbS vs AbS, RealOnly vs Real+Synthetic.

## What To Explicitly State In The Paper
- `0POS` and `[+ gram]` are excluded from supervised targets in this work.
- Provenance tracing method and ambiguity handling rules.
- Label-support thresholds used for detailed-label reporting.