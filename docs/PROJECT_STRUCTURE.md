# Project Structure

This repository is organized around one shared canonical data pipeline and two primary top-level study entry points, with additional lower-level follow-up lines retained under `studies/`.

## Top-Level Study Entry Points

- `study_01_talkbank_tool_paper/`
  Top-level entry for the frozen first study.
- `study_02_hitl_adaptation/`
  Top-level entry for the current second study.

## Shared Layers

- `data/curated/`: curated synthetic examples and manually prepared real-data supplements
- `data/processed/`: canonical training resources and provenance-preserving master files
- `data/intermediate/`: rebuildable preparation snapshots retained for traceability
- `data/audits/`: label audits, review sets, and paper-facing audit tables
- `scripts/`: shared setup, evaluation, uncertainty, OOD, and audit utilities
- `tests/`: regression tests for shared logic

## Study Layers

- `studies/01_confirmatory_annotation/`
  Lower-level navigation layer for the first study.
- `studies/02_uncertainty_and_feedback/`
  Lower-level navigation layer for the second study, including the current human-corrected adaptation line.
- `studies/03_compositional_target_recovery/`
  Additional exploratory follow-up line not yet promoted to a top-level entry point.

## Support Layers

- `docs/`: repo map, experiment registry, data lineage, and cross-study framing
- `reviews/`: active reviewed CSVs and blinded review bundles
- `archive/`: retired or superseded material kept for traceability
- `handoff/`: deployment handoff assets
- `artifacts/`: small exported artifacts that are not yet assigned to a study or are comparison-side products

## Active Surface Reduction

As of 2026-03-23:

- obsolete and paper-drafting docs were moved from `docs/` into `archive/docs_legacy_2026-03-23/`
- one-off or superseded utilities were moved from `scripts/` into `archive/scripts_legacy_2026-03-23/`

The goal is to keep the top-level working surface centered on the first study, the second study, and the shared live pipeline.

## Compatibility Note

Tracked experiment, result, review, artifact, and paper files remain at their historical top-level paths so Git can stage them without symlink-related path errors. Human-facing navigation should start in `study_01_talkbank_tool_paper/` or `study_02_hitl_adaptation/`. The `studies/` tree remains the lower-level study navigation layer that links back to the canonical tracked assets.
