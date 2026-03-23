# Project Structure

This repository is organized around one shared canonical data pipeline and three study lines.

## Shared Layers

- `data/curated/`: curated synthetic examples and manually prepared real-data supplements
- `data/processed/`: canonical training resources and provenance-preserving master files
- `data/intermediate/`: rebuildable preparation snapshots retained for traceability
- `data/audits/`: label audits, review sets, and paper-facing audit tables
- `scripts/`: shared setup, evaluation, uncertainty, OOD, and audit utilities
- `tests/`: regression tests for shared logic

## Study Layers

- `studies/01_confirmatory_annotation/`
  Current confirmatory paper line, final experiment package, final run outputs, and paper materials.
- `studies/02_uncertainty_and_feedback/`
  Uncertainty analyses, OOD review-triage work, and planned user-feedback / agrammaticality work.
- `studies/03_compositional_target_recovery/`
  Compositional probing, latent target-recovery analysis, and related follow-up experiments.

## Support Layers

- `docs/`: repo map, experiment registry, data lineage, and cross-study framing
- `archive/`: retired or superseded material kept for traceability
- `handoff/`: deployment handoff assets
- `artifacts/`: small exported artifacts that are not yet assigned to a study or are comparison-side products

## Compatibility Note

Tracked experiment, result, artifact, and paper files remain at their historical top-level paths so Git can stage them without symlink-related path errors. Human-facing navigation should start in `studies/`, whose study-specific subpaths link back to those canonical tracked assets.
