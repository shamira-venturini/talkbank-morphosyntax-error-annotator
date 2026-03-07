# Project Structure

This repository is organized around canonical data, frozen experiment packages, and small reproducible results.

- `data/curated/`: curated synthetic examples and manually prepared real-data supplements
- `data/processed/`: canonical training files used by the current experiment pipeline
- `data/intermediate/`: preparation snapshots kept for traceability and rebuilding
- `data/audits/`: label audits, review sets, and paper tables
- `data/norming/`: norming material and helper scripts
- `experiments/`: frozen split packages, notebooks, and experiment runbooks
- `results/`: GitHub-safe outputs exported from Colab runs
- `archive/`: older `FT-*` attempts and legacy artifacts retained for record
- `scripts/`: data-processing and experiment setup code

The current canonical training file is:

- `data/processed/master_training.jsonl`

The current primary experiment package is:

- `experiments/acl_rr_v1/`
