# AGENTS.md

This repository studies **automatic morphosyntactic error annotation for TalkBank CHAT data**. All Codex work in this repo must stay aligned with the project aim, research questions, annotation theory, and experiment constraints documented in `docs/PROJECT_AIM_AND_RQS.md`.

## Project Objective
Build, evaluate, and document a **reproducible, low-resource, privacy-preserving pipeline** for assigning **CHAT morphosyntactic error labels** to spoken-language utterances, with a focus on learner and clinical language data.

The core scientific goal is **not** generic grammatical error correction and **not** generic POS tagging. The task is to apply a **structured CHAT annotation scheme** to observed utterances.

## Central Framing
Treat the annotation task as **typed morphosyntactic error labeling over a produced–target contrast**.

Use this conceptual definition throughout the repo:
- `label = δ(produced_form, target_form)`

Interpretation:
- the label is a typed contrast between the observed erroneous form and the intended target form;
- many labels are **not** intrinsic properties of the observed token;
- missing morphology, overregularization, double marking, and past/perfective substitutions require comparing the produced form with an inferred target and/or with sentential obligatory context.

This means the task is **not well modeled as plain token-wise sequence labeling**, even if the output inventory is finite.

## Annotation-Scheme Assumptions
When reasoning about labels, use the following conventions.

### 1) Surface syntax of the tags
The tag format has a simple bracketed, colon-separated structure, e.g.:
- `[* m:0ed]`
- `[* m:=ed]`
- `[* m:03s:a]`
- `[* s:r:gc:pro]`

The surface syntax can be treated as a **regular language**, but the semantics are richer: tags encode a taxonomy of error domain, subtype, and grammatical feature.

### 2) Core distinction: `m:*` vs `s:*`
Use these operational definitions consistently.

- `m:*` = contrasts internal to the inflectional realization of a lexical item.
  - examples: missing inflection, superfluous inflection, overregularization, irregular/base substitution, allomorphy, double marking, agreement-sensitive morphology.
- `s:*` = substitution of one lexical or grammatical-category item for another.
  - examples: wrong preposition, pronoun substitution, derivational substitution, determiner substitution.

In short:
- `m:*` = same lexical item, wrong morphological realization
- `s:*` = different lexical or grammatical-category choice

### 3) Use of `:a`
Normalize the CHAT scheme as follows:
- `m` = broad morphological domain
- `:a` = agreement-sensitive subtype only

Do **not** treat `[* m:a]` as the required default form of every morphological tag. Reserve `:a` for regular agreement-sensitive cases such as:
- `m:03s:a`
- `m:+3s:a`
- `m:0s:a`
- `m:+s:a`

### 4) Use of `:i`
Treat `:i` as an **irregular-sensitive marker** used only in those detailed labels where the error crucially depends on irregular morphology, e.g.:
- `m:++ed:i`
- `m:++en:i`
- `m:++s:i`

Do **not** generalize `:i` beyond the labels that explicitly license it in the supported inventory.

### 5) Scheme-licensed vs corpus-attested labels
Keep a clear distinction between:
- labels licensed by the scheme,
- labels attested in the current corpus,
- labels present in training/eval.

The tool may support a broader inventory than the currently attested subset, but all reports must state clearly which labels are actually represented in the data.

## Why Existing NLP Tools Are Not Enough
Preserve this distinction in docs, code comments, and paper text.

- `%mor` / MOR / GRASP provide morphosyntactic analyses of the observed utterance.
- POS or morphological taggers assign categories/features to observed forms.
- CHAT error coding instead represents the **type of divergence** between a produced form and an intended target.

Therefore, this project should not be framed as:
- generic sequence labeling,
- generic POS tagging,
- generic grammatical error correction.

It is specifically **structured morphosyntactic error annotation in CHAT**.

## Learning Setup to Preserve
The preferred modeling interpretation is **analysis-by-synthesis plus curriculum learning**.

### Analysis-by-synthesis
When reconstruction is enabled, the model should:
1. infer or reconstruct the target form,
2. compare produced vs target,
3. output the error label.

This should be treated as a central methodological contribution, not a prompting trick.

### Curriculum learning
The curriculum should mirror the hierarchy of the annotation scheme while making hard latent contrasts available early.

Recommended progression:
1. detect whether an error is present; for morphological errors that license them in detailed form, expose **early latent flags** for agreement sensitivity (`:a`) and irregular sensitivity (`:i`),
2. classify coarse domain (`m` vs `s` or equivalent),
3. classify subtype / operation,
4. predict the full detailed label.

Operational interpretation of the early latent flags:
- `:a` should be treated as an early signal for categories whose detailed labels require agreement sensitivity, especially where bare-form ambiguity arises (e.g. `m:03s:a`, `m:+3s:a`, `m:0s:a`, `m:+s:a`, `m:vsg:a`, `m:vun:a`),
- `:i` should be treated as an early signal for categories whose detailed labels require irregular morphology, especially double-marking irregulars (e.g. `m:++ed:i`, `m:++en:i`, `m:++s:i`),
- these signals are **curriculum features**, not independent output labels unless an experiment explicitly tests that design.

If implemented, prefer to describe this as moving agreement and irregular sensitivity **earlier in the decision process**, not as introducing `[* m:a]` as a default morphological notation.

Any curriculum code or experiment packaging should preserve this logic unless explicitly testing an ablation.

## Canonical Research Questions
Base reasoning and repo changes on the current working RQs distilled in `docs/PROJECT_AIM_AND_RQS.md`:

1. **Baseline performance**: how well a direct model annotates morphosyntactic errors in controlled TalkBank CHAT splits.
2. **Reconstruction / analysis-by-synthesis**: whether target-form reconstruction improves annotation robustness.
3. **Compositional generalization**: whether the model generalizes to valid detailed labels withheld from training.
4. **Stability**: whether the observed effects hold across multiple random seeds.

In addition, current paper-facing analysis should explicitly track:
- **curriculum vs no curriculum**,
- **standard vs primed/reconstruction training**,
- **the interaction between these two factors**,
- where possible, whether earlier access to `:a` and `:i` reduces confusion in the relevant subclasses.

## Required Experiment Matrix
Treat the following as the canonical comparison set unless the user explicitly changes it:
- `std_nocurr`
- `primed_nocurr`
- `std_curr`
- `primed_curr`

Also maintain targeted diagnostics for the hardest ambiguity:
- `[* m:03s:a]` vs `[* m:0ed]`

This ambiguity matters because in child and clinical language, bare verbs may under-specify tense/finiteness. All analysis should distinguish:
- genuine model confusion,
- genuine linguistic ambiguity in the source data.

If a curriculum variant exposes `:a` early, this ambiguity should be one of the first diagnostic checks.

## Data and Evaluation Priorities
When reasoning about experiments, preserve this order of priority:

1. Preserve the canonical data and split-building pipeline.
2. Keep experiment packages frozen and reproducible.
3. Use real-only eval/test splits for primary confirmatory claims.
4. Use synthetic coverage splits for label-coverage diagnostics.
5. Use holdout splits for withheld-label generalization claims.
6. Use 3-seed replication for important comparisons.

Do not silently change split logic, label normalization, or experiment manifests.

## Evaluation Requirements
Whenever results or analysis are updated, prefer reporting:
- exact-tag accuracy,
- macro-F1,
- per-label metrics,
- confusion matrices,
- syntax-validity of produced tags,
- special confusion between `[* m:03s:a]` and `[* m:0ed]`.

When reconstruction is used, also report:
- reconstruction accuracy,
- label accuracy conditioned on correct vs incorrect reconstruction, if available.

When early `:a` / `:i` curriculum features are used, also report:
- performance on the affected subclasses,
- whether the early features reduce confusions among agreement-sensitive and irregular-sensitive labels,
- whether gains remain stable across seeds.

## Reporting and Writing Rules
When generating documentation, paper text, or comments:
- describe the task as **automatic morphosyntactic error annotation in TalkBank CHAT**;
- do not collapse it into generic GEC;
- distinguish clearly between annotation syntax and annotation semantics;
- distinguish clearly between scheme-licensed labels and corpus-attested labels;
- state when a claim concerns the regular surface form of tags rather than the semantics of the scheme.

When discussing novelty, emphasize:
- low-resource and privacy-preserving local deployment,
- analysis-by-synthesis / reconstruction,
- curriculum learning aligned with the annotation taxonomy,
- structured application of a CHAT-based annotation scheme.

Do not claim that the model has fully “learned the annotation language” in a generative formal-language sense unless there is direct evidence. Safer wording:
- “learned to apply a structured annotation scheme”
- “learned aspects of the compositional structure of the tag inventory”

## Repo-Behavior Rules for Codex
When modifying this repo:
- never edit raw data in place;
- keep all processed data reproducible from scripts;
- preserve experiment manifests and config files;
- prefer small, reviewable changes;
- add or update tests when changing label normalization, parsing, split building, curriculum staging, or evaluation logic;
- keep paper-facing tables and figures traceable to metric files.

If a requested change risks altering the scientific framing, note it explicitly before implementing.

## What to Flag Immediately
Report any of the following to the user instead of silently normalizing them away:
- duplicate labels in vocabularies or configs,
- mismatches between README/docs and actual experiment manifests,
- labels used in code that are not documented,
- documented labels missing from vocabularies or evaluation code,
- inconsistent use of `m`, `m:a`, or `:a`,
- inconsistent use or overgeneration of `:i`,
- confusion between substitutional and morphological labels,
- train/eval leakage across attested and held-out label sets,
- claims in paper text that overstate what the model learned.

## Source of Truth
Use `docs/PROJECT_AIM_AND_RQS.md` as the paper-framing source of truth and update it when the project framing changes. The current working aim and research questions there should guide all experiment packaging and reporting decisions.
