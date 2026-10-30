# Reference data and experiment archives

This document describes the inputs and saved experiments used for the paper. These files are not included in the public repository. Redistribution permissions have not been secured, and identifiable injury information requires a separate data-protection assessment. No public data archive is available.

The aggregate results in `results/` support the table and workload-figure commands listed in the [README](README.md). Refitting models, optimizing allocations and reconstructing assignment-level evaluations require the additional files inventoried below.

## File inventory

[data_manifest.json](data_manifest.json) lists reference paths, byte sizes and SHA-256 checksums. All paths are relative to the repository root.

| Profile | Files | Uncompressed size | Purpose |
|---|---:|---:|---|
| `inputs` | 35 | 535.976 MB | Model refitting and roster optimization |
| `archives` | 322 | 196.281 MB | Evaluation of saved allocations and archive-based figures |

The input profile contains 27 data files and eight metadata/result files from the four selected tuning runs. The data comprise prepared feature matrices, rider demographics and specialties, UCI point scales and race categories, teams, calendars, start lists, derived route features, histories and injury intervals. Reproduction uses the derived route features; original GPX tracks are not required.

The archive profile contains fitted models, roster summaries, per-race selections and hybrid assignment records. Archived models support evaluation of the saved allocations without repeating the tuning grids.

In the public scenario CSVs, `reward_scoring` and `evaluation_scoring` identify the scoring methods used. The ranker CSV also retains a legacy `regressor_scoring` configuration field; the applied evaluation method is recorded in `evaluation_scoring`.

## Data sources

| Source | Research inputs |
|---|---|
| ProCyclingStats | Results, rider and team records, race calendars and injury records |
| La Flamme Rouge | Route geometry |
| OpenStreetMap | Road surfaces |

The repository covers the modeling and optimization workflow from prepared inputs onward. The upstream collection pipeline and source-level records are not included. Source attribution and licensing scope are documented in [NOTICE](NOTICE).

## Reference experiments

The reproduction commands use the following tuning runs:

| Model | Directory |
|---|---|
| `STAGE` | `outputs/tuning/stage/2026-09-09_21-56-39` |
| `GC` | `outputs/tuning/gc/2026-09-10_08-06-42` |
| `STAGE_REG` | `outputs/tuning/stage_reg/2026-09-10_10-14-40` |
| `GC_REG` | `outputs/tuning/gc_reg/2026-09-11_16-39-00` |

The corresponding evaluation archives are:

| Manifest key | Directory |
|---|---|
| `ranker` | `outputs/rostering_evaluation/2026-09-11_10-20-59_072753` |
| `regressor` | `outputs/rostering_evaluation/2026-09-11_09-19-56_537799` |
| `ranker_solver` | `outputs/rostering_evaluation/2026-09-10_15-08-16_163989` |
| `hybrid` | `outputs/hybrid_evaluation/2026-09-14_16-45-11_478985` |
| `predictive` | `outputs/evaluation/2026-09-10_10-39-20_065552` |

The default hybrid and retrospective plotting commands use these reference archives. Hybrid rescoring uses `python reproduce.py hybrid`. The generic `--rescore-run` option requires nested scenario exports outside this inventory and is intended for runs containing those exports.

## Staging and verification

For an existing, lawfully held copy of the reference files, copy and verify the inventory with:

```sh
python stage_data.py /path/to/reference/files
```

Use `--profile inputs` or `--profile archives` to select a subset. The default is `all`. The script checks source hashes before copying and verifies destination sizes and hashes afterward.

To verify files already in the repository and check the historical calendars:

```sh
python stage_data.py --check
python reproduce.py check
```

Checksums identify the exact files used in the reference experiments. Access to those files and any permission to use or redistribute them must be obtained separately from the relevant rights holders.
