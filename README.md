# UCI roster optimization

Code and aggregate results accompanying **A predictive-prescriptive framework for optimal rider allocation in cycling**.

The framework combines predictive models of rider performance with seasonal roster optimization and evaluates allocations using predicted and hybrid observed/predicted UCI points. This repository contains the model fitting, tuning, optimization, evaluation and figure-generation code for the 2024 and 2025 seasons.

## Reproducing results

The included aggregate CSVs support the following analyses directly:

| Analysis | Command | Output directory |
|---|---|---|
| Predictive comparison, cross-model gains, hybrid gains and scenario sensitivity | `python reproduce.py tables` | `reproduced/` |
| Hybrid appendix table (Table D6) | `python manuscript_hybrid_table.py` | `reproduced/` |
| Workload and point-concentration figures | `python figures/workload_analysis.py` | `reproduced/` |

These commands do not require a Gurobi license. The figure command requires a LaTeX installation. [WORKLOAD_ANALYSIS.md](WORKLOAD_ANALYSIS.md) describes the workload and concentration metrics and findings.

**Full reproduction requires additional data.** Source records, prepared feature matrices, fitted models and detailed evaluation archives are not included. Redistribution permissions have not been secured, and identifiable injury information requires a separate data-protection assessment. [DATA_MANIFEST.md](DATA_MANIFEST.md) describes these inputs and the reference experiments; no public data archive is available.

## Installation

The code was tested with Python 3.11.13. Tested dependency versions are pinned in [requirements.txt](requirements.txt).

```sh
python -m venv .venv
```

Activate the virtual environment (`source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell), then install the dependencies and generate the summary tables:

```sh
python -m pip install -r requirements.txt
python reproduce.py tables
```

The reference comparisons cover 36 team-seasons: 18 teams in each of 2024 and 2025. Outputs include cross-model and hybrid improvements, predictive metrics and scenario sensitivity. When the reference archives are also available, the same command additionally generates selected hyperparameters, solver statistics, team-level sensitivity and assignment Jaccard similarities.

## Repository structure

| Path | Contents |
|---|---|
| `src/models/` | Predictive models, tuning, evaluation and SHAP analysis |
| `src/rostering/` | Roster optimization and allocation evaluation |
| `figures/` | Figure-generation scripts |
| `results/` | Aggregate reference results |
| `tests/` | Synthetic, aggregate-result and integration tests |
| `validation/` | Numerical validation records |
| `reproduce.py` | Command-line entry point for the reproduction workflow |
| `data_manifest.json` | File inventory and checksums for the undistributed reference inputs |

## Running the full workflow

The commands in this section require lawful access to the reference inputs described in [DATA_MANIFEST.md](DATA_MANIFEST.md). Place the files under `data/`, `models/` and `outputs/`, preserving the manifest paths. To copy and verify them from an existing directory:

```sh
python stage_data.py /path/to/reference/files
```

For files already in place, verify their checksums and validate the historical calendars:

```sh
python stage_data.py --check
python reproduce.py check
```

The staging option `--profile inputs` selects the 27 data files and eight tuning metadata/result files needed for refitting and optimization. `--profile archives` selects the saved evaluations, models and assignments used for retrospective evaluation and figures.

| Task | Command | Additional requirement |
|---|---|---|
| Refit all four model types for both test years | `python reproduce.py models` | Reference inputs |
| Optimize the 85-day scenario with recorded injuries | `python reproduce.py rosters --reference` | Reference inputs and Gurobi license |
| Run the full workload/injury scenario sweep | `python reproduce.py rosters` | Reference inputs and Gurobi license |
| Recompute hybrid scores from saved assignment predictions | `python reproduce.py hybrid` | Reference hybrid archive |
| Regenerate Figures 7–10 | `python reproduce.py plots` | Reference archives and LaTeX |
| Refit rankers and generate SHAP figures | `python reproduce.py shap` | Reference inputs and LaTeX |

To restrict optimization to one team, year and reward model:

```sh
python reproduce.py rosters --reference --reward ranker --year 2024 --teams alpecin-deceuninck
```

New model evaluations, roster solutions and hybrid evaluations are written to `outputs/reproduced_models/`, `outputs/reproduced_rosters/` and `outputs/reproduced_hybrid/`, respectively. The default `plots` and `hybrid` commands use the reference archives listed in the manifest. To evaluate newly optimized allocations, pass the corresponding run directories:

```sh
python -m src.rostering.hybrid_evaluation --ranker-run outputs/reproduced_rosters/RANKER_TIMESTAMP --regressor-run outputs/reproduced_rosters/REGRESSOR_TIMESTAMP
```

Tuning grids can be rerun with `python reproduce.py tune STAGE`, replacing `STAGE` with `GC`, `STAGE_REG` or `GC_REG` as needed. The selected-parameter workflow does not require retuning. Model and roster configurations are in [tuning_config.yaml](src/models/tuning_config.yaml) and [evaluation_config.yaml](src/rostering/evaluation_config.yaml).

Gurobi is included in the dependencies; optimization requires a suitable license. Summary generation, predictive evaluation and hybrid rescoring do not invoke the solver.

## Scoring conventions

- Both model families rank riders and map ranks to UCI point scales. Regression outputs determine the ranking rather than serving directly as point predictions.
- Optimization uses a fixed full-field reward matrix. Selected-field evaluation removes unselected teammates and reranks the remaining field using fixed scores and features.
- Hybrid scoring uses recorded stage and general-classification awards for historical participants and predictions for optimized nonparticipants. Under the complete-results assumption, missing results contribute zero. Entirely cancelled races score zero; nonparticipants retain whole-race predictions for partially cancelled stage races.
- The original hybrid baseline consists of recorded points, with zero for riders who did not start. Each score uses its own 20 highest-scoring riders for the season; the observed and predicted components of a hybrid score use the same hybrid top 20.
- Cross-model and hybrid means use 36 optimized team-season observations. Team-level sensitivity tables pool both seasons by organizational team; sensitivity plots show the distribution across team-seasons.
- Solver records include status and the configured 0.1% optimality tolerance. Unavailable achieved gaps remain missing.

Reported gains are retrospective model-based comparisons, rather than realized counterfactual outcomes. Hybrid scores mix recorded awards and predicted ranks and need not conserve total race points. Recorded-point baselines cover the evaluated calendar and award types, rather than the complete official annual UCI total. Injury constraints use historical availability information.

## Tests and validation

```sh
python -m pip install -r requirements-dev.txt
python -m pytest
```

Without the reference data, the suite runs synthetic and aggregate-result tests and skips checks requiring scale/category tables. Roster integration tests require the reference inputs and a Gurobi license. The end-to-end example, `python smoke_pipeline.py`, also requires the reference archives and a Gurobi license.

[VALIDATION.md](VALIDATION.md) reports the validation scope and numerical comparisons. Historical reruns reproduced all 288 yearly predictive metrics and all 288 hybrid summary rows using the reference data. The full tuning grids, 1,008-solve scenario sweep and SHAP refits were not repeated during validation.

## Citation and license

Citation metadata are available in [CITATION.cff](CITATION.cff) and through GitHub's **Cite this repository** menu.

Original code and documentation are released under the [MIT License](LICENSE). Third-party software and data retain their respective terms. [NOTICE](NOTICE) provides source attribution and the scope of the license.
