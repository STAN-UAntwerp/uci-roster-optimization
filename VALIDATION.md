# Validation and reference results

Validation covers reproduction from the public aggregate results and numerical checks using the undistributed reference inputs. The latter are historical checks performed with locally held data; their results do not imply that the full workflow can be run from the public repository alone. [DATA_MANIFEST.md](DATA_MANIFEST.md) describes the required inputs.

## Public-checkout validation: 30 October 2026

A copy containing only the public repository files was tested in a fresh Python 3.11.13 environment with the dependency versions in `requirements.txt` and `requirements-dev.txt`, including NumPy 2.4.6 and pytest 9.0.3. All 28 public-data and synthetic tests passed, including the predictive table's LaTeX export. Two checks requiring undistributed scale/category tables were skipped.

The following commands completed without reference data or a Gurobi license:

```sh
python reproduce.py tables
python manuscript_hybrid_table.py
python figures/workload_analysis.py
```

The checks reproduced the predictive comparison, cross-model and hybrid reference means, the 36-row Table D6 and all three workload/concentration PDFs. The predictive comparison matched the included reference CSV. Additional consistency checks verified the 14-scenario design for every team-season, hybrid component totals and percentage improvements, and reallocation gains reconstructed from category aggregates.

## Earlier validation: 28 October 2026

Validation on 28 October 2026 used an extracted copy of the code and aggregate results. The test suite passed 26 tests and skipped two checks requiring undistributed scale/category tables. The aggregate summary tables, hybrid appendix table (Table D6) and all three workload/concentration PDFs were regenerated successfully.

With the reference inputs available, the full test suite passed 52 tests. Coverage includes synthetic scoring cases, aggregate-result consistency, workload metrics and roster integration checks.

## Model and allocation checks

The historical full-workflow validation included:

| Check | Result |
|---|---|
| Reference file integrity | All 357 files verified against SHA-256 hashes |
| Input and calendar validation | Both seasons, 18 teams per season, 14 scenarios per team and all four model configurations |
| Predictive model refitting | Eight fits; all 288 yearly metrics matched the reference evaluation exactly |
| Hybrid evaluation from saved assignments | All 288 summary rows matched for recorded points, predicted components, totals and improvements |
| Assignment agreement between reward models | Mean Jaccard similarity of 0.348131 |
| Archive-based figures | Figures 7–10 regenerated; Figure 9 category gains matched the manuscript outputs |

The predictive metric comparison had a maximum absolute difference of 0.0. Detailed numerical records are available in [validation/](validation/).

The historical test suite passed 50 tests covering module imports, zero and missing results, cancelled races, rank-to-point conversion, top-20 scoring, injury and calendar constraints, and Gurobi objective values.

An end-to-end test used the 2024 Alpecin-Deceuninck squad at the Tour Down Under and GP d'Ouverture. It generated rewards with both model families, solved allocations with both solvers and evaluated both allocations with both evaluators. The solves returned optimal status (status 2).

## Reference improvements

The following mean percentage improvements are recomputed by `python reproduce.py tables`. Each cell summarizes 36 optimized team-seasons under the 85-day cap with recorded injury constraints.

| Reward model | Evaluator | Predicted improvement (%) | Hybrid improvement (%) |
|---|---|---:|---:|
| Ranker | Ranker | 109.682080 | 98.742151 |
| Ranker | Regressor | 90.939588 | 83.150907 |
| Regressor | Ranker | 82.297753 | 74.840478 |
| Regressor | Regressor | 124.618469 | 109.210105 |

Predicted improvements compare optimized and original selected-field predictions. Hybrid improvements compare the optimized hybrid total with the original recorded-points total. These are means of team-season percentage changes. Each score uses its own top 20 riders; the original hybrid baseline contains recorded points only.

## Scope and limitations

The historical full-workflow checks used Python 3.11.13 and NumPy 2.4.0. The public-checkout validation used a fresh installation with the current dependency pins, including NumPy 2.4.6; full model fitting and optimization were not repeated with the updated NumPy version. Figure generation used a working LaTeX installation.

The full tuning grids, the full 1,008-solve workload/injury sweep and the SHAP refits were not repeated during validation. Solver runtimes and choices among equally optimal allocations may vary across machines. Reference solver counts and timings are taken from the archived experiments; the end-to-end test timings are separate.
