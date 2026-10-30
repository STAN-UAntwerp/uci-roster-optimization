# Assigned workload and point concentration

This analysis compares assigned racing days and the concentration of UCI points
across riders in the original and optimized allocations. It covers 36 team-seasons
under the 85-day cap with recorded injury constraints, with separate optimized
allocations for ranker and regressor rewards.

The repository includes two aggregate datasets:

| File | Rows | Contents |
|---|---:|---|
| [workload_team_summary.csv](results/workload_team_summary.csv) | 108 | Seasonal workload and rolling-window peaks by team-season and allocation |
| [concentration_team_summary.csv](results/concentration_team_summary.csv) | 432 | Point concentration by team-season, allocation, evaluator and scoring method |

Both datasets contain team-level statistics without rider identities or dated
rider schedules.

## Generating the figures

After installing the Python dependencies and LaTeX, run:

```sh
python figures/workload_analysis.py
```

The script reads the included aggregate CSVs and writes three vector PDFs to
`reproduced/`: `rider_workload_season.pdf`, `rider_workload_peaks.pdf` and
`rider_points_concentration.pdf`. It also writes overall summary CSVs for workload
and concentration. Figures use the manuscript's 5.15-inch width and LaTeX-rendered
Computer Modern labels. No reference inputs or Gurobi license are required.

## Definitions

- Seasonal days sum the unique scheduled racing dates within each assignment,
  matching the optimizer's annual budget. All roster members are included.
  Stage-race rest days are excluded; withdrawals do not truncate assignments.
  Cancelled stages still listed in the planning calendar remain charged.
- Rolling peaks count distinct assigned racing dates in any consecutive
  30/60/90-day window. `max_peak_*` is the maximum over all riders in the team;
  `mean_peak_*` is the mean of individual rider maxima. The figures use the former.
  Fixed-length windows are not calendar months.
- Historical allocations contain 20 duplicate rider-date charges in total.
  Rolling peaks count each date once; seasonal totals retain assignment-day
  accounting. Optimized schedules have no duplicate dates. Historical schedules
  need not satisfy the imposed 85-day planning cap.
- `share_at_cap` means the share with at least 85 assigned days. For optimized
  schedules it is exactly the share at 85. The denominator is the full squad.
- Gini is the uncorrected population coefficient, including zero-valued riders.
  The coefficient is undefined for an all-zero population.
  Point shares use all squad points before truncation to the top 20.
- Predicted concentration uses frozen selected-field scores. Hybrid concentration
  uses recorded original points versus mixed recorded/predicted optimized points;
  it is descriptive and changes both allocation and scoring basis.
- The point-concentration figure uses predicted scores from the ranker evaluator
  for all allocations. The CSV also contains regressor-evaluated statistics.
  Comparisons of allocation effects hold the evaluator fixed.

Each box summarizes 36 team-seasons, with medians, quartiles and whiskers spanning
the observed minimum and maximum. The rolling-window panels abbreviate the
allocation labels to Orig., Rank. and Regr. Observations are paired across
allocations, and the same organizational teams recur across years; these
distributions describe the evaluated schedules rather than independent treatment
replications.

## Findings

The following means give equal weight to each team-season:

| Metric | Original | Ranker allocation | Regressor allocation |
|---|---:|---:|---:|
| Mean assigned days per rider | 60.4 | 62.9 | 63.1 |
| Riders with at least 85 days (%) | 3.4 | 24.1 | 25.0 |
| Within-team workload Gini | 0.146 | 0.186 | 0.183 |
| Team maximum in 30 days | 23.1 | 25.1 | 25.1 |
| Team maximum in 60 days | 34.0 | 38.8 | 39.2 |
| Team maximum in 90 days | 46.9 | 52.1 | 53.7 |
| Predicted points Gini, ranker evaluator | 0.764 | 0.778 | 0.786 |
| Top-five points share, ranker evaluator (%) | 75.2 | 77.2 | 78.3 |

Mean workload grows modestly, but more riders reach the annual cap and schedules
become denser within the season. Peak workloads reach 44 racing days in a 60-day
window for both reward models, and 62/61 in 90 days for ranker/regressor rewards.
These are scheduling diagnostics, not evidence of physiological overload:
training, travel, intensity and individual recovery are not represented.
The secondary objective maximizes the minimum count of race assignments, not
equality of days or recovery. A stage race and a one-day race each count as one
assignment in that secondary objective.

Point concentration is already high in original predicted scores. Its increase
is modest and evaluator-dependent: under regressor evaluation, the regressor
allocation's Gini changes from 0.771 to 0.772, while its top-five share decreases
from 76.8% to 76.4%. The hybrid/actual comparison is more concentrated (ranker
evaluation Gini 0.648 original versus 0.752/0.759 optimized), but should not be
interpreted as a pure allocation effect. Unmodeled domestique contributions
also prevent interpreting point concentration as inequality in rider value.

## Reconstructing the aggregates

Recomputing the aggregate CSVs requires the team rosters, team calendars and
reference hybrid assignment archives described in [DATA_MANIFEST.md](DATA_MANIFEST.md).
These inputs are not included in the public repository. For an existing,
lawfully held copy of the reference files:

```sh
python workload_analysis.py --source /path/to/reference/files
```

The script verifies roster membership, unique assignments, unchanged assignments
across evaluators, optimized annual caps and nonoverlap, and agreement between
reconstructed top-20 point totals and saved team summaries. Synthetic tests cover
Gini boundaries, rolling-window boundaries, and rest-day exclusion.

Reconstructed aggregate CSVs and a validation record are written to `results/`.
The optional `--private-details` flag also writes rider-level diagnostics to
`outputs/workload_analysis/`, which is excluded from version control. Rider-level
outputs have the same access and redistribution restrictions as the source data.
