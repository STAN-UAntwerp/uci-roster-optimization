"""Describe assigned workload and point concentration without refitting or solving.

Raw reconstruction requires private reference inputs. Only team-level aggregates
are written under results/; optional rider details stay under ignored outputs/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LEVELS = {'1.UWT', '1.Pro', '2.UWT', '2.Pro', '1.1', '2.1', '1.2', '2.2'}


def gini(values):
    """Uncorrected population Gini, including zero-valued squad members."""
    x = np.sort(np.asarray(values, dtype=float))
    if not len(x) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Expected a nonempty finite nonnegative population')
    if x.sum() == 0:
        return np.nan
    n = len(x)
    return float(2 * np.dot(np.arange(1, n + 1), x) / (n * x.sum()) - (n + 1) / n)


def rolling_peak(daily, window):
    """Maximum racing days in any window consecutive calendar days."""
    return int(np.convolve(daily, np.ones(window, dtype=int), mode='valid').max())


def calendar(program):
    # Same unique scheduled racing dates charged by RosterOptimizer._combine_stages.
    return {job: sorted({pd.Timestamp(s['date']) for s in info['stages']}
                        if 'stages' in info else {pd.Timestamp(info['date'])})
            for job, info in program.items() if info.get('level') in LEVELS}


def reconstruct(source, out, private_details=False):
    manifest = json.loads((ROOT / 'data_manifest.json').read_text())
    hybrid = source / manifest['runs']['hybrid']
    summaries = pd.read_csv(hybrid / 'team_summary.csv')
    rows, concentration, rider_rows = [], [], []
    audit = {'team_seasons': 0, 'original_duplicate_dates': 0, 'assignments_checked': 0}
    for year in (2024, 2025):
        programs = json.loads((source / f'data/roster/programs_{year}.json').read_text())
        squads = json.loads((source / f'data/teams/teams_{year}.json').read_text())
        days = pd.date_range(f'{year}-01-01', f'{year}-12-31')
        for team_href, program in sorted(programs.items()):
            team = team_href.split('/')[-1].removesuffix(f'-{year}')
            squad = sorted(set(squads[team_href]))
            dates = calendar(program)
            audit['team_seasons'] += 1
            for allocation, reward, roster in [('original', 'ranker', 'original'),
                                               ('ranker', 'ranker', 'optimized'),
                                               ('regressor', 'regressor', 'optimized')]:
                file = hybrid / reward / 'ranker' / str(year) / team / 'assignments.csv'
                frame = pd.read_csv(file)
                frame = frame[frame.roster.eq(roster)].copy()
                assert not frame.duplicated(['job', 'rider_href']).any()
                assert set(frame.rider_href) <= set(squad)
                assert set(frame.job) <= dates.keys()
                pairs = set(zip(frame.job, frame.rider_href))
                audit['assignments_checked'] += len(frame)
                metrics = []
                for rider in squad:
                    jobs = frame.loc[frame.rider_href.eq(rider), 'job']
                    daily = np.zeros(len(days), dtype=int)
                    for job in jobs:
                        for date in dates[job]:
                            assert date.year == year
                            daily[(date - days[0]).days] += 1
                    duplicates = int(np.maximum(daily - 1, 0).sum())
                    if roster == 'optimized':
                        assert duplicates == 0, (team, year, rider)
                        assert daily.sum() <= 85
                    else:
                        audit['original_duplicate_dates'] += duplicates
                    record = dict(year=year, team=team, allocation=allocation,
                                  rider_href=rider, race_days=int(daily.sum()),
                                  jobs=len(jobs), duplicate_dates=duplicates,
                                  **{f'peak_{w}': rolling_peak((daily > 0).astype(int), w)
                                                    for w in (30, 60, 90)})
                    metrics.append(record)
                    rider_rows.append(record)
                m = pd.DataFrame(metrics)
                rows.append(dict(year=year, team=team, allocation=allocation,
                                 n_riders=len(squad), mean_days=m.race_days.mean(),
                                 median_days=m.race_days.median(), max_days=m.race_days.max(),
                                 gini_days=gini(m.race_days), total_days=m.race_days.sum(),
                                 share_at_cap=(m.race_days >= 85).mean(),
                                 share_zero_days=m.race_days.eq(0).mean(),
                                 duplicate_dates=m.duplicate_dates.sum(),
                                 **{f'{stat}_peak_{w}': getattr(m[f'peak_{w}'], stat)()
                                    for w in (30, 60, 90) for stat in ('mean', 'max')}))
                for evaluator in ('ranker', 'regressor'):
                    frame = pd.read_csv(hybrid / reward / evaluator / str(year) / team / 'assignments.csv')
                    frame = frame[frame.roster.eq(roster)]
                    assert set(zip(frame.job, frame.rider_href)) == pairs
                    ref = summaries[(summaries.year == year) & (summaries.team == team)
                                    & (summaries.reward_model == reward)
                                    & (summaries.evaluator == evaluator) & (summaries.roster == roster)]
                    assert len(ref) == 1
                    for scoring, column in [('predicted', 'predicted_only_points'), ('hybrid', 'hybrid_points')]:
                        pts = frame.groupby('rider_href')[column].sum().reindex(squad, fill_value=0)
                        total = pts.sum()
                        assert total > 0
                        np.testing.assert_allclose(pts.nlargest(20).sum(), ref.iloc[0][column], atol=1e-6)
                        concentration.append(dict(year=year, team=team, allocation=allocation,
                                                  evaluator=evaluator, scoring=scoring,
                                                  n_riders=len(squad), gini_points=gini(pts),
                                                  top5_share=pts.nlargest(5).sum()/total,
                                                  top1_share=pts.max()/total,
                                                  top20_share=pts.nlargest(20).sum()/total,
                                                  zero_point_share=pts.eq(0).mean()))
    assert audit['team_seasons'] == 36
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'workload_team_summary.csv', index=False)
    pd.DataFrame(concentration).to_csv(out / 'concentration_team_summary.csv', index=False)
    if private_details:
        private = ROOT / 'outputs/workload_analysis'
        private.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rider_rows).to_csv(private / 'rider_workload.csv', index=False)
    (out / 'workload_validation.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Workspace with lawfully held reference inputs')
    parser.add_argument('--private-details', action='store_true')
    args = parser.parse_args()
    reconstruct(args.source.resolve(), ROOT / 'results', args.private_details)


if __name__ == '__main__':
    main()
