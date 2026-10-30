"""Generate Table D6 values from the included aggregate reference results."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CODES = {'alpecin-deceuninck': 'ADC', 'arkea-b-b-hotels': 'ARK',
         'astana-qazaqstan-team': 'AST', 'bahrain-victorious': 'TBV',
         'bora-hansgrohe': 'RBH', 'cofidis': 'COF', 'decathlon-ag2r-la-mondiale': 'DAT',
         'ef-education-easypost': 'EFE', 'groupama-fdj': 'GFC', 'ineos-grenadiers': 'IGD',
         'intermarche-wanty': 'IWA', 'lidl-trek': 'LTK', 'movistar-team': 'MOV',
         'red-bull-bora-hansgrohe': 'RBH', 'soudal-quick-step': 'SOQ',
         'team-dsm-firmenich-postnl': 'DFP', 'team-picnic-postnl': 'TPP',
         'team-jayco-alula': 'JAY', 'team-visma-lease-a-bike': 'TVL',
         'uae-team-emirates': 'UAD', 'uae-team-emirates-xrg': 'UAD', 'xds-astana-team': 'XAT'}


def table_frame():
    h = pd.read_csv(ROOT / 'results/team_summary.csv')
    h = h[h.reward_model.eq('ranker') & h.evaluator.eq('ranker')]
    original = h[h.roster.eq('original')].set_index(['year', 'team'])
    optimized = h[h.roster.eq('optimized')].set_index(['year', 'team'])
    assert len(original) == len(optimized) == 36
    assert original.index.is_unique and optimized.index.is_unique
    assert original.index.equals(optimized.index)
    frame = pd.DataFrame({'actual_original': original.actual_points,
                          'predicted_original': original.predicted_only_points,
                          'predicted_optimized': optimized.predicted_only_points,
                          'hybrid_optimized': optimized.hybrid_points,
                          'predicted_gain_pct': 100*(optimized.predicted_only_points/original.predicted_only_points-1),
                          'hybrid_gain_pct': optimized.hybrid_improvement_pct}).reset_index()
    frame['code'] = frame.team.map(CODES)
    assert frame.code.notna().all()
    return frame.sort_values(['year', 'code'])


def latex_table():
    frame = table_frame()
    lines = [r'\begin{table*}[htbp]', r'\centering',
        r'\caption{Reference-scenario scores (85-day cap, recorded injuries respected, ranker reward and evaluator). Each score sums its own 20 highest-scoring rider season totals. Actual points refer to the original allocation within the evaluated calendar; predicted points use selected-field scoring. The optimized hybrid total combines recorded points for historical participants with predictions for nonparticipants. Predicted gain compares the two predicted totals; hybrid gain compares the optimized hybrid total with the original actual total. Percentages are calculated before rounding. The final row reports unweighted means over 36 team-seasons, including means of individual percentage gains rather than percentage changes between mean totals.}',
        r'\label{tab:optimization-evaluation}', r'\small', r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{llrrrrrr}', r'\toprule',
        r'Season & Team & \shortstack{Actual\\original} & \shortstack{Predicted\\original} & \shortstack{Predicted\\optimized} & \shortstack{Hybrid\\optimized} & \shortstack{Predicted\\gain} & \shortstack{Hybrid\\gain} \\', r'\midrule']
    for year, group in frame.groupby('year'):
        if year != frame.year.min():
            lines.append(r'\midrule')
        for i, row in enumerate(group.itertuples()):
            fields = [str(year) if i == 0 else '', row.code]
            fields += [f'{getattr(row, col):,.0f}' for col in ['actual_original', 'predicted_original', 'predicted_optimized', 'hybrid_optimized']]
            fields += [f'{row.predicted_gain_pct:.1f}'+r'\%', f'{row.hybrid_gain_pct:.1f}'+r'\%']
            lines.append(' & '.join(fields)+r' \\')
    means = frame.mean(numeric_only=True)
    lines += [r'\midrule', r'\multicolumn{2}{l}{Mean} & '+ ' & '.join(
        [f'{means[c]:,.0f}' for c in ['actual_original', 'predicted_original', 'predicted_optimized', 'hybrid_optimized']]
        + [f'{means[c]:.1f}'+r'\%' for c in ['predicted_gain_pct', 'hybrid_gain_pct']])+r' \\',
        r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


if __name__ == '__main__':
    out = ROOT / 'reproduced'
    out.mkdir(exist_ok=True)
    table_frame().to_csv(out / 'table_D6.csv', index=False)
    (out / 'table_D6.tex').write_text(latex_table()+'\n', encoding='utf-8')
    print('Saved', out / 'table_D6.tex')
