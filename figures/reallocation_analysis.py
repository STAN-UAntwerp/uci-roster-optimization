import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
RUN = ROOT / "outputs/hybrid_evaluation/2026-09-14_16-45-11_478985"
summary = pd.read_csv(RUN / "team_summary.csv")
levels = ["1.UWT", "1.Pro", "1.1", "2.UWT", "2.Pro", "2.1"]
team_pairs = {
    "IWA": {2024: "intermarche-wanty", 2025: "intermarche-wanty"},
    "DFP/TPP": {2024: "team-dsm-firmenich-postnl", 2025: "team-picnic-postnl"},
    "UAD": {2024: "uae-team-emirates", 2025: "uae-team-emirates-xrg"},
}
records = []
for code, seasons in team_pairs.items():
    for year, team in seasons.items():
        programs = json.loads((ROOT / f"data/roster/programs_{year}.json").read_text())
        program = programs[f"team/{team}-{year}"]
        race_levels = {job: details["level"] for job, details in program.items()}
        assignments = pd.read_csv(RUN / "ranker/ranker" / str(year) / team / "assignments.csv")
        assignments = assignments.loc[assignments.max_race_days.eq(85)
                                      & assignments.injuries_considered
                                      & ~assignments.use_actual_race_days]
        for roster in ("original", "optimized"):
            selected = assignments.loc[assignments.roster.eq(roster)].copy()
            assert not selected.duplicated(["job", "rider_href"]).any()
            assert selected.top_n.nunique() == 1
            top_n = int(selected.top_n.iloc[0])
            # Select the team's season top 20 ONCE, not separately per category.
            # This figure uses pure predicted awards, never the hybrid ranking.
            totals = (selected.groupby("rider_href", as_index=False).predicted_only_points.sum()
                      .sort_values(["predicted_only_points", "rider_href"], ascending=[False, True]))
            counted = totals.head(top_n) if top_n else totals
            selected = selected.loc[selected.rider_href.isin(counted.rider_href)]
            expected = summary.loc[summary.year.eq(year) & summary.team.eq(team)
                                   & summary.reward_model.eq("ranker") & summary.evaluator.eq("ranker")
                                   & summary.roster.eq(roster) & summary.max_race_days.eq(85)
                                   & summary.injuries_considered & ~summary.use_actual_race_days,
                                   "predicted_only_points"]
            assert len(expected) == 1
            np.testing.assert_allclose(selected.predicted_only_points.sum(), expected.iloc[0], rtol=0, atol=1e-6)
            selected["level"] = selected.job.map(race_levels)
            assert selected.level.notna().all()
            assert set(selected.level) <= set(levels)
            by_level = selected.groupby("level").predicted_only_points.sum().reindex(levels, fill_value=0)
            for level, points in by_level.items():
                records.append({"team": code, "year": year, "roster": roster,
                                "level": level, "predicted_points": points})

category_points = pd.DataFrame(records)
by_year = category_points.pivot(index=["team", "year", "level"], columns="roster", values="predicted_points")
by_year["point_increase"] = by_year.optimized - by_year.original
# Annual mean across the two seasons, not a cumulative two-season total.
category_gains = by_year.groupby(["team", "level"]).point_increase.mean().unstack("level").reindex(index=team_pairs, columns=levels)
pts = category_gains.to_numpy().tolist()
category_points.to_csv(ROOT / "outputs/reallocation_category_points.csv", index=False)
by_year.to_csv(ROOT / "outputs/reallocation_gains_per_year.csv")
category_gains.to_csv(ROOT / "outputs/reallocation_mean_annual_gains.csv")
print("Figure 9: mean annual selected-field predicted gains, with each roster's own season top 20.")
print(category_gains.round(2).to_string())


# Values plotted below: mean annual gains, with categories in manuscript order.
category_gains


import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

plt.figure(figsize=(5.15, 3.27))
plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX to render all text
        "font.family": "serif",  # Match the LaTeX font family
        "font.serif": ["Computer Modern"],  # The standard LaTeX font
        "font.size": 10,  # Match the manuscript font size
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)
plt.bar([i-0.3 for i in range(6)], pts[0], edgecolor="black", label="IWA", width=0.3, color="#59267C")
plt.bar(
    range(6),
    pts[1],
    hatch="///",
    edgecolor="black",
    label="DFP/TPP",
    width=0.3,
    color="#C85200",
)
plt.bar(
    [i + 0.3 for i in range(6)],
    pts[2],
    hatch="ooo",
    edgecolor="black",
    label="UAD",
    width=0.3,
    color="#4EA8DE",
)
plt.xlabel("Level")
plt.xticks(range(6), ["1.UWT", "1.Pro", "1.1", "2.UWT", "2.Pro", "2.1"])
plt.xlim(-0.55, 5.55)
plt.ylabel("Mean annual predicted point increase")
plt.legend(frameon=True, edgecolor="black", framealpha=1)
plt.tight_layout()
plt.axhline(0, color="black", linewidth=0.6)
plt.savefig(ROOT / "outputs/reallocation_analysis.pdf", format="pdf")
plt.close("all")