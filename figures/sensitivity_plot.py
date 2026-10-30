import matplotlib
matplotlib.use("Agg")
import pandas as pd
from pathlib import Path
import json
root = Path.cwd()
manifest = json.loads((root / "data_manifest.json").read_text())
frame = pd.read_csv(root / manifest["runs"]["ranker_solver"] / "team_summary.csv")
frame = frame[~frame.use_actual_race_days]
max_race_days_list = list(range(65, 91, 5))
column = "eval_ranker_total_optimized_selected_field_pts"
groups = frame.groupby(["max_race_days", "injuries_considered"])[column]
mean_points_inj = [groups.get_group((d, True)).mean() for d in max_race_days_list]
std_inj = [groups.get_group((d, True)).std(ddof=1) for d in max_race_days_list]
mean_points_uninj = [groups.get_group((d, False)).mean() for d in max_race_days_list]
std_uninj = [groups.get_group((d, False)).std(ddof=1) for d in max_race_days_list]
import matplotlib.pyplot as plt

plt.figure(figsize=(5.15, 3.27))
plt.style.use("seaborn-v0_8-whitegrid")
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

# Injuries Considered: Solid black circles
plt.errorbar(
    [rd - 0.5 for rd in max_race_days_list],
    mean_points_inj,
    yerr=std_inj,
    label="With injuries",
    fmt="o",
    color="#59267C",
    markerfacecolor="white",
    markeredgewidth=1.5,
    capsize=4,
    elinewidth=1.5,
)

# Injuries Ignored: Hollow gray squares
plt.errorbar(
    [rd + 0.5 for rd in max_race_days_list],
    mean_points_uninj,
    yerr=std_uninj,
    label="Without injuries",
    fmt="s",
    color="#4EA8DE",
    markerfacecolor="white",
    markeredgewidth=1.5,
    capsize=4,
    elinewidth=1.5,
)

plt.xlabel("Max Race Days", fontsize=11)
plt.ylabel("Total Optimized Points", fontsize=11)
plt.ylim(bottom=0)

# Added a distinct edge to the legend frame to separate it from the grid
plt.legend(loc="lower right", frameon=True, edgecolor="black", fontsize=10, framealpha=1)
plt.tight_layout()
plt.savefig("outputs/sensitivity_plot.pdf", format="pdf")
plt.close("all")
