import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent

plt.style.use("seaborn-v0_8-whitegrid")

team_href_to_uci_code = {
    "alpecin-deceuninck": "ADC",
    "arkea-b-b-hotels": "ARK",
    "astana-qazaqstan-team": "AST",
    "bahrain-victorious": "TBV",
    "bora-hansgrohe": "RBH",
    "cofidis": "COF",
    "decathlon-ag2r-la-mondiale": "DAT",
    "ef-education-easypost": "EFE",
    "groupama-fdj": "GFC",
    "ineos-grenadiers": "IGD",
    "intermarche-wanty": "IWA",
    "lidl-trek": "LTK",
    "movistar-team": "MOV",
    "red-bull-bora-hansgrohe": "RBH",
    "soudal-quick-step": "SOQ",
    "team-dsm-firmenich-postnl": "DFP",
    "team-picnic-postnl": "TPP",
    "team-jayco-alula": "JAY",
    "team-visma-lease-a-bike": "TVL",
    "uae-team-emirates": "UAD",
    "uae-team-emirates-xrg": "UAD",
    "xds-astana-team": "XAT",
}

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
def create_optimization_plot(csv_file, year):
    # Load data
    df = pd.read_csv(csv_file)
    df = df[(df["year"] == int(year))
        & (df["reward_model"] == "ranker")
        & (df["evaluator"] == "ranker")]
    optimized = df[df["roster"] == "optimized"].set_index("team")
    original = df[df["roster"] == "original"].set_index("team")
    assert optimized.index.is_unique and original.index.is_unique
    assert set(optimized.index) == set(original.index)
    order = optimized.sort_values("predicted_only_points").index
    optimized = optimized.loc[order]
    original = original.loc[order]
    y = np.arange(len(order))
    # Use exactly the same team order for all plotted series.
    np.testing.assert_allclose(
        optimized["actual_points"] + optimized["predicted_not_raced_points"]
        + optimized["predicted_missing_actual_points"], optimized["hybrid_points"],
        rtol=0, atol=1e-6,
    )
    print(f"{year}: predicted fallback for unverified actuals contributes "
          f"{100 * optimized.predicted_missing_actual_points.sum() / optimized.hybrid_points.sum():.3f}% "
          "of the pooled optimized hybrid score.")

    fig, ax = plt.subplots(figsize=(5.15, 3.4))
    height = 0.3  # height of the bars

    # Create horizontal bars
    ax.barh(
        y - height,
        original["predicted_only_points"],
        height,
        hatch="//",
        edgecolor="black",
        label="Pred. pts (original)",
        color="#4EA8DE",
    )

    ax.barh(
        y,
        optimized["actual_points"],
        height,
        edgecolor="black",
        label="Actual pts (hybrid)",
        hatch=r"\\",
        color="#00E664",
    )

    ax.barh(
        y,
        optimized["hybrid_points"] - optimized["actual_points"],
        height,
        edgecolor="black",
        left=optimized["actual_points"],
        label="Pred. pts (hybrid)",
        hatch=r"\\",
        color="#E69F00",
    )

    ax.barh(
        y + height,
        optimized["predicted_only_points"],
        height,
        edgecolor="black",
        label="Pred. pts (opt.)",
        color="#59267C",
    )

    # Original allocations use actual points only, including zero for DNS.
    # It is not a complete official annual UCI points total.
    ax.scatter(
        original["hybrid_points"],
        y,
        color="black",
        marker="o",
        edgecolors="white",
        linewidths=1,
        s=20,
        zorder=3,
        label="Actual pts (original)",
    )

    ax.set_ylim(-0.5, len(order) - 0.5)

    ax.set_yticks(y)
    ax.set_yticklabels([team_href_to_uci_code.get(t, t) for t in order])
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(loc="lower right", frameon=True, edgecolor="black", fontsize=10, framealpha=1)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    ax.set_xlabel("Season points (top 20 riders)")
    plt.tight_layout()
    plt.savefig(ROOT / f"outputs/roster_optimization_{year}.pdf", format="pdf")
    plt.close(fig)


# Generate plots for both years
create_optimization_plot(
    ROOT / "outputs/hybrid_evaluation/2026-09-14_16-45-11_478985/team_summary.csv",
    "2024",
)
create_optimization_plot(
    ROOT / "outputs/hybrid_evaluation/2026-09-14_16-45-11_478985/team_summary.csv",
    "2025",
)