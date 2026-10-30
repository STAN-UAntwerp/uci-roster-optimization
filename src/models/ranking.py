import datetime as dt
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from dataclasses import dataclass
from xgboost import XGBRanker
from pathlib import Path
from src.data_processing.cumulative_specialties import add_specialty_scores

root = Path(__file__).parent.parent.parent

@dataclass
class RankingModel:
    test_year: int | None = None
    train_years: int | None = None
    model: XGBRanker | None = None
    params: dict | None = None
    feature_names: list[str] | None = None
    features: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    test_data: pd.DataFrame | None = None
    
    def set_up(self):
        if self.features is None:
            self.features = get_ranking_features()
        self.model, self.feature_names, self.X_test, self.test_data = get_ranking_model(
            test_year=self.test_year,
            train_years=self.train_years,
            params=self.params,
            features=self.features,
        )
        print(f"Ranking model set up for test year {self.test_year} with {self.train_years} training years.")


def get_ranking_features() -> pd.DataFrame:
    """Build and return the complete feature frame used by ranking-model fits."""
    features = pd.read_csv(root / "models/features-2019-2025-detailed_gpx.csv")
    features["year"] = features["date"].apply(
        lambda x: dt.datetime.strptime(x, "%Y-%m-%d").year
    )
    
    rider_general = pd.read_csv(root / "data/rider_data/general_rider_data.csv")
    rider_general = rider_general[
        ["year_of_birth", "turned_pro_year", "rider_href", "height_cm", "weight_kg"]
    ].rename(columns={"height_cm": "height_m"})
    
    features_general = features.merge(rider_general, on="rider_href", how="left")
    features_general["age"] = (
        features_general["year"] - features_general["year_of_birth"]
    )
    features_general["years_pro"] = (
        (features_general["year"] - features_general["turned_pro_year"])
        .fillna(0)
        .clip(lower=0)
    )
    features_general["bmi"] = features_general["weight_kg"] / (
        (features_general["height_m"]) ** 2
    ).fillna(0)
    features_general["bmi"] = (
        features_general["bmi"].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    
    multiday_mask = features_general["level"].str.startswith("2.")
    features_general["is_stage_race"] = multiday_mask.astype(int)
    features_general.loc[multiday_mask, "race_href"] = (
        features_general.loc[multiday_mask, "race_href"]
        + "/stage-"
        + features_general.loc[multiday_mask, "stage_number"].astype(int).astype(str)
    )
    
    add_specialty_scores(features_general)
    
    # Career-adjusted specialty features (normalize by career length)
    years_pro_safe = np.maximum(features_general["years_pro"].fillna(3), 1)
    features_general["gc_per_year"] = features_general["gc"] / years_pro_safe
    features_general["tt_per_year"] = features_general["tt"] / years_pro_safe
    features_general["hills_per_year"] = features_general["hills"] / years_pro_safe
    features_general["climber_per_year"] = features_general["climber"] / years_pro_safe
    features_general["sprint_per_year"] = features_general["sprint"] / years_pro_safe
    features_general["onedayraces_per_year"] = (
        features_general["onedayraces"] / years_pro_safe
    )
    
    # Percentile versions of career-adjusted features (within-race normalization)
    features_general["gc_per_year_pctile"] = features_general.groupby("race_href")[
        "gc_per_year"
    ].rank(pct=True)
    features_general["tt_per_year_pctile"] = features_general.groupby("race_href")[
        "tt_per_year"
    ].rank(pct=True)
    features_general["hills_per_year_pctile"] = features_general.groupby("race_href")[
        "hills_per_year"
    ].rank(pct=True)
    features_general["climber_per_year_pctile"] = features_general.groupby("race_href")[
        "climber_per_year"
    ].rank(pct=True)
    features_general["sprint_per_year_pctile"] = features_general.groupby("race_href")[
        "sprint_per_year"
    ].rank(pct=True)
    features_general["onedayraces_per_year_pctile"] = features_general.groupby(
        "race_href"
    )["onedayraces_per_year"].rank(pct=True)
    
    features_general["climber_elevation_interaction"] = (
        features_general["climber_per_year"]
        * features_general["elevation_gain_per_km_m"]
    )
    
    features_general["hills_elevation_interaction"] = (
        features_general["hills_per_year"] * features_general["ascent_last_50km_m"]
    )
    
    features_general["recent_form"] = features_general[
        "rolling_mean_uci_4w"
    ] / features_general["rolling_mean_uci_53w"].replace(0, 1)
    
    features_general["strength_vs_startlist"] = features_general[
        "rolling_mean_uci_53w"
    ] / features_general["startlist_strength_53w"].replace(0, 1)
    
    stage_race_mask = features_general["is_stage_race"] == 1
    
    for col in ["previous_rank", "best_previous_rank", "mean_previous_uci_points"]:
        features_general[col] = np.where(stage_race_mask, np.nan, features_general[col])
    
    # GC rolling percentiles (within-race normalization)
    features_general["gc_rolling_mean_uci_53w_pctile"] = features_general.groupby(
        "race_href"
    )["gc_rolling_mean_uci_53w"].rank(pct=True)
    # features_general["gc_rolling_mean_uci_4w_pctile"] = features_general.groupby(
    #     "race_href"
    # )["gc_rolling_mean_uci_4w"].rank(pct=True)
    
    max_specialty = (
        features_general[["gc", "tt", "hills", "climber", "sprint", "onedayraces"]]
        .max(axis=1)
        .replace(0, 1)
    )
    features_general[["gc", "tt", "hills", "climber", "sprint", "onedayraces"]] = (
        features_general[["gc", "tt", "hills", "climber", "sprint", "onedayraces"]]
        .div(max_specialty, axis=0)
        .fillna(0)
    )
    features_general[
        [
            "strength_x_gc",
            "strength_x_tt",
            "strength_x_hills",
            "strength_x_climber",
            "strength_x_sprint",
            "strength_x_onedayraces",
        ]
    ] = features_general[["gc", "tt", "hills", "climber", "sprint", "onedayraces"]].mul(
        features_general["rolling_mean_uci_53w"], axis=0
    )
    
    features_clean = features_general.copy()
    features_clean = features_clean.set_index(["race_href", "rider_href"])
    features_clean = features_clean.drop_duplicates(keep="first")
    features_clean = features_clean.drop(
        columns=[
            "gpx_name",
            "stage_number",
            "rider_name",
            "rider_team",
            "rank",
            "year_of_birth",
            "turned_pro_year",
            "pcs_points",
            "uci_point_penalty",
            "level",
            "sprint",
            "climber",
            "hills",
            "gc",
            "tt",
            "onedayraces",
            "distance",
            "weight_kg",
            "height_m",
            "gc_rolling_mean_uci_4w",
        ]
    )
    return features_clean

def build_ranker_data(df: pd.DataFrame):
    """Sort by race and create bounded relevance labels + group sizes for XGBRanker."""
    df_sorted = df.sort_index(level=[0, 1]).copy()

    X_rank = df_sorted.drop(columns=["rank_numeric", "uci_points"])

    y_relevance = np.round(df_sorted["uci_points"]).astype(int)

    group_sizes = df_sorted.groupby(level=0, sort=False).size().astype(int).tolist()

    group_weights = (
        df_sorted.groupby(level=0, sort=False)["uci_points"]
        .max()
        .astype(int)
        .tolist()
    )

    return X_rank, y_relevance, group_sizes, df_sorted, group_weights

def get_ranking_model(
    test_year: int | None = None,
    train_years: int = 5,
    params: dict | None = None,
    features: pd.DataFrame | None = None,
    weights: bool = True,
):
    """
    Parameters
    ----------
    test_year : int, optional
        Hold-out year for evaluation.
    train_years : int, optional
        Number of years to use for training.
    params : dict, optional
        Parameters for the XGBRanker model. If None, default parameters are used.
    features : pandas.DataFrame, optional
        Pre-built feature frame from :func:`get_ranking_features`. Reuse this across
        parameter trials to avoid rebuilding features.
    """
    features_clean = get_ranking_features() if features is None else features.copy()

    if test_year is not None:
        X_test, y_test, group_test, test_df, weight_test = build_ranker_data(
            features_clean[features_clean["year"] == test_year].drop(columns=["date", "year"])
        )
            
        test_df = test_df[["rank_numeric", "uci_points"]].copy()
        
        features_clean = features_clean[
            features_clean["year"].isin(range(test_year - train_years, test_year))
        ].drop(columns=["date", "year"])
    else:
        X_test, y_test, group_test, test_df, weight_test = None, None, None, None, None
        features_clean = features_clean.drop(columns=["date", "year"])

    X_rank_train, y_rank_train, group_train, rank_train_df, weight_train = (
        build_ranker_data(features_clean)
    )
    
    params_preset = {
                    "objective": "rank:ndcg",
                    "ndcg_exp_gain": False,
                    "n_estimators": 500,
                    "eta": 0.03,
                    "max_depth": 6,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "random_state": 23,
                    "lambdarank_pair_method": "topk",
                    "lambdarank_num_pair_per_sample": 50,
                }
    if params is None:
        params = params_preset
    else: 
        params_preset.update(params)
        params = params_preset
        
    ranker = XGBRanker(**params)

    if not weights:
        weight_train = None
        
    ranker.fit(
        X_rank_train,
        y_rank_train,
        group=group_train,
        sample_weight=weight_train,
        verbose=False,
    )
    return (ranker, X_rank_train.columns.tolist(), X_test, test_df)


class ShapExplainer:
    """SHAP-based explainer for the UCI stage ranking model.

    Usage:
        explainer = ShapExplainer(ranker, X_test, test_df)
        explainer.explain_race("race/tour-de-france/2025/stage-1", top_k=10)
    """

    def __init__(
        self,
        ranker: XGBRanker,
        X_test: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        self.ranker = ranker
        self.X_test = X_test
        self.test_df = test_df
        self.feature_names = X_test.columns.tolist()
        self.explainer = shap.TreeExplainer(ranker)

    def _get_race_data(self, race_href: str) -> tuple[pd.DataFrame, np.ndarray]:
        race_mask = self.X_test.index.get_level_values("race_href") == race_href
        if not race_mask.any():
            available = (
                self.X_test.index.get_level_values("race_href").unique().tolist()
            )
            raise ValueError(
                f"Race '{race_href}' not found. Available races ({len(available)}):\n"
                + "\n".join(available[:20])
            )
        X_race = self.X_test.loc[race_mask]
        shap_values = self.explainer.shap_values(X_race)
        return X_race, shap_values

    def explain_race(self, race_href: str, top_k: int = 10):
        """Show SHAP waterfall plots for the top-k predicted riders in a race/stage."""
        import matplotlib.font_manager as _fm
        from pathlib import Path as _P

        # Register Inter from both system and user font directories
        _font_dirs = [
            _P(r"C:\Windows\Fonts"),
            _P.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts",
        ]
        for _fdir in _font_dirs:
            if _fdir.exists():
                for _ttf in _fdir.glob("Inter*.ttf"):
                    _fm.fontManager.addfont(str(_ttf))
                for _ttf in _fdir.glob("Inter*.otf"):
                    _fm.fontManager.addfont(str(_ttf))

        _available = {f.name for f in _fm.fontManager.ttflist}
        # Prefer "Inter 18pt" (static weights) over variable "Inter"
        if "Inter" in _available:
            _font_family = "Inter"
        else:
            _font_family = "sans-serif"
        plt.rcParams.update(
            {
                "font.family": _font_family,
                "font.sans-serif": [
                    "Inter",
                    "Helvetica Neue",
                    "Helvetica",
                    "Arial",
                ],
                "font.size": 14,
                "axes.titlesize": 16,
                "axes.labelsize": 14,
                "xtick.labelsize": 13,
                "ytick.labelsize": 13,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.spines.left": False,
                "axes.linewidth": 0.6,
                "figure.dpi": 300,
            }
        )

        X_race, shap_values = self._get_race_data(race_href)
        scores = self.ranker.predict(X_race)

        order = scores.argsort()[::-1]
        top_k = min(top_k, len(order))

        race_test = self.test_df.loc[
            self.test_df.index.get_level_values("race_href") == race_href
        ]

        print(f"{'=' * 70}")
        print(f"SHAP Analysis: {race_href}")
        print(f"Riders in race: {len(X_race)}, showing top {top_k}")
        print(f"{'=' * 70}")

        for rank_pos in range(top_k):
            idx = order[rank_pos]
            rider_href = X_race.index.get_level_values("rider_href")[idx]
            rider_name = rider_href.split("/")[-1] if "/" in rider_href else rider_href

            true_rank = "N/A"
            if rider_href in race_test.index.get_level_values("rider_href"):
                true_rank = int(
                    race_test.xs(rider_href, level="rider_href")["rank_numeric"].iloc[0]
                )

            print(f"\n{'─' * 70}")
            print(
                f"Predicted #{rank_pos + 1}: {rider_name}  "
                f"(score: {scores[idx]:.2f}, true rank: {true_rank})"
            )
            print(f"{'─' * 70}")

            rider_shap = shap_values[idx]
            rider_features = X_race.iloc[idx]

            contrib = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "value": rider_features.values,
                    "shap": rider_shap,
                }
            ).sort_values("shap", key=abs, ascending=False)

            print(f"  {'Feature':<35s} {'Value':>10s} {'SHAP':>10s}")
            for _, row in contrib.head(10).iterrows():
                sign = "+" if row["shap"] >= 0 else ""
                print(
                    f"  {row['feature']:<35s} {row['value']:>10.4f} {sign}{row['shap']:>9.4f}"
                )

            plt.figure(figsize=(12, 4))
            shap.waterfall_plot(
                shap.Explanation(
                    values=rider_shap,
                    base_values=self.explainer.expected_value,
                    data=rider_features.values,
                    feature_names=self.feature_names,
                ),
                max_display=5,
                show=False,
            )
            plt.title(
                f"Predicted #{rank_pos + 1}: {rider_name} (true rank: {true_rank})",
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout()
            plt.show()

    def compare_riders(self, race_href: str, rider_hrefs: list[str]):
        """Compare SHAP contributions side-by-side for specific riders."""
        X_race, shap_values = self._get_race_data(race_href)
        scores = self.ranker.predict(X_race)
        rider_index = X_race.index.get_level_values("rider_href")

        missing = [r for r in rider_hrefs if r not in rider_index]
        if missing:
            available = rider_index.tolist()
            raise ValueError(
                f"Riders not found in race: {missing}\n"
                f"Available riders ({len(available)}):\n" + "\n".join(available[:20])
            )

        print(f"{'=' * 70}")
        print(f"Rider Comparison: {race_href}")
        print(f"{'=' * 70}\n")

        rider_data = []
        for rider_href in rider_hrefs:
            pos = list(rider_index).index(rider_href)
            rider_name = rider_href.split("/")[-1] if "/" in rider_href else rider_href
            rider_data.append(
                {
                    "name": rider_name,
                    "href": rider_href,
                    "score": scores[pos],
                    "shap": shap_values[pos],
                    "features": X_race.iloc[pos],
                }
            )

        if len(rider_data) == 2:
            r1, r2 = rider_data
            diff = r1["shap"] - r2["shap"]
            diff_df = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    f"{r1['name']}_value": r1["features"].values,
                    f"{r2['name']}_value": r2["features"].values,
                    f"{r1['name']}_shap": r1["shap"],
                    f"{r2['name']}_shap": r2["shap"],
                    "shap_diff": diff,
                }
            ).sort_values("shap_diff", key=abs, ascending=False)

            print(
                f"Score: {r1['name']} = {r1['score']:.2f}, "
                f"{r2['name']} = {r2['score']:.2f} "
                f"(diff: {r1['score'] - r2['score']:+.2f})\n"
            )
            print(f"Top features explaining the difference:\n")
            print(
                f"  {'Feature':<30s} "
                f"{'Val(' + r1['name'][:8] + ')':>12s} "
                f"{'Val(' + r2['name'][:8] + ')':>12s} "
                f"{'SHAP diff':>10s}"
            )
            for _, row in diff_df.head(15).iterrows():
                sign = "+" if row["shap_diff"] >= 0 else ""
                col1 = f"{r1['name']}_value"
                col2 = f"{r2['name']}_value"
                print(
                    f"  {row['feature']:<30s} "
                    f"{row[col1]:>12.4f} "
                    f"{row[col2]:>12.4f} "
                    f"{sign}{row['shap_diff']:>9.4f}"
                )

        for rd in rider_data:
            pos = list(rider_index).index(rd["href"])
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[pos],
                    base_values=self.explainer.expected_value,
                    data=X_race.iloc[pos].values,
                    feature_names=self.feature_names,
                ),
                max_display=15,
                show=False,
            )
            plt.title(f"{rd['name']} (score: {rd['score']:.2f})", fontsize=12)
            plt.tight_layout()
            plt.show()

    def summary(self, race_href: str | None = None):
        """SHAP summary plot (beeswarm) for a race or the full test set."""
        if race_href:
            X_data, sv = self._get_race_data(race_href)
            title = f"SHAP Summary — {race_href}"
        else:
            sv = self.explainer.shap_values(self.X_test)
            X_data = self.X_test
            # X_data.to_csv(root / "outputs/stage_2024.csv", index=False)
            # np.save(root / "outputs/stage_shap_2024.npy", sv)
            title = "SHAP Summary — Full Test Set"

        plt.title(title, fontsize=14, pad=20)
        shap.summary_plot(sv, X_data, max_display=10, show=False, plot_size=(8, 6))
        ax = plt.gca()
        ax.tick_params(axis="y", labelsize=11)
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)
        labels = ax.get_yticklabels()
        ax.set_yticklabels(labels, fontsize=11)
        plt.tight_layout()
        plt.show()
