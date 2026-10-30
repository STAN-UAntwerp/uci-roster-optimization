import datetime as dt
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRanker

from pathlib import Path

root = Path(__file__).parent.parent.parent


def engineer_gc_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to GC ranking data.
    Input should have multi-index (race_href, rider_href) and NOT include date/level columns.
    """
    df = features.copy()

    years_pro_safe = np.maximum(df["years_pro"].fillna(3), 1)  # Avoid division by zero
    df["gc_per_year"] = df["gc"] / years_pro_safe
    df["tt_per_year"] = df["tt"] / years_pro_safe
    df["hills_per_year"] = df["hills"] / years_pro_safe
    df["climber_per_year"] = df["climber"] / years_pro_safe
    df["sprint_per_year"] = df["sprint"] / years_pro_safe
    df["onedayraces_per_year"] = df["onedayraces"] / years_pro_safe

    df.drop(
        columns=["gc", "tt", "hills", "climber", "sprint", "onedayraces"],
        inplace=True,
        errors="ignore",
    )

    df["strength_vs_startlist"] = df["rolling_mean_uci_53w"] / (
        df["startlist_strength_53w"].replace(0, 1)
    )

    # 2. Specialty-profile matching features
    df["tt_interaction"] = (df["tt_per_year"] * df["tt_kms"]).fillna(0)
    df["climbing_interaction"] = (
        df["climber_per_year"] * df["profile_score_mean"]
    ).fillna(0)
    df["climbing_interaction_final"] = (
        df["climber_per_year"] * df["profile_score_final_mean"]
    ).fillna(0)
    df["gc_profile_match"] = (df["gc_per_year"] * df["profile_score_mean"]).fillna(0)
    df["climbing_interaction_max"] = (
        df["climber_per_year"] * df["profile_score_max"]
    ).fillna(0)
    df["tt_profile_match"] = (
        df["tt_per_year"] * (df["tt_kms"] / df["number_of_stages"].replace(0, 1))
    ).fillna(0)

    # 3. Team strength features
    df["strength_vs_team"] = df["rolling_mean_uci_53w"] / df[
        "team_strength_53w"
    ].replace(0, 1)
    df["team_quality_ratio"] = df["team_strength_53w"] / (
        df["startlist_strength_53w"].replace(0, 1)
    )

    # 4. Historical performance features
    # For riders with no history, use their current form percentile within the race
    df["previous_rank_filled"] = df["previous_rank"].fillna(
        df.groupby(level=0)["rolling_mean_uci_53w"].transform(
            lambda x: x.rank(ascending=False).mean()
        )
    )
    df["has_race_history"] = (~df["previous_rank"].isna()).astype(int)

    df["gc_per_year_pctile"] = df.groupby(level=0)["gc_per_year"].rank(pct=True)
    df["tt_per_year_pctile"] = df.groupby(level=0)["tt_per_year"].rank(pct=True)
    df["hills_per_year_pctile"] = df.groupby(level=0)["hills_per_year"].rank(pct=True)
    df["climber_per_year_pctile"] = df.groupby(level=0)["climber_per_year"].rank(
        pct=True
    )
    df["sprint_per_year_pctile"] = df.groupby(level=0)["sprint_per_year"].rank(pct=True)
    df["onedayraces_per_year_pctile"] = df.groupby(level=0)[
        "onedayraces_per_year"
    ].rank(pct=True)

    df["total_specialty_per_year"] = (
        df["gc_per_year"]
        + df["climber_per_year"]
        + df["tt_per_year"]
        + df["sprint_per_year"]
        + df["hills_per_year"]
        + df["onedayraces_per_year"]
    )

    return df

def get_gc_ranking_features(test_year=None, train_years: int = 5):
    features = pd.read_csv(root / "models" / "gc_features_2019_2025.csv").set_index(
        [
            "race_href",
            "rider_href",
        ],
        drop=True,
    )
    if test_year:
        features = features[features["date"] < f"{test_year+1}-01-01 00:00:00"]
        features = features[features["date"] >= f"{test_year - train_years}-01-01 00:00:00"]

    features = features[features["level"].isin(["2.UWT", "2.Pro", "2.1", "2.2"])]
    features = features[features["rank_numeric"] != 999]

    features = features.drop(columns=["level"])

    features = engineer_gc_features(features)
    return features

def get_gc_ranking_model(test_year=None, train_years: int = 5, weights=True):
    features = get_gc_ranking_features(test_year, train_years)
    features_train = features[features["date"] < f"{test_year}-01-01 00:00:00"].copy().drop(columns=["date"])

    from src.models.ranking import build_ranker_data
    X_rank_train, y_rank_train, group_train, rank_train_df, weight_train = (
        build_ranker_data(features_train)
    )

    ranker = XGBRanker(
        objective="rank:ndcg",
        ndcg_exp_gain=False,
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=23,
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=50,
        missing=np.nan,
    )

    if not weights:
        weight_train = None

    ranker.fit(
        X_rank_train,
        y_rank_train,
        group=group_train,
        sample_weight=weight_train,
        verbose=False,
    )
    return ranker, X_rank_train.columns.tolist()


class GCShapExplainer:
    """SHAP-based explainer for the GC ranking model.

    Usage:
        explainer = GCShapExplainer(ranker, X_test, test_df)
        explainer.explain_race("race/tour-de-france/2025", top_k=10)
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
        """Get features and SHAP values for a specific race."""
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
        """Show SHAP waterfall plots for the top-k predicted riders in a race.

        Parameters
        ----------
        race_href : str
            The race identifier (index level-0 value in X_test).
        top_k : int
            Number of top-predicted riders to explain.
        """
        X_race, shap_values = self._get_race_data(race_href)
        scores = self.ranker.predict(X_race)

        # Rank riders by predicted score (descending)
        order = scores.argsort()[::-1]
        top_k = min(top_k, len(order))

        # Get true ranks from test_df
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

            # True rank if available
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

            # Top contributions table
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

            # Waterfall plot
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=rider_shap,
                    base_values=self.explainer.expected_value,
                    data=rider_features.values,
                    feature_names=self.feature_names,
                ),
                max_display=15,
                show=False,
            )
            plt.title(
                f"Predicted #{rank_pos + 1}: {rider_name} (true rank: {true_rank})",
                fontsize=12,
            )
            plt.tight_layout()
            plt.show()

    def compare_riders(self, race_href: str, rider_hrefs: list[str]):
        """Compare SHAP contributions side-by-side for specific riders in a race."""
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

        # Show top differential features
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

        # Waterfall for each rider
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
            # np.save(root / "outputs/shap_summary_2025.npy", sv)
            title = "SHAP Summary — Full Test Set"

        plt.title(title, fontsize=14, pad=20)
        shap.summary_plot(
            sv, X_data, max_display=10, show=False, plot_size=(8, 6)
        )
        ax = plt.gca()
        ax.tick_params(axis="y", labelsize=11)
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)
        plt.tight_layout()
        plt.show()
