"""Reproducible SHAP analyses using the rankers selected for model evaluation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRanker

from src.models import ranking, ranking_gc
from src.models.evaluation import load_tuned_params
from src.models.tuning import _feature_fingerprint

# Same explicit runs as models/model_evaluation.ipynb. Never select an
# unfinished tuning run just because it has a newer directory timestamp.
TUNING_RUNS = {
    "STAGE": ranking.root / "outputs/tuning/stage/2026-09-09_21-56-39",
    "GC": ranking.root / "outputs/tuning/gc/2026-09-10_08-06-42",
}


def fit_explanation_model(model, test_year=2025, train_years=5, *, run_dir=None,
                          features=None):
    """Fit exactly as evaluation does: tuned rounds, past years, no weights."""
    if model not in TUNING_RUNS or train_years < 1:
        raise ValueError("Use STAGE or GC and a positive training window")
    params, provenance = load_tuned_params(
        TUNING_RUNS[model] if run_dir is None else run_dir, model
    )
    windows = provenance["tuning_metadata"]["config"]["DATES"]["test_dates"]
    if any(pd.Timestamp(end) >= pd.Timestamp(f"{test_year}-01-01")
           for _, end in windows):
        raise ValueError("Tuning validation must end before the test year")
    if features is None:
        features = (ranking.get_ranking_features().drop(columns="year")
                    if model == "STAGE" else ranking_gc.get_gc_ranking_features())
    dates = pd.to_datetime(features["date"])
    train = features.loc[(dates.dt.year >= test_year - train_years)
                         & (dates.dt.year < test_year)].drop(columns="date")
    test = features.loc[dates.dt.year == test_year].drop(columns="date")
    if train.empty or test.empty:
        raise ValueError(f"Empty training or test data for {model} {test_year}")
    X_train, y_train, groups, _, _ = ranking.build_ranker_data(train)
    X_test, _, _, labels, _ = ranking.build_ranker_data(test)
    estimator = XGBRanker(**params)
    estimator.fit(X_train, y_train, group=groups, verbose=False)
    provenance.update(test_year=test_year, train_years=train_years,
                      sample_weights=False, feature_names=X_train.columns.tolist(),
                      feature_fingerprint=_feature_fingerprint(features))
    return estimator, X_test, labels[["rank_numeric", "uci_points"]], provenance


def save_summary(values, features, path):
    """Plot actual column names; never relabel features by their plot position."""
    with plt.rc_context(
        {
            "backend": "Agg",
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    ):
        plt.figure()
        shap.summary_plot(values, features, max_display=10, show=False,
                          plot_size=(5.15, 3.4), rng=np.random.default_rng(23))
        ax = plt.gca()
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlabel(ax.get_xlabel(), fontsize=10)
        ax.set_ylabel(ax.get_ylabel(), fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
        cbar = plt.gcf().get_axes()[-1]  # Gets the colorbar ax if it exists
        if cbar.get_ylabel():
            cbar.set_ylabel(cbar.get_ylabel(), fontsize=10)
        for text in cbar.texts:
            text.set_fontsize(10)
        cbar.tick_params(labelsize=10)
        plt.tight_layout()
        plt.savefig(path, format="pdf")
        plt.close()


def regenerate(output_dir=None):
    """Refresh 2024/2025 pooled summaries and the two 2025 race summaries."""
    output = Path(output_dir) if output_dir else ranking.root / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for model, prefix, race, race_plot in (
        ("STAGE", "stage", "race/milano-sanremo/2025", "msr_shap.pdf"),
        ("GC", "gc", "/race/tour-de-france/2025", "tdf_shap.pdf"),
    ):
        features = (ranking.get_ranking_features().drop(columns="year")
                    if model == "STAGE" else ranking_gc.get_gc_ranking_features())
        frames, arrays = [], []
        for year in (2024, 2025):
            estimator, X_test, labels, provenance = fit_explanation_model(
                model, year, features=features
            )
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(X_test)
            # Check that the explanations reconstruct the actual ranking scores.
            base = explainer.expected_value
            np.testing.assert_allclose(values.sum(axis=1) + base,
                                       estimator.predict(X_test), rtol=1e-4, atol=1e-4)
            if year == 2025:
                mask = X_test.index.get_level_values("race_href") == race
                if not mask.any():
                    raise ValueError(f"Missing race: {race}")
                save_summary(values[mask], X_test.loc[mask], output / race_plot)
            if frames and list(frames[0].columns) != list(X_test.columns):
                raise ValueError("Feature columns differ between test years")
            frames.append(X_test)
            arrays.append(values)
            print(f"Refreshed {model} {year}: {len(X_test)} rows", flush=True)
        save_summary(np.vstack(arrays), pd.concat(frames), output / f"{prefix}_shap.pdf")
    (output / "shap_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    regenerate()
