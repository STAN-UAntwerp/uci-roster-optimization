"""Shared ranking-evaluation utilities for the stage and GC models.

Computes, per race and then averaged:
    - top-k accuracy       (overlap of true vs predicted top-k)
    - top-k rank correlation (Spearman and Kendall on the true top-k riders)
    - NDCG@k

Results for the two test years (2024 and 2025) are aggregated as mean +/- std
so the stage and GC models can each be reported against a baseline.
"""

from __future__ import annotations

import warnings
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning
from xgboost import Booster, DMatrix

from src.models import ranking, ranking_gc

K_VALUES: tuple[int, ...] = (1, 3, 5, 10, 20, 25)
TEST_YEARS: tuple[int, ...] = (2024, 2025)
GC_LEVELS: tuple[str, ...] = ("2.UWT", "2.Pro", "2.1", "2.2")
UNCLASSIFIED_RANK = 999


# --------------------------------------------------------------------------- #
# Per-race metric primitives
# --------------------------------------------------------------------------- #
def _classified(group: pd.DataFrame) -> pd.DataFrame:
    """Return classified riders, with predicted ranks unaffected by rank 999 rows."""
    out = group[group["true_rank"] != UNCLASSIFIED_RANK].copy()
    if "score" in out.columns:
        out["pred_rank"] = out["score"].rank(
            ascending=False, method="first"
        ).astype(int)
    return out


def _top_k_accuracy(group: pd.DataFrame, k: int) -> float:
    """Fraction of the true top-k riders that appear in the predicted top-k."""
    group = _classified(group)
    if group.empty:
        return np.nan
    kk = min(k, len(group))
    true_top = set(group.nsmallest(kk, "true_rank")["rider_href"])
    pred_top = set(group.nsmallest(kk, "pred_rank")["rider_href"])
    return len(true_top & pred_top) / kk


def _top_k_correlation(group: pd.DataFrame, k: int, method: str) -> float:
    """Rank correlation (spearman/kendall) restricted to the true top-k riders."""
    group = _classified(group)
    top = group[group["true_rank"] <= k]
    if len(top) < 2 or top["pred_rank"].nunique() < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return top["true_rank"].corr(top["pred_rank"], method=method)


def _safe_nanmean(values: list[float]) -> float:
    """nanmean that returns NaN (without warning) for an all-NaN / empty list."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def _native_ndcg(eval_df: pd.DataFrame, k_values) -> dict[str, float]:
    """Evaluate fixed scores with XGBoost itself, without training any trees.

    Base margins supply predictions to a zero-tree identity-link booster.
    Sorting and rounded labels match build_ranker_data; native evaluation
    handles ties, singleton races, and zero ideal gain exactly as in tuning.
    Keep every supplied rider, including unclassified stage riders.
    """
    if eval_df.empty:
        return {f"NDCG@{k}": np.nan for k in k_values}
    ordered = eval_df.sort_values(["race_href", "rider_href"], kind="stable")
    data = DMatrix(
        np.zeros((len(ordered), 1), dtype=np.float32),
        label=np.round(ordered["uci_points"].to_numpy()).astype(int),
        base_margin=ordered["score"].to_numpy(),
        group=ordered.groupby("race_href", sort=False).size().to_numpy(),
        nthread=1,
    )
    booster = Booster({
        "objective": "reg:squarederror", "base_score": 0,
        "ndcg_exp_gain": False, "eval_metric": [f"ndcg@{k}" for k in k_values],
        "nthread": 1,
    }, cache=[data])
    values = dict(item.split(":", 1) for item in
                  booster.eval_set([(data, "evaluation")]).split("\t")[1:])
    return {f"NDCG@{k}": float(values[f"evaluation-ndcg@{k}"]) for k in k_values}


def _ndcg_at_k(group: pd.DataFrame, k: int) -> float:
    """Native linear-gain XGBoost NDCG for a single race."""
    frame = group.copy()
    frame["race_href"] = "race"
    if "rider_href" not in frame:
        frame["rider_href"] = np.arange(len(frame))
    return _native_ndcg(frame, (k,))[f"NDCG@{k}"]


# --------------------------------------------------------------------------- #
# Aggregation across races (single year)
# --------------------------------------------------------------------------- #
def evaluate_eval_df(eval_df: pd.DataFrame, k_values: tuple[int, ...] = K_VALUES) -> dict[str, float]:
    """Compute all metrics for one prediction dataframe, averaged over races.

    `eval_df` must contain race_href, rider_href, true_rank, score, uci_points.
    NDCG uses every supplied rider and race, matching tuning. Accuracy and
    correlations exclude rank 999 and races with fewer than two classified
    riders. Correlations measure ordering within the true top-k.
    """
    required = {"race_href", "rider_href", "true_rank", "score", "uci_points"}
    if missing := required.difference(eval_df.columns):
        raise ValueError(f"Missing evaluation columns: {sorted(missing)}")
    if any(k < 1 for k in k_values):
        raise ValueError("k values must be positive")
    native_ndcg = _native_ndcg(eval_df, k_values)
    metrics: dict[str, float] = {}
    for k in k_values:
        accs, spear, kend = [], [], []
        for _, grp in eval_df.groupby("race_href"):
            if len(_classified(grp)) < 2:
                continue
            accs.append(_top_k_accuracy(grp, k))
            if k >= 2:  # correlation is undefined for a single rider
                spear.append(_top_k_correlation(grp, k, "spearman"))
                kend.append(_top_k_correlation(grp, k, "kendall"))
        metrics[f"Acc@{k}"] = _safe_nanmean(accs)
        metrics[f"NDCG@{k}"] = native_ndcg[f"NDCG@{k}"]
        metrics[f"Spearman@{k}"] = _safe_nanmean(spear)
        metrics[f"Kendall@{k}"] = _safe_nanmean(kend)
    return metrics


def _add_pred_rank(eval_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Return a copy with a `pred_rank` derived from `score_col` (higher = better)."""
    out = eval_df.copy()
    out["score"] = out[score_col]
    out["pred_rank"] = (
        out.groupby("race_href")["score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    return out


# --------------------------------------------------------------------------- #
# Model-specific eval-frame builders
# --------------------------------------------------------------------------- #
def build_stage_eval(test_year: int, train_years: int = 5) -> pd.DataFrame:
    """Train the stage model for `test_year` and return a per-rider eval frame.

    Columns: race_href, rider_href, true_rank, model_score, baseline_score.
    Baseline = trailing 53-week rolling mean UCI points.
    """
    ranker, _feat, X_test, test_df = ranking.get_ranking_model(
        test_year=test_year, train_years=train_years
    )
    eval_df = test_df[["rank_numeric", "uci_points"]].copy()
    eval_df["model_score"] = ranker.predict(X_test)
    eval_df["baseline_score"] = X_test["rolling_mean_uci_53w"].fillna(0).values
    eval_df = eval_df.reset_index().rename(columns={"rank_numeric": "true_rank"})
    return eval_df


def build_gc_eval(test_year: int, train_years: int = 5) -> pd.DataFrame:
    """Train the GC model for `test_year` and return a per-rider eval frame.

    Columns: race_href, rider_href, true_rank, model_score, baseline_score.
    Baseline = trailing 53-week rolling mean UCI points.
    """
    ranker, feature_names = ranking_gc.get_gc_ranking_model(
        test_year=test_year, train_years=train_years
    )

    data = pd.read_csv(ranking_gc.root / "models" / "gc_features_2019_2025.csv", parse_dates=["date"])
    data = data[data["date"] >= pd.Timestamp(f"{test_year}-01-01")]
    data = data[data["date"] < pd.Timestamp(f"{test_year + 1}-01-01")]
    data = data[data["level"].isin(GC_LEVELS)]
    data = data[data["rank_numeric"] != UNCLASSIFIED_RANK]
    data = data.set_index(["race_href", "rider_href"], drop=True)
    data = data.drop(columns=["date", "level"])
    data = ranking_gc.engineer_gc_features(data)

    df_sorted = data.sort_index(level=[0, 1]).copy()
    X_test = df_sorted.drop(columns=["rank_numeric", "uci_points"])[feature_names]

    eval_df = df_sorted[["rank_numeric", "uci_points"]].copy()
    eval_df["model_score"] = ranker.predict(X_test)
    eval_df["baseline_score"] = X_test["rolling_mean_uci_53w"].fillna(0).values
    eval_df = eval_df.reset_index().rename(columns={"rank_numeric": "true_rank"})
    return eval_df


def build_gc_regression_data(
    test_year: int,
    train_years: int = 5,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return GC regression train/test matrices and the aligned test labels."""
    data = pd.read_csv(
        ranking_gc.root / "models" / "gc_features_2019_2025.csv",
        parse_dates=["date"],
    )
    data = data[data["level"].isin(GC_LEVELS)]
    data = data[data["rank_numeric"] != UNCLASSIFIED_RANK]

    train = data[
        (data["date"] >= pd.Timestamp(f"{test_year - train_years}-01-01"))
        & (data["date"] < pd.Timestamp(f"{test_year}-01-01"))
    ]
    test = data[
        (data["date"] >= pd.Timestamp(f"{test_year}-01-01"))
        & (data["date"] < pd.Timestamp(f"{test_year + 1}-01-01"))
    ]

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.set_index(["race_href", "rider_href"], drop=True)
        frame = frame.drop(columns=["date", "level"])
        return ranking_gc.engineer_gc_features(frame).sort_index(level=[0, 1])

    train = prepare(train)
    test = prepare(test)
    feature_cols = [
        column for column in train.columns if column not in {"rank_numeric", "uci_points"}
    ]
    X_train = train[feature_cols]
    y_train = train["uci_points"]
    X_test = test[feature_cols]
    y_test = test["uci_points"]
    test_df = test[["rank_numeric", "uci_points"]].copy()
    return X_train, y_train, X_test, y_test, test_df


def build_regression_eval(
    test_df: pd.DataFrame,
    X_test: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Return an evaluation frame for regression-model predictions."""
    if len(test_df) != len(predictions):
        raise ValueError("test_df and predictions must have the same length")
    if not test_df.index.equals(X_test.index):
        raise ValueError("test_df and X_test must have identical ordered indices")

    eval_df = (
        test_df[["rank_numeric", "uci_points"]]
        .copy()
        .reset_index()
        .rename(columns={"rank_numeric": "true_rank"})
    )
    eval_df["model_score"] = predictions
    eval_df["baseline_score"] = X_test["rolling_mean_uci_53w"].fillna(0).to_numpy()
    return eval_df


# --------------------------------------------------------------------------- #
# Cross-year aggregation (mean +/- std)
# --------------------------------------------------------------------------- #
def evaluate_model_over_years(
    build_fn,
    test_years: tuple[int, ...] = TEST_YEARS,
    train_years: int = 5,
    k_values: tuple[int, ...] = K_VALUES,
) -> pd.DataFrame:
    """Evaluate model and baseline across `test_years` and aggregate mean +/- std.

    `build_fn` is `build_stage_eval` or `build_gc_eval`.
    Returns a dataframe indexed by metric with columns:
        model_mean, model_std, baseline_mean, baseline_std.
    """
    model_per_year: list[dict[str, float]] = []
    baseline_per_year: list[dict[str, float]] = []

    for year in test_years:
        eval_df = build_fn(year, train_years=train_years)
        model_per_year.append(
            evaluate_eval_df(_add_pred_rank(eval_df, "model_score"), k_values)
        )
        baseline_per_year.append(
            evaluate_eval_df(_add_pred_rank(eval_df, "baseline_score"), k_values)
        )

    model_df = pd.DataFrame(model_per_year)
    baseline_df = pd.DataFrame(baseline_per_year)

    summary = pd.DataFrame(
        {
            "model_mean": model_df.mean(),
            "model_std": model_df.std(ddof=0),
            "baseline_mean": baseline_df.mean(),
            "baseline_std": baseline_df.std(ddof=0),
        }
    )
    return summary


def format_summary(summary: pd.DataFrame, title: str) -> str:
    """Render a mean +/- std comparison table as text."""
    lines = [f"=== {title} (mean +/- std across test years) ===",
             f"{'Metric':>14s}  {'Model':>18s}  {'Baseline':>18s}"]
    lines.append("-" * len(lines[-1]))
    for metric, row in summary.iterrows():
        if np.isnan(row["model_mean"]):
            lines.append(f"{metric:>14s}  {'--':>18s}  {'--':>18s}")
            continue
        model = f"{row['model_mean']:.4f} +/- {row['model_std']:.4f}"
        base = f"{row['baseline_mean']:.4f} +/- {row['baseline_std']:.4f}"
        star = " *" if row["model_mean"] >= row["baseline_mean"] else "  "
        lines.append(f"{metric:>14s}  {model:>18s}  {base:>18s}{star}")
    lines.append("\n* = model >= baseline on that metric")
    return "\n".join(lines)


def load_tuned_params(run_dir: str | Path, model: str) -> tuple[dict, dict]:
    """Select by mean validation NDCG and freeze rounds before seeing test data."""
    from src.models.tuning import _summarize_results

    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if metadata["model"] != model:
        raise ValueError(f"Expected {model}, found {metadata['model']} in {run_dir}")
    results = pd.read_csv(run_dir / "results.csv")
    parameter_columns = [c for c in results if c not in {
        "validation_start", "validation_end", "ndcg_at_20", "best_iteration"
    }]
    counts = results.groupby(parameter_columns, dropna=False).size()
    expected = len(metadata["config"]["DATES"]["test_dates"])
    if expected < 1 or not counts.eq(expected).all():
        raise ValueError("Every configuration must have all validation windows")
    best = _summarize_results(results).iloc[0]
    params = {key: best[key].item() if hasattr(best[key], "item") else best[key]
              for key in parameter_columns}
    # best_iteration is zero-based. Use the rounded mean number of trees from
    # validation, then refit without early stopping on the held-out years.
    params["n_estimators"] = max(1, int(np.floor(best["mean_best_iteration"] + 1.5)))
    params.pop("early_stopping_rounds", None)
    provenance = {
        "run_dir": str(run_dir.resolve()), "tuning_metadata": metadata,
        "results_sha256": hashlib.sha256((run_dir / "results.csv").read_bytes()).hexdigest(),
        "refit_params": params,
        "validation_ndcg_at_20": float(best["mean_ndcg_at_20"]),
    }
    return params, provenance


def comparison_table(per_year: pd.DataFrame, k_values=K_VALUES) -> pd.DataFrame:
    """Six columns, equal weight per test year, with metrics in paper order."""
    means = per_year.groupby(["task", "model", "metric"])["value"].mean()
    table = means.unstack(["task", "model"])
    columns = pd.MultiIndex.from_product(
        [["Stage/one-day", "GC"], ["Ranker", "XGBReg.", "Baseline"]]
    )
    rows = [f"{metric}@{k}" for metric in ("Acc", "NDCG", "Spearman")
            for k in k_values if metric != "Spearman" or k > 1]
    return table.reindex(index=rows, columns=columns).rename_axis("Metric")


def evaluate_tuned_models(
    tuning_runs: dict[str, str | Path],
    test_years: tuple[int, ...] = TEST_YEARS,
    train_years: int = 5,
    k_values: tuple[int, ...] = K_VALUES,
    output_root: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Refit four tuned models and archive predictions, metrics and paper table.

    Training uses the same feature builders, rounded point targets and no
    sample weights as tuning. Each test year uses only preceding training
    years; 2025 is thus a rolling evaluation that may train on 2024 outcomes.
    """
    from xgboost import XGBRanker, XGBRegressor
    import xgboost
    import sklearn
    from src.models.tuning import _feature_fingerprint

    names = ("STAGE", "STAGE_REG", "GC", "GC_REG")
    if set(tuning_runs) != set(names):
        raise ValueError(f"Supply an explicit tuning run for each of {names}")
    if not test_years or len(set(test_years)) != len(test_years) or train_years < 1:
        raise ValueError("Use distinct test years and a positive training window")
    selected = {name: load_tuned_params(tuning_runs[name], name) for name in names}
    for _, provenance in selected.values():
        windows = provenance["tuning_metadata"]["config"]["DATES"]["test_dates"]
        if any(pd.Timestamp(end) >= pd.Timestamp(f"{min(test_years)}-01-01")
               for _, end in windows):
            raise ValueError("Tuning validation must end before all test years")

    predictions, race_records = [], []
    fingerprints = {}
    for task, prefix in (("Stage/one-day", "STAGE"), ("GC", "GC")):
        features = (ranking.get_ranking_features().drop(columns=["year"])
                    if prefix == "STAGE" else ranking_gc.get_gc_ranking_features())
        fingerprints[task] = _feature_fingerprint(features)
        dates = pd.to_datetime(features["date"])
        for year in test_years:
            train_frame = features.loc[
                (dates.dt.year >= year - train_years) & (dates.dt.year < year)
            ].drop(columns="date")
            test_frame = features.loc[dates.dt.year == year].drop(columns="date")
            if train_frame.empty or test_frame.empty:
                raise ValueError(f"Empty training or test data for {task} {year}")
            X_train, y_train, groups, _, _ = ranking.build_ranker_data(train_frame)
            X_test, _, _, labels, _ = ranking.build_ranker_data(test_frame)
            frame = labels[["rank_numeric", "uci_points"]].copy()
            frame["Baseline"] = X_test["rolling_mean_uci_53w"].fillna(0)
            for name, display, cls in ((prefix, "Ranker", XGBRanker),
                                       (prefix + "_REG", "XGBReg.", XGBRegressor)):
                estimator = cls(**selected[name][0])
                kwargs = {"group": groups} if display == "Ranker" else {}
                estimator.fit(X_train, y_train, verbose=False, **kwargs)
                frame[display] = estimator.predict(X_test)
            frame = frame.reset_index().rename(columns={"rank_numeric": "true_rank"})
            frame["task"], frame["year"] = task, year
            predictions.append(frame)
            for display in ("Ranker", "XGBReg.", "Baseline"):
                scored = _add_pred_rank(frame, display)
                for race, group in scored.groupby("race_href"):
                    metrics = evaluate_eval_df(group, k_values)
                    race_records.extend({"task": task, "year": year, "model": display,
                                         "race_href": race, "n_riders": len(group),
                                         "metric": metric, "value": value}
                                        for metric, value in metrics.items())
            print(f"Evaluated {task} {year}")
    per_race = pd.DataFrame(race_records)
    if per_race.empty:
        raise ValueError("No races available for evaluation")
    per_year = per_race.groupby(["task", "year", "model", "metric"], as_index=False).agg(
        value=("value", "mean"), n_races=("value", "count")
    )
    table = comparison_table(per_year, k_values)
    output_root = Path(output_root) if output_root else ranking.root / "outputs" / "evaluation"
    output_dir = output_root / dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    per_race.to_csv(output_dir / "metrics_per_race.csv", index=False)
    per_year.to_csv(output_dir / "metrics_per_year.csv", index=False)
    table.to_csv(output_dir / "comparison.csv")
    table.to_latex(output_dir / "comparison.tex", float_format="%.2f", na_rep="--",
                   multicolumn_format="c", column_format="lcccccc", escape=True)
    metadata = {
        "test_years": list(test_years), "train_years": train_years, "k_values": list(k_values),
        "aggregation": "equal-weight races within year, then equal-weight test years",
        "ndcg": "native XGBoost ndcg@k, ndcg_exp_gain=False, rounded UCI points; native ties and zero IDCG=1",
        "population": "NDCG: all rows from tuning feature builders (stage includes rank 999; GC excludes it); Acc/correlations: classified races with at least two riders",
        "gc_levels": list(GC_LEVELS), "sample_weights": False,
        "rounds_rule": "rounded mean of validation best_iteration + 1; no test early stopping",
        "feature_fingerprints": fingerprints,
        "evaluation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"xgboost": xgboost.__version__, "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "models": {name: provenance for name, (_, provenance) in selected.items()},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return table, output_dir
