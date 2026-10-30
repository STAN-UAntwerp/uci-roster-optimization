import argparse
import datetime as dt
import hashlib
import json
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

from xgboost import DMatrix, XGBRanker, train
from src.models.ranking import build_ranker_data

CONFIG_PATH = Path(__file__).with_name("tuning_config.yaml")
OUTPUTS_PATH = Path(__file__).parent.parent.parent / "outputs" / "tuning"
RANKING_MODEL_PATH = Path(__file__).with_name("ranking.py")

with CONFIG_PATH.open() as file:
    config = yaml.safe_load(file)


def _get_parameter_combinations(model_config: dict) -> list[dict]:
    """Expand the grid, including pair-method-specific pair counts."""
    params = {}
    for key, value in model_config.items():
        if not isinstance(value, (list, dict)):
            params[key] = value

    tuning_param_names = [
        key for key, value in model_config.items() if isinstance(value, list)
    ]
    tuning_param_options = [model_config[key] for key in tuning_param_names]
    all_param_combinations = []
    pair_count_options = model_config.get("lambdarank_num_pair_per_sample", {})
    for combination in product(*tuning_param_options):
        combination_dict = params.copy()
        combination_dict.update(dict(zip(tuning_param_names, combination, strict=True)))
        method = combination_dict.get("lambdarank_pair_method")
        for pair_count in pair_count_options.get(method, [None]):
            params_with_pair_count = combination_dict.copy()
            if pair_count is not None:
                params_with_pair_count["lambdarank_num_pair_per_sample"] = pair_count
            if "num_round" in params_with_pair_count:
                params_with_pair_count["n_estimators"] = params_with_pair_count.pop(
                    "num_round"
                )
            all_param_combinations.append(params_with_pair_count)
    return all_param_combinations


def _get_validation_splits(features: pd.DataFrame, dates: dict) -> list[dict]:
    """Build training and validation rank matrices once for every date window."""
    validation_windows = dates.get("test_dates", [])
    train_windows = dates.get("train_dates", [])
    if len(validation_windows) != len(train_windows):
        raise ValueError("DATES.test_dates and DATES.train_dates must have equal lengths.")

    dated_features = features.copy()
    dated_features["date"] = pd.to_datetime(dated_features["date"])
    splits = []
    for validation_window, train_window in zip(
        validation_windows, train_windows, strict=True
    ):
        validation_start, validation_end = pd.to_datetime(validation_window)
        train_start, train_end = pd.to_datetime(train_window)

        features_train = dated_features.loc[
            (dated_features["date"] >= train_start)
            & (dated_features["date"] <= train_end)
        ].drop(columns=["date"])
        features_validation = dated_features.loc[
            (dated_features["date"] >= validation_start)
            & (dated_features["date"] <= validation_end)
        ].drop(columns=["date"])
        if features_train.empty or features_validation.empty:
            raise ValueError(
                f"Empty train or validation data for {validation_start.date()} to "
                f"{validation_end.date()}."
            )

        X_train, y_train, group_train, _, weight_train = build_ranker_data(
            features_train
        )
        X_validation, y_validation, group_validation, _, _ = build_ranker_data(
            features_validation
        )
        splits.append(
            {
                "validation_start": validation_start.date().isoformat(),
                "validation_end": validation_end.date().isoformat(),
                "X_train": X_train,
                "y_train": y_train,
                "group_train": group_train,
                "weight_train": weight_train,
                "X_validation": X_validation,
                "y_validation": y_validation,
                "group_validation": group_validation,
            }
        )
    return splits


def _feature_fingerprint(features: pd.DataFrame) -> str:
    """Return a stable fingerprint for the prepared features used in this run."""
    hasher = hashlib.sha256()
    hasher.update(pd.util.hash_pandas_object(features, index=True).values.tobytes())
    hasher.update("|".join(map(str, features.columns)).encode())
    hasher.update("|".join(map(str, features.dtypes)).encode())
    return hasher.hexdigest()


def _save_results(
    results_frame: pd.DataFrame,
    features: pd.DataFrame,
    model: str,
) -> Path:
    """Save results and the configuration/data version that produced them."""
    run_timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = OUTPUTS_PATH / model.lower() / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    results_frame.to_csv(output_dir / "results.csv", index=False)
    summary = _summarize_results(results_frame)
    summary.to_csv(output_dir / "summary.csv", index=False)
    with (output_dir / "best_params.json").open("w") as file:
        json.dump({key: summary.iloc[0][key].item()
                   if hasattr(summary.iloc[0][key], "item") else summary.iloc[0][key]
                   for key in _get_parameter_combinations(config[model])[0]}, file, indent=2)
    metadata = {
        "run_timestamp": run_timestamp,
        "model": model,
        "config": config,
        "feature_columns": list(map(str, features.columns)),
        "feature_dtypes": {
            str(column): str(dtype) for column, dtype in features.dtypes.items()
        },
        "feature_rows": len(features),
        "feature_fingerprint_sha256": _feature_fingerprint(features),
        "ranking_model_sha256": hashlib.sha256(RANKING_MODEL_PATH.read_bytes()).hexdigest(),
    }
    with (output_dir / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)
    return output_dir


def _summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Select a parameter set across windows, rather than a single lucky window."""
    parameter_columns = [column for column in results if column not in {
        "validation_start", "validation_end", "ndcg_at_20", "best_iteration"
    }]
    return (
        results.groupby(parameter_columns, dropna=False, as_index=False)
        .agg(mean_ndcg_at_20=("ndcg_at_20", "mean"),
             mean_best_iteration=("best_iteration", "mean"))
        .sort_values("mean_ndcg_at_20", ascending=False, ignore_index=True)
    )


def _fit_regressor(params: dict, split: dict):
    """Fit squared-error trees with native, per-race NDCG early stopping.

    The native API lets regression validation carry group sizes, which the
    XGBRegressor sklearn interface does not expose. Groups affect evaluation;
    the regression objective still trains on individual rider point targets.
    """
    train_params = params.copy()
    rounds = train_params.pop("n_estimators")
    patience = train_params.pop("early_stopping_rounds")
    train_params["seed"] = train_params.pop("random_state", 0)
    train_params.update(eval_metric="ndcg@20", ndcg_exp_gain=False)
    return train(
        train_params,
        split["regression_train"],
        num_boost_round=rounds,
        evals=[(split["regression_validation"], "validation")],
        early_stopping_rounds=patience,
        maximize=True,
        verbose_eval=False,
    )


def do_tuning(model: str = "STAGE") -> pd.DataFrame:
    """Tune rankers or regressors using validation NDCG@20."""
    if model not in {"STAGE", "GC", "STAGE_REG", "GC_REG"} or model not in config:
        raise ValueError(f"Model {model} not found in tuning configuration.")

    dates = config.get("DATES", {})
    parameter_combinations = _get_parameter_combinations(config[model])
    if model in {"STAGE", "STAGE_REG"}:
        from src.models.ranking import get_ranking_features
        features = get_ranking_features().drop(columns=["year"])
    elif model in {"GC", "GC_REG"}:
        from src.models.ranking_gc import get_gc_ranking_features
        features = get_gc_ranking_features()
    validation_splits = _get_validation_splits(features, dates)
    if not validation_splits:
        raise ValueError("At least one validation window is required.")
    is_regressor = model.endswith("_REG")
    if is_regressor:
        for split in validation_splits:
            split["regression_train"] = DMatrix(split["X_train"], label=split["y_train"])
            split["regression_validation"] = DMatrix(
                split["X_validation"], label=split["y_validation"],
                group=split["group_validation"],
            )
    results = []

    total_params = len(parameter_combinations) * len(validation_splits)
    print("Starting tuning...")
    done = 0
    for params in parameter_combinations:
        for split in validation_splits:
            if is_regressor:
                fitted_model = _fit_regressor(params, split)
            else:
                fitted_model = XGBRanker(**params, eval_metric=["ndcg@20"])
                fitted_model.fit(
                    split["X_train"],
                    split["y_train"],
                    group=split["group_train"],
                    # sample_weight=split["weight_train"],
                    eval_set=[(split["X_validation"], split["y_validation"])],
                    eval_group=[split["group_validation"]],
                    verbose=False,
                )
            results.append(
                {
                    **params,
                    "validation_start": split["validation_start"],
                    "validation_end": split["validation_end"],
                    "ndcg_at_20": float(fitted_model.best_score),
                    "best_iteration": fitted_model.best_iteration,
                }
            )
            done += 1
            print(f"Completed {done}/{total_params}")

    results_frame = pd.DataFrame(results).sort_values(
        "ndcg_at_20", ascending=False, ignore_index=True
    )
    results_frame.attrs["output_dir"] = str(_save_results(results_frame, features, model))
    return results_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="STAGE", help="Model to tune")
    args = parser.parse_args()
    results = do_tuning(args.model)
    print(f"Saved tuning run to {results.attrs['output_dir']}")
    print(results.to_string(index=False))
    print("Best parameters across validation windows:")
    print(_summarize_results(results).head(1).to_string(index=False))
