"""Historical roster optimization and rescoring with tuned rankers or regressors.

Run ``python -m src.rostering.optimize --help`` for the CLI. No optimization
or model fitting is performed by ``--validate-only``.
"""

import argparse
import ast
import hashlib
import json
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.rostering.optimize import (
    CWD, RosterOptimizer, _get_actual_uci_points, _get_race_objects,
    _get_teams, _load_actual_points_table, _top_n_sum, _injury_intervals,
)

ROOT = CWD.parent.parent


def _evaluation_families(config):
    choice = config.get("evaluation_model", "same")
    if choice == "same":
        return [config.get("reward_model", "ranker")]
    return ["ranker", "regressor"] if choice == "both" else [choice]


def _model_names(family):
    if family not in {"ranker", "regressor"}:
        raise ValueError(f"Unknown model family: {family}")
    return ("STAGE_REG", "GC_REG") if family == "regressor" else ("STAGE", "GC")


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _select_models(config):
    from src.models.evaluation import load_tuned_params

    families = set(_evaluation_families(config))
    if not config.get("rescore_run"):
        families.add(config.get("reward_model", "ranker"))
    names = sorted({name for family in families for name in _model_names(family)})
    selected = {name: load_tuned_params(_path(config["tuning_runs"][name]), name)
                for name in names}
    cutoff = pd.Timestamp(f"{min(config['years'])}-01-01")
    for _, provenance in selected.values():
        windows = provenance["tuning_metadata"]["config"]["DATES"]
        if any(pd.Timestamp(end) >= cutoff
               for kind in ("test_dates", "train_dates") for _, end in windows[kind]):
            raise ValueError("Tuning training and validation must precede all evaluation years")
    return selected


def _training_frame(features, year, train_years):
    dates = pd.to_datetime(features["date"])
    frame = features.loc[(dates.dt.year >= year - train_years) & (dates.dt.year < year)]
    if frame.empty:
        raise ValueError(f"No training data before {year}")
    return frame.drop(columns=["date", "year"], errors="ignore")


def _fit_models(feature_frames, selected, year, train_years):
    from xgboost import XGBRanker, XGBRegressor
    from src.models.ranking import build_ranker_data

    models = {}
    for name in selected:
        frame = _training_frame(feature_frames[name.removesuffix("_REG")], year, train_years)
        X, y, groups, _, _ = build_ranker_data(frame)
        is_regressor = name.endswith("_REG")
        model = (XGBRegressor if is_regressor else XGBRanker)(**selected[name][0])
        # Match tuning exactly: rounded relevance, no race weights or test early stopping.
        model.fit(X, y, verbose=False, **({} if is_regressor else {"group": groups}))
        models[name] = (model, X.columns.tolist())
    return models


def _make_optimizer(models, family, riders, schedule, year, config):
    stage, gc = _model_names(family)
    return RosterOptimizer(
        riders, schedule, *models[stage], *models[gc], year=year,
        top_n=config["top_n"], strict_startlists=True,
        time_limit=config["time_limit"], mip_gap=config["mip_gap"],
    )


def _add_evaluation(frame, evaluators):
    """Add explicitly prefixed metrics; never change optimization selections."""
    frame = frame.copy()
    summaries = {}
    for family, evaluator in evaluators.items():
        scores, summary = evaluator.score_rosters(frame)
        prefix = f"eval_{family}_"
        for col in scores.columns.drop("job"):
            frame[prefix + col] = scores[col].to_numpy()
        summaries.update({prefix + key: value for key, value in summary.items()})
    return frame, summaries


def _scenarios(config):
    budgets = [(cap, False) for cap in config["max_race_days"]]
    if config["include_actual_race_days"]:
        budgets.append((None, True))
    for (cap, actual), injuries in product(budgets, config["injury_modes"]):
        budget_name = "actual_race_days" if actual else f"max_race_days_{cap}"
        injury_name = "injuries_considered" if injuries else "injuries_ignored"
        yield budget_name, injury_name, cap, actual, injuries


def _preflight(config):
    """Validate every requested team's calendar and historical inputs before fitting."""
    prepared = {}
    for year in config["years"]:
        required = [
            f"data/startlists/startlists_{year}.json", f"data/gpx/{year}.csv",
            f"data/results_with_gpx/{year}/race_to_gpx.json", f"data/profile_score/{year}.csv",
            f"data/unprocessed/results_{year - 1}.jsonl",
            *[f"data/unprocessed/results_gc_{y}.jsonl" for y in range(year - 3, year)],
            "data/unprocessed/rider_results.jsonl", "data/unprocessed/results_gc.jsonl",
            "data/rider_data/general_rider_data.csv", "data/rider_data/cumulative_specialties.csv",
            "models/features-2019-2025-detailed_gpx.csv", "models/gc_features_2019_2025.csv",
        ]
        if True in config["injury_modes"]:
            required.append("data/injuries/injuries.json")
        for rel in required:
            if not (ROOT / rel).is_file():
                raise FileNotFoundError(ROOT / rel)
        teams, programs = _get_teams(year)
        startlists = json.loads((ROOT / f"data/startlists/startlists_{year}.json").read_text(encoding="utf-8"))
        year_teams = {}
        for team in sorted(programs):
            name = team.split("/")[-1].removesuffix(f"-{year}")
            if config.get("teams") and name not in config["teams"] and team not in config["teams"]:
                continue
            schedule = _get_race_objects(programs[team])
            riders = sorted(set(teams.get(team, [])))
            if not riders or not schedule:
                raise ValueError(f"Empty squad or calendar for {team}")
            jobs = {r.parent_href or r.href for r in schedule}
            missing = sorted(j for j in jobs if not startlists.get(j))
            if missing:
                raise ValueError(f"Missing historical startlists for {team}: {missing}")
            if any(r.date.year != year or not r.points for r in schedule):
                raise ValueError(f"Invalid year or missing points scale in {team}'s calendar")
            year_teams[team] = (name, riders, schedule)
        if not year_teams:
            raise ValueError(f"No teams selected for {year}")
        prepared[year] = year_teams
    return prepared


def evaluate_scenarios(config):
    """Return per-race results and team summaries, and archive all run inputs."""
    from src.models import ranking, ranking_gc
    from src.models.tuning import _feature_fingerprint
    import gurobipy
    import xgboost

    config = dict(config)
    # Obsolete settings in older YAML files cannot enable raw point scoring.
    config.pop("regressor_scoring", None)
    config.pop("saved_reward_scoring", None)
    selected = _select_models(config)
    source_root = _path(config["rescore_run"]) if config.get("rescore_run") else None
    prepared = _preflight(config)
    undated_injuries = []
    if True in config["injury_modes"]:
        _, undated_injuries = _injury_intervals(json.loads(
            (ROOT / "data/injuries/injuries.json").read_text(encoding="utf-8")
        ))
        if undated_injuries:
            print(f"Injury data: {len(undated_injuries)} undated records cannot define availability; recorded in metadata.")
    if config.get("validate_only"):
        for name, (params, _) in selected.items():
            print(f"{name}: {params}")
        for year, teams in prepared.items():
            print(f"Validated {year}: {len(teams)} teams, {len(list(_scenarios(config)))} scenarios per team")
        return pd.DataFrame(), pd.DataFrame()

    output_dir = _path(config["output_root"]) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "config": config, "models": {name: data[1] for name, data in selected.items()},
        "sample_weights": False,
        "training": "preceding train_years only; 2025 may train on 2024",
        "workload": "actual_race_days is a startlist-derived scheduled-day budget; excludes rest days, includes stages after withdrawal",
        "points": "both model families rank predictions and apply UCI scales; selected_field removes unused teammates with frozen scores",
        "evaluation": "eval_<family> columns rescore identical original/optimized selections; objective columns retain the reward model's scores",
        "actual_points": "stage/one-day and GC points covered by the feature CSVs, restricted to the team program; not full official season totals",
        "injuries": "retrospective availability oracle; [injury_date, injury_date + duration weeks); missing rider record means no recorded injuries",
        "undated_injuries_omitted": undated_injuries,
        "comparison": "model-based counterfactual estimates; predicted_minus_actual is not realized improvement",
        "versions": {"xgboost": xgboost.__version__, "gurobi": gurobipy.gurobi.version(),
                     "pandas": pd.__version__, "numpy": np.__version__},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), CWD / "optimize.py", Path(ranking.__file__), Path(ranking_gc.__file__))},
        "status": "running",
    }
    if source_root:
        manifest["source_run"] = str(source_root.resolve())
        manifest["source_metadata_sha256"] = hashlib.sha256((source_root / "metadata.json").read_bytes()).hexdigest()
        manifest["solver_stats_origin"] = "copied from source run; no optimization performed"

    def save_manifest():
        (output_dir / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save_manifest()
    print(f"Saving evaluation to {output_dir}", flush=True)
    feature_frames = {"STAGE": ranking.get_ranking_features(), "GC": ranking_gc.get_gc_ranking_features()}
    manifest["feature_fingerprints"] = {
        name: _feature_fingerprint(frame.drop(columns="year", errors="ignore"))
        for name, frame in feature_frames.items()
    }
    save_manifest()
    race_frames, summaries, failures = [], [], []
    for year, teams in prepared.items():
        print(f"Fitting tuned models for {year}: {list(selected)}", flush=True)
        models = _fit_models(feature_frames, selected, year, config["train_years"])
        model_dir = output_dir / "models" / str(year)
        model_dir.mkdir(parents=True)
        for name, (model, _) in models.items():
            model.save_model(model_dir / f"{name.lower()}.json")
        actual_points = _load_actual_points_table(year)
        for team, (name, riders, schedule) in teams.items():
            reward_family = config.get("reward_model", "ranker")
            optimizer = None if source_root else _make_optimizer(models, reward_family, riders, schedule, year, config)
            print(f"Preparing {team}", flush=True)
            # Preparation failures abort rather than dropping a team from every scenario.
            if optimizer is not None:
                optimizer.prepare()
            evaluators = {
                family: optimizer if family == reward_family and optimizer is not None else
                _make_optimizer(models, family, riders, schedule, year, config)
                for family in _evaluation_families(config)
            }
            for evaluator in evaluators.values():
                if evaluator is not optimizer:
                    evaluator.prepare()
            actual_total = _top_n_sum(_get_actual_uci_points(
                actual_points, {r.parent_href or r.href for r in schedule}, set(riders)
            ), config["top_n"])
            for budget, injury_name, cap, actual, injuries in _scenarios(config):
                scenario_dir = output_dir / budget / injury_name / str(year)
                team_dir = scenario_dir / name
                team_dir.mkdir(parents=True, exist_ok=True)
                identity = {"year": year, "team": name, "team_href": team,
                            "max_race_days": cap, "use_actual_race_days": actual,
                            "injuries_considered": injuries, "top_n": config["top_n"],
                            "reward_model": reward_family,
                            "reward_scoring": config.get("source_reward_scoring", "rank") if source_root else "rank",
                            "evaluation_model": config.get("evaluation_model", "same"),
                            "evaluation_scoring": "rank"}
                if optimizer is not None:
                    optimizer.max_race_days = cap if cap is not None else 85
                    optimizer.use_actual_race_days = actual
                    optimizer.injuries = injuries
                action = "Rescoring" if source_root else "Solving"
                print(f"{action} {year} / {name} / {budget} / {injury_name}", flush=True)
                try:
                    if source_root:
                        saved_dir = source_root / budget / injury_name / str(year) / name
                        frame = pd.read_csv(saved_dir / "per_race_results.csv")
                        stats = pd.read_csv(saved_dir / "solver_stats.csv").iloc[0].to_dict()
                        if not stats.get("solution_count", 0):
                            raise ValueError("Saved scenario has no feasible roster to rescore")
                        for col in ("original_riders", "optimized_riders"):
                            frame[col] = frame[col].map(ast.literal_eval)
                        # Drop prior evaluator columns so they cannot masquerade as fresh results.
                        frame = frame.drop(columns=[c for c in frame if c.startswith("eval_")])
                        stats = {k: v for k, v in stats.items() if not k.startswith("eval_")}
                        (team_dir / "source_roster_sha256.txt").write_text(
                            hashlib.sha256((saved_dir / "per_race_results.csv").read_bytes()).hexdigest()
                        )
                    else:
                        frame = optimizer.run(team_dir)
                        stats = {**optimizer.solver_stats, **optimizer.points_summary}
                    evaluation_summary = {}
                    if stats.get("solution_count", 0):
                        frame, evaluation_summary = _add_evaluation(frame, evaluators)
                except Exception as exc:
                    failure = {**identity, "error": f"{type(exc).__name__}: {exc}"}
                    failures.append(failure)
                    pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)
                    print(f"FAILED: {failure['error']}", flush=True)
                    continue
                for key, value in identity.items():
                    frame[key] = value
                frame.to_csv(team_dir / "per_race_results.csv", index=False)
                race_frames.append(frame)
                summary = {**stats, **identity, "n_riders": len(riders),
                           "n_races": len({r.parent_href or r.href for r in schedule}),
                           "n_stages_and_one_day_races": len(schedule),
                           **evaluation_summary,
                           "actual_world_pts": stats.get("actual_world_pts", actual_total) if source_root else actual_total}
                summary["improvement_vs_predicted"] = summary["total_optimized_pts"] - summary["total_original_pts"]
                baseline = summary["total_original_pts"]
                summary["improvement_vs_predicted_pct"] = 100 * summary["improvement_vs_predicted"] / baseline if baseline else np.nan
                summary["predicted_minus_actual_pts"] = summary["total_optimized_pts"] - summary["actual_world_pts"]
                summaries.append(summary)
                pd.DataFrame([summary]).to_csv(team_dir / "solver_stats.csv", index=False)
                pd.DataFrame(summaries).to_csv(output_dir / "team_summary.csv", index=False)
    per_race = pd.concat(race_frames, ignore_index=True) if race_frames else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    per_race.to_csv(output_dir / "per_race_results.csv", index=False)
    # Retain the old per-scenario layout inside each new run for plotting notebooks.
    if not summary.empty:
        for budget, injury_name, cap, actual, injuries in _scenarios(config):
            for year in config["years"]:
                mask = ((summary["year"] == year) & (summary["use_actual_race_days"] == actual)
                        & (summary["injuries_considered"] == injuries))
                if not actual:
                    mask &= summary["max_race_days"] == cap
                subset = summary.loc[mask]
                scenario_dir = output_dir / budget / injury_name / str(year)
                scenario_dir.mkdir(parents=True, exist_ok=True)
                subset.to_csv(scenario_dir / "team_summary.csv", index=False)
                subset.to_csv(scenario_dir / "solver_stats.csv", index=False)
                race_mask = ((per_race["year"] == year) & (per_race["use_actual_race_days"] == actual)
                             & (per_race["injuries_considered"] == injuries))
                if not actual:
                    race_mask &= per_race["max_race_days"] == cap
                per_race.loc[race_mask].to_csv(scenario_dir / "per_race_results.csv", index=False)
    unsolved = int(summary["solution_count"].eq(0).sum()) if not summary.empty else 0
    manifest["status"] = "completed_with_failures" if failures or unsolved else "complete"
    manifest["failed_scenarios"] = len(failures)
    manifest["scenarios_without_solution"] = unsolved
    save_manifest()
    per_race.attrs["output_dir"] = summary.attrs["output_dir"] = str(output_dir)
    print(f"Saved evaluation to {output_dir}", flush=True)
    if failures or unsolved:
        raise RuntimeError(f"{len(failures)} scenarios failed and {unsolved} returned no solution; see {output_dir}")
    return per_race, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CWD / "evaluation_config.yaml")
    parser.add_argument("--year", "--years", dest="years", type=int, nargs="+")
    parser.add_argument("--max-race-days", "--max_race_days", dest="max_race_days", type=int, nargs="+")
    parser.add_argument("--use-actual-race-days", "--use_actual_race_days", action="store_true")
    parser.add_argument("--injuries", action="store_true", help="Run only scenarios considering injuries")
    parser.add_argument("--no-injuries", action="store_true", help="Run only scenarios ignoring injuries")
    parser.add_argument("--top-n", "--top_n", type=int)
    parser.add_argument("--train-years", type=int)
    parser.add_argument("--teams", nargs="+", help="Team slugs without year, or full team hrefs")
    parser.add_argument("--stage-tuning-run")
    parser.add_argument("--gc-tuning-run")
    parser.add_argument("--stage-reg-tuning-run")
    parser.add_argument("--gc-reg-tuning-run")
    parser.add_argument("--reward-model", choices=["ranker", "regressor"])
    parser.add_argument("--evaluation-model", choices=["same", "ranker", "regressor", "both"])
    parser.add_argument("--output-root")
    parser.add_argument("--time-limit", type=float, help="Seconds per solver run")
    parser.add_argument("--mip-gap", type=float)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--rescore-run", type=Path,
                        help="Rescore saved selections from a run directory without invoking the optimizer")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.rescore_run:
        if args.reward_model or args.max_race_days or args.use_actual_race_days or args.top_n is not None:
            parser.error("Rescoring preserves the saved reward model, workload scenarios and top_n")
        saved_config = json.loads((_path(args.rescore_run) / "metadata.json").read_text(encoding="utf-8"))["config"]
        config = {**saved_config, **{key: config.get(key, default) for key, default in (
            ("evaluation_model", "same"),
            ("tuning_runs", {}), ("output_root", "outputs/rostering_evaluation"),
        )}}
        config["rescore_run"] = str(_path(args.rescore_run).resolve())
        # Preserve the provenance of old runs; new evaluation always uses scales.
        config["source_reward_scoring"] = (
            saved_config.get("source_reward_scoring", saved_config.get("saved_reward_scoring", saved_config.get("regressor_scoring", "rank")))
            if saved_config.get("reward_model") == "regressor" else "rank"
        )
    for key in ("years", "max_race_days", "top_n", "train_years", "teams", "output_root", "time_limit", "mip_gap",
                "reward_model", "evaluation_model"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    for name in ("STAGE", "GC", "STAGE_REG", "GC_REG"):
        value = getattr(args, f"{name.lower()}_tuning_run")
        if value:
            config["tuning_runs"][name] = value
    if args.max_race_days:
        config["include_actual_race_days"] = False
    if args.use_actual_race_days:
        config["max_race_days"] = []
        config["include_actual_race_days"] = True
    if args.injuries and args.no_injuries:
        parser.error("Choose either --injuries or --no-injuries")
    if args.injuries or args.no_injuries:
        config["injury_modes"] = [args.injuries]
    config["validate_only"] = args.validate_only
    if (not config["years"] or len(set(config["years"])) != len(config["years"])
            or config["train_years"] < 1 or config["top_n"] < 0
            or any(cap < 0 for cap in config["max_race_days"])
            or config["mip_gap"] < 0
            or (config["time_limit"] is not None and config["time_limit"] <= 0)
            or not config["injury_modes"] or not list(_scenarios(config))):
        parser.error("Use distinct years, positive training/time limits, non-negative budgets/top_n/gap and at least one scenario")
    return evaluate_scenarios(config)


if __name__ == "__main__":
    main()
