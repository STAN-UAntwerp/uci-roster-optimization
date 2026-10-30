"""Regression tests for historical roster scoring and scenario evaluation."""

import json
from datetime import datetime
from types import SimpleNamespace

import gurobipy as gp
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

_data = Path(__file__).resolve().parents[1] / "data"
if not all((_data / name).is_file() for name in ("pcs_races_all.csv", "pcs_scales_all.csv")):
    pytest.skip("Roster integration tests require the undistributed reference scale/category tables.",
                allow_module_level=True)

from src.rostering import optimize, evaluation


def race(href, day, parent=None):
    return SimpleNamespace(href=href, date=datetime(2024, 1, day),
                           parent_href=parent, n_per_team=1, points=[100, 50], tt_kms=0)


def optimizer(races, **kwargs):
    return optimize.RosterOptimizer(["a"], races, None, [], None, [], year=2024,
                                    injuries=False, **kwargs)


def test_proseries_stage_scale_uses_explicit_historical_category():
    for year in (2024, 2025):
        points = optimize._get_uci_points(f"race/tour-of-belgium/{year}", True, year)
        expected = optimize.point_scales.loc[
            (optimize.point_scales.year == year)
            & (optimize.point_scales.category == "UCI.WR.Pro.Stage - 2023-2025")
        ].sort_values("rank")["points"].tolist()
        assert points == expected
        assert points[0] > 0


def test_tied_scores_award_each_place_once_and_are_order_independent():
    assert optimize._ranked_points(["b", "a", "c"], np.array([1, 1, 0]), [100, 50]) == {
        "a": 100, "b": 50, "c": 0,
    }
    assert optimize._ranked_points(["a", "b", "c"], np.array([1, 1, 0]), [100, 50]) == {
        "a": 100, "b": 50, "c": 0,
    }
    with pytest.raises(ValueError, match="finite"):
        optimize._ranked_points(["a"], np.array([np.nan]), [100])


def test_selected_field_removes_unused_teammates():
    o = optimize.RosterOptimizer(["a", "b"], [race("race", 1)], None, [], None, [])
    o._race_scores = {"race": (["a", "b", "other"], np.array([3, 2, 1]), [100, 50, 10])}
    o._gc_scores = {}
    totals, jobs = o._score_selection({"race": ["b"]})
    assert totals == {"a": 0, "b": 100}
    assert jobs == {"race": 100}
    impact = o._compute_other_team_impact(
        {"race": [["other", "opponent"]]}, {"race": ["a", "b"]}, {"race": ["b"]}
    )
    assert impact.iloc[0]["pts_with_original"] == 10
    assert impact.iloc[0]["pts_with_optimized"] == 50


def test_primary_points_objective_does_not_trade_points_for_participation():
    o = optimizer([race("tour/1", 1, "tour"), race("tour/2", 2, "tour"),
                   race("one", 1), race("two", 2)], max_race_days=2, top_n=0)
    model = o._create_milp_model(np.array([[5, 5, 4.9, 4.9]]))
    try:
        model.Params.OutputFlag = 0
        model.optimize()
        assert model.Status == gp.GRB.OPTIMAL
        assert o._assignment_vars["a", "tour"].X == 1
        assert o._assignment_vars["a", "one"].X == 0
        assert o._assignment_vars["a", "two"].X == 0
    finally:
        model.dispose()


def test_rest_days_block_overlap_without_consuming_race_day_budget():
    o = optimizer([race("tour/1", 1, "tour"), race("tour/2", 3, "tour"),
                   race("one", 2)], max_race_days=2)
    model = o._create_milp_model(np.array([[50, 50, 10]]))
    try:
        model.Params.OutputFlag = 0
        model.optimize()
        assert o._assignment_vars["a", "tour"].X == 1
        assert o._assignment_vars["a", "one"].X == 0
        assert o.rider_race_day_limits == {"a": 2}
    finally:
        model.dispose()


def test_injury_recovery_date_is_available(tmp_path, monkeypatch):
    injury_dir = tmp_path / "data" / "injuries"
    injury_dir.mkdir(parents=True)
    (injury_dir / "injuries.json").write_text(json.dumps({
        "a": [{"injury_date": "2024-01-01", "duration": "1"}]
    }))
    monkeypatch.setattr(optimize, "CWD", tmp_path / "src" / "rostering")
    o = optimizer([race("blocked", 7), race("available", 8)])
    o.injuries = True
    model = o._create_milp_model(np.array([[100, 100]]))
    try:
        model.Params.OutputFlag = 0
        model.optimize()
        assert o._assignment_vars["a", "blocked"].X == 0
        assert o._assignment_vars["a", "available"].X == 1
    finally:
        model.dispose()


def test_startlist_budget_never_silently_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(optimize, "CWD", tmp_path / "src" / "rostering")
    o = optimizer([race("one", 1)], use_actual_race_days=True)
    with pytest.raises(FileNotFoundError):
        o._get_rider_race_day_limits(["one"], {"one": 1})
    path = tmp_path / "data" / "startlists"
    path.mkdir(parents=True)
    (path / "startlists_2024.json").write_text('{}')
    with pytest.raises(ValueError, match="missing startlists"):
        o._get_rider_race_day_limits(["one"], {"one": 1})
    (path / "startlists_2024.json").write_text('{"one": [["a", "team"]]}')
    assert o._get_rider_race_day_limits(["one"], {"one": 1}) == {"a": 1}


def test_training_window_excludes_evaluation_and_future_years():
    frame = pd.DataFrame({"date": ["2018-01-01", "2019-01-01", "2023-12-31", "2024-01-01", "2025-01-01"],
                          "value": range(5)})
    assert evaluation._training_frame(frame, 2024, 5)["value"].tolist() == [1, 2]
    assert evaluation._training_frame(frame, 2025, 5)["value"].tolist() == [2, 3]


def test_undated_injury_is_audited_without_dropping_dated_absences():
    intervals, undated = optimize._injury_intervals({"a": [
        {"injury_date": "0000-00-00", "duration": "16"},
        {"injury_date": "2024-01-01", "duration": "1"},
    ]})
    assert intervals["a"] == [(datetime(2024, 1, 1), datetime(2024, 1, 8))]
    assert undated == [{"rider_href": "a", "injury_date": "0000-00-00", "duration": "16"}]
    with pytest.raises(ValueError, match="Invalid injury record"):
        optimize._injury_intervals({"a": [{"injury_date": "2024-01-01", "duration": -1}]})


def test_default_and_legacy_cli_scenario_selection(monkeypatch):
    monkeypatch.setattr(evaluation, "evaluate_scenarios", lambda c: c)
    config = evaluation.main([])
    assert config["years"] == [2024, 2025]
    assert len(list(evaluation._scenarios(config))) == 14
    config = evaluation.main(["--year", "2024", "--max_race_days", "75", "--injuries"])
    assert list(evaluation._scenarios(config)) == [
        ("max_race_days_75", "injuries_considered", 75, False, True)
    ]
    config = evaluation.main(["--use_actual_race_days", "--no-injuries"])
    assert list(evaluation._scenarios(config)) == [
        ("actual_race_days", "injuries_ignored", None, True, False)
    ]


def test_prepare_reuses_predictions_across_constraint_scenarios(monkeypatch):
    o = optimizer([race("one", 1)])
    calls = []
    monkeypatch.setattr(o, "_set_other_riders", lambda: calls.append("startlists"))
    weights = (np.ones((1, 1)), np.zeros((1, 1)))
    monkeypatch.setattr(o, "_build_weight_matrix", lambda: calls.append("predict") or weights)
    assert o.prepare() is weights
    o.injuries = True
    o.max_race_days = 75
    assert o.prepare() is weights
    assert calls == ["startlists", "predict"]


def test_leap_day_rolling_features_use_previous_february_end(tmp_path, monkeypatch):
    path = tmp_path / "data" / "unprocessed"
    path.mkdir(parents=True)
    for filename in ("results_2023.jsonl", "results_gc_2023.jsonl"):
        (path / filename).write_text(json.dumps({"a": [["2023-02-27", "10"]]}) + "\n")
    team_dir = tmp_path / "data" / "teams"
    team_dir.mkdir()
    (team_dir / "teams_2024.json").write_text('{"team": ["a"]}')
    monkeypatch.setattr(optimize, "CWD", tmp_path / "src" / "rostering")
    frame = pd.DataFrame({"rider_href": ["a"], "race_href": ["race"],
                          "race_date": [pd.Timestamp("2024-02-29")]})
    stage, gc = optimize._approx_rolling_features(frame.copy(), frame.copy(), 2024)
    assert stage.loc[0, "rolling_mean_uci_4w"] == 10
    assert gc.loc[0, "rolling_mean_uci_4w"] == 10
    assert stage.loc[0, "strength_vs_startlist"] == 1


def test_run_exports_consistent_top_n_totals(tmp_path, monkeypatch):
    startlists_dir = tmp_path / "data" / "startlists"
    startlists_dir.mkdir(parents=True)
    (startlists_dir / "startlists_2024.json").write_text('{"one": [["a", "team"]]}')
    monkeypatch.setattr(optimize, "CWD", tmp_path / "src" / "rostering")
    o = optimizer([race("one", 1)], top_n=1)
    o._prepared_weights = (np.array([[100.0]]), np.array([[0.0]]))
    o._race_scores = {"one": (["a"], np.array([1.0]), [100.0])}
    o._gc_scores = {}
    output = tmp_path / "output"
    result = o.run(output)
    assert result.iloc[0]["optimized_selected_field_pts"] == 100
    assert o.points_summary["total_optimized_pts"] == 100
    assert o.points_summary["total_original_pts"] == 100
    assert o.solver_stats["solution_count"] > 0
    assert json.loads((output / "optimized_roster.json").read_text()) == {"one": ["a"]}
    assert pd.read_csv(output / "per_rider_points.csv").iloc[0]["race_day_limit"] == 85


def test_scenario_runner_archives_inputs_and_reuses_models(tmp_path, monkeypatch):
    from src.models import ranking, ranking_gc
    import yaml

    config = yaml.safe_load((evaluation.CWD / "evaluation_config.yaml").read_text())
    config.update(years=[2024], max_race_days=[70, 85], include_actual_race_days=False,
                  output_root=str(tmp_path / "runs"))
    monkeypatch.setattr(evaluation, "_select_models", lambda c: {
        "STAGE": ({}, {}), "GC": ({}, {})
    })
    schedule = [race("one", 1)]
    monkeypatch.setattr(evaluation, "_preflight", lambda c: {
        2024: {"team/test-2024": ("test", ["a"], schedule)}
    })
    frame = pd.DataFrame({"date": ["2023-01-01"], "value": [1]})
    monkeypatch.setattr(ranking, "get_ranking_features", lambda: frame)
    monkeypatch.setattr(ranking_gc, "get_gc_ranking_features", lambda: frame)
    calls = []

    class Model:
        def save_model(self, path):
            path.write_text('{}')

    def fit(*args):
        calls.append("fit")
        return {"STAGE": (Model(), []), "GC": (Model(), [])}

    class Optimizer:
        def __init__(self, *args, **kwargs):
            pass

        def prepare(self):
            calls.append("prepare")

        def run(self, path):
            calls.append((self.max_race_days, self.injuries))
            self.solver_stats = {"solution_count": 1, "status": 2}
            self.points_summary = {"total_original_pts": 100, "total_optimized_pts": 150}
            return pd.DataFrame({"job": ["one"], "original_riders": [["a"]],
                                 "optimized_riders": [["a"]]})

        def score_rosters(self, frame):
            return frame[["job"]].assign(original_pts=100, optimized_pts=150), {
                "total_original_pts": 100, "total_optimized_pts": 150,
            }

    monkeypatch.setattr(evaluation, "_fit_models", fit)
    monkeypatch.setattr(evaluation, "RosterOptimizer", Optimizer)
    monkeypatch.setattr(evaluation, "_load_actual_points_table", lambda y: pd.DataFrame({
        "rider_href": ["a"], "race_href": ["one"], "uci_points": [90]
    }))
    per_race, summary = evaluation.evaluate_scenarios(config)
    assert calls == ["fit", "prepare", (70, False), (70, True), (85, False), (85, True)]
    assert len(per_race) == len(summary) == 4
    assert summary["actual_world_pts"].tolist() == [90] * 4
    assert summary["improvement_vs_predicted"].tolist() == [50] * 4
    from pathlib import Path
    output = Path(summary.attrs["output_dir"])
    assert json.loads((output / "metadata.json").read_text())["status"] == "complete"
    assert (output / "max_race_days_70/injuries_ignored/2024/team_summary.csv").is_file()
    assert (output / "models/2024/stage.json").is_file()

    # A second pass must read selections from disk and never invoke the solver.
    def must_not_solve(*args):
        raise AssertionError("Rescoring must not optimize")
    monkeypatch.setattr(Optimizer, "run", must_not_solve)
    config["rescore_run"] = str(output)
    config["evaluation_model"] = "regressor"
    monkeypatch.setattr(evaluation, "_fit_models", lambda *args: {
        "STAGE_REG": (Model(), []), "GC_REG": (Model(), [])
    })
    rescored, rescored_summary = evaluation.evaluate_scenarios(config)
    assert rescored["optimized_riders"].tolist() == per_race["optimized_riders"].tolist()
    assert "eval_ranker_optimized_pts" not in rescored
    assert rescored_summary["eval_regressor_total_optimized_pts"].tolist() == [150] * 4
    assert rescored_summary["total_optimized_pts"].tolist() == [150] * 4


def test_regression_rewards_use_scales_and_count_gc_once(monkeypatch):
    class Predictor:
        def predict(self, X):
            return X["p"].to_numpy()

    races = [race("s1", 1, "tour"), race("s2", 2, "tour")]
    o = optimize.RosterOptimizer(["a", "b"], races, Predictor(), ["p"], Predictor(), ["p"])
    stage = pd.DataFrame({"p": [-5, 3, 4, 6]}, index=pd.MultiIndex.from_tuples(
        [("a", "s1"), ("b", "s1"), ("a", "s2"), ("b", "s2")], names=["rider_href", "race_href"]))
    gc = pd.DataFrame({"p": [8, 7]}, index=pd.MultiIndex.from_tuples(
        [("a", "tour"), ("b", "tour")], names=["rider_href", "race_href"]))
    monkeypatch.setattr(o, "_get_features", lambda: (stage, gc))
    monkeypatch.setattr(optimize, "_get_uci_points", lambda *args, **kwargs: [100, 50])
    weights, gc_weights = o._build_weight_matrix()
    np.testing.assert_array_equal(weights, [[150, 50], [150, 100]])
    np.testing.assert_array_equal(gc_weights, [[100, 0], [50, 0]])
    o._prepared_weights = weights, gc_weights
    scores, totals = o.score_rosters(pd.DataFrame({"job": ["tour"],
                                                 "original_riders": [["a"]], "optimized_riders": [["b"]]}))
    assert totals["total_original_pts"] == 200
    assert totals["total_optimized_pts"] == 250
    assert totals["total_optimized_selected_field_pts"] == 300
    # Changing magnitudes without changing order cannot change the rewards.
    stage["p"] = stage["p"] * 100 + 1000
    gc["p"] = gc["p"] * 100 + 1000
    changed_weights, changed_gc = o._build_weight_matrix()
    np.testing.assert_array_equal(changed_weights, weights)
    np.testing.assert_array_equal(changed_gc, gc_weights)
    # Changing scales does change the rewards, without changing model predictions.
    for event in races:
        event.points = [200, 100]
    monkeypatch.setattr(optimize, "_get_uci_points", lambda *args, **kwargs: [200, 100])
    scaled_weights, _ = o._build_weight_matrix()
    np.testing.assert_array_equal(scaled_weights, weights * 2)


def test_alternative_evaluation_preserves_roster_and_can_reverse_improvement():
    o = optimize.RosterOptimizer(["a", "b"], [race("one", 1)], None, [], None, [])
    o._prepared_weights = np.array([[100.0], [50.0]]), np.zeros((2, 1))
    o._race_scores = {"one": (["a", "b"], np.array([100.0, 50.0]), [100, 50])}
    o._gc_scores = {}
    frame = pd.DataFrame({"job": ["one"], "original_riders": [["a"]], "optimized_riders": [["b"]],
                          "original_expected_pts": [50], "optimized_expected_pts": [100]})
    scored, summary = evaluation._add_evaluation(frame, {"regressor": o})
    pd.testing.assert_frame_equal(scored[frame.columns], frame)
    assert summary["eval_regressor_improvement_pts"] == -50
    assert summary["eval_regressor_improvement_pts_pct"] == -50


def test_ranker_and_regressor_fits_use_their_respective_api(monkeypatch):
    import xgboost
    calls = []

    class Estimator:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y, **kwargs):
            calls.append((self.params, X.copy(), y.copy(), kwargs))

    monkeypatch.setattr(xgboost, "XGBRanker", Estimator)
    monkeypatch.setattr(xgboost, "XGBRegressor", Estimator)
    frame = pd.DataFrame({"date": ["2023-01-01", "2024-01-01"], "x": [1, 2],
                          "rank_numeric": [1, 1], "uci_points": [12.2, 100]},
                         index=pd.MultiIndex.from_tuples([("race1", "a"), ("race2", "a")],
                                                        names=["race_href", "rider_href"]))
    models = evaluation._fit_models({"STAGE": frame}, {
        "STAGE": ({"objective": "rank:ndcg"}, {}),
        "STAGE_REG": ({"objective": "reg:squarederror"}, {}),
    }, 2024, 5)
    assert set(models) == {"STAGE", "STAGE_REG"}
    assert calls[0][3] == {"verbose": False, "group": [1]}
    assert calls[1][3] == {"verbose": False}
    assert calls[1][1]["x"].tolist() == [1]
    assert calls[1][2].tolist() == [12]


@pytest.mark.parametrize("reward,assessment,expected", [
    ("ranker", "same", {"STAGE", "GC"}),
    ("regressor", "same", {"STAGE_REG", "GC_REG"}),
    ("ranker", "regressor", {"STAGE", "GC", "STAGE_REG", "GC_REG"}),
    ("regressor", "both", {"STAGE", "GC", "STAGE_REG", "GC_REG"}),
])
def test_model_selection_loads_required_tuning_runs_only(monkeypatch, reward, assessment, expected):
    from src.models import evaluation as model_evaluation
    seen = []

    def load(path, name):
        seen.append(name)
        return {}, {"tuning_metadata": {"config": {"DATES": {
            "train_dates": [["2019-01-01", "2022-12-31"]],
            "test_dates": [["2023-01-01", "2023-12-31"]],
        }}}}

    monkeypatch.setattr(model_evaluation, "load_tuned_params", load)
    config = {"years": [2024], "reward_model": reward, "evaluation_model": assessment,
              "tuning_runs": {name: name for name in expected}}
    assert set(evaluation._select_models(config)) == expected
    assert set(seen) == expected


@pytest.mark.parametrize("family", ["ranker", "regressor"])
def test_reward_model_factory_routes_models(family):
    models = {name: (object(), []) for name in ("STAGE", "GC", "STAGE_REG", "GC_REG")}
    o = evaluation._make_optimizer(models, family, ["a"], [race("one", 1)], 2024,
                                   {"top_n": 20, "time_limit": None, "mip_gap": 0.001})
    stage, gc = evaluation._model_names(family)
    assert o.ranker is models[stage][0]
    assert o.gc_ranker is models[gc][0]
    assert not hasattr(o, "prediction_mode")


def test_rescore_cli_keeps_source_reward_scoring_and_top_n(tmp_path, monkeypatch):
    import yaml
    config = yaml.safe_load((evaluation.CWD / "evaluation_config.yaml").read_text())
    config.update(reward_model="regressor", regressor_scoring="points", top_n=10,
                  max_race_days=[71], years=[2024])
    (tmp_path / "metadata.json").write_text(json.dumps({"config": config}))
    monkeypatch.setattr(evaluation, "evaluate_scenarios", lambda c: c)
    result = evaluation.main(["--rescore-run", str(tmp_path), "--evaluation-model", "both"])
    assert result["reward_model"] == "regressor"
    assert result["source_reward_scoring"] == "points"
    assert result["top_n"] == 10
    assert result["max_race_days"] == [71]


def test_raw_prediction_scoring_option_is_removed():
    with pytest.raises(SystemExit):
        evaluation.main(["--regressor-scoring", "points"])
    with pytest.raises(TypeError):
        optimize.RosterOptimizer(["a"], [race("one", 1)], None, [], None, [], prediction_mode="points")
