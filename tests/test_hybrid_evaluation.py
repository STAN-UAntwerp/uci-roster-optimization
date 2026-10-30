import numpy as np
import pytest

from src.rostering.hybrid_evaluation import hybrid_assignments, summarize_assignments


def test_zero_actual_replaces_prediction_and_unknown_remains_visible():
    predictions = {("race", "zero"): 100, ("race", "absent"): 40, ("race", "unknown"): 20}
    actual = {
        ("race", "zero"): {"actual_known": True, "actual_points": 0, "actual_status": "classified"},
        ("race", "unknown"): {"actual_known": False, "actual_points": np.nan, "actual_status": "missing"},
    }
    rows = hybrid_assignments(predictions, actual).set_index("rider_href")
    assert rows.loc["zero", "hybrid_points"] == 0
    assert rows.loc["absent", "predicted_not_raced_points"] == 40
    assert rows.loc["unknown", "predicted_missing_actual_points"] == 20
    assert rows.loc["unknown", "predicted_not_raced_points"] == 0


def test_top_n_uses_hybrid_season_total_then_splits_same_riders():
    predictions = {("r1", "a"): 100, ("r1", "b"): 50, ("r2", "a"): 20, ("r2", "b"): 30}
    actual = {("r1", "a"): {"actual_known": True, "actual_points": 0, "actual_status": "DNF"},
              ("r1", "b"): {"actual_known": True, "actual_points": 60, "actual_status": "finished"}}
    rows = hybrid_assignments(predictions, actual)
    riders, totals = summarize_assignments(rows, 1)
    assert totals["hybrid_points"] == 90
    assert totals["actual_points"] == 60
    assert totals["predicted_not_raced_points"] == 30
    assert totals["predicted_only_points"] == 120
    assert riders.loc[riders.counted_in_hybrid_top_n, "rider_href"].tolist() == ["b"]
    _, all_riders = summarize_assignments(rows, 0)
    assert all_riders["hybrid_points"] == 110


def test_nonfinite_actual_is_not_silently_counted():
    with pytest.raises(ValueError, match="Known actual"):
        hybrid_assignments({("r", "a"): 10}, {("r", "a"): {
            "actual_known": True, "actual_points": np.nan, "actual_status": "bad"}})


def test_cancelled_event_zero_even_for_new_assignment():
    rows = hybrid_assignments({("r", "new_rider"): 100}, {("r", "old_rider"): {
        "actual_known": True, "actual_points": 0, "actual_status": "race_cancelled_assumed", "did_race": False}})
    assert rows.hybrid_points.iloc[0] == 0
    assert rows.predicted_not_raced_points.iloc[0] == 0
    assert not rows.historical_starter.iloc[0]


def test_original_dns_is_zero_but_optimized_nonparticipant_retains_prediction():
    prediction = {("r", "a"): 100}
    actual = {("r", "a"): {"actual_known": True, "actual_points": 0,
              "actual_status": "observed", "did_race": False}}
    original = hybrid_assignments(prediction, actual, original=True)
    optimized = hybrid_assignments(prediction, actual)
    assert original.hybrid_points.iloc[0] == 0
    assert original.predicted_not_raced_points.iloc[0] == 0
    assert optimized.hybrid_points.iloc[0] == 100
    with pytest.raises(ValueError, match="complete actual coverage"):
        hybrid_assignments(prediction, {}, original=True)
