import json

import pandas as pd

from src.rostering.actual_results import canonical_job, load_actual_results


def _fixture(tmp_path, stage, gc=None, job="race/test/2024", riders=("rider/a",), feature_points=True):
    (tmp_path / "data/startlists").mkdir(parents=True)
    raw = tmp_path / "data/unprocessed"
    raw.mkdir()
    (tmp_path / "data/startlists/startlists_2024.json").write_text(json.dumps({job: [[r, "team/t"] for r in riders]}))
    (raw / "rider_results.jsonl").write_text(json.dumps(stage) + "\n")
    (raw / "results_gc.jsonl").write_text(json.dumps(gc or {}) + "\n")
    if feature_points:
        models = tmp_path / "models"
        models.mkdir()
        for filename, source, index in (("features-2019-2025-detailed_gpx.csv", stage, 3), ("gc_features_2019_2025.csv", gc or {}, 2)):
            rows = [{"rider_href": rider, "race_href": canonical_job(href, 2024), "date": row[0], "rank": row[1], "uci_points": row[index]}
                    for rider, races in source.items() for href, records in races.items() for row in records
                    if href.split("/")[-1] not in {"points", "kom", "youth"}]
            pd.DataFrame(rows, columns=["rider_href", "race_href", "date", "rank", "uci_points"]).to_csv(models / filename, index=False)
    return load_actual_results(2024, tmp_path, assume_complete=False).set_index("rider_href")


def test_dnf_one_day_is_verified_zero_and_missing_is_unknown(tmp_path):
    result = _fixture(tmp_path, {"rider/a": {"race/test": [["2024-01-01", "DNF", "100", "0"]]}}, riders=("rider/a", "rider/b"))
    assert result.loc["rider/a", "actual_points"] == 0
    assert result.loc["rider/a", "actual_known"]
    assert not result.loc["rider/b", "actual_known"]
    assert pd.isna(result.loc["rider/b", "actual_points"])


def test_withdrawal_keeps_earned_stage_points_and_verifies_later_zero(tmp_path):
    result = _fixture(tmp_path, {
        "rider/a": {"race/test/2024/stage-1": [["2024-01-01", "1", "100", "10"]],
                    "race/test/2024/stage-2": [["2024-01-02", "DNF", "100", "0"]]},
        "rider/b": {"race/test/2024/stage-3": [["2024-01-03", "2", "100", "5"]]},
    })
    assert result.loc["rider/a", "actual_points"] == 10
    assert result.loc["rider/a", "actual_status"] == "observed_with_withdrawal"


def test_missing_before_withdrawal_stays_unknown(tmp_path):
    result = _fixture(tmp_path, {
        "rider/a": {"race/test/2024/stage-2": [["2024-01-02", "DNF", "100", "0"]]},
        "rider/b": {"race/test/2024/stage-1": [["2024-01-01", "1", "100", "10"]]},
    })
    assert not result.loc["rider/a", "actual_known"]


def test_gc_required_and_other_classifications_excluded(tmp_path):
    result = _fixture(tmp_path, {"rider/a": {
        "race/test/2024/stage-1": [["2024-01-01", "1", "100", "10"]],
        "race/test/2024/points": [["2024-01-01", "1", "100", "1000"]],
    }}, {"rider/a": {"race/test": [["2024-01-01", "1", "50"]]}})
    assert result.loc["rider/a", "actual_points"] == 60


def test_unexplained_missing_gc_is_not_zero(tmp_path):
    result = _fixture(tmp_path, {"rider/a": {"race/test/2024/stage-1": [["2024-01-01", "1", "100", "10"]]}})
    assert result.loc["rider/a", "actual_status"] == "missing_gc_result"


def test_verified_dauphine_alias(tmp_path):
    result = _fixture(tmp_path, {"rider/a": {"race/tour-auvergne-rhone-alpes/2024/stage-1": [["2024-01-01", "DNF", "100", "0"]]}}, job="race/dauphine/2024")
    assert result.loc["rider/a", "actual_known"]
    assert result.loc["rider/a", "job"] == "race/dauphine/2024"


def test_first_dns_not_raced_but_later_dns_raced(tmp_path):
    result = _fixture(tmp_path, {
        "rider/a": {"race/test/2024/stage-1": [["2024-01-01", "DNS", "100", "0"]]},
        "rider/b": {"race/test/2024/stage-1": [["2024-01-01", "1", "100", "10"]],
                    "race/test/2024/stage-2": [["2024-01-02", "DNS", "100", "0"]]},
    }, riders=("rider/a", "rider/b"))
    assert not result.loc["rider/a", "did_race"]
    assert result.loc["rider/b", "did_race"]
    assert result.loc["rider/b", "actual_points"] == 10


def test_blank_cancelled_result_not_verified_zero(tmp_path):
    result = _fixture(tmp_path, {"rider/a": {"race/test": [["2024-01-01", "", "100", "0"]]}})
    assert not result.loc["rider/a", "actual_known"]
    assert result.loc["rider/a", "actual_status"] == "unverified_cancelled_or_unclassified_result"


def test_feature_results_fill_missing_raw_observation(tmp_path):
    _fixture(tmp_path, {"rider/b": {"race/test": [["2024-01-01", "1", "100", "10"]]}})
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    pd.DataFrame([{"rider_href": "rider/a", "race_href": "race/test/2024", "date": "2024-01-01", "rank": "DNF", "uci_points": 0}]).to_csv(models / "features-2019-2025-detailed_gpx.csv", index=False)
    result = load_actual_results(2024, tmp_path, assume_complete=False).iloc[0]
    assert result.actual_known
    assert result.actual_points == 0


def test_conflicting_raw_awards_flagged(tmp_path):
    result = _fixture(tmp_path, {"rider/a": {"race/test": [["2024-01-01", "1", "100", "10"], ["2024-01-01", "1", "100", "20"]]}})
    assert not result.loc["rider/a", "actual_known"]
    assert result.loc["rider/a", "actual_status"] == "conflicting_raw_results"


def test_raw_positive_awards_without_feature_confirmation_unknown(tmp_path):
    result = _fixture(tmp_path, {"rider/a": {"race/test": [["2024-01-01", "1", "100", "10"]]}}, feature_points=False)
    assert result.loc["rider/a", "actual_status"] == "unverified_raw_positive_awards"


def test_whole_missing_scheduled_stage_unknown(tmp_path):
    _fixture(tmp_path, {"rider/a": {"race/test/2024/stage-1": [["2024-01-01", "DNF", "100", "0"]]}})
    roster = tmp_path / "data/roster"
    roster.mkdir()
    (roster / "programs_2024.json").write_text(json.dumps({"team/t": {"race/test/2024": {"stages": [{"href": "race/test/2024/stage-2", "date": "2024-01-02"}]}}}))
    result = load_actual_results(2024, tmp_path, assume_complete=False).iloc[0]
    assert result.actual_status == "unverified_scheduled_component"


def test_raw_leader_award_excluded(tmp_path):
    result = _fixture(tmp_path, {"rider/a": {"race/test/2024/stage-1": [["2024-01-01", "DNF", "100", "+10"]]}}, feature_points=False)
    assert result.loc["rider/a", "actual_points"] == 0


def test_complete_results_missing_rider_and_gc_earn_zero(tmp_path):
    _fixture(tmp_path, {"rider/a": {"race/test/2024/stage-1": [["2024-01-01", "1", "100", "10"]]}},
             riders=("rider/a", "rider/b"))
    result = load_actual_results(2024, tmp_path).set_index("rider_href")
    assert result.actual_known.all()
    assert result.loc["rider/a", "actual_points"] == 10
    assert result.loc["rider/b", "actual_points"] == 0


def test_complete_results_cancelled_stage_retains_earned_points(tmp_path):
    _fixture(tmp_path, {"rider/a": {"race/test/2024/stage-1": [["2024-01-01", "1", "100", "10"]]}},
             {"rider/a": {"race/test": [["2024-01-01", "1", "50"]]}})
    directory = tmp_path / "data/roster"
    directory.mkdir()
    (directory / "programs_2024.json").write_text(json.dumps({"team/t": {"race/test/2024": {"stages": [
        {"href": "race/test/2024/stage-2", "date": "2024-01-02"}]}}}))
    result = load_actual_results(2024, tmp_path).iloc[0]
    assert result.actual_known
    assert result.actual_points == 60


def test_complete_results_whole_event_cancelled(tmp_path):
    _fixture(tmp_path, {})
    result = load_actual_results(2024, tmp_path).iloc[0]
    assert result.actual_known and result.actual_points == 0
    assert result.actual_status == "race_cancelled_assumed"
