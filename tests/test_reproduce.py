from pathlib import Path
import json
import pandas as pd


def test_archived_results_are_present_and_nonempty():
    root = Path(__file__).parents[1]
    ranker = pd.read_csv(root / "results/all_scenarios_ranker.csv")
    regressor = pd.read_csv(root / "results/all_scenarios_regressor.csv")
    hybrid = pd.read_csv(root / "results/team_summary.csv")
    assert len(ranker) == len(regressor) == 504
    assert len(hybrid) == 288
    assert ranker["eval_ranker_improvement_selected_field_pts_pct"].notna().all()


def test_hybrid_complete_results_metadata():
    metadata = json.loads((Path(__file__).parents[1] / "results/metadata.json").read_text())
    assert "complete" in metadata["completeness_assumption"]


def test_public_json_uses_standard_constants():
    root = Path(__file__).parents[1]

    def reject_constant(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")

    for path in [root / "data_manifest.json", *sorted((root / "results").glob("*.json")),
                 *sorted((root / "validation").glob("*.json"))]:
        json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def test_predictive_comparison_exports_latex():
    from src.models.evaluation import comparison_table

    root = Path(__file__).parents[1]
    yearly = pd.read_csv(root / "results/metrics_per_year.csv")
    table = comparison_table(yearly)
    latex = table.to_latex(float_format="%.2f", na_rep="--",
                           multicolumn_format="c", column_format="lcccccc", escape=True)
    assert "Stage/one-day" in latex and "GC" in latex
    assert "NDCG@20" in latex and "Spearman@20" in latex


def test_reference_statistics_and_actual_baseline():
    import numpy as np
    import reproduce

    reproduce.summaries()
    root = Path(__file__).parents[1]
    cross = pd.read_csv(root / "reproduced/cross_model_summary.csv")
    np.testing.assert_allclose(cross.mean_pct, [109.682080429, 90.939588477, 82.297752698, 124.618468625])
    assert cross.n.eq(36).all()
    hybrid = pd.read_csv(root / "reproduced/hybrid_summary.csv")
    np.testing.assert_allclose(hybrid["mean"], [98.742150712, 83.150907212, 74.840478100, 109.210104809])
    assert hybrid["count"].eq(36).all()


def test_imports_resolve_inside_release_package():
    import pytest
    root = Path(__file__).parents[1].resolve()
    if not all((root / "data" / name).is_file() for name in ("pcs_races_all.csv", "pcs_scales_all.csv")):
        pytest.skip("Optimizer import requires the undistributed reference scale/category tables.")
    from src.models import ranking, ranking_gc, evaluation
    from src.rostering import optimize, hybrid_evaluation
    root = Path(__file__).parents[1].resolve()
    for module in (ranking, ranking_gc, evaluation, optimize, hybrid_evaluation):
        assert Path(module.__file__).resolve().is_relative_to(root)
