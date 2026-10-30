"""Entry point for numerical reproduction from derived data or frozen archives."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def summaries():
    import numpy as np
    import pandas as pd
    from src.models.evaluation import comparison_table, load_tuned_params

    out = ROOT / "reproduced"
    out.mkdir(exist_ok=True)
    cross, sensitivity = [], []
    for reward in ("ranker", "regressor"):
        frame = pd.read_csv(ROOT / f"results/all_scenarios_{reward}.csv")
        reference = frame[frame.max_race_days.eq(85) & frame.injuries_considered & ~frame.use_actual_race_days]
        assert len(reference) == 36 and not reference.duplicated(["year", "team"]).any()
        for evaluator in ("ranker", "regressor"):
            values = reference[f"eval_{evaluator}_improvement_selected_field_pts_pct"]
            cross.append(dict(reward_model=reward, evaluator=evaluator, mean_pct=values.mean(),
                              sd_percentage_points=values.std(ddof=1), n=len(values), minimum_pct=values.min()))
        for key, group in frame.groupby(["max_race_days", "injuries_considered", "use_actual_race_days"], dropna=False):
            values = group[f"eval_{reward}_total_optimized_selected_field_pts"]
            sensitivity.append(dict(reward_model=reward, max_race_days=key[0], injuries_considered=key[1],
                                    use_actual_race_days=key[2], mean_points=values.mean(),
                                    sd_points=values.std(ddof=1), n=len(values)))
    pd.DataFrame(cross).to_csv(out / "cross_model_summary.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(out / "sensitivity_summary.csv", index=False)
    hybrid = pd.read_csv(ROOT / "results/team_summary.csv")
    original, optimized = hybrid[hybrid.roster.eq("original")], hybrid[hybrid.roster.eq("optimized")]
    assert original.predicted_not_raced_points.eq(0).all()
    np.testing.assert_allclose(original.hybrid_points, original.actual_points)
    np.testing.assert_allclose(optimized.hybrid_points, optimized.actual_points + optimized.predicted_not_raced_points)
    result = optimized.groupby(["reward_model", "evaluator"]).hybrid_improvement_pct.agg(["mean", "std", "count"])
    assert result["count"].eq(36).all()
    result.to_csv(out / "hybrid_summary.csv")
    yearly = pd.read_csv(ROOT / "results/metrics_per_year.csv")
    comparison_table(yearly).to_csv(out / "predictive_comparison.csv")
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    selected = []
    for name, path in manifest["tuning_runs"].items():
        if (ROOT / path / "results.csv").exists():
            params, _ = load_tuned_params(ROOT / path, name)
            selected.append({"model": name, **params})
    if selected:
        pd.DataFrame(selected).to_csv(out / "selected_hyperparameters.csv", index=False)
    solver = ROOT / manifest["runs"]["ranker_solver"] / "team_summary.csv"
    if solver.exists():
        frame = pd.read_csv(solver)
        reference = frame[frame.max_race_days.eq(85) & frame.injuries_considered & ~frame.use_actual_race_days]
        reference.to_csv(out / "solver_reference_rows.csv", index=False)
        reference.groupby("year").agg(
            teams=("team", "size"), variables=("n_variables", "sum"),
            constraints=("n_constraints", "sum"), mean_seconds=("solve_time_s", "mean"),
            maximum_seconds=("solve_time_s", "max"),
            optimal_status_count=("status", lambda s: s.eq(2).sum()),
        ).to_csv(out / "solver_summary.csv")
        aliases = {"astana-qazaqstan-team": "xds-astana-team", "bora-hansgrohe": "red-bull-bora-hansgrohe",
                   "team-dsm-firmenich-postnl": "team-picnic-postnl", "uae-team-emirates": "uae-team-emirates-xrg"}
        fixed = frame[~frame.use_actual_race_days].copy()
        fixed["team"] = fixed.team.replace(aliases)
        points = fixed.groupby(["team", "max_race_days", "injuries_considered"])[
            "eval_ranker_total_optimized_selected_field_pts"].sum().unstack(["max_race_days", "injuries_considered"])
        pd.DataFrame({"75_vs_85_pct": 100 * (points[75, True] - points[85, True]) / points[85, True],
                      "injuries_vs_no_injuries_pct": 100 * (points[85, True] - points[85, False]) / points[85, False],
                      }).to_csv(out / "team_sensitivity.csv")
    original[original.reward_model.eq("ranker") & original.evaluator.eq("ranker")][
        ["year", "team", "actual_points", "predicted_only_points"]
    ].to_csv(out / "original_actual_baseline.csv", index=False)
    # Assignment robustness compares sets of (race, rider), once per team-season.
    import ast
    sets = {}
    for family in ("ranker", "regressor"):
        path = ROOT / manifest["runs"][family] / "per_race_results.csv"
        if not path.exists():
            break
        races = pd.read_csv(path)
        races = races[races.max_race_days.eq(85) & races.injuries_considered & ~races.use_actual_race_days]
        sets[family] = {key: {(r.job, rider) for r in group.itertuples()
                             for rider in ast.literal_eval(r.optimized_riders)}
                        for key, group in races.groupby(["year", "team"])}
    if len(sets) == 2:
        assert sets["ranker"].keys() == sets["regressor"].keys()
        records = []
        for (year, team), left in sets["ranker"].items():
            right = sets["regressor"][year, team]
            records.append({"year": year, "team": team, "jaccard": len(left & right) / len(left | right)})
        pd.DataFrame(records).to_csv(out / "assignment_jaccard.csv", index=False)
    print(pd.DataFrame(cross).to_string(index=False))
    print(result.to_string())
    print("Saved", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tables", help="Recompute manuscript summaries from included frozen results")
    sub.add_parser("check", help="Check the full staged data and historical calendars without fitting")
    sub.add_parser("models", help="Refit all four tuned models for both test years and recompute metrics")
    sub.add_parser("plots", help="Regenerate Figures 7–10 from staged archives (requires LaTeX)")
    sub.add_parser("shap", help="Refit rankers and regenerate SHAP figures (requires LaTeX)")
    tune = sub.add_parser("tune", help="Rerun a full tuning grid; expensive and optional")
    tune.add_argument("model", choices=["STAGE", "GC", "STAGE_REG", "GC_REG"])
    roster = sub.add_parser("rosters", help="Fit models and solve seasonal allocations")
    roster.add_argument("--reward", choices=["ranker", "regressor", "both"], default="both")
    roster.add_argument("--reference", action="store_true", help="85-day, injuries respected only")
    roster.add_argument("--year", type=int, nargs="+", default=[2024, 2025])
    roster.add_argument("--teams", nargs="+")
    hybrid = sub.add_parser("hybrid", help="Reapply observed awards to saved assignment predictions, no fitting")
    hybrid.add_argument("--source", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    if args.command == "tables":
        summaries()
    elif args.command == "check":
        subprocess.run([sys.executable, "stage_data.py", "--check", "--profile", "inputs"], check=True)
        from src.rostering.evaluation import main as preflight
        preflight(["--validate-only", "--evaluation-model", "both"])
    elif args.command == "models":
        from src.models.evaluation import evaluate_tuned_models
        evaluate_tuned_models({k: ROOT / v for k, v in manifest["tuning_runs"].items()},
                              output_root=ROOT / "outputs/reproduced_models")
    elif args.command == "rosters":
        from src.rostering.evaluation import main as evaluate
        for reward in (["ranker", "regressor"] if args.reward == "both" else [args.reward]):
            argv = ["--reward-model", reward, "--evaluation-model", "both", "--years", *map(str, args.year),
                    "--output-root", "outputs/reproduced_rosters"]
            if args.reference:
                argv += ["--max-race-days", "85", "--injuries"]
            if args.teams:
                argv += ["--teams", *args.teams]
            evaluate(argv)
    elif args.command == "hybrid":
        from src.rostering.hybrid_evaluation import refresh_hybrid_run
        refresh_hybrid_run(args.source or ROOT / manifest["runs"]["hybrid"], ROOT / "outputs/reproduced_hybrid")
    elif args.command == "tune":
        from src.models.tuning import do_tuning
        do_tuning(args.model)
    elif args.command == "shap":
        from src.models.shap_analysis import regenerate
        regenerate()
    elif args.command == "plots":
        import runpy
        for name in ["bar_plot", "reallocation_analysis", "sensitivity_plot"]:
            runpy.run_path(str(ROOT / "figures" / (name + ".py")), run_name="__main__")


if __name__ == "__main__":
    main()
