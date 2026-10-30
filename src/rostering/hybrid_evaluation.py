"""Evaluate archived allocations with observed awards for historical starters.

Example: python -m src.rostering.hybrid_evaluation
Defaults to the latest complete archive per reward family, the 85-day/injury
reference scenario, and both archived evaluators. No fitting or solving occurs.
"""
import argparse
import ast
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def hybrid_assignments(predicted, actual, *, original=False):
    """Replace predictions only for confirmed observations; retain unknowns.

    Original allocations use actual points only, including zero for DNS.
    actual is indexed by (job, rider). Absence means not historically entered;
    an entered rider with unknown results has an explicit unverified component.
    """
    rows = []
    cancelled_jobs = {job for (job, _), record in actual.items()
                      if record["actual_status"] == "race_cancelled_assumed"}
    for (job, rider), points in sorted(predicted.items()):
        if not np.isfinite(points) or points < 0:
            raise ValueError("Predicted scale points must be finite and nonnegative")
        record = actual.get((job, rider))
        entered = record is not None
        if job in cancelled_jobs:
            record = {"actual_known": True, "actual_points": 0.0,
                      "actual_status": "race_cancelled_assumed", "did_race": False}
        if original:
            if not entered or not record["actual_known"]:
                raise ValueError("Original roster requires complete actual coverage")
            if not record.get("did_race", True):
                record = {**record, "actual_points": 0.0, "actual_known": True}
        raced = entered and bool(record.get("did_race", True))
        known = (original or raced or job in cancelled_jobs) and bool(record["actual_known"])
        observed = float(record["actual_points"]) if known else 0.0
        if not np.isfinite(observed) or observed < 0:
            raise ValueError("Known actual points must be finite and nonnegative")
        rows.append({"job": job, "rider_href": rider, "historical_starter": entered, "did_race": raced,
            "actual_known": known, "predicted_only_points": points,
            "actual_points": observed,
            "predicted_not_raced_points": 0.0 if raced or known else points,
            "predicted_missing_actual_points": points if raced and not known else 0.0,
            "hybrid_points": observed if known else points,
            "actual_status": record["actual_status"] if entered or job in cancelled_jobs else "not_in_historical_startlist"})
    return pd.DataFrame(rows, columns=["job", "rider_href", "historical_starter", "did_race", "actual_known",
        "predicted_only_points", "actual_points", "predicted_not_raced_points",
        "predicted_missing_actual_points", "hybrid_points", "actual_status"])


def summarize_assignments(assignments, top_n):
    """Select top N on combined season totals, then decompose the same riders."""
    if top_n < 0:
        raise ValueError("top_n must be nonnegative (0 means all riders)")
    cols = ["actual_points", "predicted_not_raced_points", "predicted_missing_actual_points", "hybrid_points", "predicted_only_points"]
    totals = assignments.groupby("rider_href")[cols].sum().reset_index()
    totals = totals.sort_values(["hybrid_points", "rider_href"], ascending=[False, True])
    totals["counted_in_hybrid_top_n"] = True
    if top_n:
        totals.loc[totals.index[top_n:], "counted_in_hybrid_top_n"] = False
    counted = totals.loc[totals.counted_in_hybrid_top_n]
    summary = {col: float(counted[col].sum()) for col in cols[:-1]}
    pure = totals.predicted_only_points.sort_values(ascending=False)
    summary["predicted_only_points"] = float((pure.iloc[:top_n] if top_n else pure).sum())
    summary.update(n_assignments=len(assignments),
        n_historical_starters=int(assignments.historical_starter.sum()),
        n_actual_known=int(assignments.actual_known.sum()),
        n_missing_actual=int((assignments.did_race & ~assignments.actual_known).sum()))
    return totals, summary


def latest_runs(root):
    chosen = {}
    for path in sorted(root.iterdir()):
        meta_path = path / "metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        family = meta.get("config", {}).get("reward_model")
        if meta.get("status") == "complete" and family in {"ranker", "regressor"}:
            chosen[family] = path
    if set(chosen) != {"ranker", "regressor"}:
        raise ValueError("A complete saved run is required for each reward family")
    return chosen


def plot_reference(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = summary.loc[summary.max_race_days.eq(85) & summary.injuries_considered]
    for evaluator, data in selected.groupby("evaluator"):
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        for ri, year in enumerate(sorted(data.year.unique())):
            for ci, reward in enumerate(("ranker", "regressor")):
                ax = axes[ri, ci]
                part = data.loc[data.year.eq(year) & data.reward_model.eq(reward) & data.roster.eq("optimized")].sort_values("team")
                y = np.arange(len(part))
                bottom = np.zeros(len(part))
                for col, color, label in (
                    ("actual_points", "#0072B2", "Recorded points: historically raced"),
                    ("predicted_not_raced_points", "#E69F00", "Predicted points: did not race"),
                    ("predicted_missing_actual_points", "#999999", "Predicted points: actual unverified"),
                ):
                    values = part[col].to_numpy()
                    if values.sum() or col != "predicted_missing_actual_points":
                        ax.barh(y, values, left=bottom, color=color, label=label)
                    bottom += values
                original = data.loc[data.year.eq(year) & data.reward_model.eq(reward) & data.roster.eq("original")].set_index("team")
                ax.scatter(original.loc[part.team, "hybrid_points"], y, marker="|", color="black", s=90, label="Original roster actual total")
                ax.set_yticks(y, part.team.str.replace("-", " "))
                ax.invert_yaxis()
                ax.set_title(f"{year} | reward: XGB{reward.capitalize()}", loc="left")
                ax.set_xlabel("Top-20 season points")
                ax.grid(axis="x", alpha=.2)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
        handles = {}
        for ax in axes.flat:
            hs, ls = ax.get_legend_handles_labels()
            handles.update(zip(ls, hs))
        fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=2, frameon=False)
        fig.suptitle(f"Hybrid evaluation using XGB{evaluator.capitalize()} for unobserved assignments", y=.945)
        fig.tight_layout(rect=(0, 0, 1, .92))
        fig.savefig(output / f"hybrid_reference_{evaluator}.pdf", bbox_inches="tight")
        fig.savefig(output / f"hybrid_reference_{evaluator}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranker-run", type=Path)
    parser.add_argument("--regressor-run", type=Path)
    parser.add_argument("--run-root", type=Path, default=ROOT / "outputs/rostering_evaluation")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/hybrid_evaluation")
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--rescore-hybrid-run", type=Path,
                        help="Refresh actual-point replacements using saved per-assignment predictions")
    parser.add_argument("--evaluation-model", choices=["ranker", "regressor", "both"], default="both")
    args = parser.parse_args(argv)
    if args.rescore_hybrid_run:
        return refresh_hybrid_run(args.rescore_hybrid_run, args.output_root)
    from src.rostering.evaluation import _make_optimizer
    from src.rostering.optimize import _get_teams, _get_race_objects
    from xgboost import XGBRanker, XGBRegressor
    from src.rostering.actual_results import load_actual_results
    sources = latest_runs(args.run_root)
    for family in sources:
        override = getattr(args, f"{family}_run")
        if override:
            sources[family] = override
    out = args.output_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    out.mkdir(parents=True)
    manifest = {"status": "running", "sources": {k: str(v.resolve()) for k, v in sources.items()},
        "scoring": "Selected-field rank-to-UCI predictions replaced by recorded stage/GC points for historical participants. Result sets are assumed complete: missing components and rider results contribute zero; wholly cancelled races contribute zero for every assignment.",
        "original_baseline": "Actual awards only; DNS contributes zero. Select the top N by actual season totals.",
        "aggregation": "Choose top N by each roster's hybrid season total, then sum both components over the same riders. Pure prediction comparison chooses its own top N.",
        "interpretation": "Retrospective hybrid sensitivity check; not realized counterfactual outcomes. Mixing recorded awards and predicted ranks need not conserve race points.",
        "all_scenarios": args.all_scenarios}
    manifest["source_code_sha256"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("hybrid_evaluation.py", "actual_results.py", "optimize.py")}
    (out / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summaries = []
    scorer_cache = {}
    evaluator_names = ("ranker", "regressor") if args.evaluation_model == "both" else (args.evaluation_model,)
    for reward, source in sources.items():
        config = json.loads((source / "metadata.json").read_text(encoding="utf-8"))["config"]
        if config["reward_model"] != reward:
            raise ValueError("Archive reward model does not match requested source")
        manifest.setdefault("source_summary_sha256", {})[reward] = hashlib.sha256((source / "team_summary.csv").read_bytes()).hexdigest()
        source_summary = pd.read_csv(source / "team_summary.csv")
        saved = pd.read_csv(source / "per_race_results.csv")
        if not args.all_scenarios:
            saved = saved.loc[saved.max_race_days.eq(85) & saved.injuries_considered & ~saved.use_actual_race_days]
        if saved.empty:
            raise ValueError(f"No requested scenarios in {source}")
        for year, season in saved.groupby("year"):
            actual = load_actual_results(int(year), ROOT)
            actual.to_csv(out / f"actual_coverage_{year}.csv", index=False)
            lookup = {(row["job"], row["rider_href"]): row for row in actual.to_dict("records")}
            teams, programs = _get_teams(int(year))
            models = {}
            model_hashes = {}
            for evaluator in evaluator_names:
                for kind in ("stage", "gc"):
                    name = kind + ("_reg" if evaluator == "regressor" else "")
                    path = source / "models" / str(year) / f"{name}.json"
                    model_hashes[name.upper()] = hashlib.sha256(path.read_bytes()).hexdigest()
                    model = (XGBRegressor if evaluator == "regressor" else XGBRanker)()
                    model.load_model(path)
                    models[name.upper()] = (model, model.get_booster().feature_names)
            manifest.setdefault("model_sha256", {})[f"{reward}/{year}"] = model_hashes
            for team, team_frame in season.groupby("team_href"):
                schedule = _get_race_objects(programs[team])
                riders = sorted(set(teams[team]))
                shared_features = {}
                for evaluator in evaluator_names:
                    print(f"Preparing {reward} / {evaluator} / {team}", flush=True)
                    suffix = "_REG" if evaluator == "regressor" else ""
                    names = [kind + suffix for kind in ("STAGE", "GC")]
                    cache_key = (int(year), team, *(model_hashes[name] for name in names))
                    scorer = scorer_cache.get(cache_key)
                    if scorer is None:
                        scorer = _make_optimizer(models, evaluator, riders, schedule, int(year), config)
                        scorer._set_other_riders()
                        schema = tuple(tuple(models[name][1]) for name in names)
                        if schema not in shared_features:
                            shared_features[schema] = scorer._get_features()
                        scorer.prepare(prepared_features=shared_features[schema])
                        scorer_cache[cache_key] = scorer
                    details, rider_details = [], []
                    for (cap, actual_budget, injuries), frame in team_frame.groupby(["max_race_days", "use_actual_race_days", "injuries_considered"], dropna=False):
                        identity = {"year": year, "team": frame.team.iloc[0], "reward_model": reward,
                            "evaluator": evaluator, "max_race_days": cap, "use_actual_race_days": actual_budget,
                            "injuries_considered": injuries, "top_n": config["top_n"]}
                        for roster in ("original", "optimized"):
                            selection = dict(zip(frame.job, frame[f"{roster}_riders"].map(ast.literal_eval)))
                            if len(selection) != len(frame):
                                raise ValueError("Duplicate saved race assignments")
                            assignments = hybrid_assignments(scorer.selected_assignment_points(selection), lookup, original=roster == "original")
                            # Original entries must all be represented by the actual-results loader.
                            if roster == "original" and not assignments.historical_starter.all():
                                raise ValueError("Original roster contains entries absent from actual coverage")
                            per_rider, totals = summarize_assignments(assignments, config["top_n"])
                            check = source_summary.loc[source_summary.year.eq(year) & source_summary.team_href.eq(team)
                                & source_summary.use_actual_race_days.eq(actual_budget) & source_summary.injuries_considered.eq(injuries)]
                            if not actual_budget:
                                check = check.loc[check.max_race_days.eq(cap)]
                            column = f"eval_{evaluator}_total_{roster}_selected_field_pts"
                            if column in check:
                                if len(check) != 1 or not np.isclose(totals["predicted_only_points"], float(check[column].iloc[0]), rtol=0, atol=1e-6):
                                    raise ValueError(f"Recreated predictions differ from archive for {team}/{evaluator}/{roster}")
                            ident = {**identity, "roster": roster}
                            summaries.append({**ident, **totals})
                            details.append(assignments.assign(**ident))
                            rider_details.append(per_rider.assign(**ident))
                    directory = out / reward / evaluator / str(year) / frame.team.iloc[0]
                    directory.mkdir(parents=True)
                    pd.concat(details).to_csv(directory / "assignments.csv", index=False)
                    pd.concat(rider_details).to_csv(directory / "rider_totals.csv", index=False)
                    pd.DataFrame(summaries).to_csv(out / "team_summary.csv", index=False)
    summary = pd.DataFrame(summaries)
    keys = ["year", "team", "reward_model", "evaluator", "max_race_days", "use_actual_race_days", "injuries_considered"]
    baseline = summary.loc[summary.roster.eq("original"), keys + ["hybrid_points"]].rename(columns={"hybrid_points": "original_hybrid_points"})
    summary = summary.merge(baseline, on=keys, validate="many_to_one")
    summary["hybrid_improvement_pct"] = 100 * (summary.hybrid_points - summary.original_hybrid_points) / summary.original_hybrid_points.replace(0, np.nan)
    summary.to_csv(out / "team_summary.csv", index=False)
    plot_reference(summary, out)
    manifest["status"] = "complete"
    manifest["n_summary_rows"] = len(summary)
    (out / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved hybrid evaluation to {out}", flush=True)
    return out


def refresh_hybrid_run(source, output_root):
    """Reapply actual-result policy without rebuilding frozen predictions."""
    from src.rostering.actual_results import load_actual_results
    source, output_root = Path(source), Path(output_root)
    previous = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if previous["status"] != "complete":
        raise ValueError("Refresh requires a complete hybrid archive")
    out = output_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    out.mkdir(parents=True)
    manifest = {**previous, "status": "running", "refreshed_from": str(source.resolve()),
        "original_baseline": "Actual awards only; DNS contributes zero. Select the top N by actual season totals.",
        "scoring": "Frozen selected-field predictions for nonparticipants; recorded stage/GC totals for participants under the complete-results assumption; wholly cancelled races score zero.",
        "completeness_assumption": "Author confirms result sets complete. Missing scheduled events are cancelled; missing rider results earn zero. Existing predictions and rosters remain frozen.",
        "source_code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                               for name in ("actual_results.py", "hybrid_evaluation.py")}}
    (out / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    keys = ["year", "team", "reward_model", "evaluator", "max_race_days", "use_actual_race_days", "injuries_considered", "top_n", "roster"]
    lookups, summaries = {}, []
    for path in sorted(source.glob("*/*/*/*/assignments.csv")):
        saved = pd.read_csv(path)
        details, rider_details = [], []
        for identity, frame in saved.groupby(keys, dropna=False):
            ident = dict(zip(keys, identity))
            year = int(ident["year"])
            if year not in lookups:
                actual = load_actual_results(year, ROOT)
                actual.to_csv(out / f"actual_coverage_{year}.csv", index=False)
                lookups[year] = {(r["job"], r["rider_href"]): r for r in actual.to_dict("records")}
            if frame.duplicated(["job", "rider_href"]).any():
                raise ValueError("Duplicate archived assignment")
            predicted = {(r.job, r.rider_href): r.predicted_only_points for r in frame.itertuples()}
            assignments = hybrid_assignments(predicted, lookups[year], original=ident["roster"] == "original")
            riders, totals = summarize_assignments(assignments, int(ident["top_n"]))
            details.append(assignments.assign(**ident))
            rider_details.append(riders.assign(**ident))
            summaries.append({**ident, **totals})
        target = out / path.relative_to(source).parent
        target.mkdir(parents=True)
        pd.concat(details).to_csv(target / "assignments.csv", index=False)
        pd.concat(rider_details).to_csv(target / "rider_totals.csv", index=False)
    if not summaries:
        raise ValueError("No archived assignments found")
    summary = pd.DataFrame(summaries)
    merge_keys = [k for k in keys if k != "roster"]
    baseline = summary.loc[summary.roster.eq("original"), merge_keys + ["hybrid_points"]].rename(columns={"hybrid_points": "original_hybrid_points"})
    summary = summary.merge(baseline, on=merge_keys, validate="many_to_one")
    summary["hybrid_improvement_pct"] = 100 * (summary.hybrid_points - summary.original_hybrid_points) / summary.original_hybrid_points.replace(0, np.nan)
    summary.to_csv(out / "team_summary.csv", index=False)
    plot_reference(summary, out)
    manifest.update(status="complete", n_summary_rows=len(summary))
    (out / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved hybrid evaluation to {out}", flush=True)
    return out


if __name__ == "__main__":
    main()
