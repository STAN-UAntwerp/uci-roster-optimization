"""Exercise real reward construction, both solvers and cross-scoring on two races."""
from pathlib import Path
import json
import os
import numpy as np
from xgboost import XGBRanker, XGBRegressor
from src.rostering.optimize import _get_teams, _get_race_objects
from src.rostering.evaluation import _make_optimizer, _add_evaluation

ROOT = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    models = {}
    for name in ["STAGE", "GC", "STAGE_REG", "GC_REG"]:
        model = (XGBRegressor if name.endswith("_REG") else XGBRanker)()
        model.load_model(ROOT / manifest["runs"]["ranker"] / "models/2024" / (name.lower() + ".json"))
        models[name] = model, model.get_booster().feature_names
    teams, programs = _get_teams(2024)
    team = next(t for t in programs if t.endswith("alpecin-deceuninck-2024"))
    full = _get_race_objects(programs[team])
    stage_job = next(r.parent_href for r in full if r.parent_href)
    one_day = next(r.href for r in full if not r.parent_href)
    schedule = [r for r in full if r.parent_href == stage_job or r.href == one_day]
    config = {"top_n": 20, "time_limit": 60, "mip_gap": 0.001}
    evaluators = {family: _make_optimizer(models, family, teams[team], schedule, 2024, config)
                  for family in ["ranker", "regressor"]}
    out = ROOT / "reproduced/smoke_pipeline"
    out.mkdir(parents=True, exist_ok=True)
    report = {"team": team, "jobs": [stage_job, one_day], "families": {}}
    for family, evaluator in evaluators.items():
        print("Preparing", family, flush=True)
        weights, gc_weights = evaluator.prepare()
        assert np.isfinite(weights).all() and (weights >= 0).all()
        assert np.isfinite(gc_weights).all() and (gc_weights >= 0).all()
    for family, optimizer in evaluators.items():
        frame = optimizer.run(out / family)
        assert optimizer.solver_stats["status"] == 2
        capacities = {r.parent_href or r.href: r.n_per_team for r in schedule}
        for row in frame.itertuples():
            assert row.n_optimized <= capacities[row.job]
        scored, totals = _add_evaluation(frame, evaluators)
        scored.to_csv(out / (family + "_cross_scores.csv"), index=False)
        solver = {key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value
                  for key, value in optimizer.solver_stats.items()}
        report["families"][family] = {"solver": solver, "scores": totals}
    (out / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("Both real-data reward models, solves and cross-evaluators passed", flush=True)


if __name__ == "__main__":
    main()
