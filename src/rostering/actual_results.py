"""Audited historical stage/one-day and GC awards for hybrid evaluation.

By default the author-confirmed complete result set is authoritative: missing
events were cancelled and missing rider results earned zero. Strict coverage
auditing remains available with assume_complete=False.
The output universe is the recorded parent-race startlists, not all rider pairs.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
# The local historical stage export uses PCS's renamed Dauphine URL, while
# historical startlists and feature exports retain the original race slug.
RACE_ALIASES = {"tour-auvergne-rhone-alpes": "dauphine"}
WITHDRAWAL = {"DNF", "DNS", "OTL", "DSQ", "DQ"}


def canonical_job(href: str, year: int | None = None) -> str:
    parts = href.replace("https://www.procyclingstats.com/", "").strip("/").split("/")
    if len(parts) < 2 or parts[0] != "race":
        return ""
    parts[1] = RACE_ALIASES.get(parts[1], parts[1])
    if len(parts) >= 3 and re.fullmatch(r"\d{4}", parts[2]):
        return "/".join(parts[:3])
    return "/".join(parts[:2] + [str(year)]) if year is not None else ""


def _records(path):
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield from json.loads(line).items()


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError):
        return None


def load_actual_results(year: int, root: Path | None = None, *, assume_complete: bool = True) -> pd.DataFrame:
    """Return one audited actual-points total per historical startlist pair.

    Columns: rider_href, job, actual_points (NaN when unknown), actual_known,
    actual_status, actual_source, did_race. ``did_race`` is false for an explicit
    DNS in the first race component; otherwise startlist participation is retained.
    ``job`` preserves the startlist's URL; aliases
    are used internally only. Totals include stage/one-day and GC awards, not
    points/ KOM classifications, jerseys, national events or official penalties.

    With assume_complete=True, missing observations contribute zero and missing
    scheduled stages do not invalidate the recorded total. Otherwise,
    completeness is checked against the union of observed stages for a race.
    A missing stage before withdrawal makes the total unknown. Stages following
    an explicit DNF/DNS/OTL/DSQ and the GC then contribute a justified zero.
    Missing GC without withdrawal remains unknown, even if stages are present.
    """
    root = Path(root) if root is not None else ROOT
    startlists = json.loads((root / f"data/startlists/startlists_{year}.json").read_text(encoding="utf-8"))
    requested = {canonical_job(job, year) for job in startlists}
    # (rider, job, component) -> set of (date, points, result status).
    observations = {}
    components = {}
    conflicts = set()
    feature_verified = set()

    def add(rider, job, component, date, points, rank):
        if job not in requested or points is None:
            return
        key = rider, job, component
        value = (date, points, str(rank).strip().upper())
        previous = observations.get(key)
        if previous is not None and previous != value:
            conflicts.add((rider, job))
        else:
            observations[key] = value
        if component != "gc":
            components.setdefault(job, {})[component] = date

    for rider, races in _records(root / "data/unprocessed/rider_results.jsonl"):
        for href, rows in races.items():
            for row in rows:
                if len(row) < 4 or not re.fullmatch(rf"{year}-\d{{2}}-\d{{2}}", str(row[0])):
                    continue
                job = canonical_job(href, year)
                tail = href.strip("/").split("/")[-1]
                if tail.startswith("stage-") or tail == "prologue":
                    component = tail
                elif len(href.strip("/").split("/")) == 2 or tail in {str(year), "result"}:
                    component = "one_day"
                else:
                    continue  # Exclude secondary classifications and malformed GC dates.
                # PCS rider history uses a leading '+' for leader-jersey
                # awards. These are outside the stage-finish/GC scope.
                points = 0.0 if str(row[3]).strip().startswith("+") else _number(row[3])
                add(rider, job, component, row[0], points, row[1])

    for rider, races in _records(root / "data/unprocessed/results_gc.jsonl"):
        for href, rows in races.items():
            for row in rows:
                if len(row) >= 3 and str(row[0]).startswith(f"{year}-"):
                    add(rider, canonical_job(href, year), "gc", row[0], _number(row[2]), row[1])

    # Feature exports preserve additional historical results not present in the
    # rider-history scrape. Match stage observations by race and date only when
    # that date identifies exactly one component (never merge split stages).
    for filename, is_gc in (("features-2019-2025-detailed_gpx.csv", False),
                            ("gc_features_2019_2025.csv", True)):
        path = root / "models" / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=lambda c: c in {"rider_href", "race_href", "date", "uci_points", "rank"})
        if "rank" not in frame:
            frame["rank"] = "classified"
        frame = frame[frame["date"].astype(str).str.startswith(str(year))]
        for item in frame.itertuples(index=False):
            job = canonical_job(item.race_href, year)
            if job not in requested:
                continue
            date = str(item.date)[:10]
            if is_gc:
                component = "gc"
            else:
                candidates = [c for c, d in components.get(job, {}).items() if d == date]
                if len(candidates) != 1:
                    continue
                component = candidates[0]
            key = item.rider_href, job, component
            points = _number(item.uci_points)
            if points is None:
                continue
            # Feature-table points are the covered finishing/GC awards, whereas
            # raw rider-history figures can incorporate additional awards.
            previous = observations.get(key)
            observations[key] = (date, points, previous[2] if previous else str(item.rank).strip().upper())
            feature_verified.add(key)

    # A whole unobserved stage cannot be discovered from per-rider result unions.
    # Compare with the independently stored program; without cancellation proof,
    # keep such events unverified instead of silently assigning the missing stage 0.
    missing_scheduled = set()
    program_path = root / f"data/roster/programs_{year}.json"
    if program_path.exists():
        for races in json.loads(program_path.read_text(encoding="utf-8")).values():
            for href, event in races.items():
                job = canonical_job(href, year)
                for stage in event.get("stages", []):
                    component = stage["href"].rstrip("/").split("/")[-1]
                    if components.get(job, {}).get(component) != stage["date"]:
                        missing_scheduled.add(job)

    rows = []
    for original_job, entrants in startlists.items():
        job = canonical_job(original_job, year)
        event_components = components.get(job, {})
        for rider in dict.fromkeys(entry[0] for entry in entrants):
            records = {component: observations[(rider, job, component)]
                       for component in event_components
                       if (rider, job, component) in observations}
            withdrawal = min((date for date, _, rank in records.values() if rank in WITHDRAWAL), default=None)
            missing = [component for component, date in event_components.items()
                       if component not in records and (withdrawal is None or date <= withdrawal)]
            gc = observations.get((rider, job, "gc"))
            stage_race = (bool(event_components) and "one_day" not in event_components) or gc is not None
            first_date = min(event_components.values(), default=None)
            did_race = not any(date == first_date and rank == "DNS" for date, _, rank in records.values())
            status = "observed"
            if (rider, job) in conflicts:
                status = "conflicting_raw_results"
            elif not event_components:
                status = "missing_race_results"
            elif job in missing_scheduled:
                status = "unverified_scheduled_component"
            elif not records or missing:
                status = "missing_stage_or_one_day_results"
            elif any(not rank or rank in {"-", "CANCELLED", "CANCELED"} for _, _, rank in records.values()):
                status = "unverified_cancelled_or_unclassified_result"
            elif stage_race and gc is None and withdrawal is None:
                status = "missing_gc_result"
            elif any(value[1] > 0 and (rider, job, component) not in feature_verified
                     for component, value in list(records.items()) + ([("gc", gc)] if stage_race and gc else [])):
                status = "unverified_raw_positive_awards"
            elif withdrawal is not None:
                status = "observed_with_withdrawal"
            # Author-confirmed result completeness: missing events did not run;
            # missing rider/component entries earned no points. Keep the strict
            # audit available for other datasets, and still flag contradictions.
            if assume_complete and status != "conflicting_raw_results":
                if not event_components and gc is None:
                    status = "race_cancelled_assumed"
                    did_race = False
                elif status.startswith("missing") or status.startswith("unverified"):
                    status = "observed_complete_results_assumption"
            known = status.startswith("observed") or status == "race_cancelled_assumed"
            points = sum(record[1] for record in records.values()) + (gc[1] if stage_race and gc is not None else 0)
            rows.append({"rider_href": rider, "job": original_job,
                         "actual_points": points if known else float("nan"),
                         "actual_known": known, "actual_status": status,
                         "actual_source": ("complete_results_assumption" if assume_complete else "raw_and_feature_results") if known else "unverified_result_coverage",
                         "did_race": did_race})
    return pd.DataFrame(rows, columns=["rider_href", "job", "actual_points", "actual_known", "actual_status", "actual_source", "did_race"])
