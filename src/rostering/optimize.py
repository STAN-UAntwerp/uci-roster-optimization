import json
from functools import lru_cache
import numpy as np
import pandas as pd
import gurobipy
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from pathlib import Path
from src.data_processing.cumulative_specialties import add_specialty_scores

CWD = Path(__file__).parent

race_categories = pd.read_csv(CWD.parent.parent / "data" / "pcs_races_all.csv")
point_scales = pd.read_csv(CWD.parent.parent / "data" / "pcs_scales_all.csv")

WINDOW_4W = timedelta(days=28)


@lru_cache(maxsize=None)
def _get_uci_points(
    race_href: str = None, stage_race: bool = False, year: int = 2026
) -> list:
    cats = race_categories[race_categories["race_name"] == race_href]["category"].values
    if stage_race:
        # PCS exports explicit stage categories, including historical suffixes
        # such as 'UCI.WR.Pro.Stage - 2023-2025'. Appending '.Stage' to the GC
        # category silently lost all ProSeries stage points in earlier runs.
        explicit = [c for c in cats if ".Stage" in c]
        cats = explicit or [c + ".Stage" for c in cats]
    else:
        cats = [c for c in cats if not any(
            part in c for part in (".Stage", ".Leader", ".Mountain", ".Points")
        )]
    available = set(point_scales.loc[point_scales["year"] == int(year), "category"])
    cats = list(dict.fromkeys(c for c in cats if c in available))
    if len(cats) != 1:
        raise ValueError(f"Expected one points category for {race_href}, stage={stage_race}, {year}; found {cats}")
    category = cats[0]

    scale = point_scales[
            (point_scales["category"] == category) & (point_scales["year"] == int(year))
        ].sort_values("rank")
    if scale.empty:
        raise ValueError(f"Missing points scale for {race_href}: {category}, {year}")
    # Index by finishing place, even if the source omits zero-point ranks.
    return scale.set_index("rank")["points"].reindex(
        range(1, int(scale["rank"].max()) + 1), fill_value=0.0
    ).tolist()


def _ranked_points(riders, scores, points_table) -> dict[str, float]:
    """Assign each place once, breaking score ties by rider href reproducibly."""
    if len(set(riders)) != len(riders) or not np.isfinite(scores).all():
        raise ValueError("Predictions require unique riders and finite scores")
    order = sorted(range(len(riders)), key=lambda i: (-scores[i], riders[i]))
    return {
        riders[i]: float(points_table[place]) if place < len(points_table) else 0.0
        for place, i in enumerate(order)
    }


def _injury_intervals(records):
    """Parse recorded absences; keep undated records in an explicit audit list."""
    intervals, undated = {}, []
    for rider, injuries in records.items():
        for injury in injuries:
            if injury.get("injury_date") in (None, "", "0000-00-00"):
                undated.append({"rider_href": rider, **injury})
                continue
            try:
                start = datetime.strptime(injury["injury_date"], "%Y-%m-%d")
                duration = float(injury["duration"])
                if not np.isfinite(duration) or duration < 0:
                    raise ValueError("Invalid injury duration")
                end = start + timedelta(weeks=duration)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid injury record for {rider}: {injury}") from exc
            intervals.setdefault(rider, []).append((start, end))
    return intervals, undated


class Race:
    """Holds relevant race information."""

    def __init__(
        self,
        href: str,
        date: datetime,
        level: str,
        tt_kms: float = 0.0,
        parent_href: str | None = None,
        n_per_team: int | None = None,
    ):
        self.href = href
        self.date = date
        self.level = level  # e.g. '1.Pro', '2.1', etc.
        self.tt_kms = tt_kms
        self.parent_href = parent_href  # for stages: the overall race href
        self.n_per_team = n_per_team or (
            7 if level.startswith("1.") else 8
        )  # default max riders per team
        self.points = _get_uci_points(
            self.parent_href or self.href,
            stage_race=bool(parent_href),
            year=self.date.year,
        )

    def __repr__(self):
        return f"Race(href='{self.href}', date='{self.date.strftime('%Y-%m-%d')}', level='{self.level}')"


def _get_new_team_href(team_href: str) -> str:
    with open(
        CWD.parent.parent / "data" / "teams" / "team_history_2026.json",
        "r",
        encoding="utf-8",
    ) as f:
        team_history_2026 = json.load(f)

    for new_href, history in team_history_2026.items():
        if team_href in history:
            return new_href


def _randomly_get_riders(team_hrefs: str, n: int, seed: int) -> set[str]:
    np.random.seed(seed)

    with open(
        CWD.parent.parent / "data" / "teams" / "teams_2026.json", "r", encoding="utf-8"
    ) as f:
        teams_2026 = json.load(f)

    selected_riders = set()
    for team in team_hrefs:
        riders = teams_2026.get(team, [])
        if riders:
            selected = np.random.choice(riders, size=min(n, len(riders)), replace=False)
            selected_riders.update(selected)
    return selected_riders


def _add_general_rider_features(
    features: pd.DataFrame, features_gc: pd.DataFrame, year: int
):
    general_rider_features = pd.read_csv(
        CWD.parent.parent / "data" / "rider_data" / "general_rider_data.csv"
    ).drop(
        columns=["nationality", "onedayraces", "gc", "tt", "sprint", "climber", "hills"]
    )
    general_rider_features["age"] = year - general_rider_features["year_of_birth"]
    general_rider_features["years_pro"] = np.maximum(
        year - general_rider_features["turned_pro_year"], 0
    )
    general_rider_features.drop(
        columns=["year_of_birth", "turned_pro_year"], inplace=True
    )
    general_rider_features["bmi"] = (
        (general_rider_features["weight_kg"] / general_rider_features["height_cm"] ** 2)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    general_rider_features.drop(columns=["weight_kg", "height_cm"], inplace=True)

    features = features.merge(
        general_rider_features,
        on="rider_href",
        how="left",
    )
    features_gc = features_gc.merge(
        general_rider_features,
        on="rider_href",
        how="left",
    )

    actual_date_column = features["date"]
    actual_date_column_gc = features_gc["date"]

    features["date"] = datetime(year, 1, 1)
    add_specialty_scores(features)
    features["date"] = actual_date_column

    features_gc["date"] = datetime(year, 1, 1)
    add_specialty_scores(features_gc)
    features_gc["date"] = actual_date_column_gc

    years_pro_safe = np.maximum(
        features["years_pro"].fillna(3), 1
    )  # Avoid division by zero
    features[
        [
            "onedayraces_per_year",
            "gc_per_year",
            "tt_per_year",
            "sprint_per_year",
            "climber_per_year",
            "hills_per_year",
        ]
    ] = features[["onedayraces", "gc", "tt", "sprint", "climber", "hills"]].div(
        years_pro_safe, axis=0
    )
    features["total_specialty_per_year"] = features[
        [
            "onedayraces_per_year",
            "gc_per_year",
            "tt_per_year",
            "sprint_per_year",
            "climber_per_year",
            "hills_per_year",
        ]
    ].sum(axis=1)
    features[
        [
            "gc_per_year_pctile",
            "tt_per_year_pctile",
            "hills_per_year_pctile",
            "climber_per_year_pctile",
            "sprint_per_year_pctile",
            "onedayraces_per_year_pctile",
        ]
    ] = features.groupby("race_href")[
        [
            "gc_per_year",
            "tt_per_year",
            "hills_per_year",
            "climber_per_year",
            "sprint_per_year",
            "onedayraces_per_year",
        ]
    ].rank(
        pct=True
    )

    years_pro_safe_gc = np.maximum(features_gc["years_pro"].fillna(3), 1)
    features_gc[
        [
            "onedayraces_per_year",
            "gc_per_year",
            "tt_per_year",
            "sprint_per_year",
            "climber_per_year",
            "hills_per_year",
        ]
    ] = features_gc[["onedayraces", "gc", "tt", "sprint", "climber", "hills"]].div(
        years_pro_safe_gc, axis=0
    )
    features_gc["total_specialty_per_year"] = features_gc[
        [
            "onedayraces_per_year",
            "gc_per_year",
            "tt_per_year",
            "sprint_per_year",
            "climber_per_year",
            "hills_per_year",
        ]
    ].sum(axis=1)

    features_gc[
        [
            "gc_per_year_pctile",
            "tt_per_year_pctile",
            "hills_per_year_pctile",
            "climber_per_year_pctile",
            "sprint_per_year_pctile",
            "onedayraces_per_year_pctile",
        ]
    ] = features_gc.groupby("race_href")[
        [
            "gc_per_year",
            "tt_per_year",
            "hills_per_year",
            "climber_per_year",
            "sprint_per_year",
            "onedayraces_per_year",
        ]
    ].rank(
        pct=True
    )

    return features, features_gc


def _approx_rolling_features(
    features: pd.DataFrame, features_gc: pd.DataFrame, year: int = 2026
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Approximate rolling features using previous year one-day race results.

    Preprocess each rider's previous year results once, then compute:
    - season_mean_points_previous_year: mean points across all one-day races in the previous year
    - rolling_mean_uci_4w_approx: mean points in the 4-week window before the
      equivalent previous year date (race_date shifted back 1 year)
    """
    with open(
        CWD.parent.parent / "data" / "unprocessed" / f"results_{year-1}.jsonl",
        "r",
        encoding="utf-8",
    ) as f:
        results_previous_year_raw = {}
        for line in f:
            results_previous_year_raw.update(json.loads(line))

    with open(
        CWD.parent.parent / "data" / "unprocessed" / f"results_gc_{year-1}.jsonl",
        "r",
        encoding="utf-8",
    ) as f:
        results_gc_previous_year_raw = {}
        for line in f:
            results_gc_previous_year_raw.update(json.loads(line))

    # Pre-process: rider -> sorted list of (datetime, points)
    rider_history: dict[str, list[tuple[datetime, float]]] = {}
    rider_season_mean: dict[str, float] = {}
    rider_history_gc: dict[str, list[tuple[datetime, float]]] = {}
    rider_season_mean_gc: dict[str, float] = {}

    for rider_href, results in results_previous_year_raw.items():
        oneday = []
        for res in results:
            try:
                dt = datetime.strptime(res[0], "%Y-%m-%d")
            except:
                continue
            pts = float(res[1].split()[0])  # exclude point penalties and bonuses
            oneday.append((dt, pts))
        if oneday:
            oneday.sort(key=lambda x: x[0])
            rider_history[rider_href] = oneday
            rider_season_mean[rider_href] = float(np.mean([p for _, p in oneday]))

    for rider_href, results in results_gc_previous_year_raw.items():
        gc = []
        for res in results:
            dt = datetime.strptime(res[0], "%Y-%m-%d")
            pts = float(res[1])
            gc.append((dt, pts))
        if gc:
            gc.sort(key=lambda x: x[0])
            rider_history_gc[rider_href] = gc
            rider_season_mean_gc[rider_href] = float(np.mean([p for _, p in gc]))

    features["rolling_mean_uci_53w"] = features["rider_href"].map(rider_season_mean)
    features["gc_rolling_mean_uci_53w"] = features["rider_href"].map(
        rider_season_mean_gc
    )
    features_gc["rolling_mean_uci_53w"] = features_gc["rider_href"].map(
        rider_season_mean_gc
    )

    # Approximate 156w (3-year) GC rolling mean using multiple previous seasons
    _all_gc_pts: dict[str, list[float]] = {}
    for _prev_year in [year - 1, year - 2, year - 3]:
        _path = (
            CWD.parent.parent
            / "data"
            / "unprocessed"
            / f"results_gc_{_prev_year}.jsonl"
        )
        if _path.exists():
            with open(_path, "r", encoding="utf-8") as f:
                for line in f:
                    for _rh, _results in json.loads(line).items():
                        _all_gc_pts.setdefault(_rh, []).extend(
                            float(r[1]) for r in _results
                        )
    rider_season_mean_gc_156w = {
        rh: float(np.mean(pts)) for rh, pts in _all_gc_pts.items() if pts
    }
    features_gc["rolling_mean_uci_156w"] = features_gc["rider_href"].map(
        rider_season_mean_gc_156w
    )

    rolling_4w = np.full(len(features), np.nan)
    rolling_4w_gc = np.full(len(features_gc), np.nan)

    for rider_href, group in features.groupby("rider_href"):
        history = rider_history.get(rider_href)
        if history is None:
            continue

        unique_dates = group["race_date"].unique()
        date_to_mean: dict = {}
        for race_date in unique_dates:
            ref_date = pd.Timestamp(race_date) - pd.DateOffset(years=1)
            cutoff = ref_date - WINDOW_4W
            window_pts = [pts for dt, pts in history if cutoff <= dt < ref_date]
            if window_pts:
                date_to_mean[race_date] = float(np.mean(window_pts))

        if date_to_mean:
            for idx, row_date in zip(group.index, group["race_date"]):
                if row_date in date_to_mean:
                    rolling_4w[idx] = date_to_mean[row_date]

    for rider_href, group in features_gc.groupby("rider_href"):
        history_gc = rider_history_gc.get(rider_href)
        if history_gc is None:
            continue

        unique_dates = group["race_date"].unique()
        date_to_mean: dict = {}
        for race_date in unique_dates:
            ref_date = pd.Timestamp(race_date) - pd.DateOffset(years=1)
            cutoff = ref_date - WINDOW_4W
            window_pts = [pts for dt, pts in history_gc if cutoff <= dt < ref_date]
            if window_pts:
                date_to_mean[race_date] = float(np.mean(window_pts))

        if date_to_mean:
            for idx, row_date in zip(group.index, group["race_date"]):
                if row_date in date_to_mean:
                    rolling_4w_gc[idx] = date_to_mean[row_date]

    features["rolling_mean_uci_4w"] = rolling_4w
    features_gc["rolling_mean_uci_4w"] = rolling_4w_gc

    rm53 = features["rolling_mean_uci_53w"]
    rm4 = features["rolling_mean_uci_4w"]

    features["recent_form"] = rm4 / rm53.replace(0, 1)

    race_grp = features.groupby("race_href")
    features["startlist_strength_53w"] = race_grp["rolling_mean_uci_53w"].transform(
        "mean"
    )
    features["startlist_strength_4w"] = race_grp["rolling_mean_uci_4w"].transform(
        "mean"
    )
    features["rider_vs_field_strength"] = rm53 / features[
        "startlist_strength_53w"
    ].replace(0, 1)
    features["strength_vs_startlist"] = features["rider_vs_field_strength"]
    features["rolling_mean_uci_53w_pctile"] = race_grp["rolling_mean_uci_53w"].rank(
        pct=True
    )
    features["rolling_mean_uci_4w_pctile"] = race_grp["rolling_mean_uci_4w"].rank(
        pct=True
    )
    features["gc_rolling_mean_uci_53w_pctile"] = race_grp[
        "gc_rolling_mean_uci_53w"
    ].rank(pct=True)

    with open(CWD.parent.parent / "data" / "teams" / f"teams_{year}.json", "r", encoding="utf-8") as f:
        teams_current_year = json.load(f)

    rider_to_team: dict[str, str] = {}
    for team_url, rider_list in teams_current_year.items():
        for rh in rider_list:
            rider_to_team[rh] = team_url
    features["team_href"] = features["rider_href"].map(rider_to_team)
    team_race_grp = features.groupby(["race_href", "team_href"])
    features["team_strength_53w"] = team_race_grp["rolling_mean_uci_53w"].transform(
        "mean"
    )
    features["team_strength_4w"] = team_race_grp["rolling_mean_uci_4w"].transform(
        "mean"
    )

    rm53_gc = features_gc["rolling_mean_uci_53w"]
    rm4_gc = features_gc["rolling_mean_uci_4w"]

    features_gc["recent_form"] = rm4_gc / rm53_gc.replace(0, 1)

    race_grp = features_gc.groupby("race_href")
    features_gc["field_size"] = race_grp["rider_href"].transform("count")
    features_gc["startlist_strength_53w"] = race_grp["rolling_mean_uci_53w"].transform(
        "mean"
    )
    features_gc["startlist_strength_4w"] = race_grp["rolling_mean_uci_4w"].transform(
        "mean"
    )
    features_gc["strength_vs_startlist"] = rm53_gc / features_gc[
        "startlist_strength_53w"
    ].replace(0, 1)
    features_gc["rolling_mean_uci_53w_pctile"] = race_grp["rolling_mean_uci_53w"].rank(
        pct=True
    )
    features_gc["rolling_mean_uci_4w_pctile"] = race_grp["rolling_mean_uci_4w"].rank(
        pct=True
    )

    rider_to_team: dict[str, str] = {}
    for team_url, rider_list in teams_current_year.items():
        for rh in rider_list:
            rider_to_team[rh] = team_url
    features_gc["team_href"] = features_gc["rider_href"].map(rider_to_team)
    team_race_grp = features_gc.groupby(["race_href", "team_href"])
    features_gc["team_strength_53w"] = team_race_grp["rolling_mean_uci_53w"].transform(
        "mean"
    )
    features_gc["team_strength_4w"] = team_race_grp["rolling_mean_uci_4w"].transform(
        "mean"
    )
    features_gc["strength_vs_team"] = rm53_gc / features_gc[
        "team_strength_53w"
    ].replace(0, 1)
    features_gc["team_quality_ratio"] = features_gc["team_strength_53w"] / features_gc[
        "startlist_strength_53w"
    ].replace(0, 1)
    features_gc["team_strength_156w"] = team_race_grp[
        "rolling_mean_uci_156w"
    ].transform("mean")
    features_gc["startlist_strength_156w"] = features_gc.groupby("race_href")[
        "rolling_mean_uci_156w"
    ].transform("mean")
    features_gc["rolling_mean_uci_156w_pctile"] = features_gc.groupby("race_href")[
        "rolling_mean_uci_156w"
    ].rank(pct=True)

    return features, features_gc


class RosterOptimizer:
    """
    Optimize race-by-race roster selection to maximise total UCI points.
    """

    def __init__(
        self,
        riders: list[str],
        races: list[Race],
        ranker,
        feature_cols: list[str],
        gc_ranker,
        gc_feature_cols: list[str],
        other_riders: dict[int, set[str]] | None = None,
        year: int = 2026,
        max_race_days: int = 85,
        use_actual_race_days: bool = False,
        top_n: int = 20,
        injuries: bool = True,
        strict_startlists: bool = False,
        time_limit: float | None = None,
        mip_gap: float = 0.001,
    ):
        if max_race_days < 0 or top_n < 0 or mip_gap < 0:
            raise ValueError("Race-day limits, top_n and mip_gap must be non-negative")
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be positive")
        self.riders = sorted(set(riders))
        self.races = list(races)
        if not self.riders or not self.races:
            raise ValueError("A non-empty squad and schedule are required")
        if len({r.href for r in self.races}) != len(self.races):
            raise ValueError("Duplicate race/stage hrefs in schedule")
        self.ranker = ranker
        self.feature_cols = feature_cols
        self.gc_ranker = gc_ranker
        self.gc_feature_cols = gc_feature_cols
        self.other_riders: dict[int, set[str]] = other_riders or {}
        self.year = year
        self.max_race_days = max_race_days
        self.use_actual_race_days = use_actual_race_days
        self.top_n = top_n
        self.injuries = injuries
        self.strict_startlists = strict_startlists
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self._prepared_weights = None
        self.points_summary = {
            "total_original_pts": np.nan,
            "total_optimized_pts": np.nan,
        }

    def _set_other_riders(self):
        """Populate self.other_riders from actual startlists where available.

        For each race, the startlist for self.year is checked first. If the race
        is present, every rider listed is used directly. When a race is missing
        from the startlist (e.g. future races in 2026), the method falls back to
        randomly sampling riders from the 2025 proxy teams.
        """
        startlists_path = (
            CWD.parent.parent / "data" / "startlists" / f"startlists_{self.year}.json"
        )
        with open(startlists_path, "r", encoding="utf-8") as f:
            startlists: dict[str, list] = json.load(f)

        if self.strict_startlists or self.year != 2026:
            for i, race in enumerate(self.races):
                key = race.parent_href or race.href
                if not startlists.get(key):
                    raise ValueError(f"Missing historical startlist: {key}")
                self.other_riders[i] = {entry[0] for entry in startlists[key]}
            return

        try:
            fallback_year = self.year - 1
            fallback_tpr_path = (
                CWD.parent.parent
                / "data"
                / "teams"
                / f"teams_per_race_{fallback_year}.json"
            )
            with open(fallback_tpr_path, "r", encoding="utf-8") as f:
                teams_per_race_fallback = json.load(f)
        except FileNotFoundError:
            teams_per_race_fallback = {}
            print(
                f"Fallback teams-per-race file not found for year {fallback_year}. Using empty fallback."
            )

        parent_riders: dict[str, set[str]] = {}
        for i, race in enumerate(self.races):
            # The startlist is keyed by the parent href for stage races, race href for one-day races
            startlist_key = race.parent_href or race.href

            # Reuse already-resolved riders for all stages of the same stage race
            if startlist_key in parent_riders:
                self.other_riders[i] = parent_riders[startlist_key]
                continue

            entries = startlists.get(startlist_key)
            if entries:
                # Startlist entries are [rider_href, team_href] pairs
                other_riders = {entry[0] for entry in entries}
            else:
                if self.strict_startlists or self.year != 2026:
                    raise ValueError(f"Missing historical startlist: {startlist_key}")
                # Fallback: random sample from previous-year proxy teams
                print(
                    f"Warning: no startlist found for race href {startlist_key}. Using fallback."
                )
                lookup_href = startlist_key.replace(str(self.year), str(fallback_year))
                teams = teams_per_race_fallback.get(lookup_href, [])
                new_year_team_hrefs = [_get_new_team_href(t) for t in teams]
                other_riders = _randomly_get_riders(
                    new_year_team_hrefs, race.n_per_team, seed=i
                )

            self.other_riders[i] = other_riders
            if race.parent_href:
                parent_riders[startlist_key] = other_riders

    def _get_rider_race_day_limits(
        self, jobs: list[str], job_race_days: dict[str, int]
    ) -> dict[str, int]:
        """Return the per-rider race-day cap keyed by rider_href.

        When ``use_actual_race_days`` is enabled, each rider is capped at the
        number of race days they actually did within the roster of races being
        optimized, counted from the team startlists for ``self.year``. For each
        job the rider appears on the startlist of, that job's race days are added
        to their budget. These are scheduled days, including stages after a
        rider's withdrawal, not observed days completed. Missing startlists fail
        explicitly instead of silently changing the workload scenario.
        """
        if not self.use_actual_race_days:
            return {r: self.max_race_days for r in self.riders}

        startlists_path = (
            CWD.parent.parent / "data" / "startlists" / f"startlists_{self.year}.json"
        )
        with open(startlists_path, "r", encoding="utf-8") as f:
            startlists: dict[str, list] = json.load(f)
        missing = [j for j in jobs if not startlists.get(j)]
        if missing:
            raise ValueError(f"Cannot derive race-day limits: missing startlists for {missing}")

        # Startlist entries are [rider_href, team_href] pairs, keyed by job.
        startlist_riders = {
            j: {entry[0] for entry in startlists.get(j, [])} for j in jobs
        }

        limits: dict[str, int] = {}
        for r in self.riders:
            limits[r] = sum(
                job_race_days[j] for j in jobs if r in startlist_riders[j]
            )
        return limits

    def _build_rider_race_pairings(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build a DataFrame of rider-race pairings from self.riders and self.other_riders."""
        rows = []
        rows_gc = []

        gc_first_stage_dict = {}
        for race in self.races:
            if race.parent_href and race.parent_href not in gc_first_stage_dict:
                gc_first_stage_dict[race.parent_href] = race.href

        stage_count: dict[str, int] = {}
        total_tt_kms: dict[str, float] = {}
        for race in self.races:
            if race.parent_href:
                stage_count[race.parent_href] = stage_count.get(race.parent_href, 0) + 1
                total_tt_kms[race.parent_href] = (
                    total_tt_kms.get(race.parent_href, 0.0) + race.tt_kms
                )

        own_riders = set(self.riders)
        for i, race in enumerate(self.races):
            # Other riders for this race (exclude own riders to avoid duplicates)
            others = self.other_riders.get(i, set()) - own_riders
            race_riders = sorted(own_riders | others)
            for rider_href in race_riders:
                rows.append(
                    {
                        "rider_href": rider_href,
                        "race_href": race.href,
                        "race_date": race.date,
                        "is_stage_race": 1 if race.parent_href else 0,
                        "is_own_rider": rider_href in own_riders,
                    }
                )
                if (
                    race.parent_href
                    and gc_first_stage_dict.get(race.parent_href) == race.href
                ):
                    rows_gc.append(
                        {
                            "rider_href": rider_href,
                            "race_href": race.parent_href,
                            "race_date": race.date,
                            "number_of_stages": stage_count.get(race.parent_href, 0),
                            "tt_kms": total_tt_kms.get(race.parent_href, 0.0),
                            "is_own_rider": rider_href in own_riders,
                        }
                    )

        return pd.DataFrame(rows), pd.DataFrame(rows_gc, columns=[
            "rider_href", "race_href", "race_date", "number_of_stages", "tt_kms", "is_own_rider"
        ])

    def _get_features(self):
        """Set up features DataFrame with rider-race pairings"""
        features, features_gc = self._build_rider_race_pairings()

        """Add GPX features where available"""
        with open(
            CWD.parent.parent
            / "data"
            / "results_with_gpx"
            / f"{self.year}"
            / "race_to_gpx.json",
            "r",
            encoding="utf-8",
        ) as f:
            race_to_gpx = json.load(f)

        features["gpx_name"] = features["race_href"].map(race_to_gpx)

        gpx_features = pd.read_csv(
            CWD.parent.parent / "data" / "gpx" / f"{self.year}.csv"
        ).rename(columns={"name": "gpx_name"})

        features = features.merge(
            gpx_features,
            on="gpx_name",
            how="left",
        )

        parent_href_to_gpx: dict[str, list[str]] = {}
        for race in self.races:
            if race.parent_href:
                gpx_name = race_to_gpx.get(race.href)
                if gpx_name:
                    parent_href_to_gpx.setdefault(race.parent_href, []).append(gpx_name)

        profile_scores = pd.read_csv(
            CWD.parent.parent / "data" / "profile_score" / f"{self.year}.csv"
        )

        parent_profile: dict[str, dict] = {}
        for parent_href, gpx_list in parent_href_to_gpx.items():
            mask = profile_scores["name"].isin(gpx_list)
            if mask.any():
                scores = profile_scores[mask]
                parent_profile[parent_href] = {
                    "profile_score_mean": scores["profile_score"].mean(),
                    "profile_score_max": scores["profile_score"].max(),
                    "profile_score_final_mean": scores["profile_score_final"].mean(),
                    "profile_score_final_max": scores["profile_score_final"].max(),
                }

        for col in [
            "profile_score_mean",
            "profile_score_max",
            "profile_score_final_mean",
            "profile_score_final_max",
        ]:
            features_gc[col] = features_gc["race_href"].map(
                {k: v[col] for k, v in parent_profile.items()}
            )

        """Merge with general rider features (age, speciality, etc.)"""
        # add_specialty_scores (called inside) requires a 'date' column
        features["date"] = features["race_date"]
        features_gc["date"] = features_gc["race_date"]
        features, features_gc = _add_general_rider_features(
            features, features_gc, year=self.year
        )

        """Rider-Race history"""
        with open(
            CWD.parent.parent / "data" / "unprocessed" / "rider_results.jsonl",
            "r",
            encoding="utf-8",
        ) as f:
            rider_results_dict = {}
            for line in f:
                rider_results_dict.update(json.loads(line))

        with open(
            CWD.parent.parent / "data" / "unprocessed" / "results_gc.jsonl",
            "r",
            encoding="utf-8",
        ) as f:
            rider_results_gc_dict = {}
            for line in f:
                rider_results_gc_dict.update(json.loads(line))

        def _previous_rank_features(row, gc=False):
            if gc:
                race_href = "/".join(row["race_href"].split("/")[:2])
                previous_results = rider_results_gc_dict.get(row["rider_href"], {}).get(
                    race_href, []
                )
            else:
                if row["is_stage_race"]:
                    return [np.nan, np.nan, np.nan]
                race_href = "/".join(row["race_href"].split("/")[:2])
                previous_results = rider_results_dict.get(row["rider_href"], {}).get(
                    race_href, []
                )
            if not previous_results:
                return [np.nan, np.nan, np.nan]
            previous_results = sorted(previous_results, key=lambda res: res[0])
            previous_ranks = [
                int(res[1]) if str(res[1]).isdigit() else 999
                for res in previous_results
                if len(res) >= 2
                and datetime.strptime(res[0], "%Y-%m-%d").year < self.year
            ]
            if previous_ranks == []:
                return [np.nan, np.nan, np.nan]

            best_rank = min(previous_ranks)
            last_rank = previous_ranks[-1]
            # GC results: [date, rank, uci_points] (3 elements, pts at index 2)
            # One-day results: [date, rank, distance, uci_points] (4 elements, pts at index 3)
            pts_idx = 2 if gc else 3
            min_len = 3 if gc else 4
            mean_uci_points = np.mean(
                [
                    float(res[pts_idx])
                    for res in previous_results
                    if len(res) >= min_len
                    and datetime.strptime(res[0], "%Y-%m-%d").year < self.year
                ]
            )
            return [last_rank, best_rank, mean_uci_points]

        def _previous_rank_features_gc(row):
            return _previous_rank_features(row, gc=True)

        features[
            ["previous_rank", "best_previous_rank", "mean_previous_uci_points"]
        ] = features.apply(_previous_rank_features, axis=1, result_type="expand")

        history_cols = ["previous_rank", "best_previous_rank", "mean_previous_uci_points"]
        if features_gc.empty:
            features_gc[history_cols] = pd.DataFrame(columns=history_cols, dtype=float)
        else:
            features_gc[history_cols] = features_gc.apply(
                _previous_rank_features_gc, axis=1, result_type="expand"
            )
        """Rolling feature approximation"""
        features, features_gc = _approx_rolling_features(
            features, features_gc, year=self.year
        )

        features_gc["previous_rank_filled"] = features_gc["previous_rank"].fillna(
            features_gc.groupby("race_href")["rolling_mean_uci_53w"].transform(
                lambda x: x.rank(ascending=False).mean()
            )
        )
        features_gc["has_race_history"] = (~features_gc["previous_rank"].isna()).astype(
            int
        )

        """Rider-GPX interaction features"""
        features["climber_elevation_interaction"] = (
            features["climber_per_year"] * features["elevation_gain_per_km_m"]
        )
        features["hills_elevation_interaction"] = (
            features["hills_per_year"] * features["ascent_last_50km_m"]
        )

        """Strength × specialty features"""
        _max_specialty = (
            features[["gc", "tt", "hills", "climber", "sprint", "onedayraces"]]
            .max(axis=1)
            .replace(0, 1)
        )
        _specialty_norm = (
            features[["gc", "tt", "hills", "climber", "sprint", "onedayraces"]]
            .div(_max_specialty, axis=0)
            .fillna(0)
        )
        for _spec_col in ["gc", "tt", "hills", "climber", "sprint", "onedayraces"]:
            features[f"strength_x_{_spec_col}"] = (
                _specialty_norm[_spec_col] * features["rolling_mean_uci_53w"]
            )

        features_gc["tt_interaction"] = (
            features_gc["tt_per_year"] * features_gc["tt_kms"]
        ).fillna(0)
        features_gc["climbing_interaction"] = (
            features_gc["climber_per_year"] * features_gc["profile_score_mean"]
        ).fillna(0)
        features_gc["climbing_interaction_final"] = (
            features_gc["climber_per_year"] * features_gc["profile_score_final_mean"]
        ).fillna(0)
        features_gc["climbing_interaction_max"] = (
            features_gc["climber_per_year"] * features_gc["profile_score_max"]
        ).fillna(0)
        features_gc["gc_profile_match"] = (
            features_gc["gc_per_year"] * features_gc["profile_score_mean"]
        ).fillna(0)
        features_gc["tt_profile_match"] = (
            features_gc["tt_per_year"]
            * (features_gc["tt_kms"] / features_gc["number_of_stages"].replace(0, 1))
        ).fillna(0)

        for label, frame, columns in (("STAGE", features, self.feature_cols),
                                      ("GC", features_gc, self.gc_feature_cols)):
            missing = sorted(set(columns) - set(frame.columns))
            if missing:
                raise ValueError(f"{label} optimization features missing model columns: {missing}")
            if frame.duplicated(["rider_href", "race_href"]).any():
                raise ValueError(f"Duplicate {label} rider/race features after merges")
        return (
            features.set_index(["rider_href", "race_href"]).loc[:, self.feature_cols],
            features_gc.set_index(["rider_href", "race_href"]).loc[
                :, self.gc_feature_cols
            ],
        )

    def _build_weight_matrix(self, prepared_features=None) -> tuple[np.ndarray, np.ndarray]:
        """Predict fixed point coefficients for each (rider, race) pair.

        For each race, all riders (own + other teams) are ranked together
        so that own-team riders receive distinct predicted ranks, avoiding
        double-counting of the same finishing position. These coefficients
        include unused teammates; selected-field rescoring is reported separately.

        Returns
        -------
        weight_matrix : np.ndarray
            Total expected points (stage + GC).
        gc_weight_matrix : np.ndarray
            GC-only expected points (zero for one-day races).
        """
        features, features_gc = self._get_features() if prepared_features is None else prepared_features
        rider_to_idx = {r: i for i, r in enumerate(self.riders)}

        weight_matrix = np.zeros((len(self.riders), len(self.races)))
        gc_weight_matrix = np.zeros((len(self.riders), len(self.races)))

        # Store raw scores per race for later impact analysis
        self._race_scores: dict[str, tuple[list, np.ndarray, list]] = {}
        self._gc_scores: dict[str, tuple[list, np.ndarray, list]] = {}

        for j, race in enumerate(self.races):
            race_mask = features.index.get_level_values("race_href") == race.href
            if not race_mask.any():
                continue

            race_features = features.loc[race_mask]
            rider_hrefs = race_features.index.get_level_values("rider_href")
            points_table = race.points

            # Predict once for the full field (own + other riders)
            X_race = race_features
            scores = self.ranker.predict(X_race)
            self._race_scores[race.href] = (
                list(rider_hrefs),
                scores.copy(),
                points_table,
            )

            rider_points = _ranked_points(list(rider_hrefs), scores, points_table)
            for rider_href in self.riders:
                weight_matrix[rider_to_idx[rider_href], j] = rider_points[rider_href]

        processed_stage_races = []
        for j, race in enumerate(self.races):
            if race.parent_href:
                if race.parent_href in processed_stage_races:
                    continue
                race_mask = (
                    features_gc.index.get_level_values("race_href") == race.parent_href
                )
                if not race_mask.any():
                    continue

                gc_race_features = features_gc.loc[race_mask]
                gc_rider_hrefs = gc_race_features.index.get_level_values("rider_href")
                gc_points_table = _get_uci_points(
                    race.parent_href, stage_race=False, year=race.date.year
                )

                # Predict once for the full GC field
                X_gc = gc_race_features
                scores = self.gc_ranker.predict(X_gc)
                self._gc_scores[race.parent_href] = (
                    list(gc_rider_hrefs),
                    scores.copy(),
                    gc_points_table,
                )

                rider_points = _ranked_points(list(gc_rider_hrefs), scores, gc_points_table)
                for rider_href in self.riders:
                    gc_pts = rider_points[rider_href]
                    weight_matrix[rider_to_idx[rider_href], j] += gc_pts
                    gc_weight_matrix[rider_to_idx[rider_href], j] = gc_pts

                processed_stage_races.append(race.parent_href)

        return weight_matrix, gc_weight_matrix

    def score_rosters(self, frame):
        """Score fixed original/optimized selections without building a solver model.

        Returns per-job scores and top-N season totals on both the full candidate
        field and the selected field. Both model families rank predictions and
        obtain points from the UCI scale; raw predictions are never rewards.
        """
        weights, _ = self.prepare()
        weights, jobs, _, _ = self._combine_stages(weights)
        rider_index = {r: i for i, r in enumerate(self.riders)}
        job_index = {j: i for i, j in enumerate(jobs)}
        result = frame[["job"]].copy()
        summary = {}
        for label in ("original", "optimized"):
            selection = dict(zip(frame["job"], frame[f"{label}_riders"]))
            if frame["job"].duplicated().any() or set(selection) != set(jobs):
                raise ValueError("Roster jobs must match the evaluation calendar exactly")
            totals = {r: 0.0 for r in self.riders}
            job_totals = {}
            for job, riders in selection.items():
                if not isinstance(riders, list) or len(set(riders)) != len(riders):
                    raise ValueError(f"Invalid {label} selection for {job}")
                if set(riders) - set(rider_index):
                    raise ValueError(f"Unknown riders in {label} selection for {job}")
                job_totals[job] = 0.0
                for rider in riders:
                    points = float(weights[rider_index[rider], job_index[job]])
                    totals[rider] += points
                    job_totals[job] += points
            field_totals, field_jobs = self._score_selection(selection)
            result[f"{label}_pts"] = result["job"].map(job_totals)
            result[f"{label}_selected_field_pts"] = result["job"].map(field_jobs)
            summary[f"total_{label}_pts"] = _top_n_sum(totals, self.top_n)
            summary[f"total_{label}_selected_field_pts"] = _top_n_sum(field_totals, self.top_n)
        for suffix in ("pts", "selected_field_pts"):
            baseline = summary[f"total_original_{suffix}"]
            gain = summary[f"total_optimized_{suffix}"] - baseline
            summary[f"improvement_{suffix}"] = gain
            summary[f"improvement_{suffix}_pct"] = 100 * gain / baseline if baseline else np.nan
        return result, summary

    def _score_selection(self, selection):
        """Re-rank a selected squad against opponents, removing unused teammates.

        This is a diagnostic for the fixed full-squad coefficients used in the
        MILP. Feature values and model scores stay frozen across selections.
        """
        by_assignment = self.selected_assignment_points(selection)
        totals = {r: 0.0 for r in self.riders}
        job_points = {job: 0.0 for job in selection}
        for (job, rider), points in by_assignment.items():
            totals[rider] += points
            job_points[job] = job_points.get(job, 0.0) + points
        return totals, job_points

    def selected_assignment_points(self, selection):
        """Frozen-score, selected-field UCI awards per rider and parent race.

        Stage awards and GC awards accumulate into a single race assignment.
        This decomposition supports hybrid evaluation without solving again.
        """
        by_assignment = {(job, r): 0.0 for job, riders in selection.items() for r in riders}
        own = set(self.riders)
        race_to_job = {r.href: r.parent_href or r.href for r in self.races}
        components = [(race_to_job[href], values) for href, values in self._race_scores.items()]
        components += list(self._gc_scores.items())
        for job, (riders, scores, scale) in components:
            selected = set(selection.get(job, []))
            active = [i for i, r in enumerate(riders) if r not in own or r in selected]
            points = _ranked_points([riders[i] for i in active], scores[active], scale)
            for r in selected:
                by_assignment[job, r] += points.get(r, 0.0)
        return by_assignment

    def _compute_other_team_impact(
        self,
        startlists: dict,
        original_selection: dict[str, list[str]],
        optimized_selection: dict[str, list[str]],
    ) -> pd.DataFrame:
        """Compute how each other team's expected points change due to own roster optimisation.

        For every race / GC classification the other-team riders are re-ranked under
        the *original* own-team selection and the MILP *optimised* selection.
        The difference in scored points is the impact.

        Parameters
        ----------
        startlists : dict
            Loaded startlists JSON (job → list of [rider_href, team_href, ...]).
        original_selection : dict[str, list[str]]
            job → own-team riders that actually started (from startlists).
        optimized_selection : dict[str, list[str]]
            job → own-team riders chosen by the MILP.

        Returns
        -------
        pd.DataFrame
            Columns: job, component, other_team_href,
                     pts_with_original, pts_with_optimized, pts_delta.
        """
        if not hasattr(self, "_race_scores"):
            return pd.DataFrame()

        own_riders_set = set(self.riders)

        # Map race_href → job (parent_href or self for one-day)
        race_to_job: dict[str, str] = {}
        for race in self.races:
            race_to_job[race.href] = race.parent_href or race.href

        rows: list[dict] = []

        def _append_rows(
            job: str,
            component: str,
            rider_hrefs: list[str],
            scores: np.ndarray,
            points_table: list,
            startlist_entries: list,
        ) -> None:
            score_by_rider: dict[str, float] = dict(zip(rider_hrefs, scores))

            orig_own = set(original_selection.get(job, []))
            opt_own = set(optimized_selection.get(job, []))

            # Other-team riders present in the feature set, grouped by team
            team_riders: dict[str, list[str]] = {}
            for entry in startlist_entries:
                rider_h, team_h = entry[0], entry[1]
                if rider_h in score_by_rider and rider_h not in own_riders_set:
                    team_riders.setdefault(team_h, []).append(rider_h)

            if not team_riders:
                return

            def score_selection(selection):
                active = [r for r in rider_hrefs if r not in own_riders_set or r in selection]
                return _ranked_points(active, np.array([score_by_rider[r] for r in active]), points_table)

            rider_pts_orig = score_selection(orig_own)
            rider_pts_opt = score_selection(opt_own)

            for team_h, their_riders in team_riders.items():
                pts_o = sum(rider_pts_orig.get(rh, 0.0) for rh in their_riders)
                pts_op = sum(rider_pts_opt.get(rh, 0.0) for rh in their_riders)
                rows.append(
                    {
                        "job": job,
                        "component": component,
                        "other_team_href": team_h,
                        "pts_with_original": pts_o,
                        "pts_with_optimized": pts_op,
                        # positive → they lose pts because our optimised riders push them down
                        "pts_delta": pts_o - pts_op,
                    }
                )

        # Per-stage impact
        for race_href, (rider_hrefs, scores, points_table) in self._race_scores.items():
            job = race_to_job.get(race_href)
            if job is None:
                continue
            _append_rows(
                job,
                "stage",
                rider_hrefs,
                scores,
                points_table,
                startlists.get(job, []),
            )

        # GC impact
        for parent_href, (rider_hrefs, scores, points_table) in self._gc_scores.items():
            _append_rows(
                parent_href,
                "gc",
                rider_hrefs,
                scores,
                points_table,
                startlists.get(parent_href, []),
            )

        return pd.DataFrame(rows)

    def _combine_stages(
        self, weight_matrix: np.ndarray
    ) -> tuple[np.ndarray, list[str], dict[str, int], dict[str, set[datetime]]]:
        """Collapse stage race columns into one column per stage race.

        Returns:
            combined_weights: (n_riders, n_combined_jobs) matrix with stage points summed
            combined_jobs: list of job identifiers (parent_href for stage races, href for one-day)
            combined_limits: {job: max riders per team}
            job_dates: {job: set of dates the job spans}
        """
        # Group race indices by their combined job key
        job_key_to_indices: dict[str, list[int]] = {}
        for j, race in enumerate(self.races):
            key = race.parent_href or race.href
            job_key_to_indices.setdefault(key, []).append(j)

        combined_jobs = list(job_key_to_indices.keys())
        n_riders = weight_matrix.shape[0]
        combined_weights = np.zeros((n_riders, len(combined_jobs)))
        combined_limits: dict[str, int] = {}
        job_dates: dict[str, set[datetime]] = {}

        for cj, job_key in enumerate(combined_jobs):
            stage_indices = job_key_to_indices[job_key]
            combined_weights[:, cj] = weight_matrix[:, stage_indices].sum(axis=1)
            combined_limits[job_key] = self.races[stage_indices[0]].n_per_team
            job_dates[job_key] = {self.races[si].date for si in stage_indices}

        return combined_weights, combined_jobs, combined_limits, job_dates

    def _create_milp_model(self, weight_matrix: np.ndarray) -> gurobipy.Model:
        """Create and return MILP model with decision variables and constraints."""
        combined_weights, jobs, job_limits, job_dates = self._combine_stages(
            weight_matrix
        )

        injuries_path = CWD.parent.parent / "data" / "injuries" / "injuries.json"
        injuries = {}
        self.undated_injuries = []
        if self.injuries:
            with open(injuries_path, "r", encoding="utf-8") as f:
                injuries, undated = _injury_intervals(json.load(f))
            self.undated_injuries = [record for record in undated if record["rider_href"] in self.riders]

        def adjust_date_set(dates: set[datetime]) -> set[datetime]:
            min_date = min(dates)
            max_date = max(dates)
            dates = set(
                min_date + timedelta(days=i)
                for i in range((max_date - min_date).days + 1)
            )
            return dates

        adjusted_job_dates = {
            job: adjust_date_set(dates) for job, dates in job_dates.items()
        }

        all_dates: set[datetime] = set()
        for dates in adjusted_job_dates.values():
            all_dates |= dates
        date_jobs: dict[datetime, list[str]] = {d: [] for d in all_dates}
        for job, dates in adjusted_job_dates.items():
            for d in dates:
                date_jobs[d].append(job)

        model = gurobipy.Model("roster_optimization")
        model.Params.Seed = 23
        model.Params.MIPGap = self.mip_gap
        if self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit
        resources = self.riders

        matching_scores = {
            (resources[r], jobs[j]): combined_weights[r, j]
            for r in range(len(resources))
            for j in range(len(jobs))
        }
        x = model.addVars(matching_scores.keys(), vtype=gurobipy.GRB.BINARY, name="x")
        self._assignment_vars = x
        min_races = model.addVar(name="min_races", vtype=gurobipy.GRB.INTEGER)
        for r in resources:
            model.addConstr(
                gurobipy.quicksum(x[r, j] for j in jobs) >= min_races,
                name=f"min_races_{r}",
            )

        # Constraint: max riders per team per race/stage-race
        model.addConstrs(
            (x.sum("*", j) <= job_limits[j] for j in jobs),
            name="max_riders_per_race",
        )
        model.addConstrs(
            (
                gurobipy.quicksum(x[r, j] for j in date_jobs[date]) <= 1
                for r in resources
                for date in date_jobs
            ),
            name="rider_time_overlap",
        )
        # Scenario-specific workload budget; stage-race rest days only block overlap.
        job_race_days = {j: len(job_dates[j]) for j in jobs}
        rider_race_day_limits = self._get_rider_race_day_limits(jobs, job_race_days)
        self.rider_race_day_limits = rider_race_day_limits
        model.addConstrs(
            (
                gurobipy.quicksum(job_race_days[j] * x[r, j] for j in jobs) <= rider_race_day_limits[r]
                for r in resources
            ),
            name="max_race_days",
        )
        
        if self.injuries:
            # Block the entire race if any day (including rest days) is unavailable.
            for rider_href in self.riders:
                for injury_start, injury_end in injuries.get(rider_href, []):
                    for job in jobs:
                        if any(
                            injury_start <= d < injury_end for d in adjusted_job_dates[job]
                        ):
                            model.addConstr(
                                x[rider_href, job] == 0, name=f"injury_{rider_href}_{job}"
                            )

        model.setObjectiveN(
            self._top_n_points_objective(model, matching_scores, x, jobs, resources),
            index=0,
            priority=2,
            abstol=0.0,
            reltol=0.0,
            name="total_expected_points",
        )
        model.setObjectiveN(min_races, index=1, priority=1, name="maximize_min_races")
        model.ModelSense = gurobipy.GRB.MAXIMIZE

        return model

    def _top_n_points_objective(self, model, matching_scores, x, jobs, resources):
        """Return the objective expr: sum of the top-N riders' expected points.

        Linearized (points are non-negative): binary counts[r] selects which
        riders count, and counted_points[r] equals that rider's total only when
        selected. Falls back to the sum over all riders when top_n is unset or
        at least the squad size.
        """
        n_riders = len(resources)
        if not self.top_n or self.top_n <= 0 or self.top_n >= n_riders:
            return gurobipy.quicksum(matching_scores[k] * x[k] for k in x)

        rider_points = {
            r: gurobipy.quicksum(matching_scores[(r, j)] * x[r, j] for j in jobs)
            for r in resources
        }
        big_m = {r: sum(matching_scores[(r, j)] for j in jobs) for r in resources}
        counts = model.addVars(
            resources, vtype=gurobipy.GRB.BINARY, name="counts_towards_score"
        )
        counted_points = model.addVars(resources, lb=0.0, name="counted_points")
        model.addConstr(
            gurobipy.quicksum(counts[r] for r in resources) <= self.top_n,
            name="top_n_count",
        )
        model.addConstrs(
            (counted_points[r] <= rider_points[r] for r in resources),
            name="counted_le_points",
        )
        model.addConstrs(
            (counted_points[r] <= big_m[r] * counts[r] for r in resources),
            name="counted_le_selected",
        )
        return gurobipy.quicksum(counted_points[r] for r in resources)

    def prepare(self, prepared_features=None):
        """Predict once for reuse across workload and injury scenarios."""
        if self._prepared_weights is None:
            if not self.other_riders:
                self._set_other_riders()
            self._prepared_weights = (self._build_weight_matrix() if prepared_features is None
                                      else self._build_weight_matrix(prepared_features))
        return self._prepared_weights

    def run(self, output_dir: Path | None = None) -> pd.DataFrame:
        """Run optimization and return a per-race comparison DataFrame.

        Columns returned:
            job                     — race / stage-race href
            original_expected_pts   — expected pts from the actual startlist selection
            original_riders         — riders from self.riders that actually started
            n_original              — number of actual starters from the squad
            optimized_expected_pts  — expected pts from the MILP-optimal selection
            optimized_riders        — MILP-selected riders
            n_optimized             — number of MILP-selected riders

        Parameters
        ----------
        output_dir : Path, optional
            Directory to write optimized_roster.json and expected_points_matrix.csv.
            Defaults to outputs/<timestamp>/.
        """
        print("Estimating points...")
        self.points_summary = {"total_original_pts": np.nan, "total_optimized_pts": np.nan}
        self.solver_stats = {}
        expected_points_matrix, gc_points_matrix = self.prepare()
        combined_weights, combined_jobs, _, _ = self._combine_stages(
            expected_points_matrix
        )
        combined_gc_weights, _, _, _ = self._combine_stages(gc_points_matrix)

        rider_to_idx = {r: i for i, r in enumerate(self.riders)}
        own_riders_set = set(self.riders)

        # ── Original roster expected points ──────────────────────────────────
        startlists_path = (
            CWD.parent.parent / "data" / "startlists" / f"startlists_{self.year}.json"
        )
        with open(startlists_path, "r", encoding="utf-8") as f:
            startlists: dict[str, list] = json.load(f)

        original_rows: list[dict] = []
        for cj, job in enumerate(combined_jobs):
            entries = startlists.get(job, [])
            actual_starters = {e[0] for e in entries} & own_riders_set
            original_pts = sum(
                combined_weights[rider_to_idx[r], cj]
                for r in actual_starters
                if r in rider_to_idx
            )
            original_rows.append(
                {
                    "job": job,
                    "original_expected_pts": (
                        original_pts if actual_starters else np.nan
                    ),
                    "original_riders": sorted(actual_starters),
                    "n_original": len(actual_starters),
                }
            )

        # ── MILP optimisation ─────────────────────────────────────────────────
        print("Building and solving MILP...")
        model = self._create_milp_model(expected_points_matrix)
        try:
            model.optimize()
        except Exception:
            model.dispose()
            raise

        # Multi-objective models may not expose a gap. Do not invent a zero gap.
        try:
            mip_gap = float(model.MIPGap)
        except (AttributeError, gurobipy.GurobiError):
            mip_gap = float("nan")
        self.solver_stats = {
            "n_variables": int(model.NumVars),
            "n_constraints": int(model.NumConstrs),
            "solve_time_s": float(model.Runtime),
            "mip_gap": mip_gap,
            "status": int(model.Status),
            "solution_count": int(model.SolCount),
            "is_optimal": model.Status == gurobipy.GRB.OPTIMAL,
            "requested_mip_gap": self.mip_gap,
            "undated_injury_records": len(self.undated_injuries),
        }

        if not model.SolCount:
            print(f"Optimization ended without a feasible solution (status {model.status})")
            df = pd.DataFrame(original_rows)
            df[["optimized_expected_pts", "optimized_riders", "n_optimized"]] = np.nan
            self.points_summary["total_original_pts"] = _top_n_sum({
                r: sum(combined_weights[rider_to_idx[r], cj]
                       for cj, row in enumerate(original_rows) if r in row["original_riders"])
                for r in self.riders
            }, self.top_n)
            model.dispose()
            return df

        # Extract variable assignments
        selected: dict[str, list[str]] = {}
        for (rider, job), var in self._assignment_vars.items():
            if var.X > 0.5:
                selected.setdefault(job, []).append(rider)
        model.dispose()

        optimized_lookup: dict[str, dict] = {}
        for cj, job in enumerate(combined_jobs):
            opt_riders = selected.get(job, [])
            opt_pts = sum(
                combined_weights[rider_to_idx[r], cj]
                for r in opt_riders
                if r in rider_to_idx
            )
            optimized_lookup[job] = {
                "optimized_expected_pts": opt_pts,
                "optimized_riders": sorted(opt_riders),
                "n_optimized": len(opt_riders),
            }

        results = [
            {**row, **optimized_lookup.get(row["job"], {})} for row in original_rows
        ]
        df = pd.DataFrame(results)

        # Headline totals use the top-N aggregation matching the MILP objective:
        # per-rider season totals, then sum of the top-N riders.
        orig_totals = {r: 0.0 for r in self.riders}
        opt_totals = {r: 0.0 for r in self.riders}
        for row in original_rows:
            cj = combined_jobs.index(row["job"])
            for r in row["original_riders"]:
                if r in rider_to_idx:
                    orig_totals[r] += combined_weights[rider_to_idx[r], cj]
        for cj, job in enumerate(combined_jobs):
            for r in selected.get(job, []):
                if r in rider_to_idx:
                    opt_totals[r] += combined_weights[rider_to_idx[r], cj]

        total_original = _top_n_sum(orig_totals, self.top_n)
        total_optimized = _top_n_sum(opt_totals, self.top_n)
        self.points_summary = {
            "total_original_pts": total_original,
            "total_optimized_pts": total_optimized,
        }
        label = (
            f"top-{self.top_n}"
            if self.top_n and 0 < self.top_n < len(self.riders)
            else "all riders"
        )
        print(
            f"\nOriginal roster expected pts ({label}) : {total_original:.1f}"
            f"\nOptimised roster expected pts ({label}): {total_optimized:.1f}"
            f"  ({total_optimized - total_original:+.1f})"
        )

        # ── Impact on other teams ─────────────────────────────────────────────
        original_selection = {
            row["job"]: row["original_riders"] for row in original_rows
        }
        original_field_totals, original_field_jobs = self._score_selection(original_selection)
        optimized_field_totals, optimized_field_jobs = self._score_selection(selected)
        df["original_selected_field_pts"] = df["job"].map(original_field_jobs)
        df["optimized_selected_field_pts"] = df["job"].map(optimized_field_jobs)
        self.points_summary.update({
            "total_original_selected_field_pts": _top_n_sum(original_field_totals, self.top_n),
            "total_optimized_selected_field_pts": _top_n_sum(optimized_field_totals, self.top_n),
        })
        impact_df = self._compute_other_team_impact(
            startlists, original_selection, selected
        )

        # ── Persist outputs ───────────────────────────────────────────────────
        if output_dir is None:
            output_dir = (
                CWD.parent.parent / "outputs" / f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
            )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "optimized_roster.json", "w", encoding="utf-8") as f:
            json.dump(selected, f, indent=2)

        pd.DataFrame(combined_weights, index=self.riders, columns=combined_jobs).to_csv(
            output_dir / "expected_points_matrix.csv"
        )
        pd.DataFrame(
            combined_gc_weights, index=self.riders, columns=combined_jobs
        ).to_csv(output_dir / "gc_expected_points_matrix.csv")

        if not impact_df.empty:
            impact_df.to_csv(output_dir / "other_team_impact.csv", index=False)
            plot_other_team_impact(impact_df, output_dir, title=output_dir.name)

        # ── Per-rider points breakdown ────────────────────────────────────────
        _, _, _, job_dates = self._combine_stages(expected_points_matrix)
        rider_rows: list[dict] = []
        for rider in self.riders:
            ridx = rider_to_idx[rider]
            assigned_jobs = [j for j, rlist in selected.items() if rider in rlist]
            opt_pts = sum(
                combined_weights[ridx, combined_jobs.index(j)] for j in assigned_jobs
            )
            race_days = sum(len(job_dates[j]) for j in assigned_jobs)

            # Original: races this rider actually started
            orig_jobs = [
                job
                for cj, job in enumerate(combined_jobs)
                if rider in {e[0] for e in startlists.get(job, [])}
            ]
            orig_pts = sum(
                combined_weights[ridx, combined_jobs.index(j)] for j in orig_jobs
            )
            orig_race_days = sum(len(job_dates[j]) for j in orig_jobs)

            rider_rows.append(
                {
                    "rider": rider,
                    "optimized_pts": opt_pts,
                    "optimized_jobs": len(assigned_jobs),
                    "optimized_race_days": race_days,
                    "original_pts": orig_pts,
                    "original_jobs": len(orig_jobs),
                    "original_race_days": orig_race_days,
                    "race_day_limit": self.rider_race_day_limits[rider],
                    "original_selected_field_pts": original_field_totals[rider],
                    "optimized_selected_field_pts": optimized_field_totals[rider],
                }
            )

        rider_df = pd.DataFrame(rider_rows).sort_values(
            "optimized_pts", ascending=False
        )
        rider_df.to_csv(output_dir / "per_rider_points.csv", index=False)

        return df


def plot_other_team_impact(
    impact_df: pd.DataFrame,
    output_dir: Path,
    title: str = "",
) -> None:
    """Bar chart showing which other teams lose / gain pts due to our roster optimisation.

    Red bars = they lose expected points (we send stronger riders).
    Blue bars = they gain expected points (our optimised roster is weaker for them).

    The plot is saved to *output_dir / other_team_impact.png*.
    """
    if impact_df.empty:
        return

    summary = (
        impact_df.groupby("other_team_href")["pts_delta"]
        .sum()
        .sort_values(ascending=False)
    )
    if summary.empty:
        return

    def _short_name(href: str) -> str:
        name = href.rstrip("/").split("/")[-1]
        if len(name) > 4 and name[-5] == "-" and name[-4:].isdigit():
            name = name[:-5]
        return name.replace("-", " ").title()

    display_names = [_short_name(h) for h in summary.index]
    colors = ["#d73027" if v > 0 else "#4575b4" for v in summary.values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(summary) * 0.35)))
    ax.barh(display_names[:20], summary.values[:20], color=colors[:20])
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(
        "Expected pts delta (positive = they lose pts due to our optimisation)"
    )
    suffix = f": {title}" if title else ""
    ax.set_title(f"Impact of optimised roster on other teams{suffix}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "other_team_impact.png", dpi=150)
    plt.close(fig)


def _get_teams(year: int) -> tuple[dict, dict]:
    with open(
        CWD.parent.parent / "data" / "teams" / f"teams_{year}.json",
        "r",
        encoding="utf-8",
    ) as f:
        teams_data = json.load(f)

    with open(
        CWD.parent.parent / "data" / "roster" / f"programs_{year}.json",
        "r",
        encoding="utf-8",
    ) as f:
        programs_data = json.load(f)

    return teams_data, programs_data


@lru_cache(maxsize=None)
def _get_n_per_team(race_href: str) -> int | None:
    year = int(race_href.split("/")[-1])
    startlists_path = (
        CWD.parent.parent / "data" / "startlists" / f"startlists_{year}.json"
    )
    if not startlists_path.exists():
        return None
    with open(startlists_path, "r", encoding="utf-8") as f:
        startlists = json.load(f)

    entries = startlists.get(race_href, [])
    if not entries:
        return None
    team_counts: dict[str, int] = {}
    for entry in {tuple(e[:2]) for e in entries}:
        team_counts[entry[1]] = team_counts.get(entry[1], 0) + 1
    npt = max(team_counts.values())
    return npt


def _get_race_objects(team_program: dict) -> list[Race]:
    program_list = []
    for race, details in team_program.items():
        if details.get("level", "") not in [
            "1.UWT",
            "1.Pro",
            "2.UWT",
            "2.Pro",
            "1.1",
            "2.1",
            "1.2",
            "2.2",
        ]:
            continue
        if "stages" in details:
            for stage in details["stages"]:
                try:
                    program_list.append(
                        Race(
                            href=stage["href"],
                            date=datetime.strptime(stage["date"], "%Y-%m-%d"),
                            level=details["level"],
                            tt_kms=stage.get("tt_kms", 0.0),
                            parent_href=race,
                            n_per_team=_get_n_per_team(race),
                        )
                    )
                except (ValueError, KeyError, TypeError) as e:
                    raise ValueError(f"Error parsing stage {stage} of race {race}") from e
        else:
            try:
                program_list.append(
                    Race(
                        href=race,
                        date=datetime.strptime(details["date"], "%Y-%m-%d"),
                        level=details["level"],
                        n_per_team=_get_n_per_team(race),
                    )
                )
            except (ValueError, KeyError, TypeError) as e:
                raise ValueError(f"Error parsing race {race}") from e

    return sorted(program_list, key=lambda r: (r.date, r.href))


def _top_n_sum(totals: dict[str, float], top_n: int) -> float:
    """Sum the values of the top-N contributors (all of them when top_n <= 0)."""
    vals = sorted(totals.values(), reverse=True)
    if top_n and 0 < top_n < len(vals):
        vals = vals[:top_n]
    return float(sum(vals))


def _norm_race_href(href: str) -> str:
    """Normalise a race href to the bare ``race/.../<year>`` form."""
    href = href.replace("https://www.procyclingstats.com", "")
    return href.lstrip("/")


def _load_actual_points_table(year: int) -> pd.DataFrame:
    """Load real-world UCI points for *year* from the model-training CSVs.

    Combines the stage / one-day points (``features-*-detailed_gpx.csv``) with
    the GC points (``gc_features_*.csv``) — the exact data the ranking models
    are trained on — into a long table ``[rider_href, race_href, uci_points]``.
    """
    root = CWD.parent.parent
    frames: list[pd.DataFrame] = []
    for rel in (
        "models/features-2019-2025-detailed_gpx.csv",
        "models/gc_features_2019_2025.csv",
    ):
        path = root / rel
        df = pd.read_csv(path, usecols=["rider_href", "race_href", "uci_points", "date"])
        df = df[df["date"].astype(str).str.startswith(str(year))]
        df = df.assign(race_href=df["race_href"].map(_norm_race_href))
        frames.append(df[["rider_href", "race_href", "uci_points"]])
    if not frames:
        return pd.DataFrame(columns=["rider_href", "race_href", "uci_points"])
    return pd.concat(frames, ignore_index=True)


def _get_actual_uci_points(
    actual_df: pd.DataFrame,
    race_hrefs: set[str],
    riders: set[str],
) -> dict[str, float]:
    """Sum real-world UCI points for a team's riders, restricted to its program.

    Only races in ``race_hrefs`` (the team's scheduled parent races) count.
    Stage, one-day and GC points are already combined in ``actual_df``.

    Parameters
    ----------
    actual_df : pd.DataFrame
        Long table from :func:`_load_actual_points_table` for the season.
    race_hrefs : set[str]
        Parent race hrefs from the team's schedule (any href form is accepted).
    riders : set[str]
        Team rider hrefs (e.g. ``rider/mathieu-van-der-poel``).

    Returns
    -------
    dict[str, float]
        ``{rider_href: total_uci_points}`` for every rider in ``riders``.
    """
    norm_hrefs = {_norm_race_href(h) for h in race_hrefs}
    sub = actual_df[
        actual_df["rider_href"].isin(riders)
        & actual_df["race_href"].isin(norm_hrefs)
    ]
    totals = sub.groupby("rider_href")["uci_points"].sum()
    return {r: float(totals.get(r, 0.0)) for r in riders}


def evaluate_season(argv=None):
    """Run the historical scenario sweep (also the legacy CLI entry)."""
    from src.rostering.evaluation import main
    return main(argv)


if __name__ == "__main__":
    evaluate_season()
