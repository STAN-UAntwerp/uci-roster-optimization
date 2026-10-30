import pandas as pd
from datetime import datetime
from pathlib import Path

root = Path(__file__).parent.parent.parent


def cumulative_specialties(specialty_scores: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative specialty points for each rider over time."""
    specialty_scores["date"] = pd.to_datetime(specialty_scores["date"], errors="coerce")
    specialty_scores = specialty_scores.sort_values("date")
    specialty_scores["points"] = pd.to_numeric(
        specialty_scores["points"], errors="coerce"
    ).fillna(0)

    specialty_scores = specialty_scores[["rider_href", "specialty", "date", "points"]]

    cumulative = (
        specialty_scores.groupby(["rider_href", "specialty"])
        .apply(lambda x: x.assign(cumulative_points=x["points"].cumsum()))
        .reset_index(drop=True)
    )
    cumulative = cumulative[["rider_href", "specialty", "date", "cumulative_points"]]
    return cumulative


def add_specialty_scores(df: pd.DataFrame) -> None:
    """Add cumulative specialty scores to the given DataFrame and edit in place."""
    if "rider_href" not in df.columns or "date" not in df.columns:
        print("DataFrame must contain 'rider_href' and 'date' columns.")
        raise ValueError("Missing required columns")

    cumulative_specialties_df = pd.read_csv(
        root / "data/rider_data/cumulative_specialties.csv"
    )
    cumulative_specialties_df["date"] = pd.to_datetime(
        cumulative_specialties_df["date"], errors="coerce"
    ).astype("datetime64[ns]")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").astype("datetime64[ns]")
    for specialty in ["sprint", "climber", "hills", "gc", "tt", "onedayraces"]:
        specialty_points = cumulative_specialties_df[
            cumulative_specialties_df["specialty"] == specialty
        ].sort_values("date")
        left = df[["rider_href", "date"]].copy()
        left["_orig_idx"] = left.index
        left_sorted = left.sort_values("date")
        merged = pd.merge_asof(
            left_sorted,
            specialty_points[["rider_href", "date", "cumulative_points"]].rename(
                columns={"cumulative_points": specialty}
            ),
            on="date",
            by="rider_href",
            direction="backward",
            allow_exact_matches=False,
        )
        merged = merged.set_index("_orig_idx")
        df[specialty] = merged[specialty].reindex(df.index).fillna(0)


if __name__ == "__main__":
    specialties = pd.read_csv(root / "data/rider_data/specialty_scores.csv")
    cumulative_specialties_df = cumulative_specialties(specialties)
    cumulative_specialties_df.to_csv(
        root / "data/rider_data/cumulative_specialties.csv", index=False
    )
