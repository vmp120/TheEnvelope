import os
import pandas as pd
import numpy as np
from collections import defaultdict

CLEANED_PATH  = "data/processed/cleaned_dataset.csv"
FEATURES_PATH = "data/processed/features_dataset.csv"

# oct–dec is "awards season"
AWARDS_SEASON_MONTHS = {10, 11, 12}

# MPAA ratings ordered from most to least restrictive for ordinal encoding
MPAA_ORDER = ["G", "PG", "PG-13", "R", "NC-17", "NR"]

# TMDb genre list (semicolon-separated in the 'genres' column after ETL)
KNOWN_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
    "Thriller", "TV Movie", "War", "Western",
]


def load_cleaned(path: str = CLEANED_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cleaned dataset not found at '{path}'.\n"
            "Run etl.py first."
        )
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded cleaned dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df

def add_commercial_features(df: pd.DataFrame) -> pd.DataFrame:

    # profit margin, ROI, and log-scaled budget/revenue
    # log scaling compresses extreme right tail of hollywood budgets (tree models and logistic regression not dominated by blockbuster outliers)

    if "budget" in df.columns and "revenue" in df.columns:
        df["profit_margin"] = df["revenue"] - df["budget"]

        # avoid division by zero (budget==NaN handled by Pandas naturally)
        df["roi"] = df["profit_margin"] / df["budget"].replace(0, np.nan)

        df["log_budget"]  = np.log1p(df["budget"].clip(lower=0))
        df["log_revenue"] = np.log1p(df["revenue"].clip(lower=0))

    print("Added commercial features: profit_margin, roi, log_budget, log_revenue")
    return df


def add_release_timing_features(df: pd.DataFrame) -> pd.DataFrame:

    # awards season flag and release quarter
    if "release_month" in df.columns:
        df["awards_season"] = df["release_month"].isin(AWARDS_SEASON_MONTHS).astype(int)
        df["release_quarter"] = ((df["release_month"] - 1) // 3 + 1).astype("Int64")

    print("Added release timing features: awards_season, release_quarter")
    return df


def add_genre_features(df: pd.DataFrame) -> pd.DataFrame:

    # one-hot encode TMDb genres and compute genre nomination rate from training data
    if "genres" not in df.columns:
        return df

    # one-hot
    for genre in KNOWN_GENRES:
        col = f"genre_{genre.lower().replace(' ', '_')}"
        df[col] = df["genres"].str.contains(genre, na=False).astype(int)

    # genre nomination rate
    genre_cols = [f"genre_{g.lower().replace(' ', '_')}" for g in KNOWN_GENRES]
    if "nominated" in df.columns:
        genre_nom_rates = {}
        for col, genre in zip(genre_cols, KNOWN_GENRES):
            mask = df[col] == 1
            if mask.sum() > 0:
                genre_nom_rates[genre] = df.loc[mask, "nominated"].mean()
            else:
                genre_nom_rates[genre] = 0.0

        def _mean_genre_nom_rate(genres_str):
            if pd.isna(genres_str):
                return np.nan
            genres = [g.strip() for g in genres_str.split(";")]
            rates = [genre_nom_rates[g] for g in genres if g in genre_nom_rates]
            return np.mean(rates) if rates else np.nan

        df["genre_nom_rate"] = df["genres"].apply(_mean_genre_nom_rate)

    print(f"Added genre features: {len(genre_cols)} one-hot columns + genre_nom_rate")
    return df

def add_mpaa_features(df: pd.DataFrame) -> pd.DataFrame:

    # ordinal-encode MPAA rating (G=0 means NR=5)
    # R-rated films dominate oscar drama nominees (G/PG skew toward animated feature nominees) so ordinal integer preserves ordering

    if "mpaa_rating" not in df.columns:
        return df

    rating_map = {r: i for i, r in enumerate(MPAA_ORDER)}
    df["mpaa_encoded"] = (
        df["mpaa_rating"]
        .str.upper()
        .str.strip()
        .map(rating_map)
        .fillna(rating_map["NR"])
        .astype(int)
    )

    print("Added MPAA feature: mpaa_encoded")
    return df

def _build_prior_nom_counts(df: pd.DataFrame, name_col: str, year_col: str = "release_year", target_col: str = "nominated", sep: str = "|") -> pd.Series:

    # for each film, count how many prior nominations people listed in name_col had before film's release year (computed in chronological order to avoid label leakage)

    if name_col not in df.columns or year_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    # build (name to set of years where they were nominated) as we iterate chronologically
    df_sorted = df[[name_col, year_col, target_col]].copy()
    df_sorted[year_col] = pd.to_numeric(df_sorted[year_col], errors="coerce")
    df_sorted = df_sorted.sort_values(year_col, na_position="last")

    # cumulative nomination counts per person, keyed by year
    # structure is {name: [(year, cumulative count), ...]}
    person_cum_noms: dict[str, int] = defaultdict(int)
    # need per-row prior counts, so process year by year
    # group rows by year, then for each group compute prior (before this year) then update cumulative counts
    prior_counts = pd.Series(np.nan, index=df_sorted.index, dtype=float)

    years = df_sorted[year_col].dropna().unique()
    years.sort()

    # snapshot of cumulative noms BEFORE each year
    snapshot_before: dict[float, dict[str, int]] = {}
    running: dict[str, int] = defaultdict(int)

    for yr in years:
        snapshot_before[yr] = dict(running)  # prior to this year
        year_mask = df_sorted[year_col] == yr
        for idx, row in df_sorted[year_mask].iterrows():
            if pd.isna(row[name_col]):
                continue
            people = [p.strip() for p in str(row[name_col]).split(sep) if p.strip()]
            if row[target_col] == 1:
                for person in people:
                    running[person] += 1

    # second pass (assign prior counts)
    for yr in years:
        snap = snapshot_before[yr]
        year_mask = df_sorted[year_col] == yr
        for idx, row in df_sorted[year_mask].iterrows():
            if pd.isna(row[name_col]):
                prior_counts.at[idx] = 0.0
                continue
            people = [p.strip() for p in str(row[name_col]).split(sep) if p.strip()]
            total = sum(snap.get(person, 0) for person in people)
            prior_counts.at[idx] = float(total)

    # re-align to original df index
    return prior_counts.reindex(df.index)


def add_talent_features(df: pd.DataFrame) -> pd.DataFrame:

    # director and lead-cast prior nomination counts (no label leakage)

    print("Computing director prior nomination counts (this may take a moment)...")
    df["director_prior_noms"] = _build_prior_nom_counts(
        df, name_col="director", year_col="release_year"
    )

    print("Computing cast prior nomination counts...")
    df["cast_prior_noms"] = _build_prior_nom_counts(
        df, name_col="top_cast", year_col="release_year"
    )

    # binary flags (has any previously-nominated talent attached)
    df["has_nominated_director"] = (df["director_prior_noms"] > 0).astype(int)
    df["has_nominated_cast"]     = (df["cast_prior_noms"] > 0).astype(int)

    print("Added talent features: director_prior_noms, cast_prior_noms, "
          "has_nominated_director, has_nominated_cast")
    return df

def add_engagement_features(df: pd.DataFrame) -> pd.DataFrame:

    # log-scaled vote counts and combined engagement score (raw vote counts span several orders of magnitude, log scaling stops blockbusters from crowding out everything else in tree splits)

    for col in ["tmdb_vote_count", "imdb_vote_count"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # simple composite: mean of available normalised rating signals
    rating_cols = [c for c in ["tmdb_vote_avg", "imdb_rating"] if c in df.columns]
    if rating_cols:
        df["mean_rating"] = df[rating_cols].mean(axis=1)

    print("Added engagement features: log vote counts, mean_rating")
    return df

# raw columns either encoded above or too noisy to hand directly to model without transformation
ENGINEERED_AWAY = [
    "release_date", # decomposed into release_month, release_year, etc.
    "genres", # one-hot encoded
    "imdb_genres", # redundant with TMDb genres
    "mpaa_rating", # ordinally encoded above
    "title_tmdb", # not a predictive feature
    "title_imdb", # not a predictive feature
    "imdb_id", # identifier
    "top_cast", # encoded via cast_prior_noms
    "director", # encoded via director_prior_noms
    "original_language", # low-cardinality
]

def drop_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in ENGINEERED_AWAY if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} raw columns superseded by engineered features")
    return df

def check(df: pd.DataFrame):
    print("\n--- Feature Set Summary ---")
    print(f"Rows:             {len(df):,}")
    print(f"Columns:          {len(df.columns)}")
    print(f"Nominated=1:      {df['nominated'].sum():,}")
    print(f"Nominated=0:      {(df['nominated'] == 0).sum():,}")
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    if null_counts.empty:
        print("Null cells:       0")
    else:
        print(f"Columns with nulls ({len(null_counts)}):")
        for col, cnt in null_counts.items():
            print(f"  {col}: {cnt:,}")
    print("---------------------------\n")
    print("Feature columns:")
    for col in sorted(df.columns):
        print(f"  {col}")

def build_features(
    input_path:  str = CLEANED_PATH,
    output_path: str = FEATURES_PATH,
) -> pd.DataFrame:
    os.makedirs("data/processed", exist_ok=True)

    df = load_cleaned(input_path)

    df = add_commercial_features(df)
    df = add_release_timing_features(df)
    df = add_genre_features(df)
    df = add_mpaa_features(df)
    df = add_talent_features(df)
    df = add_engagement_features(df)
    df = drop_raw_columns(df)

    check(df)

    df.to_csv(output_path, index=False)
    print(f"Saved feature dataset to {output_path}")

    return df


if __name__ == "__main__":
    build_features()