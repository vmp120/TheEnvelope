import os
import pandas as pd
import numpy as np

def load_merged(path="data/merged/merged_dataset.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError (
            f"Merged dataset not found at '{path}'.\n"
            "Run data_collection.py first."
        )
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded merged dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df

# columns that have no use for prediction
drop_columns = [
    "overview", # unnecessary text not used in this model
    "film", # duplicate of title_tmdb from oscar merge
    "nomid", # oscar internal ID
    "detail", # oscar ceremony detail text
    "note", # oscar note text
    "citation", # oscar citation text
    "nomineeids", # raw ID strings
    "nominees", # raw name strings
    "multifilmnomination", # edge case flag not useful for prediction
    "tmdb_id", # internal ID not feature
    "director_ids", # raw nconst strings resolved in features.py
]

def drop_uninformative_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in drop_columns if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} uninformative columns")
    return df

def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    # release_date to datetime, extract release_month and release_year
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_month"] = df["release_date"].dt.month
        df["release_year"] = df["release_date"].dt.year

    # numeric columns that may have been read as strings
    numeric_cols = [
        "budget", "revenue", "runtime",
        "tmdb_popularity", "tmdb_vote_avg", "tmdb_vote_count",
        "imdb_rating", "imdb_vote_count", "imdb_runtime",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # year columns as integers
    for col in ["release_year", "year_tmdb", "year_imdb", "year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    print("Fixed column types")
    return df

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    initial_rows = len(df)

    # drop rows with no title at all since completely unusable
    df = df.dropna(subset=["title_tmdb"])

    # budget and revenue of 0 means "not reported" and not actually zero
    # replace with NaN so can be imputed properly
    for col in ["budget", "revenue"]:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # impute budget and revenue with median within the same release year (more accurate than a global median since production costs change over time)
    for col in ["budget", "revenue"]:
        if col in df.columns:
            df[col] = df.groupby("release_year")[col].transform(
                lambda x: x.fillna(x.median())
            )
            # if still missing fill with global median
            df[col] = df[col].fillna(df[col].median())

    # Runtime - fill with median
    if "runtime" in df.columns:
        df["runtime"] = df["runtime"].fillna(df["runtime"].median())

    # IMDb rating - fill with median
    if "imdb_rating" in df.columns:
        df["imdb_rating"] = df["imdb_rating"].fillna(df["imdb_rating"].median())

    # IMDb vote count - fill with 0 (no votes = unknown)
    if "imdb_vote_count" in df.columns:
        df["imdb_vote_count"] = df["imdb_vote_count"].fillna(0)

    # MPAA rating - fill unknown with "NR" (not rated)
    if "mpaa_rating" in df.columns:
        df["mpaa_rating"] = df["mpaa_rating"].fillna("NR")

    # Release month - fill with 0 (unknown)
    if "release_month" in df.columns:
        df["release_month"] = df["release_month"].fillna(0).astype(int)

    # Genres - fill with "Unknown"
    for col in ["genres", "imdb_genres"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    dropped = initial_rows - len(df)
    print(f"Handled missing values — dropped {dropped:,} untitled rows")
    print(f"  Remaining nulls per column:")

    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    if null_counts.empty:
        print("    None")
    else:
        for col, count in null_counts.items():
            print(f"    {col}: {count:,}")

    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # remove duplicate on imdb_id (keep first occurrence)
    if "imdb_id" in df.columns:
        df = df.drop_duplicates(subset="imdb_id", keep="first")

    # remove movies with unrealistically short runtime (< 40 min = not a feature film)
    if "runtime" in df.columns:
        df = df[df["runtime"] >= 40]

    # remove movies outside 2010–2024 (target window)
    if "release_year" in df.columns:
        df = df[df["release_year"].between(2010, 2024)]

    dropped = before - len(df)
    print(f"Removed {dropped:,} duplicate/junk rows — {len(df):,} remaining")
    return df

def check(df: pd.DataFrame):
    print("\n--- Check ---")
    print(f"Total rows:       {len(df):,}")
    print(f"Nominated=1:      {df['nominated'].sum():,}")
    print(f"Nominated=0:      {(df['nominated'] == 0).sum():,}")
    print(f"Imbalance ratio:  1:{int((df['nominated'] == 0).sum() / max(df['nominated'].sum(), 1))}")
    print(f"Columns:          {len(df.columns)}")
    print(f"Null cells:       {df.isnull().sum().sum():,}")
    print("--------------------\n")

def run_etl(input_path="data/merged/merged_dataset.csv",output_path="data/processed/cleaned_dataset.csv") -> pd.DataFrame:

    os.makedirs("data/processed", exist_ok=True)

    df = load_merged(input_path)
    df = drop_uninformative_columns(df)
    df = fix_types(df)
    df = handle_missing(df)
    df = remove_duplicates(df)

    check(df)

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return df

if __name__ == "__main__":
    run_etl()