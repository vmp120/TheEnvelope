import os
import time
import threading
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import shutil
import time
from tqdm import tqdm

DATA_DIR = "data/raw"
MAX_RETRIES = 3
CHUNK_SIZE = 1024

TMDB_API_KEY = "86772112cfd4d268b4c07a24ea06b9b8"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

IMDB_URLS = {
    "name.basics.tsv.gz": "https://datasets.imdbws.com/name.basics.tsv.gz",
    "title.basics.tsv.gz": "https://datasets.imdbws.com/title.basics.tsv.gz",
    "title.crew.tsv.gz": "https://datasets.imdbws.com/title.crew.tsv.gz",
    "title.ratings.tsv.gz": "https://datasets.imdbws.com/title.ratings.tsv.gz",
}

OSCARS_URL = "https://drive.google.com/uc?export=download&id=1r0hR64vWMkx0uDFIWbtMLwBcGYxeK2hS"

def ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def download_file(url, output_path):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(output_path, "wb") as file, tqdm(
                desc=os.path.basename(output_path),
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        file.write(chunk)
                        bar.update(len(chunk))

            return

        except Exception as e:
            print(f"[Retry {attempt}/{MAX_RETRIES}] Failed: {e}")
            time.sleep(2 * attempt)

    print(f"Failed to download {url} after {MAX_RETRIES} attempts.")


def unzip_file(gz_path):
    tsv_path = gz_path.replace(".gz", "")

    if os.path.exists(tsv_path):
        print(f"Skipping unzip (exists): {tsv_path}")
        return

    print(f"Unzipping {os.path.basename(gz_path)}...")

    with gzip.open(gz_path, "rb") as f_in:
        with open(tsv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

def download_imdb(unzip=True):
    for filename, url in IMDB_URLS.items():
        output_path = os.path.join(DATA_DIR, filename)

        if os.path.exists(output_path):
            print(f"Skipping (exists): {filename}")
        else:
            print(f"Downloading {filename}...")
            download_file(url, output_path)

        if unzip:
            unzip_file(output_path)


def download_oscars():
    if not OSCARS_URL:
        return

    output_path = os.path.join(DATA_DIR, "the_oscar_award.csv")

    if os.path.exists(output_path):
        print("Skipping Oscars dataset (exists)")
        return

    print("Downloading Oscars dataset...")
    download_file(OSCARS_URL, output_path)

def _tmdb_get(endpoint: str, params: dict) -> dict:
    params["api_key"] = TMDB_API_KEY
    url = f"{TMDB_BASE_URL}/{endpoint}"

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                print(f"Rate limited. Retrying after {retry_after}s...")
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            print(f"TMDb request failed (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)

    return {}

def get_tmdb_movie_ids_by_year(year, pages=10):
    ids = []

    for page in range(1, pages + 1):
        data = _tmdb_get("discover/movie", {
            "primary_release_year": year,
            "sort_by": "popularity.desc",
            "page": page
        })

        results = data.get("results", [])
        if not results:
            break

        ids.extend([m["id"] for m in results])

    print(f"{year}: {len(ids)} IDs")
    return ids

def get_tmdb_movie_details(movie_id):
    data = _tmdb_get(f"movie/{movie_id}", {
        "append_to_response": "credits,release_dates"
    })

    if not data:
        return {}

    cast = data.get("credits", {}).get("cast", [])[:5]
    crew = data.get("credits", {}).get("crew", [])

    directors = [m["name"] for m in crew if m["job"] == "Director"]
    top_cast = [m["name"] for m in cast]

    mpaa_rating = None
    for entry in data.get("release_dates", {}).get("results", []):
        if entry["iso_3166_1"] == "US":
            for rd in entry.get("release_dates", []):
                if rd.get("certification"):
                    mpaa_rating = rd["certification"]
                    break

    genres = [g["name"] for g in data.get("genres", [])]

    return {
        "tmdb_id": data.get("id"),
        "imdb_id": data.get("imdb_id"),
        "title": data.get("title"),
        "release_date": data.get("release_date"),
        "runtime": data.get("runtime"),
        "budget": data.get("budget"),
        "revenue": data.get("revenue"),
        "genres": ";".join(genres),
        "mpaa_rating": mpaa_rating,
        "original_language": data.get("original_language"),
        "tmdb_popularity": data.get("popularity"),
        "tmdb_vote_avg": data.get("vote_average"),
        "tmdb_vote_count": data.get("vote_count"),
        "director": "|".join(directors),
        "top_cast": "|".join(top_cast),
        "overview": data.get("overview"),
    }

def collect_tmdb(years, max_workers=12, rate_limit_per_sec=35):
    checkpoint_path = "data/raw/tmdb_raw.csv"
    os.makedirs("data/raw", exist_ok=True)

    all_records = []
    seen_ids = set()

    # Resume
    if os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        if not existing.empty:
            all_records = existing.to_dict("records")
            seen_ids = set(existing["tmdb_id"].dropna().astype(int))

            completed_years = set(
                pd.to_datetime(existing["release_date"], errors="coerce")
                .dt.year.dropna().astype(int)
            )
        else:
            completed_years = set()
    else:
        completed_years = set()

    # rate limiter
    lock = threading.Lock()
    request_times = []

    def rate_limited_call(mid):
        nonlocal request_times
        with lock:
            now = time.time()
            request_times = [t for t in request_times if now - t < 1]

            if len(request_times) >= rate_limit_per_sec:
                sleep_time = 1 - (now - request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)

            request_times.append(time.time())

        return get_tmdb_movie_details(mid)

    # main loop
    for year in years:
        if year in completed_years:
            print(f"Skipping {year}")
            continue

        movie_ids = get_tmdb_movie_ids_by_year(year)
        movie_ids = [mid for mid in movie_ids if mid not in seen_ids]

        print(f"{year}: fetching {len(movie_ids)} movies...")

        new_records = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(rate_limited_call, mid): mid for mid in movie_ids}

            for future in as_completed(futures):
                mid = futures[future]
                try:
                    data = future.result()
                    if data:
                        new_records.append(data)
                        seen_ids.add(mid)
                except Exception as e:
                    print(f"Error {mid}: {e}")

        all_records.extend(new_records)

        pd.DataFrame(all_records).drop_duplicates("tmdb_id").to_csv(
            checkpoint_path, index=False
        )

        print(f"Saved checkpoint after {year} (+{len(new_records)})")

    df = pd.DataFrame(all_records).drop_duplicates("tmdb_id")
    print(f"TMDb complete: {len(df):,}")
    return df

def _imdb_path(filename, data_dir="data/raw"):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing IMDb file: {path}")
    return path


def load_imdb(data_dir="data/raw"):
    basics = pd.read_csv(
        _imdb_path("title.basics.tsv", data_dir),
        sep="\t", na_values="\\N", low_memory=False,
        usecols=["tconst", "titleType", "primaryTitle", "startYear", "runtimeMinutes", "genres"]
    )
    basics = basics[basics["titleType"] == "movie"].drop(columns="titleType")

    ratings = pd.read_csv(_imdb_path("title.ratings.tsv", data_dir),
        sep="\t", na_values="\\N"
    )

    crew = pd.read_csv(_imdb_path("title.crew.tsv", data_dir),sep="\t", na_values="\\N",usecols=["tconst", "directors"])

    df = basics.merge(ratings, on="tconst", how="left").merge(crew, on="tconst", how="left")

    df.rename(columns={
        "tconst": "imdb_id",
        "primaryTitle": "title",
        "startYear": "year",
        "runtimeMinutes": "imdb_runtime",
        "genres": "imdb_genres",
        "averageRating": "imdb_rating",
        "numVotes": "imdb_vote_count",
        "directors": "director_ids"
    }, inplace=True)

    print(f"IMDb loaded: {len(df):,}")
    return df

def load_oscars(path="data/raw/the_oscar_award.csv"):
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lower().str.strip()

    # convert string to int
    df["year"] = df["year"].str.split("/").str[0].astype(int)

    df["nominated"] = 1
    df["winner"] = df["winner"].astype(bool)

    print(f"Oscars loaded: {len(df):,}")
    return df

def collect_all(years=range(2010, 2025)):
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/merged", exist_ok=True)

    # TMDb
    tmdb_df = collect_tmdb(years)
    tmdb_df.to_csv("data/raw/tmdb_raw.csv", index=False)

    # IMDb
    imdb_df = load_imdb()
    imdb_df.to_csv("data/raw/imdb_movies.csv", index=False)

    # oscars
    oscars_df = load_oscars()
    oscars_df.to_csv("data/raw/oscars_clean.csv", index=False)

    merged = tmdb_df.merge(imdb_df, on="imdb_id", how="left", suffixes=("_tmdb", "_imdb"))

    oscars_per_film = (oscars_df.groupby("film").agg(nominated=("nominated", "max"),oscar_wins=("winner", "sum"),oscar_nominations=("film", "count"),).reset_index())

    merged = merged.merge(oscars_per_film, left_on="title_tmdb", right_on="film", how="left")

    merged["nominated"] = merged["nominated"].fillna(0).astype(int)
    merged["oscar_wins"] = merged["oscar_wins"].fillna(0).astype(int)
    merged["oscar_nominations"] = merged["oscar_nominations"].fillna(0).astype(int)

    merged.to_csv("data/merged/merged_dataset.csv", index=False)

    print("\nFINAL STATS")
    print(f"TMDb: {len(tmdb_df):,}")
    print(f"IMDb: {len(imdb_df):,}")
    print(f"Merged: {len(merged):,}")
    print(f"Nominated films: {merged['nominated'].sum():,}")

    print("\nSaved in data/merged/merged_dataset.csv")

    return merged

if __name__ == "__main__":

    ensure_dir()
    download_imdb(unzip=True)
    download_oscars()
    print("Data collection complete.")

    collect_all(range(2010, 2025))