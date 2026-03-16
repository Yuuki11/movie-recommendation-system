from ast import literal_eval
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DATA_PATH = BASE_DIR / "data" / "movies.csv"
KAGGLE_DIR = BASE_DIR / "data" / "kaggle"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "movies_catalog.csv"
KAGGLE_REQUIRED_FILES = (
    KAGGLE_DIR / "movies_metadata.csv",
    KAGGLE_DIR / "links_small.csv",
    KAGGLE_DIR / "ratings_small.csv",
)


def _kaggle_dataset_available() -> bool:
    return all(path.exists() for path in KAGGLE_REQUIRED_FILES)


def dataset_source() -> str:
    if _kaggle_dataset_available():
        return "Kaggle: The Movies Dataset"
    return "Bundled sample movie catalog"


def _parse_genres(raw_genres: str) -> List[str]:
    if not isinstance(raw_genres, str) or not raw_genres.strip():
        return []

    try:
        parsed = literal_eval(raw_genres)
    except (ValueError, SyntaxError):
        return []

    return [_normalize_genre_name(item["name"]) for item in parsed if isinstance(item, dict) and item.get("name")]


def _normalize_genre_name(genre_name: str) -> str:
    replacements = {
        "Science Fiction": "Sci-Fi",
    }
    return replacements.get(genre_name, genre_name)


def _build_kaggle_catalog() -> pd.DataFrame:
    metadata = pd.read_csv(
        KAGGLE_DIR / "movies_metadata.csv",
        low_memory=False,
        usecols=["id", "title", "genres", "release_date", "popularity", "vote_average", "vote_count"],
    )
    links_small = pd.read_csv(KAGGLE_DIR / "links_small.csv")
    ratings_small = pd.read_csv(KAGGLE_DIR / "ratings_small.csv")

    ratings_summary = (
        ratings_small.groupby("movieId")
        .agg(user_rating=("rating", "mean"), rating_count=("rating", "size"))
        .reset_index()
    )

    metadata["tmdbId"] = pd.to_numeric(metadata["id"], errors="coerce")
    metadata["release_date"] = pd.to_datetime(metadata["release_date"], errors="coerce")
    metadata["vote_average"] = pd.to_numeric(metadata["vote_average"], errors="coerce")
    metadata["vote_count"] = pd.to_numeric(metadata["vote_count"], errors="coerce")
    metadata["popularity"] = pd.to_numeric(metadata["popularity"], errors="coerce")
    links_small["tmdbId"] = pd.to_numeric(links_small["tmdbId"], errors="coerce")

    catalog = (
        ratings_summary.merge(links_small, on="movieId", how="inner")
        .merge(
            metadata[
                [
                    "tmdbId",
                    "title",
                    "genres",
                    "release_date",
                    "popularity",
                    "vote_average",
                    "vote_count",
                ]
            ],
            on="tmdbId",
            how="inner",
        )
        .copy()
    )

    catalog["genre_list"] = catalog["genres"].apply(_parse_genres)
    catalog["genres"] = catalog["genre_list"].apply(lambda genres: "|".join(genres))
    catalog["year"] = catalog["release_date"].dt.year

    vote_average = catalog["vote_average"].fillna(0)
    user_rating = catalog["user_rating"].fillna(0) * 2
    catalog["imdb_rating"] = (vote_average * 0.7 + user_rating * 0.3).round(1)

    popularity_signal = (
        catalog["popularity"].fillna(0)
        + np.log1p(catalog["rating_count"]) * 10
        + catalog["vote_count"].fillna(0).clip(upper=2000) / 50
    )
    catalog["popularity"] = (
        popularity_signal.rank(method="average", pct=True).mul(99).add(1).round().astype(int)
    )

    catalog = catalog.loc[
        catalog["title"].notna()
        & catalog["year"].notna()
        & catalog["genres"].ne("")
        & catalog["imdb_rating"].gt(0)
        & catalog["rating_count"].ge(20)
    ].copy()

    catalog["year"] = catalog["year"].astype(int)
    catalog = (
        catalog.sort_values(
            by=["rating_count", "imdb_rating", "popularity"],
            ascending=[False, False, False],
        )
        .drop_duplicates(subset=["movieId"])
        .rename(columns={"movieId": "movie_id"})
    )

    return catalog[["movie_id", "title", "genres", "year", "imdb_rating", "popularity"]]


@lru_cache(maxsize=1)
def load_movies() -> pd.DataFrame:
    if _kaggle_dataset_available():
        if PROCESSED_DATA_PATH.exists():
            movies = pd.read_csv(PROCESSED_DATA_PATH)
        else:
            movies = _build_kaggle_catalog()
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            movies.to_csv(PROCESSED_DATA_PATH, index=False)
    else:
        movies = pd.read_csv(SAMPLE_DATA_PATH)

    movies["genre_list"] = movies["genres"].str.split("|").apply(
        lambda genres: [_normalize_genre_name(genre) for genre in genres]
    )
    movies["genres"] = movies["genre_list"].apply(lambda genres: "|".join(genres))
    return movies


@lru_cache(maxsize=1)
def available_genres() -> List[str]:
    movies = load_movies()
    genres = sorted({genre for genres in movies["genre_list"] for genre in genres})
    return genres


def build_feature_frame(movies: pd.DataFrame) -> pd.DataFrame:
    genre_features = movies["genres"].str.get_dummies(sep="|")
    numeric_features = movies[["year", "imdb_rating", "popularity"]].copy()
    return pd.concat([genre_features, numeric_features], axis=1)
