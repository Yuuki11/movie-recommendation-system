from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from recommender.data_loader import build_feature_frame, load_movies


@dataclass
class RecommendationBundle:
    recommendations: List[Dict[str, object]]
    metrics: Dict[str, object]
    summary: str


class MovieRecommender:
    def __init__(self):
        self.movies = load_movies().copy()
        self.feature_frame = build_feature_frame(self.movies)

    def recommend(
        self,
        preferred_genres,
        avoided_genres,
        minimum_rating,
        top_n=5,
    ):
        preferred = set(preferred_genres)
        avoided = set(avoided_genres)

        labels, liked_mask, avoided_mask = self._build_labels(
            preferred=preferred,
            avoided=avoided,
            minimum_rating=minimum_rating,
        )
        model, training_accuracy, test_accuracy, train_size, test_size = self._fit_model(labels)
        probabilities = model.predict_proba(self.feature_frame)[:, 1]

        ranked_movies = self.movies.copy()
        ranked_movies["match_probability"] = probabilities
        ranked_movies["liked_match_count"] = ranked_movies["genre_list"].apply(
            lambda genres: len(preferred.intersection(genres))
        )
        ranked_movies["avoided_match_count"] = ranked_movies["genre_list"].apply(
            lambda genres: len(avoided.intersection(genres))
        )

        candidate_mask = (~avoided_mask) & (
            ranked_movies["imdb_rating"] >= minimum_rating
        )
        if preferred:
            candidate_mask = candidate_mask & ranked_movies["liked_match_count"].gt(0)

        candidates = ranked_movies.loc[candidate_mask].copy()
        candidates = candidates.sort_values(
            by=[
                "match_probability",
                "liked_match_count",
                "imdb_rating",
                "popularity",
            ],
            ascending=[False, False, False, False],
        )

        recommendations = [
            {
                "title": row.title,
                "genres": row.genres,
                "genre_display": ", ".join(row.genre_list),
                "year": int(row.year),
                "imdb_rating": round(float(row.imdb_rating), 1),
                "popularity": int(row.popularity),
                "match_probability": round(float(row.match_probability) * 100, 1),
                "reason": self._build_reason(
                    genre_list=row.genre_list,
                    liked_matches=row.liked_match_count,
                    preferred=preferred,
                    minimum_rating=minimum_rating,
                    imdb_rating=float(row.imdb_rating),
                ),
            }
            for row in candidates.head(top_n).itertuples()
        ]

        metrics = {
            "training_accuracy": round(float(training_accuracy) * 100, 1),
            "test_accuracy": round(float(test_accuracy) * 100, 1),
            "train_size": int(train_size),
            "test_size": int(test_size),
            "candidate_pool": int(len(candidates)),
            "positive_examples": int(labels.sum()),
            "negative_examples": int((1 - labels).sum()),
        }
        summary = self._build_summary(preferred, avoided, minimum_rating)
        return RecommendationBundle(
            recommendations=recommendations,
            metrics=metrics,
            summary=summary,
        )

    def _build_labels(
        self,
        preferred,
        avoided,
        minimum_rating,
    ):
        liked_mask = self.movies["genre_list"].apply(
            lambda genres: bool(preferred.intersection(genres))
        )
        avoided_mask = self.movies["genre_list"].apply(
            lambda genres: bool(avoided.intersection(genres))
        )
        high_quality_mask = self.movies["imdb_rating"] >= minimum_rating
        high_popularity_mask = self.movies["popularity"] >= self.movies["popularity"].median()

        preference_score = (
            liked_mask.astype(int) * 3
            - avoided_mask.astype(int) * 4
            + high_quality_mask.astype(int)
            + high_popularity_mask.astype(int)
        )

        labels = (preference_score >= 3).astype(int)
        if labels.nunique() == 1:
            labels = (self.movies["imdb_rating"] >= self.movies["imdb_rating"].median()).astype(int)

        if labels.sum() < 3:
            fallback_positive = self.movies["imdb_rating"].rank(method="dense", ascending=False) <= 6
            labels = fallback_positive.astype(int)

        if (1 - labels).sum() < 3:
            fallback_positive = self.movies["popularity"].rank(method="dense", ascending=False) <= 6
            labels = fallback_positive.astype(int)

        return labels, liked_mask, avoided_mask

    def _fit_model(self, labels):
        pipeline = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.2, max_iter=1000)),
            ]
        )
        stratify_labels = labels if labels.nunique() > 1 else None

        if labels.value_counts().min() >= 2:
            train_features, test_features, train_labels, test_labels = train_test_split(
                self.feature_frame,
                labels,
                test_size=0.2,
                random_state=42,
                stratify=stratify_labels,
            )
            pipeline.fit(train_features, train_labels)
            training_accuracy = pipeline.score(train_features, train_labels)
            test_accuracy = pipeline.score(test_features, test_labels)
            return (
                pipeline,
                training_accuracy,
                test_accuracy,
                len(train_features),
                len(test_features),
            )

        pipeline.fit(self.feature_frame, labels)
        fallback_accuracy = pipeline.score(self.feature_frame, labels)
        return pipeline, fallback_accuracy, fallback_accuracy, len(self.feature_frame), 0

    @staticmethod
    def _build_reason(
        genre_list,
        liked_matches,
        preferred,
        minimum_rating,
        imdb_rating,
    ):
        matching_genres = sorted(preferred.intersection(genre_list))
        if matching_genres:
            genre_text = ", ".join(matching_genres[:2])
            return f"Matches your interest in {genre_text}."
        if imdb_rating >= minimum_rating:
            return "Strong overall fit based on quality and popularity."
        if liked_matches > 0:
            return "Contains genres you leaned toward."
        return "Recommended as a nearby fit to your profile."

    @staticmethod
    def _build_summary(
        preferred,
        avoided,
        minimum_rating,
    ):
        if preferred:
            focus = ", ".join(sorted(preferred))
        else:
            focus = "high-quality popular movies"

        if avoided:
            exclusions = ", ".join(sorted(avoided))
            return (
                f"Ranked movies around {focus}, filtered away from {exclusions}, "
                f"with a target IMDb rating of {minimum_rating:.1f}+."
            )

        return f"Ranked movies around {focus} with a target IMDb rating of {minimum_rating:.1f}+."
