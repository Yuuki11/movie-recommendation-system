from flask import Flask, redirect, render_template, request, url_for
from typing import List, Optional

from recommender.data_loader import available_genres, dataset_source
from recommender.model import MovieRecommender


app = Flask(__name__)
recommender = MovieRecommender()


def _parse_genre_field(raw_value: Optional[str]) -> List[str]:
    if not raw_value:
        return []
    cleaned = [item.strip() for item in raw_value.split(",")]
    return [item for item in cleaned if item]


def _build_selections(values):
    preferred_genres = _parse_genre_field(values.get("preferred_genres"))
    avoided_genres = _parse_genre_field(values.get("avoided_genres"))

    try:
        minimum_rating = float(values.get("minimum_rating", 7.0))
    except (TypeError, ValueError):
        minimum_rating = 7.0

    try:
        top_n = int(values.get("top_n", 5))
    except (TypeError, ValueError):
        top_n = 5

    return {
        "preferred_genres": preferred_genres,
        "avoided_genres": avoided_genres,
        "minimum_rating": minimum_rating,
        "top_n": max(3, min(top_n, 10)),
    }


@app.route("/", methods=["GET"])
def index():
    selections = _build_selections(request.args)
    has_filters = any(
        [
            selections["preferred_genres"],
            selections["avoided_genres"],
            request.args.get("minimum_rating"),
            request.args.get("top_n"),
        ]
    )
    results = None
    if has_filters:
        results = recommender.recommend(
            preferred_genres=selections["preferred_genres"],
            avoided_genres=selections["avoided_genres"],
            minimum_rating=selections["minimum_rating"],
            top_n=selections["top_n"],
        )

    return render_template(
        "index.html",
        dataset_source=dataset_source(),
        genres=available_genres(),
        results=results,
        selections=selections,
    )


@app.route("/recommend", methods=["POST"])
def recommend():
    selections = _build_selections(request.form)
    return redirect(
        url_for(
            "index",
            preferred_genres=",".join(selections["preferred_genres"]),
            avoided_genres=",".join(selections["avoided_genres"]),
            minimum_rating="{0:.1f}".format(selections["minimum_rating"]),
            top_n=selections["top_n"],
        )
    )


if __name__ == "__main__":
    app.run(debug=True)
