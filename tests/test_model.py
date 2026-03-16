from recommender.model import MovieRecommender


def test_recommender_returns_requested_number_of_movies():
    recommender = MovieRecommender()
    results = recommender.recommend(
        preferred_genres=["Action", "Sci-Fi"],
        avoided_genres=["Horror"],
        minimum_rating=7.0,
        top_n=5,
    )

    assert len(results.recommendations) == 5
    assert all("Horror" not in movie["genres"] for movie in results.recommendations)
    assert all(movie["recommendation_score"] < 100 for movie in results.recommendations)


def test_recommender_falls_back_without_explicit_preferences():
    recommender = MovieRecommender()
    results = recommender.recommend(
        preferred_genres=[],
        avoided_genres=[],
        minimum_rating=7.5,
        top_n=3,
    )

    assert len(results.recommendations) == 3
    assert results.metrics["positive_examples"] > 0
    assert "validation_accuracy" in results.metrics
