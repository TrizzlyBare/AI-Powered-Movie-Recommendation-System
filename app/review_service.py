import requests
import json

TMDB_API_KEY = "8ca02aa7e60b4c6ed8fa3ec582f847d5"


def get_movie_reviews(movie_id):
    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={TMDB_API_KEY}"
    )
    response = requests.get(url)
    data = response.json()

    # Parse reviews
    reviews = []
    for review in data.get("results", []):
        reviews.append(
            {
                "author": review["author"],
                "content": review["content"],
                "rating": review.get("rating", "No rating available"),
            }
        )
    return reviews
