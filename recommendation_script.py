import torch
from app.recommendation import recommend_movies, MovieRecommender
import pandas as pd


def load_model(filename="model.pth"):
    model = MovieRecommender(num_users=1000, num_movies=1000)
    model.load_state_dict(torch.load(filename))
    return model


def get_movie_titles():
    movies = pd.read_csv("data/movies.csv")
    return {row["movieId"]: row["title"] for _, row in movies.iterrows()}


def main(user_id):
    model = load_model("model.pth")
    movie_titles = get_movie_titles()
    top_movie_ids, movie_reviews = recommend_movies(user_id, model, num_movies=100)

    print(f"Recommendations for user {user_id}:")
    for movie_id in top_movie_ids:
        print(f"Movie: {movie_titles[movie_id]}")
        print("Reviews:")
        for review in movie_reviews[movie_id]:
            print(f"  Author: {review['author']}")
            print(f"  Content: {review['content']}")
            print(f"  Rating: {review['rating']}")
            print()


if __name__ == "__main__":
    user_id = int(input("Enter user ID: "))
    main(user_id)
