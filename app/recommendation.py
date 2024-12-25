import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from app.review_service import get_movie_reviews


class MovieRecommender(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super(MovieRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, user, movie):
        user_embedded = self.user_embedding(user)
        movie_embedded = self.movie_embedding(movie)
        x = user_embedded * movie_embedded
        return self.fc(x)


def train_model(ratings, num_users, num_movies):
    model = MovieRecommender(num_users, num_movies)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        optimizer.zero_grad()
        predictions = model(ratings["userId"], ratings["movieId"])
        loss = criterion(predictions, ratings["rating"])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    return model


def recommend_movies(user_id, model, num_movies):
    model.eval()
    movie_ids = torch.arange(0, num_movies)
    user_ids = torch.full_like(movie_ids, user_id)

    with torch.no_grad():
        predictions = model(user_ids, movie_ids).squeeze().numpy()

    top_movie_ids = predictions.argsort()[-5:][::-1]
    movie_reviews = {}

    for movie_id in top_movie_ids:
        reviews = get_movie_reviews(movie_id)
        movie_reviews[movie_id] = reviews

    return top_movie_ids, movie_reviews
