import torch
from app.recommendation import MovieRecommender


def save_model(model, filename="model.pth"):
    torch.save(model.state_dict(), filename)


def load_model(filename="model.pth"):
    model = MovieRecommender(num_users=1000, num_movies=1000)
    model.load_state_dict(torch.load(filename))
    return model
