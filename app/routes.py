from flask import Blueprint, request, jsonify
from app.recommendation import recommend_movies
import torch

recommender_bp = Blueprint("recommender", __name__)


@recommender_bp.route("/recommend/<int:user_id>", methods=["GET"])
def get_recommendations(user_id):
    model = torch.load("model.pth")

    top_movie_ids, movie_reviews = recommend_movies(user_id, model, num_movies=100)

    response_data = {
        "user_id": user_id,
        "recommendations": top_movie_ids.tolist(),
        "reviews": movie_reviews,
    }

    return jsonify(response_data)
