from flask import Flask
from app.routes import recommender_bp

app = Flask(__name__)
app.register_blueprint(recommender_bp)

if __name__ == "__main__":
    app.run(debug=True)
