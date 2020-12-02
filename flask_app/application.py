from flask import Flask
from ml_models import nmf_recommand, get_recommendations, user_rating
from flask import render_template
from nmf import ratings_pivot
import joblib


svd = joblib.load("svd_model.sav")
nmf = joblib.load("nmf.sav")

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('main.html', title='Hello, World!')


@app.route('/recommender')
def recommender():

    return render_template('recommendations.html', movies=recs)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
