from flask import render_template, request
from recommender import input_movies
import joblib
from nmf import ratings_pivot
from ml_models import nmf_recommand, get_recommendations
from flask import Flask


svd = joblib.load("svd_model.sav")
nmf = joblib.load("nmf.sav")

app = Flask(__name__)


@app.route('/')
def index():
    # return render_template('main.html', title='Movie Recommender')
    recs = input_movies()
    return render_template('main.html', movies=recs)


@app.route('/recommender')
def recommender():
    user_input = dict(request.args)
    print(user_input)
    recs = input_movies()
    return render_template('recommendations.html', movies=recs)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
