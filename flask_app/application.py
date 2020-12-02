from flask import Flask
from ml_models import nmf_recommand, get_recommendations
from flask import render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('main.html', title='Hello, World!')


@app.route('/recommender')
def recommender():
    recs = get_recommendations()
    return render_template('recommendations.html', movies=recs)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
