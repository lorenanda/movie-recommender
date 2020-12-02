from flask import Flask 
from recommender import get_recommendations
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
    #return render_template('main.html', title='Movie Recommender')
    recs = get_recommendations()
    return render_template('main.html', movies=recs)


@app.route('/recommender')
def recommender():
    recs = get_recommendations()
    return render_template('recommendations.html', movies=recs)