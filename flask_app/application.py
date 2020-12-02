from flask import Flask 
from recommender import input_movies
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
    #return render_template('main.html', title='Movie Recommender')
    recs = input_movies()
    return render_template('main.html', movies=recs)
  
@app.route('/recommender')
def recommender():
    recs = input_movies()
    return render_template('recommendations.html', movies=recs)