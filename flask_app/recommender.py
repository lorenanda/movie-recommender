"""
This is the code for the ML model
that makes recommendatins based on movie ratings.
"""

import pandas as pd
import random
from nmf import ratings_pivot
import joblib

model=joblib.load("nmf.sav")

movies_df = pd.read_csv('./data/movies.csv')
movies = movies_df['title']

def get_recommendations():
    random.shuffle(movies)
    return movies[:3]

def nmf_recommender(model, orig_data):
    pass