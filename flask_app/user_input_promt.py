"""
This is the code for the ML model
that makes recommendatins based on movie ratings.
"""

import pandas as pd
import random
from nmf import ratings_pivot
import joblib

movies_df = pd.read_csv('./data/movies.csv')
movies = movies_df['title']

most_rated = pd.DataFrame(ratings_pivot.isin(
    [0.0]).sum().sort_values().head(10))
most_rated = pd.merge(most_rated, movies_df, on='movieId')


def input_movies():
    return most_rated[['title', 'movieId']]
