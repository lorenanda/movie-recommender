"""
This is the code for the ML model
that makes recommendatins based on movie ratings.
"""
import numpy as np
import pandas as pd
import random
from nmf import ratings_pivot
from ml_models import movies
import joblib

get_most_rated = pd.DataFrame(ratings_pivot.isin(
    [0.0]).sum().sort_values())
all_most_rated = pd.merge(get_most_rated, movies, on='movieId')

old = all_most_rated[all_most_rated["year"] < 2010]["movieId"].head(25)
new = all_most_rated[all_most_rated["year"] > 2010]["movieId"].head(25)


def input_movies():
    old_choice = np.random.choice(old, 4, replace=False)
    new_choice = np.random.choice(new, 6, replace=False)
    m1 = all_most_rated[all_most_rated["movieId"].isin(old_choice)]
    m2 = all_most_rated[all_most_rated["movieId"].isin(new_choice)]
    most_rated = m2.append(m1, ignore_index=True)
    return most_rated[['title', 'movieId']]


most_rated = input_movies()
