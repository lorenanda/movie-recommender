"""
This is the code for the ML model
that makes recommendatins based on movie ratings.
"""
import numpy as np
import pandas as pd
import random
from svd_train_model import user_rating_matrix
from ml_models import movies
import joblib


user_rating_matrix["sum_rating"] = user_rating_matrix.groupby("movieId")[
    "rating"].transform("sum")
user_rating_matrix["std_rating"] = user_rating_matrix.groupby(
    "movieId")["rating"].transform("std")

sorted_matrix = user_rating_matrix.sort_values(
    by="sum_rating", ascending=False)

sorted_matrix = sorted_matrix.groupby(
    "movieId")["std_rating", "sum_rating"].mean()

complex_ratings = pd.merge(sorted_matrix, movies, on='movieId')


old = complex_ratings[complex_ratings["year"] <= 2010].head(30)
new = complex_ratings[complex_ratings["year"] > 2010].head(30)
old.sort_values(by="std_rating", ascending=False, inplace=True)
new.sort_values(by="std_rating", ascending=False, inplace=True)
old = old["movieId"].unique()
new = new["movieId"].unique()


def input_movies():

    old_choice = np.random.choice(old, 7, replace=False)
    new_choice = np.random.choice(new, 8, replace=False)
    m1 = all_most_rated[all_most_rated["movieId"].isin(old_choice)]
    m2 = all_most_rated[all_most_rated["movieId"].isin(new_choice)]
    most_rated = m2.append(m1, ignore_index=True)
    return most_rated[['title', 'movieId']]


most_rated = input_movies()
