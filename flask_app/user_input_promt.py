"""
This is a helper function that renders the most popular movie and then
displayes to the user those movies with the highest standard deviation of the rating.
It solves the cold start problem
"""
import random
import numpy as np
import pandas as pd

from train_svd_model import user_rating_matrix
from ml_models import movies


# data preparation

user_rating_matrix["sum_rating"] = user_rating_matrix.groupby("movieId")[
    "rating"].transform("sum")
user_rating_matrix["std_rating"] = user_rating_matrix.groupby(
    "movieId")["rating"].transform("std")

# removes duplicated crated by the transform function
sorted_matrix = user_rating_matrix.groupby(
    "movieId")["std_rating", "sum_rating"].mean().sort_values(by='sum_rating', ascending=False)


# merges data frame with movies data frame containg titles
movies_with_title = pd.merge(sorted_matrix, movies, on='movieId')

# catalogs movies into old and new category, then selects those with highest standard dev
old = movies_with_title[movies_with_title["year"] <= 2010].head(25)
new = movies_with_title[movies_with_title["year"] > 2010].head(25)
old.sort_values(by="std_rating", ascending=False, inplace=True)
new.sort_values(by="std_rating", ascending=False, inplace=True)
old = old["movieId"].unique()
new = new["movieId"].unique()


def input_movies(old=old, new=new, movies_with_title=movies_with_title):
    """
    Returns a data frame with 15 movie titles and their movieId link
    ##Parameters##:
    old - MovieId of old movies 
    new- MovieIds of new movies
    movies_with_tile - data frame containg user ratings, movieId and title of the movies

    ##Returns##:
    Data frame of 15 movies containg MovieId and title

    """

    old_choice = np.random.choice(old, 7, replace=False)
    new_choice = np.random.choice(new, 8, replace=False)
    m1 = movies_with_title[movies_with_title["movieId"].isin(old_choice)]
    m2 = movies_with_title[movies_with_title["movieId"].isin(new_choice)]
    most_rated = m2.append(m1, ignore_index=True)
    return most_rated[['title', 'movieId']]


most_rated = input_movies(
    old=old, new=new, movies_with_title=movies_with_title)
