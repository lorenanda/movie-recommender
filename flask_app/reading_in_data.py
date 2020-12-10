import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine

import flask_app.config2 as config2

# establishing SQL connection
engine = create_engine(
    f"postgres://postgres:{config2.psql_pw}@127.0.0.1:5432/movies")

# read in movie rating data
select_query = """
    SELECT * FROM movie_ratings;
    """
engine_output = engine.execute(select_query)

final_dict = []
for row in engine_output:
    dict1 = {"userId": int((row["userid"])), "movieId": int((row["movieid"])),
             "rating": float((row["rating"]))}
    final_dict.append(dict1)

user_rating_matrix = pd.DataFrame(final_dict)


# pivot data
user_rating = user_rating_matrix.pivot(
    index="userId", columns="movieId", values="rating")

# set the NANs to zero
ratings_pivot = user_rating.replace(np.nan, 0)

# read in data for movies and title
select_movies = """
    SELECT * FROM movies;
    """
engine_output_movies = engine.execute(select_movies)

movies_final = []
for row in engine_output_movies:
    dict1 = {"movieId": int((row["movieid"])),
             "title": str((row["title"]))}
    movies_final.append(dict1)

movies_df = pd.DataFrame(movies_final)

movies_df['year'] = 0
for i, data in movies_df.iterrows():
    try:
        movies_df.loc[i, 'year'] = int(
            data['title'].split('(')[-1].replace(')', ''))
    except:
        movies_df.loc[i, 'year'] = 0

movies = movies_df[['title', 'year', 'movieId']]

# read in the link data for IMDB and TMDB
select_link = """
    SELECT * FROM link;
    """
engine_output_link = engine.execute(select_link)

movies_link = []
for row in engine_output_link:
    dict1 = {"movieId": int((row["movieid"])),
             "tmdbId": str((row["tmdbid"]))}
    movies_link.append(dict1)

link = pd.DataFrame(movies_link)
