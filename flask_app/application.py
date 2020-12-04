from flask import Flask, render_template, request
import joblib
import pandas as pd
import tmdbv3api
from tmdbv3api import TMDb, Movie
import config

from ml_models import svd, nmf, nmf_recommand, ratings_pivot, user_rating, calculate_similarity_matrix, recomandations_similar_users, collaborative_filtering
from user_input_promt import input_movies, movies_df
from get_TMDB_info import TMDBInfo

tmdb = TMDb()
tmdb.api_key = config.API_KEY
link = pd.read_csv("./data/links.csv")

app = Flask(__name__)


@app.route('/')
def index():
    """Display the most rated movies to the user
       and promt user to rate them: solves cold start problem 
    """
    top10 = input_movies()

    return render_template(
        'main.html',
        title="Movie Recommender",
        movie0=top10['title'][0],
        movie1=top10['title'][1],
        movie2=top10['title'][2],
        movie3=top10['title'][3],
        movie4=top10['title'][4],
        movie5=top10['title'][5],
        movie6=top10['title'][6],
        movie7=top10['title'][7],
        movie8=top10['title'][8],
        movie9=top10['title'][9]
    )


@app.route('/recommender')
def recommender():
    """Intercept user input and make recomandations based on his 
        initial rating
    """
    top10 = input_movies()
    top10 = pd.DataFrame(top10.reset_index())
    top10["label"] = 'movie' + top10["index"].astype(str)

    # intercept user input
    user_input = dict(request.args)
    input_frame = pd.DataFrame(columns=["label", "rating"])
    for label, rating in user_input.items():
        label1 = label
        rating1 = [r for r in rating]
        agg = {'label': label1, 'rating': rating1[0]}
        input_frame = input_frame.append(
            agg, ignore_index=True)

    input_frame = input_frame.merge(top10, on="label")[["movieId", "rating"]]
    input_frame["rating"] = input_frame["rating"].astype(int)

    # select only the rated movies and calucte how many ratings the user has inputed
    # return dictionary of the form {movieId: rating}
    input_frame = input_frame[input_frame["rating"] > 0]
    len_ratings = len(input_frame)
    input_frame.set_index("movieId", inplace=True)

    input_frame = input_frame.to_dict()["rating"]

    # make recomandations to the user
    if len_ratings > 7:
        rec = nmf_recommand(model=nmf, new_user=input_frame,
                            n=5, orig_data=ratings_pivot)

    else:
        sim_matrix = calculate_similarity_matrix(
            input_frame, df=user_rating.fillna(user_rating.mean().mean()), n_users=5)
        rec_for_sim_users = recomandations_similar_users(
            sim_matrix, user_rating)

        rec = collaborative_filtering(
            rec_for_sim_users, 5, new_user_input=input_frame)

    # display only the titles
    rec = pd.merge(rec, movies_df, on='movieId')
    recs = rec["title"]

    # get information from TMDB
    rec_link = pd.merge(rec, link, on='movieId')
    rec_link["tmdbId"] = rec_link["tmdbId"].astype(int)

    movie_info = pd.DataFrame(columns=["title", "overview", "image_url", "popularity",
                                       "release_date", "video_url"])
    for i in rec_link["tmdbId"]:
        t = TMDBInfo(movieId=i, api_key=tmdb.api_key, tmdb=TMDb())
        overview, image_url, title, popularity, release_date = t.get_details()
        print(overview, image_url, title, popularity, release_date)
        t.get_movie_trailer()
        video_url = t.get_video_url()
        print(video_url)
        args = {"title": title, "overview": overview, "image_url": image_url, "popularity": popularity,
                "release_date": release_date, "video_url": video_url}
        movie_info = movie_info.append(args, ignore_index=True)

    movie_info.set_index("title", inplace=True)
    return render_template('recommendations.html', movies=recs)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
