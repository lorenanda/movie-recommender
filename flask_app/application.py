from flask import Flask, render_template, request
import joblib
import pandas as pd

from ml_models import svd, nmf, nmf_recommand, ratings_pivot, user_rating, calculate_similarity_matrix, recomandations_similar_users, collaborative_filtering
from user_input_promt import input_movies, movies_df


app = Flask(__name__)


@app.route('/')
def index():

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
    top10 = input_movies()
    top10 = pd.DataFrame(top10.reset_index())
    top10["label"] = 'movie' + top10["index"].astype(str)

    user_input = dict(request.args)
    input_frame = pd.DataFrame(columns=["label", "rating"])
    for label, rating in user_input.items():
        label1 = label
        rating1 = [r for r in rating]
        print(label1, rating1)
        agg = {'label': label1, 'rating': rating1[0]}
        input_frame = input_frame.append(
            agg, ignore_index=True)

    input_frame = input_frame.merge(top10, on="label")[["movieId", "rating"]]
    input_frame["rating"] = input_frame["rating"].astype(int)
    input_frame = input_frame[input_frame["rating"] > 1]  # to change to zero
    len_ratings = len(input_frame)
    input_frame.set_index("movieId", inplace=True)

    input_frame = input_frame.to_dict()["rating"]

    if len_ratings > 7:
        rec = nmf_recommand(model=nmf, new_user=input_frame,
                            n=5, orig_data=ratings_pivot)

        rec = pd.merge(rec, movies_df, on='movieId')

    else:
        sim_matrix = calculate_similarity_matrix(
            input_frame, df=user_rating.fillna(user_rating.mean().mean()), n_users=5)
        rec_for_sim_users = recomandations_similar_users(
            sim_matrix, user_rating)

        rec = collaborative_filtering(
            rec_for_sim_users, 5, new_user_input=input_frame)
        rec = pd.merge(rec, movies_df, on='movieId')

    recs = rec["title"]
    return render_template('recommendations.html', movies=recs)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
