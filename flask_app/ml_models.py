"""
Comprises all ML_models for recommendation systems:
- SVD from surpise
- NMF from sklearn
- combination of collaborative filtering with SVD
"""

import random
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine
import joblib
import pickle
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def split_data(thrsh, movies):
    """
    Splits data in old and new movies based on a threshold
    ####Parameters###:
        - thrsh: year 
        - movies: data frame containing movie Ids, year and title (title optional)
    ####Returns####:
        - 2 lists of Movie Ids: one with movie above the threshold, 
            one with movies below the threshold
    """
    cols_above = [iid for iid in movies[movies["year"] >= thrsh]["movieId"]]
    cols_below = [iid for iid in movies[movies["year"] < thrsh]["movieId"]]
    cols_none = [iid for iid in movies[movies["year"] == 0]["movieId"]]

    return cols_above, cols_below, cols_none


# baseline

def get_recommendations(movies):
    random.shuffle(movies)
    return movies[:3]

# SVD


def predict_new_user_input(algo, user_input, orig_data, user_id=None):
    """We  need to predict user ratings based on SVD in order to make recommendations:
     # Parameters#####:
       - algo = SVD algo (other surpise algos)
       - user_input = users initial ratings
       - orig_data = original training data, user-rating matrix (NANs are allowed)
       - user_id = default False, if True a specific user id is provided
                 True to be used only when predicting based on collaborative filtering
      ######Returns#####
      - prediction dictionary for all movie in the training set
    """
    if not user_id:
        new_user = pd.DataFrame(user_input, index=[random.randint(
            1, 610)], columns=orig_data.columns)
    else:
        new_user = pd.DataFrame(
            user_input, index=[user_id], columns=orig_data.columns)

    user_input = pd.DataFrame(new_user.unstack().reset_index())
    user_input.columns = ["movieId", "userId", "rating"]

    pred = []
    for i in range(len(user_input)):

        pred1 = algo.predict(
            uid=user_input["userId"].iloc[i], iid=user_input["movieId"].iloc[i],
            r_ui=user_input["rating"].iloc[i])

        pred.append(pred1)
    return pred


def recommand_n(predictions, n=10, rating=False, uid=0):
    """Recommends n best movies based on SVD algo from surpise
        # Parameters###:

           -predictions = given by function predict_new_user_input (dictionary)
           -n = number of items to recommend (int)
           -rating = default False, if True also the ratings of the movie will be outputted

        # Returns#####:
            -data frame containing user Id, n recommended movies and the ratings (if rating is True)

    """
    # First map the predictions to each user.
    top_n = defaultdict(list)
    if uid == 0:
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))
    else:
        for iid, est in predictions.items():
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    recommendation = []
    cols = ["userId", "movieId", "rating"]
    recommendations = pd.DataFrame(columns=cols)
    for uid, user_ratings in top_n.items():
        recommendation = [iid for (iid, _) in user_ratings]
        rating = [est for (_, est) in user_ratings]
        for rec, ratings in zip(recommendation, rating):
            agg = {'userId': uid, 'movieId': rec, 'rating': ratings}
            recommendations = recommendations.append(
                agg, ignore_index=True)

    if rating:
        recommendations = recommendations
    else:
        recommendations = recommendations[["movieId"]]

    return recommendations

# NMF


def nmf_recommand(model, new_user, n, orig_data, cols_above, cols_below, selection=3):
    """
    Recommender system based on NMF.
    # Parameters#####:
        - model = NMF ML_models
        - new_user = user input (dict in from movie Id and ratings)
        - n = number of recommendations (int)
        - orig_data = original data (must not contain NANs)
        - selection = vals: 1-3 (1-new movies, 2-old movies , 3-indifferent)
        - cols_above, cols_below = input from function split_data

    # Return#####:
        - data frame containing movie Id recommendations 

    """
    userid = random.randint(1, 610)
    new_user_input = pd.DataFrame(
        new_user, index=[userid], columns=orig_data.columns)
    new_user_input.fillna(3, inplace=True)
    P_new_user = model.transform(new_user_input)
    user_pred = pd.DataFrame(np.dot(P_new_user, model.components_), columns=orig_data.columns,
                             index=[new_user_input.index.unique()[0]])

    list1 = list(new_user.keys())
    if selection == 1:
        cols_above = set(cols_above).intersection(set(orig_data.columns))
        list_drop = list(set(list1) | set(cols_above))
    elif selection == 2:
        cols_below = set(cols_below).intersection(set(orig_data.columns))
        list_drop = list(set(list1) | set(cols_below))
    else:
        list_drop = list1

    user_pred.drop(columns=list_drop, inplace=True)

    user_pred.sort_values(
        by=userid, ascending=False, axis=1, inplace=True)

    return pd.DataFrame(user_pred.T[:n].reset_index(), columns=["movieId"])

# collaborative filtering


def calculate_similarity_matrix(new_user_input, orig_data, n_users=5):
    """Calculates similarity matrix using cosine similarity.
        This is a user similarity matrix.
        # Parameters####:
            - new_user = new user profile (dict)
            - orig_data = data frame containing other users and ratings (must not contain NANS)
            - n_users = how many similar users should be picked (int)

        # Returns#####:
            - data frame containing similar users (ids) and their similarity index

    """
    new_user = pd.DataFrame(new_user_input, index=[random.randint(1_000, 2_000)],
                            columns=orig_data.columns)
    new_user.fillna(orig_data.mean().mean(), inplace=True)
    orig_data = orig_data.append(new_user, ignore_index=True)
    sim_matrix = pd.DataFrame(cosine_similarity(orig_data)).iloc[-1]
    sim_matrix.index = list(range(1, len(orig_data)+1))
    sim_matrix.sort_values(ascending=False, inplace=True)
    similar_users = sim_matrix[1:(n_users+1)]

    return similar_users


def recomandations_similar_users(similar_users, orig_data, cols_above, cols_below, selection=3):
    """Makes recommendations for similar users based on SVD algo.
    # Parameters####:
        - similar_users = data frame containing user Id and similarity index
        - orig_data = reconstructed data using SVD() (original ratings remain in place)
    # Returns###:
        - data frame containing movieId and their relative importance to the new user
    """
    final_recomand = pd.DataFrame(
        columns=["userId", "movieId", "rating", "rating_sim", "sim"])
    for usr, sim in zip(similar_users.index, similar_users.values):
        pred = orig_data.loc[usr]
        pred = pd.DataFrame(pred).T
        if selection == 1:
            list_drop = set(cols_above).intersection(set(orig_data.columns))
            pred.drop(columns=list_drop, inplace=True)
        elif selection == 2:
            list_drop = set(cols_below).intersection(set(orig_data.columns))
            pred.drop(columns=list_drop, inplace=True)
        pred = pred.T
        pred = pred.squeeze()
        recomand = recommand_n(pred.to_dict(), 10, True, uid=usr)
        recomand["rating_sim"] = recomand["rating"] * sim
        recomand["sim"] = sim
        final_recomand = final_recomand.append(recomand, ignore_index=True)

    return final_recomand


def collaborative_filtering(final_recomand, n, new_user_input):
    """Selects the most relevant movies for a new user, based on their similarity with other users.
    # Parameters###:
        - final_recomand = output of function recomandations_similar_users
        - n = number of movies to be recommended
    # Returns###:
        - pandas data frame containing top movie recommendation (iids) for new user
    """

    final_recomand["rating_sim"] = final_recomand["rating_sim"].astype(float)
    recomand_sum = final_recomand.groupby(
        "movieId")[["rating_sim", "sim"]].sum().reset_index()
    recomand_sum["most_similar"] = recomand_sum["rating_sim"] / \
        recomand_sum["sim"]

    user_recomand = recomand_sum[~recomand_sum["movieId"].isin(list(new_user_input.keys()))].sort_values(
        by="most_similar", ascending=False)["movieId"][:n]
    user_recomand = pd.DataFrame(
        pd.Series(user_recomand, name="movieId"), columns=["movieId"])

    user_recomand["movieId"] = user_recomand["movieId"].astype(int)

    return user_recomand
