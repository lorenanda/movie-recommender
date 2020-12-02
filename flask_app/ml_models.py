import random
import pandas as pd
from collections import defaultdict
from svd_train_model import user_rating
import joblib
from nmf import ratings_pivot
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# load the model from disk

svd = joblib.load("svd_model.sav")
nmf = joblib.load("nmf.sav")

movies_df = pd.read_csv('./data/movies.csv')
movies = movies_df['title']

# basline


def get_recommendations():
    random.shuffle(movies)
    return movies[:3]

# SVD


def predict_new_user_input(algo, user_input, orig_data, user_id=None):

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


def recommand_n(predictions, n=10, rating=False):

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
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


def nmf_recommand(model, new_user, n, orig_data):
    userid = random.randint(1, 610)
    new_user_input = pd.DataFrame(
        new_user, index=[userid], columns=orig_data.columns)
    new_user_input.fillna(3, inplace=True)
    P_new_user = model.transform(new_user_input)
    user_pred = pd.DataFrame(np.dot(P_new_user, model.components_), columns=ratings_pivot.columns,
                             index=[new_user_input.index.unique()[0]])
    user_pred.drop(columns=new_user.keys(), inplace=True)
    user_pred.sort_values(
        by=userid, ascending=False, axis=1, inplace=True)
    return user_pred.T[:n]

# collaborative filtering


def calculate_similarity_matrix(new_user_input, df):
    new_user = pd.DataFrame(new_user_input, index=[random.randint(1_000, 2_000)],
                            columns=user_rating.columns)
    new_user.fillna(user_rating.mean().mean(), inplace=True)
    df = df.append(new_user, ignore_index=True)
    sim_matrix = pd.DataFrame(cosine_similarity(df)).iloc[-1]
    sim_matrix.index = list(range(1, len(df)+1))
    sim_matrix.sort_values(ascending=False, inplace=True)
    similar_users = sim_matrix[1:6]

    return similar_users


def recomandations_similar_users(similar_users, orig_data):
    final_recomand = pd.DataFrame(
        columns=["userId", "movieId", "rating", "rating_sim", "sim"])
    for usr, sim in zip(similar_users.index, similar_users.values):
        user_input = user_rating.loc[usr].dropna().to_dict()
        pred = predict_new_user_input(
            algo=svd, user_input=user_input, orig_data=user_rating, user_id=usr)
        recomand = recommand_n(pred, 10, True)
        recomand["rating_sim"] = recomand["rating"] * sim
        recomand["sim"] = sim
        final_recomand = final_recomand.append(recomand, ignore_index=True)

    return final_recomand


def collaborative_filtering(final_recomand):

    final_recomand["rating_sim"] = final_recomand["rating_sim"].astype(float)
    recomand = final_recomand.groupby(
        "movieId")[["rating_sim", "sim"]].sum().reset_index()
    recomand["most_similar"] = recomand["rating_sim"] / recomand["sim"]

    return recomand[~recomand["movieId"].isin(list(new_user_input.keys()))].sort_values(
        by="most_similar", ascending=False)["movieId"][:5]


if __name__ == "__main__":
    new_user_input = {1: 3, 50: 5}
    pred = predict_new_user_input(
        algo=svd, user_input=new_user_input, orig_data=user_rating)
    print(recommand_n(pred, 5, True))
    print(nmf_recommand(model=nmf, new_user=new_user_input,
                        n=4, orig_data=ratings_pivot))
    sim_matrix = calculate_similarity_matrix(
        new_user_input, df=user_rating.fillna(user_rating.mean().mean()))
    rec_for_sim_users = recomandations_similar_users(sim_matrix, user_rating)
    print(collaborative_filtering(rec_for_sim_users))
