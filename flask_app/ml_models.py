import random
import pandas as pd
from collections import defaultdict
from svd_train_model import user_rating
import joblib
from nmf import ratings_pivot
import numpy as np

# load the model from disk

svd = joblib.load("flask_app/svd_model.sav")
nmf = joblib.load("nmf.sav")


def predict_new_user_input(algo, user_input, orig_data):

    new_user = pd.DataFrame(user_input, index=[random.randint(
        1, 610)], columns=orig_data.columns)

    user_input = pd.DataFrame(new_user.unstack().reset_index())
    user_input.columns = ["movieId", "userId", "rating"]

    pred = []
    for i in range(len(user_input)):

        pred1 = algo.predict(
            uid=user_input["userId"].iloc[i], iid=user_input["movieId"].iloc[i],
            r_ui=user_input["rating"].iloc[i])

        pred.append(pred1)
    return pred


def recommand_n(predictions, n=10):

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    recommendation = []
    cols = ["userId", "movieId"]
    recommendations = pd.DataFrame(columns=cols)
    for uid, user_ratings in top_n.items():
        recommendation = [iid for (iid, _) in user_ratings]
        for rec in recommendation:
            agg = {'userId': uid, 'movieId': rec}
            recommendations = recommendations.append(
                agg, ignore_index=True)

    return recommendations


def nmf_recommand(model, new_user, n):
    userid = random.randint(1, 610)
    new_user_input = pd.DataFrame(
        new_user, index=[userid], columns=ratings_pivot.columns)
    new_user_input.fillna(3, inplace=True)
    P_new_user = model.transform(new_user_input)
    user_pred = pd.DataFrame(np.dot(P_new_user, model.components_), columns=ratings_pivot.columns,
                             index=[new_user_input.index.unique()])
    user_pred.drop(columns=new_user.keys(), inplace=True)
    user_pred.T.sort_values(
        by=userid, ascending=False, axis=1, inplace=True)
    return user_pred.T[:n]


if __name__ == "__main__":
    new_user_input = {1: 3, 50: 5}
    pred = predict_new_user_input(
        algo=svd, user_input=new_user_input, orig_data=user_rating)
    print(recommand_n(pred, 10))
    print(nmf_recommand(model=nmf, new_user=new_user_input, n=5))