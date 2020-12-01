import random
import pandas as pd
from collections import defaultdict
from nmf_train_model import user_rating
import joblib

# load the model from disk
svd = joblib.load("svd_model.sav")


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

    recomendacion = []
    columnas = ["userId", "movieId"]
    recomendaciones = pd.DataFrame(columns=columnas)
    for uid, user_ratings in top_n.items():
        recomendacion = [iid for (iid, _) in user_ratings]
        for rec in recomendacion:
            agregar = {'userId': uid, 'movieId': rec}
            recomendaciones = recomendaciones.append(
                agregar, ignore_index=True)

    return recomendaciones


if __name__ == "__main__":
    new_user_input = {20: 3, 50: 5}
    pred = predict_new_user_input(
        algo=svd, user_input=new_user_input, orig_data=user_rating)
    print(recommand_n(pred, 10))
