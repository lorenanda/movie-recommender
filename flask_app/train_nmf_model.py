"""
Train the NMF model from sklearn.
Save the model and use it in the Flask application.

"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import NMF
import joblib
import random

movies = pd.read_csv('./data/movies.csv')
ratings = pd.read_csv('./data/ratings.csv')
ratings_pivot = ratings.pivot(
    index='userId', columns='movieId', values='rating')
ratings_pivot.replace(np.nan, 0, inplace=True)


if __name__ == "__main__":

    model = NMF(
        n_components=20,
        init='random',
        random_state=10,
        max_iter=1000
    )
    model.fit(ratings_pivot)
    model.reconstruction_err_
    # save model
    joblib.dump(model, "nmf.sav")

    P = model.transform(ratings_pivot)
    Q = model.components_.T

    # recreate matrix
    ratings_pred = Q.dot(P.T)

    new_user = {1: 3, 50: 4}
    new_user = pd.DataFrame(
        new_user, index=[random.randint(1, 610)], columns=ratings_pivot.columns)
    new_user.fillna(3, inplace=True)

    P_new_user = model.transform(new_user)

    user_pred = pd.DataFrame(np.dot(P_new_user, Q.T), columns=ratings_pivot.columns,
                             index=[new_user.index.unique()])

    print(user_pred)
