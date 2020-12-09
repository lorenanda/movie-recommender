"""Train the SVD model from scikit surpise. 
Calculate the imputed user_rating matrix
Use this in other recommendation systems
"""
import random
import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader, dump
from surprise import SVD
from surprise import accuracy
import pickle

user_rating_matrix = pd.read_csv(
    "./data/ratings.csv", parse_dates=True)


R = user_rating_matrix[["userId", "movieId", "rating"]]
user_rating = R.pivot(index="userId", columns="movieId", values="rating")


if '__name__' == '__main__':
    """train the SV model from scikit predict 
        and validate the model """
    # prepare the data for surpise
    reader = Reader(rating_scale=(0.5, 5))

    R_surpise = Dataset.load_from_df(
        R[['userId', 'movieId', 'rating']], reader)

    svd = SVD(random_state=43, n_factors=25, n_epochs=500)

    # train algo on the whole data
    R_surpise = R_surpise.build_full_trainset()
    svd.fit(R_surpise)

    # test and fill in the missing ratings
    testset = R_surpise.build_anti_testset()
    predictions = svd.test(testset)
    # score model
    print(accuracy.rmse(predictions))
    # 0.7426613175948654
    print(accuracy.mse(predictions))
    # 0.5515458326517415

    # Reconstruction of original matrix

    original = np.zeros((R_surpise.n_users, R_surpise.n_items))

    for (u, i, r) in R_surpise.all_ratings():
        original[u][i] = r

    known_entries = (original != 0)
    mean = R_surpise.global_mean
    bi = svd.bi.reshape(svd.bi.shape[0], 1)
    bu = svd.bu.reshape(svd.bu.shape[0], 1)
    qi = svd.qi
    pu = svd.pu
    reconstruct = mean + bu + bi.T + (pu).dot((qi).T)
    reconstruct = np.clip(reconstruct, 1, 5)
    reconstruct[known_entries] = original[known_entries]
    reconstructed_frame = pd.DataFrame(
        reconstruct, columns=user_rating.columns, index=user_rating.index)
    reconstructed_frame.to_pickle("R_hat.pkl")

    # save model
    # filename = 'svd_model.sav'

    # joblib.dump(svd, filename)
