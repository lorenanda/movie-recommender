import random
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split, cross_validate
from surprise.model_selection import KFold
import joblib

user_rating_matrix = pd.read_csv(
    "./data/ratings.csv", parse_dates=True)


R = user_rating_matrix[["userId", "movieId", "rating"]]
user_rating = R.pivot(index="userId", columns="movieId", values="rating")


if '__name__' == '__main__':
    """validatig the model """
    # prepare the data for surpise
    reader = Reader(rating_scale=(0.5, 5))

    R_surpise = Dataset.load_from_df(
        R[['userId', 'movieId', 'rating']], reader)

    svd = SVD(random_state=43, n_factors=20, n_epochs=500)

    # train algo on the whole data
    R_surpise = R_surpise.build_full_trainset()
    svd.fit(R_surpise)

    # test and fill in the missing ratings
    testset = R_surpise.build_anti_testset()
    predictions = svd.test(testset)
    print(predictions)
    # score model
    print(accuracy.rmse(predictions))
    # 0.7426613175948654
    print(accuracy.mse(predictions))
    # 0.5515458326517415

    # save model
    # filename = 'svd_model.sav'
    # joblib.dump(svd, filename)