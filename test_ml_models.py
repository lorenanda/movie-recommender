import pytest
import numpy as np
import joblib
import flask_app
from flask_app import split_data, ratings_pivot, movies_df
from flask_app import ml_models


nmf = joblib.load("./flask_app/saved_models/nmf.sav")


@pytest.mark.parametrize(['input', 'expected_output'],
                         # Pass the parameters as a list of tuples
                         [({1: 4, 2: 5, 100: 1}, 5), ({}, 5)])
def test_nmf(input, expected_output):
    cols_above, cols_below, _ = split_data(2010, movies_df)
    results = ml_models.nmf_recommand(model=nmf, new_user=input, n=5, orig_data=ratings_pivot,
                                      cols_above=cols_above, cols_below=cols_below, selection=3)
    assert results.shape[0] == expected_output


@pytest.mark.parametrize(['input', 'expected_output'],
                         # Pass the parameters as a list of tuples
                         [({1: 4, 2: 5, 100: 1}, 5), ({}, 5)])
def test_similarity_matrix(input, expected_output):
    results = ml_models.calculate_similarity_matrix(
        new_user_input=input, orig_data=ratings_pivot, n_users=5)
    assert len(results.values) == expected_output
