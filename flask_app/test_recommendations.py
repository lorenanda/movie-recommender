import pytest
import 

def test_single_movie():

    x = random_recommend(MOVIES, 1)
    assert x[0] in MOVIES