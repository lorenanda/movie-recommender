"""
This is the code for the ML model
that makes recommendatins based on movie ratings.
"""

import random

movies = [
          "The Shawshank Redemption",
          "Star Wars: Episode IV - A New Hope",
          "Pulp Fiction",
          "The Dark Knight",
          "Forrest Gump",
          "Inception",
          "The Matrix",
          "Saving Private Ryan",
          "Casablanca",
          "The Lion King"
]

def get_recommendations():
    random.shuffle(movies)
    return movies[:3]

def nmf_recommender():
    pass

    """
    ratings = pd.read_csv('./data/ratings.csv')
    ratings_pivot = ratings.pivot(index='userId', columns='movieId', values='rating')
    ratings_pivot.replace(np.nan, 0, inplace=True)
    
    model = NMF(n_components=2, init='random', random_state=10)
    model.fit(ratings_pivot)

    Q = model.components_  
    P = model.transform(ratings_pivot) 

    nR = np.dot(P, Q)
    
    """