import numpy as np


def process_dataframe(df):
    user_ids = df["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    movie_ids = df["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

    df["user"] = df["userId"].map(user2user_encoded).astype('int64')
    df["movie"] = df["movieId"].map(movie2movie_encoded).astype('int64')
    df["rating"] = df["rating"].values.astype(np.float32)

    num_users = len(user2user_encoded)
    num_movies = len(movie_encoded2movie)
    min_rating = min(df["rating"])
    max_rating = max(df["rating"])

    return df, user2user_encoded, userencoded2user, movie2movie_encoded, movie_encoded2movie, num_users, num_movies, min_rating, max_rating
