import pandas as pd
import numpy as np
from json import loads
from tensorflow import keras


def give_recommendation(df, user2user_encoded, userencoded2user, movie2movie_encoded, movie_encoded2movie, loaded_model, user_id, movie_df, links_df):
    # Merge movie_df and links_df on 'movieId' to include 'tmdbId' in movie_df
    movie_df = movie_df.merge(links_df[['movieId', 'tmdbId']], on='movieId')

    # Check if the user_id exists in the user2user_encoded mapping
    if user_id not in user2user_encoded:
        raise ValueError(f"User ID {user_id} not found in the dataset")

    # Get movies watched by the user
    movies_watched_by_user = df[df.userId == user_id]

    # Get the list of movie IDs that the user has not watched
    movies_not_watched = movie_df[~movie_df["movieId"].isin(
        movies_watched_by_user.movieId.values)]["movieId"]
    movies_not_watched = list(set(movies_not_watched).intersection(
        set(movie2movie_encoded.keys())))
    movies_not_watched = [
        [movie2movie_encoded.get(x)] for x in movies_not_watched]

    # Get the user encoder
    user_encoder = user2user_encoded.get(user_id)

    # Create an array of user and movie pairs
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movies_not_watched), movies_not_watched))

    # Predict ratings for the movies the user has not watched
    ratings = loaded_model.predict(user_movie_array).flatten()

    # Get the top 10 movie indices
    top_ratings_indices = ratings.argsort()[-10:][::-1]

    # Get the top 10 recommended movie IDs
    recommended_movie_ids = [movie_encoded2movie.get(
        movies_not_watched[x][0]) for x in top_ratings_indices]

    # Filter the recommended movies from the movie_df
    recommended_movies = movie_df[movie_df["movieId"].isin(
        recommended_movie_ids)]

    # Convert the recommended movies to JSON format
    result = recommended_movies.to_json(orient="records")
    parsed = loads(result)
    return parsed
