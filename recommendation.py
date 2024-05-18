import numpy as np
from json import loads


def give_recommendation(df, user2user_encoded, userencoded2user, movie2movie_encoded, movie_encoded2movie, loaded_model, user_id, movie_df):
    movies_watched_by_user = df[df.userId == user_id]
    movies_not_watched = movie_df[~movie_df["movieId"].isin(
        movies_watched_by_user.movieId.values)]["movieId"]
    movies_not_watched = list(set(movies_not_watched).intersection(
        set(movie2movie_encoded.keys())))
    movies_not_watched = [
        [movie2movie_encoded.get(x)] for x in movies_not_watched]
    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movies_not_watched), movies_not_watched))
    ratings = loaded_model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [movie_encoded2movie.get(
        movies_not_watched[x][0]) for x in top_ratings_indices]
    recommended_movies = movie_df[movie_df["movieId"].isin(
        recommended_movie_ids)]
    result = recommended_movies.to_json(orient="records")
    parsed = loads(result)
    return parsed
