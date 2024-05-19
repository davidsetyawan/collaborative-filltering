import pandas as pd


def load_data(ratings_file, movies_file, links_file):
    df_ratings = pd.read_csv(ratings_file)
    df_movies = pd.read_csv(movies_file)
    df_links = pd.read_csv(links_file)
    return df_ratings, df_movies, df_links
