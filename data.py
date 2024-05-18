import pandas as pd


def load_data(ratings_file, movies_file, links_file):
    df_m = pd.read_csv(movies_file)
    df = pd.read_csv(ratings_file)
    df_l = pd.read_csv(links_file)
    df_l['tmdbId'] = df_l['tmdbId'].apply("int64")
    return df, df_m, df_l
