import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from data import load_data
from recommender import RecommenderNet
from processing import process_dataframe
from recommendation import give_recommendation

ratings_file = "input/ratings.csv"
movies_file = "input/movies.csv"
links_file = "input/links.csv"

df, df_m, df_l = load_data(ratings_file, movies_file, links_file)

df, user2user_encoded, userencoded2user, movie2movie_encoded, movie_encoded2movie, num_users, num_movies, min_rating, max_rating = process_dataframe(
    df)

print(df.tail(5))
print(
    f"Number of users: {num_users}, Number of Movies: {num_movies}, Min rating: {min_rating}, Max rating: {max_rating}")

df = df.sample(frac=1, random_state=42)
x = df[["user", "movie"]].values
y = df["rating"].apply(lambda x: (x - min_rating) /
                       (max_rating - min_rating)).values

train_indices = int(0.9 * df.shape[0])
x_train = x[:train_indices]
y_train = y[:train_indices]
x_val = x[train_indices:]
y_val = y[train_indices:]

model = RecommenderNet(num_users, num_movies)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=keras.optimizers.Adam(learning_rate=0.001))

path = "model/my_model.keras"
loaded_model = load_model(path)
loaded_model.summary()

user_id = 1  # Example user_id
recommendations = give_recommendation(df, user2user_encoded, userencoded2user,
                                      movie2movie_encoded, movie_encoded2movie, loaded_model, user_id, df_m)
print(recommendations)
