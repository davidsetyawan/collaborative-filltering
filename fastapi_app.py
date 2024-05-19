from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from keras.models import load_model
from recommendation import give_recommendation
from recommender import RecommenderNet
from processing import process_dataframe
from data import load_data

app = FastAPI()

ratings_file = "input/ratings.csv"
movies_file = "input/movies.csv"
links_file = "input/links.csv"

df, df_m, df_l = load_data(ratings_file, movies_file, links_file)
df, user2user_encoded, userencoded2user, movie2movie_encoded, movie_encoded2movie, num_users, num_movies, min_rating, max_rating = process_dataframe(
    df)

model_path = "model/my_model.keras"
loaded_model = load_model(model_path)


class RecommendationRequest(BaseModel):
    user_id: int


@app.get("/")
def read_root():
    return {"message": "Welcome to the Movie Recommendation API"}


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    user_id = request.user_id
    if user_id not in user2user_encoded:
        raise HTTPException(status_code=404, detail="User not found")

    recommendations = give_recommendation(df, user2user_encoded, userencoded2user,
                                          movie2movie_encoded, movie_encoded2movie, loaded_model, user_id, df_m, df_l)
    return recommendations
