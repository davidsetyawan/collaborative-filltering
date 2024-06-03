from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_login import LoginManager  # Loginmanager Class
from fastapi_login.exceptions import InvalidCredentialsException  # Exception class
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
# Module Imports
from utils.recommendation import give_recommendation
from models.recommender import RecommenderNet
from utils.processing import process_dataframe
from data.data_loader import load_data
from utils.tmdb import get_movie_details

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET = "mysecretkey"

manager = LoginManager(SECRET, token_url="/auth/login", use_cookie=True)
manager.cookie_name = "username"
fake_db = {"david": {"password": "david", "userId": '123555'}}   # unhashed"


@manager.user_loader()
def load_user(username: str):
    user = fake_db.get(username)
    return user


@app.post("/auth/login")
def login(data: OAuth2PasswordRequestForm = Depends()):
    username = data.username
    password = data.password
    user = load_user(username)
    if not user:
        raise InvalidCredentialsException
    elif password != user['password']:
        raise InvalidCredentialsException

    access_token = manager.create_access_token(
        data={"userId": user["userId"]}
    )
    resp = RedirectResponse(url="/private", status_code=status.HTTP_302_FOUND)
    manager.set_cookie(resp, access_token)
    return {"status": "success", "status_code": "200", "access_token": access_token, "token_type": "bearer"}


@app.get("/private")
def getPrivateendpoint(_=Depends(manager)):
    return "You are an authentciated user"


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
def recommend(user_id: int = Query(..., description="The ID of the user to get recommendations for")):
    if user_id not in user2user_encoded:
        raise HTTPException(status_code=404, detail="User not found")

    recommendations = give_recommendation(df, user2user_encoded, userencoded2user,
                                          movie2movie_encoded, movie_encoded2movie, loaded_model, user_id, df_m, df_l)
    return recommendations


@app.get("/movie/{tmdb_id}")
def movie_details(tmdb_id: int):
    try:
        details = get_movie_details(tmdb_id)
        return details
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code,
                            detail="Data tidak ditemukan pada TMDB. Error: " + str(e))
