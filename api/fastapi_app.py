from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi_users import FastAPIUsers
from pydantic import BaseModel
from typing import List
from keras.models import load_model
import requests

from auth.user import fastapi_users, auth_backend  # Import from auth.user
from auth.database import User, get_async_session, init_db
from utils.recommendation import give_recommendation
from models.recommender import RecommenderNet
from utils.processing import process_dataframe
from data.data_loader import load_data
from utils.tmdb import get_movie_details


app = FastAPI()

fastapi_users = FastAPIUsers(..., [auth_backend])


@app.on_event("startup")
async def on_startup():
    await init_db()

ratings_file = "input/ratings.csv"
movies_file = "input/movies.csv"
links_file = "input/links.csv"

df, df_m, df_l = load_data(ratings_file, movies_file, links_file)
df, user2user_encoded, userencoded2user, movie2movie_encoded, movie_encoded2movie, num_users, num_movies, min_rating, max_rating = process_dataframe(
    df)

model_path = "model/my_model.keras"
loaded_model = load_model(model_path)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Movie Recommendation API"}


@app.middleware("http")
async def debug_request(request: Request, call_next):
    print(request.headers)
    response = await call_next(request)
    return response


@app.post("/recommend", current_user=Depends(fastapi_users.current_user()))
def recommend(user_id: int = Query(..., description="The ID of the user to get recommendations for")):
    if user_id not in user2user_encoded:
        raise HTTPException(status_code=404, detail="User not found")

    recommendations = give_recommendation(df, user2user_encoded, userencoded2user,
                                          movie2movie_encoded, movie_encoded2movie, loaded_model, user_id, df_m, df_l)
    return recommendations


@app.get("/movie/{tmdb_id}", current_user=Depends(fastapi_users.current_user()))
def movie_details(tmdb_id: int):
    try:
        details = get_movie_details(tmdb_id)
        return details
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))


app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
)
app.include_router(
    fastapi_users.get_register_router(), prefix="/auth", tags=["auth"]
)
