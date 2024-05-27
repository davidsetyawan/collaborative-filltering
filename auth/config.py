from fastapi_users.authentication import JWTStrategy, AuthenticationBackend
from fastapi_users import FastAPIUsers
from auth.user import get_user_manager
from auth.database import User
import os

SECRET = os.getenv("SECRET_KEY", "SECRET")


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)


# Check if AuthenticationBackend is properly configured
auth_backend = AuthenticationBackend(
    name="jwt", transport=None, get_strategy=get_jwt_strategy)

fastapi_users = FastAPIUsers(
    get_user_manager,
    [auth_backend],
    User
)
