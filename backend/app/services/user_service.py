# backend/app/services/user_service.py
import datetime
from typing import Dict, Optional
from app.models.user import UserCreate, UserProfile, UserInDB
from app.core.security import get_password_hash, verify_password
from app.services.health_service import calculate_bmi

# In-memory storage for users (replace with a database in a real application)
fake_users_db: Dict[str, UserInDB] = {}

async def create_user(user_in: UserCreate) -> Optional[UserProfile]:
    """Creates a new user."""
    if user_in.username in fake_users_db:
        return None # Username already exists

    hashed_password = get_password_hash(user_in.password)
    bmi = calculate_bmi(user_in.weight, user_in.height)
    join_date = datetime.datetime.now().strftime("%Y-%m-%d")

    user_db = UserInDB(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hashed_password,
        age=user_in.age,
        height=user_in.height,
        weight=user_in.weight,
        bmi=bmi,
        gender=user_in.gender,
        activity_level=user_in.activity_level,
        health_conditions=user_in.health_conditions,
        join_date=join_date
    )
    fake_users_db[user_in.username] = user_db

    # Return the public profile, excluding the hashed password
    user_profile = UserProfile(**user_db.dict(exclude={"hashed_password"}))
    return user_profile

async def get_user(username: str) -> Optional[UserInDB]:
    """Retrieves a user by username (including hashed password)."""
    return fake_users_db.get(username)

async def get_user_profile(username: str) -> Optional[UserProfile]:
    """Retrieves a user's public profile by username."""
    user_db = fake_users_db.get(username)
    if user_db:
        return UserProfile(**user_db.dict(exclude={"hashed_password"}))
    return None

async def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticates a user."""
    user = await get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

