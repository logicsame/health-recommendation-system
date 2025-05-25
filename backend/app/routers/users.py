# backend/app/routers/users.py
from fastapi import APIRouter, HTTPException, status, Depends
from typing import Optional

from app.models.user import UserCreate, UserProfile
from app.services.user_service import create_user, get_user_profile
# Import authentication dependency if needed for profile endpoint
# from app.core.auth import get_current_active_user # Assuming you create this

router = APIRouter()

@router.post("/signup", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
async def register_user(user_in: UserCreate):
    """Creates a new user account."""
    user = await create_user(user_in)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )
    return user

@router.get("/{username}/profile", response_model=UserProfile)
async def read_user_profile(username: str):
    # Optional: Add authentication check here if profile should be private
    # current_user: UserProfile = Depends(get_current_active_user)
    # if current_user.username != username:
    #     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this profile")

    """Retrieves the public profile of a user."""
    user_profile = await get_user_profile(username)
    if not user_profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user_profile

# You might add endpoints to update profile, delete user, etc.

