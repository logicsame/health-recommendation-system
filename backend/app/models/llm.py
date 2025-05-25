# backend/app/models/llm.py
from pydantic import BaseModel
from .user import UserProfile # Import UserProfile for context

class RecommendationRequest(BaseModel):
    food_info: str
    health_condition: str
    user_profile: UserProfile # Pass the full profile for context

class RecommendationResponse(BaseModel):
    recommendation: str

class MealPlanRequest(BaseModel):
    user_profile: UserProfile
    health_condition: str

class MealPlanResponse(BaseModel):
    meal_plan: str

