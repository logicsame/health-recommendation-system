# backend/app/routers/llm.py
from fastapi import APIRouter, HTTPException, status, Depends

from app.models.llm import RecommendationRequest, RecommendationResponse, MealPlanRequest, MealPlanResponse
from app.services.llm_service import generate_recommendation, generate_meal_plan, model_loading_error
# Import authentication dependency if needed
# from app.core.auth import get_current_active_user
# from app.models.user import UserProfile

router = APIRouter()

# Dependency to check if LLM loaded correctly
def check_llm_status():
    if model_loading_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service is unavailable: {model_loading_error}"
        )

@router.post("/recommendation", response_model=RecommendationResponse, dependencies=[Depends(check_llm_status)])
async def get_food_recommendation(request: RecommendationRequest):
    # Optional: Add authentication check here
    # current_user: UserProfile = Depends(get_current_active_user)
    # You might want to ensure the user_profile in the request matches the logged-in user
    """Generates a food recommendation based on food info, health condition, and user profile."""
    try:
        recommendation = await generate_recommendation(request)
        if recommendation.startswith("Error:"):
            # Handle errors reported by the service function
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=recommendation
            )
        return RecommendationResponse(recommendation=recommendation)
    except Exception as e:
        print(f"Error in /recommendation endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while generating the recommendation: {e}"
        )

@router.post("/meal-plan", response_model=MealPlanResponse, dependencies=[Depends(check_llm_status)])
async def get_meal_plan(request: MealPlanRequest):
    # Optional: Add authentication check here
    # current_user: UserProfile = Depends(get_current_active_user)
    # You might want to ensure the user_profile in the request matches the logged-in user
    """Generates a one-day meal plan based on user profile and health condition."""
    try:
        meal_plan = await generate_meal_plan(request)
        if meal_plan.startswith("Error:"):
            # Handle errors reported by the service function
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=meal_plan
            )
        return MealPlanResponse(meal_plan=meal_plan)
    except Exception as e:
        print(f"Error in /meal-plan endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while generating the meal plan: {e}"
        )

