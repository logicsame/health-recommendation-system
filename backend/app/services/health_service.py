# backend/app/services/health_service.py

def calculate_bmi(weight: float, height: float) -> float:
    """Calculates Body Mass Index (BMI)."""
    if height <= 0:
        return 0.0
    return round(weight / ((height / 100) ** 2), 2)

def calculate_water_intake(weight: float) -> float:
    """Calculates recommended daily water intake in liters."""
    if not isinstance(weight, (int, float)) or weight <= 0:
        return 0.0
    # Recommended intake: 33 ml per kg
    return round(weight * 0.033, 2)

def calculate_calorie_needs(weight: float, height: float, age: int, gender: str, activity_level: str) -> float:
    """Calculates estimated daily calorie needs (TDEE) using Mifflin-St Jeor equation."""
    if not all([isinstance(val, (int, float)) and val > 0 for val in [weight, height, age]]):
         return 0.0

    # Basal Metabolic Rate (BMR) - Mifflin-St Jeor Equation
    if gender.lower() == 'male':
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    elif gender.lower() == 'female':
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    else: # Use average if gender is 'Other' or unspecified
        bmr_male = (10 * weight) + (6.25 * height) - (5 * age) + 5
        bmr_female = (10 * weight) + (6.25 * height) - (5 * age) - 161
        bmr = (bmr_male + bmr_female) / 2

    activity_factors = {
        'sedentary': 1.2,
        'lightly active': 1.375,
        'moderately active': 1.55,
        'very active': 1.725,
        'extra active': 1.9
    }
    # Normalize activity level string (lowercase, remove spaces)
    normalized_activity = activity_level.lower().replace(" ", "")
    factor = activity_factors.get(normalized_activity, 1.2) # Default to sedentary if unknown

    # Total Daily Energy Expenditure (TDEE)
    tdee = bmr * factor
    return round(max(0, tdee), 0) # Return non-negative integer calories

