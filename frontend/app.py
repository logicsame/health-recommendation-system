# frontend/app.py
import streamlit as st
import requests
from PIL import Image
import pandas as pd
import plotly.express as px
import datetime
import os
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables from .env file if it exists (for local dev)
load_dotenv()

# --- Configuration ---
# Use environment variables for backend URL, default to Docker service name
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://backend:8000/api/v1")

# --- API Client Functions ---
def api_request(method, endpoint, **kwargs):
    """Helper function to make requests to the backend API."""
    url = f"{BACKEND_URL}{endpoint}"
    headers = kwargs.get("headers", {})
    # Add authorization header if token exists in session state
    if st.session_state.get("auth_token"):
        headers["Authorization"] = f"Bearer {st.session_state.auth_token}"
        kwargs["headers"] = headers

    try:
        response = requests.request(method, url, **kwargs)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        # Check if response has JSON content before trying to decode
        if response.headers.get("Content-Type") == "application/json":
            return response.json()
        else:
            return response.text # Or handle non-JSON responses as needed
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        # Try to parse error detail from response if available
        try:
            error_detail = e.response.json().get("detail", str(e))
            st.error(f"Backend Error Detail: {error_detail}")
        except Exception:
            pass # Ignore if error response is not JSON or doesn't have detail
        return None

# --- Authentication Functions ---
def login(username, password):
    """Login user via API."""
    # FastAPI OAuth2 expects form data
    data = {"username": username, "password": password}
    response = api_request("post", "/auth/token", data=data)
    if response and "access_token" in response:
        st.session_state.auth_token = response["access_token"]
        st.session_state.logged_in = True
        st.session_state.current_user = username
        # Fetch user profile after successful login
        fetch_user_profile(username)
        return True
    return False

def signup(username, email, password, age, height, weight, gender, activity_level, health_conditions):
    """Signup user via API."""
    data = {
        "username": username,
        "email": email,
        "password": password,
        "age": age,
        "height": height,
        "weight": weight,
        "gender": gender,
        "activity_level": activity_level,
        "health_conditions": health_conditions
    }
    response = api_request("post", "/users/signup", json=data)
    return response is not None

def logout():
    """Logs out the user by clearing session state."""
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.auth_token = None
    st.session_state.user_profile = None
    st.session_state.extracted_text = ""
    st.session_state.recommendation = ""
    st.session_state.meal_plan = ""
    st.success("Logged out successfully.")
    st.rerun()

def fetch_user_profile(username):
    """Fetches user profile from API and stores it in session state."""
    response = api_request("get", f"/users/{username}/profile")
    if response:
        st.session_state.user_profile = response
    else:
        st.error("Failed to fetch user profile.")
        st.session_state.user_profile = None # Clear profile on error

# --- Feature Functions (API Calls) ---
def get_ocr_text(image_bytes, filename):
    """Get OCR text from image via API."""
    files = {"file": (filename, image_bytes, "image/jpeg")} # Assume jpeg, adjust if needed
    # Note: Don't pass json= or data= when sending files
    response = api_request("post", "/ocr/extract-text", files=files)
    if response and "extracted_text" in response:
        return response["extracted_text"]
    return None

def get_recommendation(food_info, health_condition, user_profile):
    """Get food recommendation via API."""
    if not user_profile:
        st.error("User profile not available. Cannot get recommendation.")
        return None
    data = {
        "food_info": food_info,
        "health_condition": health_condition,
        "user_profile": user_profile
    }
    response = api_request("post", "/llm/recommendation", json=data)
    if response and "recommendation" in response:
        return response["recommendation"]
    return None

def get_meal_plan(user_profile, health_condition):
    """Get meal plan via API."""
    if not user_profile:
        st.error("User profile not available. Cannot get meal plan.")
        return None
    data = {
        "user_profile": user_profile,
        "health_condition": health_condition
    }
    response = api_request("post", "/llm/meal-plan", json=data)
    if response and "meal_plan" in response:
        return response["meal_plan"]
    return None

# --- Health Calculation Functions (moved to backend, kept for potential frontend display) ---
def calculate_bmi(weight, height):
    if height <= 0:
        return 0
    return round(weight / ((height / 100) ** 2), 2)

def calculate_water_intake(weight):
    if not isinstance(weight, (int, float)) or weight <= 0:
        return 0
    return round(weight * 0.033, 2)

# Calorie needs calculation is complex and now handled by backend if needed
# We can display the results fetched from the profile if the backend calculates it.

# --- Initialize Session State ---
def init_session_state():
    defaults = {
        "logged_in": False,
        "current_user": None,
        "auth_token": None,
        "user_profile": None,
        "extracted_text": "",
        "recommendation": "",
        "meal_plan": "",
        "recommendation_history": {} # Keep history client-side for now
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- UI Components ---
def display_login_signup():
    st.header("User Account")
    if not st.session_state.logged_in:
        login_option = st.radio("Choose an option", ["Login", "Sign Up"], key="login_signup_radio", horizontal=True)

        if login_option == "Login":
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    if login(username, password):
                        st.success("Logged in successfully!")
                        st.rerun() # Rerun to update UI based on logged_in state
                    else:
                        st.error("Invalid username or password")
        else: # Sign Up
            with st.form("signup_form"):
                st.subheader("Create New Account")
                new_username = st.text_input("New Username")
                new_email = st.text_input("Email")
                new_password = st.text_input("New Password", type="password")
                age = st.number_input("Age", min_value=1, max_value=120, value=30)
                height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
                weight = st.number_input("Weight (kg)", min_value=20, max_value=500, value=70)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                activity_level = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extra Active"])
                health_conditions = st.text_area("Health Conditions (e.g., Diabetes, High Blood Pressure)", "None")
                submitted = st.form_submit_button("Sign Up")
                if submitted:
                    if not all([new_username, new_email, new_password]):
                         st.error("Username, email, and password are required.")
                    elif signup(new_username, new_email, new_password, age, height, weight, gender, activity_level, health_conditions):
                        st.success("Account created successfully! Please log in.")
                    else:
                        st.error("Signup failed. Username might already exist or backend error occurred.")
    else:
        st.subheader(f"Welcome, {st.session_state.current_user}!")
        if st.button("Logout"):

            logout()

def display_user_profile():
    st.header("👤 Your Profile")
    profile = st.session_state.user_profile
    if profile:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Age", profile["age"])
            st.metric("Height (cm)", profile["height"])
            st.metric("Weight (kg)", profile["weight"])
            st.metric("BMI", f"{profile["bmi"]:.1f}")
        with col2:
            st.metric("Gender", profile["gender"])
            st.metric("Activity Level", profile["activity_level"])
            st.metric("Water Intake (L/day)", f"{calculate_water_intake(profile["weight"]):.2f}")
            # Calorie needs could be added here if calculated/stored in profile by backend
            # st.metric("Est. Calorie Needs", profile.get("calorie_needs", "N/A"))

        st.subheader("Health Conditions")
        st.write(profile["health_conditions"] or "None specified")
    else:
        st.warning("Profile data not available. Please log in.")

def display_ocr_section():
    st.header("📷 Analyze Food Label (OCR)")
    uploaded_file = st.file_uploader("Upload an image of a food label", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Food Label", use_column_width=True)

            # Process OCR when button is clicked
            if st.button("Extract Text from Image"):
                with st.spinner("Extracting text using OCR..."):
                    # Get bytes from the uploaded file
                    img_bytes = uploaded_file.getvalue()
                    extracted_text = get_ocr_text(img_bytes, uploaded_file.name)

                if extracted_text is not None:
                    st.session_state.extracted_text = extracted_text
                    st.success("Text extracted successfully!")
                else:
                    st.error("Failed to extract text. Check backend logs or API key.")
                    st.session_state.extracted_text = "" # Clear on failure
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.session_state.extracted_text = ""

    # Display extracted text if available
    if st.session_state.extracted_text:
        st.subheader("Extracted Text")
        st.text_area("OCR Result", st.session_state.extracted_text, height=200)

def display_llm_features():
    st.header("🤖 AI Nutrition Advisor")
    if not st.session_state.user_profile:
        st.warning("Please log in and ensure your profile is loaded to use AI features.")
        return

    tab1, tab2 = st.tabs(["Food Recommendation", "Generate Meal Plan"])

    with tab1:
        st.subheader("Get Food Recommendation")
        food_info_input = st.text_area("Enter Food Information (or use text from OCR)", value=st.session_state.get("extracted_text", ""), height=150)
        # Use health conditions from profile by default
        health_condition_input = st.text_input("Relevant Health Condition", value=st.session_state.user_profile.get("health_conditions", "None"))

        if st.button("Get Recommendation"):
            if not food_info_input:
                st.warning("Please enter food information.")
            else:
                with st.spinner("Generating recommendation..."):
                    recommendation = get_recommendation(food_info_input, health_condition_input, st.session_state.user_profile)
                if recommendation:
                    st.session_state.recommendation = recommendation
                    # Save to history (client-side)
                    save_to_history(st.session_state.current_user, food_info_input, recommendation)
                else:
                    st.error("Failed to get recommendation.")
                    st.session_state.recommendation = ""

        if st.session_state.recommendation:
            st.subheader("Recommendation Result")
            st.markdown(st.session_state.recommendation)

    with tab2:
        st.subheader("Generate One-Day Meal Plan")
        # Use health conditions from profile by default
        health_condition_plan = st.text_input("Primary Health Condition for Meal Plan", value=st.session_state.user_profile.get("health_conditions", "None"), key="meal_plan_condition")

        if st.button("Generate Meal Plan"):
            with st.spinner("Generating meal plan..."):
                meal_plan = get_meal_plan(st.session_state.user_profile, health_condition_plan)
            if meal_plan:
                st.session_state.meal_plan = meal_plan
            else:
                st.error("Failed to generate meal plan.")
                st.session_state.meal_plan = ""

        if st.session_state.meal_plan:
            st.subheader("Generated Meal Plan")
            st.markdown(st.session_state.meal_plan)

def display_history():
    st.header("📜 Recommendation History")
    username = st.session_state.current_user
    if username and username in st.session_state.recommendation_history:
        history = st.session_state.recommendation_history[username]
        if not history:
            st.info("No recommendations saved yet.")
            return

        # Display history in reverse chronological order
        for i, entry in enumerate(reversed(history)):
            with st.expander(f"{entry["timestamp"]} - Recommendation {len(history)-i}"):
                st.subheader("Food Info Provided:")
                st.text(entry["food_info"])
                st.subheader("Recommendation Given:")
                st.markdown(entry["recommendation"])
    else:
        st.info("Log in to view your recommendation history.")

# Function to save recommendation to history (client-side session state)
def save_to_history(username, food_info, recommendation):
    if username not in st.session_state.recommendation_history:
        st.session_state.recommendation_history[username] = []
    st.session_state.recommendation_history[username].append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "food_info": food_info,
        "recommendation": recommendation
    })

# --- Main App Layout ---
def main():
    st.set_page_config(page_title="Smart Food Advisor", layout="wide", initial_sidebar_state="expanded")
    st.title("🌟 Smart Food Advisor")

    # Initialize session state variables if they don't exist
    init_session_state()

    # Sidebar for Login/Signup and Profile
    with st.sidebar:
        display_login_signup()
        if st.session_state.logged_in:
            st.divider()
            display_user_profile()

    # Main content area
    if not st.session_state.logged_in:
        st.info("Please log in or sign up using the sidebar to use the application features.")
    else:
        # Use columns for layout
        col_ocr, col_llm = st.columns(2)

        with col_ocr:
            display_ocr_section()

        with col_llm:
            display_llm_features()

        st.divider()
        display_history()

if __name__ == "__main__":
    main()

