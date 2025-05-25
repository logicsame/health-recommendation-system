# backend/app/core/config.py
import os
from dotenv import load_dotenv
from pydantic import BaseSettings

# Load environment variables from .env file, if it exists
load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Smart Food Advisor API"
    API_V1_STR: str = "/api/v1"

    # LLM Configuration (adjust model paths as needed)
    LLM_MODEL_ID: str = "/app/saved_model" # Path inside the Docker container
    BASE_MODEL_ID: str = "/app/saved_model" # Path inside the Docker container

    # Mistral API Key (loaded from environment)
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "YOUR_MISTRAL_API_KEY_HERE")
    MISTRAL_OCR_MODEL: str = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")

    # CORS Origins (adjust if frontend runs on a different port/domain)
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost", "http://localhost:8501", "http://frontend:8501"]

    class Config:
        case_sensitive = True
        # If using a .env file:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

