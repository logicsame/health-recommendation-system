# Smart Food Advisor

## Overview

This project is a web application designed to provide personalized nutrition advice. It combines Optical Character Recognition (OCR) to extract text from food labels, a Large Language Model (LLM) for generating dietary recommendations and meal plans, and user profile management to tailor the advice.

The application features a FastAPI backend handling the core logic (OCR, LLM inference, user management) and a Streamlit frontend providing an interactive user interface.

## Features

*   **User Authentication**: Secure user signup and login.
*   **User Profiles**: Stores user details like age, weight, height, health conditions, etc.
*   **OCR Food Label Analysis**: Upload food label images to extract nutritional information using Mistral AI OCR.
*   **LLM-Powered Recommendations**: Get personalized food suitability recommendations based on extracted text and user profile.
*   **AI Meal Plan Generation**: Generate one-day meal plans tailored to user profiles and health conditions.
*   **Recommendation History**: View past recommendations.
*   **Dockerized**: Easy setup and deployment using Docker and Docker Compose.

## Project Structure

```
smart-food-advisor/
├── backend/
│   ├── app/
│   │   ├── core/         # Core logic (config, security)
│   │   ├── models/       # Pydantic models (data validation)
│   │   ├── routers/      # API endpoint definitions
│   │   ├── services/     # Business logic (OCR, LLM, User, Health)
│   │   ├── __init__.py
│   │   └── main.py       # FastAPI application entrypoint
│   ├── saved_model/    # <--- PLACE YOUR LLM MODEL FILES HERE
│   ├── Dockerfile      # Backend Docker build instructions
│   └── requirements.txt # Backend Python dependencies
├── frontend/
│   ├── pages/          # (Optional) Streamlit multi-page app structure
│   ├── app.py          # Streamlit application entrypoint
│   ├── Dockerfile      # Frontend Docker build instructions
│   └── requirements.txt # Frontend Python dependencies
├── .env.example      # Example environment variables
├── docker-compose.yml # Docker Compose configuration
└── README.md         # This file
```

## Technology Stack

*   **Backend**: FastAPI, Python 3.11, Uvicorn
*   **Frontend**: Streamlit, Python 3.11
*   **LLM**: Hugging Face Transformers, BitsAndBytes (for quantization)
*   **OCR**: Mistral AI API
*   **Containerization**: Docker, Docker Compose
*   **Authentication**: JWT (JSON Web Tokens), Passlib (for hashing)
*   **Data Validation**: Pydantic

## Setup and Installation

### Prerequisites

*   [Docker](https://docs.docker.com/get-docker/)
*   [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)
*   [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
*   A Mistral AI API Key (for OCR functionality)
*   A pre-trained Large Language Model (LLM) compatible with Hugging Face `AutoModelForCausalLM`.

### Docker Setup (Recommended)

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace with the actual URL if hosted
    cd smart-food-advisor
    ```

2.  **Prepare the LLM Model:**
    *   Download or obtain your desired LLM files (e.g., from Hugging Face Hub).
    *   Place the complete model directory (containing `config.json`, `pytorch_model.bin` or `.safetensors` files, `tokenizer.json`, etc.) inside the `backend/saved_model/` directory.
    *   **Important**: The application expects the model files to be directly within `backend/saved_model/`. Ensure the paths in `backend/app/core/config.py` (`LLM_MODEL_ID` and `BASE_MODEL_ID`) point correctly to `/app/saved_model` within the container.

3.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file with your actual values:
        *   `MISTRAL_API_KEY`: Your API key from Mistral AI.
        *   `SECRET_KEY`: A strong, random secret key for JWT tokens. You can generate one using `openssl rand -hex 32`.

4.  **Build and Run with Docker Compose:**
    ```bash
    docker-compose up --build -d
    ```
    *   `--build`: Forces Docker to rebuild the images if Dockerfiles have changed.
    *   `-d`: Runs the containers in detached mode (in the background).

5.  **Access the Application:**
    *   Frontend (Streamlit): Open your web browser and navigate to `http://localhost:8501`.
    *   Backend API Docs (FastAPI): Navigate to `http://localhost:8000/docs`.

6.  **Stopping the Application:**
    ```bash
    docker-compose down
    ```

### Manual Setup (Advanced)

Running manually requires setting up separate Python environments for the backend and frontend.

1.  **Clone the repository** (as above).
2.  **Prepare the LLM Model** (as above, place in `backend/saved_model/`).
3.  **Backend Setup:**
    *   Navigate to the `backend` directory: `cd backend`
    *   Create and activate a virtual environment (e.g., using `venv`):
        ```bash
        python3 -m venv venv
        source venv/bin/activate # On Windows use `venv\Scripts\activate`
        ```
    *   Install dependencies: `pip install -r requirements.txt`
    *   Set environment variables (e.g., by exporting them or using a `.env` file with `python-dotenv` locally).
    *   Run the FastAPI server: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload` (`--reload` for development).
4.  **Frontend Setup:**
    *   Navigate to the `frontend` directory: `cd ../frontend`
    *   Create and activate a separate virtual environment.
    *   Install dependencies: `pip install -r requirements.txt`
    *   Set the `BACKEND_API_URL` environment variable: `export BACKEND_API_URL="http://localhost:8000/api/v1"` (or set in `.env`).
    *   Run the Streamlit app: `streamlit run app.py`

## Environment Variables

The following environment variables are used by the application (configure in `.env` file when using Docker Compose):

*   `MISTRAL_API_KEY` (Required): Your API key for Mistral AI, used for OCR.
*   `SECRET_KEY` (Required): A secret key used for signing JWT authentication tokens. Keep this secure.
*   `MISTRAL_OCR_MODEL` (Optional): The specific Mistral OCR model to use (defaults to `mistral-ocr-latest`).
*   `BACKEND_API_URL` (Frontend - Optional): The URL of the backend API. This is automatically set in `docker-compose.yml` for the frontend service. Set manually if running the frontend separately.

## LLM Model Requirement

This project **does not include** a pre-trained Large Language Model (LLM). You must provide your own model files and place them in the `backend/saved_model/` directory.

The backend is configured to load a model from this directory using Hugging Face Transformers (`AutoModelForCausalLM` and `AutoTokenizer`). Ensure the model you provide is compatible and that the directory structure is correct.

## API Documentation

When the backend service is running, interactive API documentation (provided by FastAPI and Swagger UI) is available at `/docs`. Example: `http://localhost:8000/docs`.

## Usage

1.  Start the application using Docker Compose (recommended) or manually.
2.  Access the Streamlit frontend in your browser (default: `http://localhost:8501`).
3.  Sign up for a new account or log in.
4.  Upload food label images via the OCR section to extract text.
5.  Use the extracted text (or manually entered info) in the AI Nutrition Advisor section to get recommendations or generate meal plans.
6.  View your past recommendations in the History section.

