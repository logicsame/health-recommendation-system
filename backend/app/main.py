# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import auth, users, ocr, llm
# Import the service modules to trigger model loading on startup
from app.services import llm_service, ocr_service

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(auth.router, prefix=settings.API_V1_STR + "/auth", tags=["auth"])
app.include_router(users.router, prefix=settings.API_V1_STR + "/users", tags=["users"])
app.include_router(ocr.router, prefix=settings.API_V1_STR + "/ocr", tags=["ocr"])
app.include_router(llm.router, prefix=settings.API_V1_STR + "/llm", tags=["llm"])

@app.on_event("startup")
async def startup_event():
    print("Starting up FastAPI application...")
    # Attempt to load models if they haven't been loaded yet
    # (They should load on import, but this is a fallback/check)
    if not llm_service.llm_model or not llm_service.llm_tokenizer:
        print("LLM Model/Tokenizer not loaded on import, attempting load on startup...")
        llm_service.load_llm_model_and_tokenizer()
    if not ocr_service.mistral_ocr_extractor.client:
         print("Mistral OCR client not initialized (check API key).")
    print("Startup complete.")

@app.get("/")
async def root():
    return {"message": f"Welcome to the {settings.PROJECT_NAME}! Visit /docs for API documentation."}

# If running directly using uvicorn for development:
# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

