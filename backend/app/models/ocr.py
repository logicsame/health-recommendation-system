# backend/app/models/ocr.py
from pydantic import BaseModel
from fastapi import UploadFile, File

class OCRRequest(BaseModel):
    # We'll handle the file upload directly in the endpoint
    pass

class OCRResponse(BaseModel):
    extracted_text: str

