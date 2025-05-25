# backend/app/routers/ocr.py
from fastapi import APIRouter, HTTPException, status, File, UploadFile
from typing import Optional

from app.models.ocr import OCRResponse
from app.services.ocr_service import get_text_from_image, mistral_ocr_extractor # Import extractor to check status

router = APIRouter()

@router.post("/extract-text", response_model=OCRResponse)
async def extract_text_from_image(file: UploadFile = File(...)):
    """Extracts text from an uploaded image file using Mistral OCR."""
    # Check if OCR service is available (API key configured)
    if not mistral_ocr_extractor.client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR service is not configured or unavailable. Please check the MISTRAL_API_KEY.",
        )

    # Validate file type (optional but recommended)
    allowed_content_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Only {', '.join(allowed_content_types)} allowed.",
        )

    try:
        image_bytes = await file.read()
        extracted_text = await get_text_from_image(image_bytes)

        if extracted_text is None:
            # This indicates an error during OCR processing (logged in service)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process image with OCR service. Check backend logs.",
            )

        return OCRResponse(extracted_text=extracted_text)

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        # Catch other potential errors during file reading or processing
        print(f"Error processing image upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while processing the image: {e}",
        )
    finally:
        await file.close()

