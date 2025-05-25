# backend/app/services/ocr_service.py
import os
import base64
from io import BytesIO
from PIL import Image
from mistralai import Mistral
from typing import Dict, Any, Optional
from app.core.config import settings

class MistralOCRTextExtractor:
    """
    A class to extract text from images using Mistral AI's OCR capabilities.
    Adapted for backend FastAPI service.
    """
    VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self):
        """
        Initialize the MistralOCRTextExtractor client using settings.
        """
        self.api_key = settings.MISTRAL_API_KEY
        self.model = settings.MISTRAL_OCR_MODEL

        if not self.api_key or self.api_key == "YOUR_MISTRAL_API_KEY_HERE":
            # Log an error or warning, but don't raise an exception that stops the server
            # The endpoint should handle the case where the extractor is not available
            print("ERROR: Mistral API key not found or not configured. OCR functionality will be disabled.")
            self.client = None
            # raise ValueError("Mistral API key is required.")
        else:
            try:
                self.client = Mistral(api_key=self.api_key)
            except Exception as e:
                print(f"ERROR: Failed to initialize Mistral client: {e}")
                self.client = None
                # raise

    def _process_ocr(self, document_source: Dict[str, str]) -> Optional[Any]:
        """
        Process OCR on a document.
        Args:
            document_source: Dictionary containing document source information
        Returns:
            OCR processing result or None if client is not initialized or API error occurs.
        """
        if not self.client:
            print("ERROR: Mistral client not initialized. Cannot process OCR.")
            return None
        try:
            return self.client.ocr.process(
                model=self.model,
                document=document_source,
                include_image_base64=False
            )
        except Exception as e:
            print(f"ERROR: Mistral API Error during OCR processing: {e}")
            # Don't raise, let the calling function handle None return
            # raise RuntimeError(f"Mistral API Error: {e}") from e
            return None

    def extract_text_from_image_bytes(self, image_bytes: bytes) -> Optional[str]:
        """
        Extract text from image bytes.
        Args:
            image_bytes: Image content as bytes
        Returns:
            Extracted text or None if an error occurs.
        """
        if not self.client:
             print("ERROR: Mistral client not initialized. Cannot extract text.")
             return None
        try:
            img = Image.open(BytesIO(image_bytes))
            buffered = BytesIO()
            # Determine format, default to PNG if unknown or unsupported by Mistral
            img_format = img.format if img.format else "PNG"
            if img_format.upper() not in ["JPEG", "PNG"]:
                 print(f"Warning: Image format {img_format} might not be optimal for Mistral OCR. Converting to PNG.")
                 img_format = "PNG"

            # Ensure image mode is RGB(A) - common requirement for APIs
            if img.mode not in ["RGB", "RGBA"]:
                print(f"Warning: Converting image mode from {img.mode} to RGB.")
                img = img.convert("RGB")
                img_format = "JPEG" # Prefer JPEG for RGB conversion unless original was PNG

            img.save(buffered, format=img_format)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            document_source = {"type": "image_url", "image_url": f"data:image/{img_format.lower()};base64,{img_str}"}
            return self._extract_text_from_source(document_source)
        except Exception as e:
            print(f"ERROR: Error preparing image for OCR: {e}")
            # raise RuntimeError(f"Error preparing image: {str(e)}") from e
            return None

    def _extract_text_from_source(self, document_source: Dict[str, str]) -> Optional[str]:
        """
        Internal method to process document source and extract text.
        Args:
            document_source: Dictionary containing document source information
        Returns:
            Extracted text or None if an error occurs.
        """
        ocr_response = self._process_ocr(document_source)
        if ocr_response and hasattr(ocr_response, 'pages') and ocr_response.pages:
            text = "\n\n".join(page.markdown for page in ocr_response.pages)
            return text.strip()
        else:
            print("Warning: OCR processing did not return any pages or text.")
            return "" # Return empty string if no text found, None if API error

# Create a single instance for the application
mistral_ocr_extractor = MistralOCRTextExtractor()

async def get_text_from_image(image_bytes: bytes) -> Optional[str]:
    """Service function to extract text using the singleton extractor."""
    return mistral_ocr_extractor.extract_text_from_image_bytes(image_bytes)

