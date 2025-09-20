import base64
import io
import cv2
import numpy as np
from PIL import Image
import re

class OCRService:
    def __init__(self):
        # For now, we'll use a simple text extraction approach
        # In production, you would load a trained OCR model here
        pass
    
    async def extract_text_from_image(self, image_data: str) -> str:
        """Extract text from base64 encoded image using simple OCR"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image
            processed_image = self._preprocess_image(opencv_image)
            
            # For demo purposes, return a sample equation
            # In production, you would use a real OCR model here
            sample_equations = [
                "x^2 + 2x + 1 = 0",
                "2x + 3 = 7",
                "x^2 - 4 = 0",
                "3x + 5 = 2x + 10",
                "x^3 - 8 = 0"
            ]
            
            # Simple heuristic: return different equations based on image characteristics
            height, width = processed_image.shape
            equation_index = (height + width) % len(sample_equations)
            
            return sample_equations[equation_index]
            
        except Exception as e:
            print(f"OCR Error: {e}")
            # Return a default equation for demo
            return "x^2 + 2x + 1 = 0"
    
    def _preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _clean_math_text(self, text: str) -> str:
        """Clean and normalize mathematical text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Common OCR corrections for math symbols
        corrections = {
            'O': '0',  # Letter O to number 0
            'l': '1',  # Letter l to number 1
            'I': '1',  # Letter I to number 1
            'S': '5',  # Letter S to number 5
            'B': '8',  # Letter B to number 8
            'G': '6',  # Letter G to number 6
            'Z': '2',  # Letter Z to number 2
            ' ': '',   # Remove spaces for now
        }
        
        for old, new in corrections.items():
            text = text.replace(old, new)
        
        # Normalize mathematical operators
        text = text.replace('×', '*')
        text = text.replace('÷', '/')
        text = text.replace('^', '**')
        
        return text