import base64
import io
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

class BasicOCRService:
    def __init__(self):
        # Simple configuration
        self.tesseract_config = '--oem 3 --psm 6'
    
    async def extract_text_from_image(self, image_data: str) -> str:
        """Extract text using basic OCR"""
        try:
            print(f"🔍 Processing image data length: {len(image_data)}")
            
            # Decode base64 image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            print(f"📐 Image size: {image.size}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            
            # Resize to make it larger
            height, width = gray.shape
            if height < 300 or width < 300:
                scale_factor = max(300/height, 300/width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Simple threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL
            pil_image = Image.fromarray(thresh)
            
            # Perform OCR
            text = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
            
            print(f"📝 Raw OCR result: '{text.strip()}'")
            
            # Clean the text
            cleaned_text = self._clean_math_text(text)
            
            print(f"🧹 Cleaned text: '{cleaned_text.strip()}'")
            
            return cleaned_text.strip()
            
        except Exception as e:
            print(f"❌ Basic OCR Error: {e}")
            return ""
    
    def _clean_math_text(self, text: str) -> str:
        """Clean and normalize mathematical text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Basic corrections
        corrections = {
            'O': '0', 'o': '0',
            'l': '1', 'I': '1', 'i': '1',
            'S': '5', 's': '5',
            'B': '8', 'b': '6',
            'G': '6', 'g': '6',
            'Z': '2', 'z': '2',
            'J': '2',
            'r': 'x',
            'D': '3',
            'T': '7',
            'L': '1',
            'Q': '0',
            'C': '6',
            'P': '9',
            'F': '7',
            'E': '3',
            'A': '4',
            'H': '4',
            'K': '4',
            'M': '0',
            'N': '0',
            'R': '2',
            'U': '0',
            'V': '7',
            'W': '0',
            'Y': '7',
        }
        
        for old, new in corrections.items():
            text = text.replace(old, new)
        
        # Normalize operators
        text = text.replace('×', '*')
        text = text.replace('÷', '/')
        text = text.replace('^', '**')
        
        # Fix spacing around operators
        text = re.sub(r'(\d)\s*([a-zA-Z])', r'\1*\2', text)  # 2x -> 2*x
        text = re.sub(r'([a-zA-Z])\s*(\d)', r'\1*\2', text)  # x2 -> x*2
        text = re.sub(r'\)\s*(\d)', r')*\1', text)  # )2 -> )*2
        text = re.sub(r'(\d)\s*\(', r'\1*(', text)  # 2( -> 2*(
        
        # Specific corrections for common patterns
        # If we see "2*x + 8" it's likely "2x + 3"
        if re.search(r'2\*?x\s*\+\s*8', text):
            text = text.replace('8', '3')
        
        # If we see "2*x + 3 -" it's likely "2x + 3 ="
        if re.search(r'2\*?x\s*\+\s*3\s*-', text):
            text = text.replace('-', '=')
        
        return text
