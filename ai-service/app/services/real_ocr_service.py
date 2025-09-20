import base64
import io
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

class RealOCRService:
    def __init__(self):
        # Configure Tesseract for better math recognition
        # PSM 6: Single uniform block of text
        # PSM 8: Single word
        # PSM 13: Raw line. Treat the image as a single text line
        self.tesseract_configs = [
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/=()[]{}^xXyYzZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789+-*/=()[]{}^xXyYzZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
            r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789+-*/=()[]{}^xXyYzZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        ]
    
    async def extract_text_from_image(self, image_data: str) -> str:
        """Extract text from base64 encoded image using real OCR"""
        try:
            # Decode base64 image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format for preprocessing
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image for better OCR
            processed_image = self._preprocess_image(opencv_image)
            
            # Convert back to PIL for Tesseract
            pil_image = Image.fromarray(processed_image)
            
            # Try multiple OCR configurations to get the best result
            best_text = ""
            best_confidence = 0
            
            for config in self.tesseract_configs:
                try:
                    # Get text and confidence
                    text = pytesseract.image_to_string(pil_image, config=config)
                    data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Calculate average confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    # Clean the text
                    cleaned_text = self._clean_math_text(text)
                    
                    # If this result is better, keep it
                    if avg_confidence > best_confidence and len(cleaned_text.strip()) > 0:
                        best_text = cleaned_text
                        best_confidence = avg_confidence
                        
                except Exception as e:
                    print(f"OCR config error: {e}")
                    continue
            
            # If no good result, try without character whitelist
            if not best_text or best_confidence < 30:
                try:
                    text = pytesseract.image_to_string(pil_image, config='--oem 3 --psm 6')
                    best_text = self._clean_math_text(text)
                except:
                    pass
            
            return best_text.strip()
            
        except Exception as e:
            print(f"Real OCR Error: {e}")
            return ""
    
    def _preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize image if too small (Tesseract works better with larger images)
        height, width = gray.shape
        if height < 200 or width < 200:
            scale_factor = max(200/height, 200/width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive threshold to get binary image
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert the image (Tesseract works better with black text on white background)
        thresh = cv2.bitwise_not(thresh)
        
        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        cleaned = cv2.medianBlur(cleaned, 3)
        
        # Dilate to make text thicker
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)
        
        return cleaned
    
    def _clean_math_text(self, text: str) -> str:
        """Clean and normalize mathematical text"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Common OCR corrections for math symbols
        corrections = {
            'O': '0',  # Letter O to number 0
            'l': '1',  # Letter l to number 1
            'I': '1',  # Letter I to number 1
            'S': '5',  # Letter S to number 5
            'B': '8',  # Letter B to number 8
            'G': '6',  # Letter G to number 6
            'Z': '2',  # Letter Z to number 2
            'o': '0',  # lowercase o to 0
            'i': '1',  # lowercase i to 1
            's': '5',  # lowercase s to 5
            'b': '6',  # lowercase b to 6
            'z': '2',  # lowercase z to 2
        }
        
        for old, new in corrections.items():
            text = text.replace(old, new)
        
        # Normalize mathematical operators
        text = text.replace('×', '*')
        text = text.replace('÷', '/')
        text = text.replace('^', '**')
        text = text.replace('=', ' = ')
        
        # Fix common OCR issues
        text = re.sub(r'(\d)\s*([a-zA-Z])', r'\1*\2', text)  # 2x -> 2*x
        text = re.sub(r'([a-zA-Z])\s*(\d)', r'\1*\2', text)  # x2 -> x*2
        text = re.sub(r'\)\s*(\d)', r')*\1', text)  # )2 -> )*2
        text = re.sub(r'(\d)\s*\(', r'\1*(', text)  # 2( -> 2*(
        
        # Remove extra spaces around operators
        text = re.sub(r'\s*([+\-*/=])\s*', r'\1', text)
        
        return text
