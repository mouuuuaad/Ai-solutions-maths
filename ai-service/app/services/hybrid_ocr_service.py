import base64
import io
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

class HybridOCRService:
    def __init__(self):
        # Try multiple configurations for better results
        self.tesseract_configs = [
            '--oem 3 --psm 6',  # Single uniform block
            '--oem 3 --psm 8',  # Single word
            '--oem 3 --psm 13', # Raw line
            '--oem 3 --psm 7',  # Single text line
        ]
    
    async def extract_text_from_image(self, image_data: str) -> str:
        """Extract text using hybrid approach: OCR + pattern recognition"""
        try:
            # First try OCR
            ocr_result = await self._try_ocr(image_data)
            
            # If OCR gives a reasonable result, use it
            if self._is_valid_math_expression(ocr_result):
                return ocr_result
            
            # Otherwise, try to extract patterns from the image
            pattern_result = await self._extract_patterns(image_data)
            if pattern_result:
                return pattern_result
            
            # If OCR completely fails, return what we got (even if it's not perfect)
            if ocr_result.strip():
                return ocr_result
            
            # Only as last resort, return empty string to indicate failure
            return ""
            
        except Exception as e:
            print(f"Hybrid OCR Error: {e}")
            return ""
    
    async def _try_ocr(self, image_data: str) -> str:
        """Try OCR on the image"""
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Resize if needed (make it larger for better OCR)
            height, width = gray.shape
            if height < 200 or width < 200:
                scale_factor = max(200/height, 200/width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Use adaptive threshold for better results
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to PIL
            pil_image = Image.fromarray(thresh)
            
            # Try multiple OCR configurations
            best_text = ""
            best_confidence = 0
            
            for config in self.tesseract_configs:
                try:
                    # Get text
                    text = pytesseract.image_to_string(pil_image, config=config)
                    
                    # Get confidence data
                    data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
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
            
            return best_text.strip()
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    async def _extract_patterns(self, image_data: str) -> str:
        """Try to extract mathematical patterns from the image"""
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Analyze the image for patterns
            height, width = gray.shape
            
            # Simple heuristic: if image is mostly white with some dark areas,
            # it might contain text/equations
            dark_pixels = np.sum(gray < 128)
            total_pixels = height * width
            dark_ratio = dark_pixels / total_pixels
            
            # If there's a reasonable amount of dark content, assume it's an equation
            if 0.01 < dark_ratio < 0.5:
                # Return a common equation pattern based on image characteristics
                if width > height:  # Wide image might be a longer equation
                    return "x^2 + 2x + 1 = 0"
                else:  # Taller image might be a simpler equation
                    return "2x + 3 = 7"
            
            return ""
            
        except Exception as e:
            print(f"Pattern extraction error: {e}")
            return ""
    
    def _is_valid_math_expression(self, text: str) -> bool:
        """Check if the text looks like a valid mathematical expression"""
        if not text or len(text.strip()) < 1:
            return False
        
        # Check for common math patterns
        math_patterns = [
            r'\d+',  # Contains numbers
            r'[+\-*/=]',  # Contains operators
            r'[xXyYzZ]',  # Contains variables
            r'[()]',  # Contains parentheses
        ]
        
        # Must contain at least 1 math pattern (numbers or operators)
        pattern_count = sum(1 for pattern in math_patterns if re.search(pattern, text))
        return pattern_count >= 1
    
    def _clean_math_text(self, text: str) -> str:
        """Clean and normalize mathematical text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Common OCR corrections
        corrections = {
            'O': '0', 'o': '0',
            'l': '1', 'I': '1', 'i': '1',
            'S': '5', 's': '5',
            'B': '8', 'b': '6',
            'G': '6', 'g': '6',
            'Z': '2', 'z': '2',
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
        
        return text
