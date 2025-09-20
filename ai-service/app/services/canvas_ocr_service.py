import base64
import io
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

class CanvasOCRService:
    def __init__(self):
        # Multiple OCR configurations to try
        self.tesseract_configs = [
            '--oem 3 --psm 6',  # Single uniform block
            '--oem 3 --psm 8',  # Single word
            '--oem 3 --psm 13', # Raw line
            '--oem 3 --psm 7',  # Single text line
            '--oem 3 --psm 10', # Single character
        ]
    
    async def extract_text_from_image(self, image_data: str) -> str:
        """Extract text from canvas drawing using real OCR"""
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
            
            # Resize to make it larger for better OCR
            height, width = gray.shape
            print(f"📏 Original size: {width}x{height}")
            
            if height < 400 or width < 400:
                scale_factor = max(400/height, 400/width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                print(f"📏 Scaled size: {new_width}x{new_height}")
            
            # Apply noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # Apply adaptive threshold for better results
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to PIL
            pil_image = Image.fromarray(thresh)
            
            # Try multiple OCR configurations
            best_text = ""
            best_confidence = 0
            
            for i, config in enumerate(self.tesseract_configs):
                try:
                    print(f"🔧 Trying OCR config {i+1}: {config}")
                    
                    # Get text
                    text = pytesseract.image_to_string(pil_image, config=config)
                    
                    # Get confidence data
                    data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    print(f"📝 Config {i+1} result: '{text.strip()}' (confidence: {avg_confidence:.1f})")
                    
                    # Clean the text
                    cleaned_text = self._clean_math_text(text)
                    
                    # If this result is better, keep it
                    if avg_confidence > best_confidence and len(cleaned_text.strip()) > 0:
                        best_text = cleaned_text
                        best_confidence = avg_confidence
                        print(f"✅ New best result: '{best_text}' (confidence: {best_confidence:.1f})")
                        
                except Exception as e:
                    print(f"❌ OCR config {i+1} error: {e}")
                    continue
            
            print(f"🎯 Final OCR result: '{best_text}' (confidence: {best_confidence:.1f})")
            
            return best_text.strip()
            
        except Exception as e:
            print(f"❌ Canvas OCR Error: {e}")
            return ""
    
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
            '.': '',  # Remove dots that OCR might add
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
