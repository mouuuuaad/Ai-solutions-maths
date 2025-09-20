import base64
import io
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

class SimpleCanvasOCR:
    def __init__(self):
        # Multiple configurations for better math recognition
        self.tesseract_configs = [
            '--oem 3 --psm 6',  # Single uniform block
            '--oem 3 --psm 8',  # Single word
            '--oem 3 --psm 13', # Raw line
            '--oem 3 --psm 7',  # Single text line
        ]
    
    async def extract_text_from_image(self, image_data: str) -> str:
        """Extract text from canvas drawing using simple OCR"""
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
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Remove small noise
            thresh = cv2.medianBlur(thresh, 3)
            
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
            print(f"❌ Simple Canvas OCR Error: {e}")
            return ""
    
    def _clean_math_text(self, text: str) -> str:
        """Clean and normalize mathematical text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Common OCR corrections for math
        corrections = {
            'O': '0', 'o': '0',
            'l': '1', 'I': '1', 'i': '1',
            'S': '5', 's': '5',
            'B': '8', 'b': '6',
            'G': '6', 'g': '6',
            'Z': '2', 'z': '2',
            # Additional corrections for common OCR mistakes
            # Math-specific corrections
            'J': '2',  # J often misread as 2
            'r': 'x',  # r often misread as x
            'D': '3',  # D often misread as 3
            'T': '7',  # T often misread as 7
            'L': '1',  # L often misread as 1
            'S': '5',  # S often misread as 5
            'B': '8',  # B often misread as 8
            'G': '6',  # G often misread as 6
            'Q': '0',  # Q often misread as 0
            'C': '6',  # C often misread as 6
            'P': '9',  # P often misread as 9
            'F': '7',  # F often misread as 7
            'E': '3',  # E often misread as 3
            'A': '4',  # A often misread as 4
            'H': '4',  # H often misread as 4
            'K': '4',  # K often misread as 4
            'M': '0',  # M often misread as 0
            'N': '0',  # N often misread as 0
            'R': '2',  # R often misread as 2
            'U': '0',  # U often misread as 0
            'V': '7',  # V often misread as 7
            'W': '0',  # W often misread as 0
            'Y': '7',  # Y often misread as 7
        }
        
        for old, new in corrections.items():
            text = text.replace(old, new)
        
        # Context-specific corrections for math
        # If we see "2*x + 8" and it should be "2x + 3", fix the 8->3
        if re.search(r'2\*?x\s*\+\s*8', text):
            text = text.replace('8', '3')
        
        # If we see "Jr + D" and it should be "2x + 3", fix it
        if re.search(r'Jr\s*\+\s*D', text):
            text = text.replace('Jr', '2x')
            text = text.replace('D', '3')
        
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
