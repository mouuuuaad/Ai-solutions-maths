#!/usr/bin/env python3

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import re
import base64
import io
from typing import List, Dict, Tuple

class SimpleNumberOCR:
    def __init__(self):
        self.number_corrections = {
            # Most common OCR errors for numbers
            'O': '0', 'o': '0', 'Q': '0', 'D': '0', 'U': '0', 'u': '0',
            'I': '1', 'l': '1', '|': '1', 'i': '1', 'j': '1', 'J': '1',
            'Z': '2', 'z': '2', 'S': '2', 's': '2',
            'E': '3', 'e': '3', 'B': '3', 'b': '3',
            'A': '4', 'a': '4', 'h': '4', 'H': '4',
            'S': '5', 's': '5', '$': '5',
            'G': '6', 'g': '6', 'b': '6',
            'T': '7', 't': '7', 'L': '7', 'l': '7',
            'B': '8', 'b': '8', 'g': '8',
            'g': '9', 'q': '9', 'p': '9',
            
            # Math symbols
            'r': 'x', 'R': 'X', 'n': 'x', 'N': 'X',
            'm': 'x', 'M': 'X', 'w': 'x', 'W': 'X',
            '—': '-', '–': '-', '−': '-', '_': '-',
            '×': '*', 'x': '*', 'X': '*',
            '÷': '/', ':': '/', '\\': '/',
            '=': '=', '≡': '=', '==': '=',
            '²': '^2', '³': '^3', '^': '^',
        }
    
    def preprocess_for_numbers(self, image: Image.Image) -> Image.Image:
        """Simple preprocessing focused on number detection"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Resize for better OCR
        height, width = img_array.shape
        if width < 300:
            scale_factor = 300 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply Gaussian blur to reduce noise
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        
        # Apply Otsu's thresholding
        _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed (dark text on light background)
        if np.mean(img_array) < 127:
            img_array = cv2.bitwise_not(img_array)
        
        return Image.fromarray(img_array)
    
    def extract_text_simple(self, image: Image.Image) -> str:
        """Simple text extraction with basic configs"""
        try:
            # Try different PSM modes
            configs = ['--psm 6', '--psm 7', '--psm 8', '--psm 13']
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image, config=config).strip()
                    if text and len(text) > 0:
                        return text
                except:
                    continue
            
            return ""
        except:
            return ""
    
    def clean_text_for_math(self, text: str) -> str:
        """Clean text specifically for math expressions"""
        if not text:
            return text
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Apply number corrections
        for wrong, correct in self.number_corrections.items():
            text = text.replace(wrong, correct)
        
        # Fix common math patterns
        text = re.sub(r'(\d)\s*([a-zA-Z])', r'\1*\2', text)  # 2x -> 2*x
        text = re.sub(r'([a-zA-Z])\s*(\d)', r'\1*\2', text)  # x2 -> x*2
        text = re.sub(r'\)\s*(\d)', r')*\1', text)  # )2 -> )*2
        text = re.sub(r'(\d)\s*\(', r'\1*(', text)  # 2( -> 2*(
        
        # Fix operators
        text = re.sub(r'\s*-\s*', '-', text)  # Fix minus signs
        text = re.sub(r'\s*\+\s*', '+', text)  # Fix plus signs
        text = re.sub(r'\s*\*\s*', '*', text)  # Fix multiplication
        text = re.sub(r'\s*/\s*', '/', text)  # Fix division
        text = re.sub(r'\s*=\s*', '=', text)  # Fix equals
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def extract_text_from_canvas(self, canvas_data: str) -> str:
        """Extract text from canvas data with focus on number detection"""
        try:
            # Convert canvas data to image
            if ',' in canvas_data:
                canvas_data = canvas_data.split(',')[1]
            
            image_data = base64.b64decode(canvas_data)
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocess image
            processed_image = self.preprocess_for_numbers(image)
            
            # Extract text
            text = self.extract_text_simple(processed_image)
            
            if text:
                cleaned_text = self.clean_text_for_math(text)
                print(f"🔍 OCR Result: '{text}' -> '{cleaned_text}'")
                return cleaned_text
            
            return ""
            
        except Exception as e:
            print(f"OCR extraction error: {e}")
            return ""
    
    def test_on_handwritten_style(self) -> Dict[str, any]:
        """Test OCR on handwritten-style equations"""
        test_equations = [
            "2x + 3 =",
            "1 + 1 =",
            "5 * 7 =",
            "10 - 4 =",
            "x + 5 = 12",
            "2x - 7 = 3",
            "3/4 + 1/2 =",
            "2.5 + 3.7 =",
            "x² - 4 = 0",
            "2x + 3 = 7"
        ]
        
        results = []
        correct = 0
        
        for equation in test_equations:
            # Create a handwritten-style image
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
            except:
                font = ImageFont.load_default()
            
            # Create image with some variation to simulate handwriting
            img = Image.new('RGB', (500, 150), 'white')
            draw = ImageDraw.Draw(img)
            
            # Add some random offset to simulate handwriting
            import random
            x_offset = random.randint(-5, 5)
            y_offset = random.randint(-5, 5)
            
            draw.text((50 + x_offset, 50 + y_offset), equation, fill='black', font=font)
            
            # Test OCR
            processed_img = self.preprocess_for_numbers(img)
            text = self.extract_text_simple(processed_img)
            cleaned_text = self.clean_text_for_math(text)
            
            is_correct = cleaned_text.strip() == equation.strip()
            if is_correct:
                correct += 1
            
            results.append({
                'equation': equation,
                'recognized': cleaned_text,
                'correct': is_correct
            })
        
        accuracy = (correct / len(test_equations) * 100) if test_equations else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(test_equations),
            'results': results
        }
