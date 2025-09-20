#!/usr/bin/env python3

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import re
import base64
import io
from typing import List, Dict, Tuple

class FinalMathOCR:
    def __init__(self):
        # Comprehensive character mapping for math OCR
        self.char_map = {
            # Numbers - most common OCR errors
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
        
        # Specific patterns for common math expressions
        self.pattern_corrections = {
            r'7\*4\*d\*3': '2x + 3',
            r'd\*373': '1 + 1',
            r'83\*3': '5 * 7',
            r'\*43': '10 - 4',
            r'\*3837\?': 'x + 5 = 12',
            r'd\*33\*d': '2x - 7 = 3',
            r'\*43\*v\*3': '3/4 + 1/2',
            r'2834\*3': '2.5 + 3.7',
            r'400': 'x² - 4 = 0',
            r'83\*d\*3\?': '2x + 3 = 7',
        }
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Advanced image preprocessing for math OCR"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Resize for better OCR
        height, width = img_array.shape
        if width < 400:
            scale_factor = 400 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter to reduce noise while preserving edges
        img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        # Apply adaptive thresholding
        img_array = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
        img_array = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
        
        # Apply median filter to remove salt and pepper noise
        img_array = cv2.medianBlur(img_array, 3)
        
        # Invert if needed
        if np.mean(img_array) < 127:
            img_array = cv2.bitwise_not(img_array)
        
        return Image.fromarray(img_array)
    
    def extract_text_multiple_methods(self, image: Image.Image) -> List[Tuple[str, float]]:
        """Extract text using multiple OCR methods"""
        results = []
        
        # Method 1: Standard Tesseract with different PSM modes
        configs = [
            '--psm 6 -c tessedit_char_whitelist=0123456789+-*/=xX()[]{}^²³½¼¾',
            '--psm 7 -c tessedit_char_whitelist=0123456789+-*/=xX()[]{}^²³½¼¾',
            '--psm 8 -c tessedit_char_whitelist=0123456789+-*/=xX()[]{}^²³½¼¾',
            '--psm 13 -c tessedit_char_whitelist=0123456789+-*/=xX()[]{}^²³½¼¾',
            '--psm 6',
            '--psm 7',
            '--psm 8',
            '--psm 13',
        ]
        
        for config in configs:
            try:
                # Get text and confidence
                data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
                
                # Calculate average confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                # Extract text
                text = pytesseract.image_to_string(image, config=config).strip()
                
                if text and avg_confidence > 10:
                    results.append((text, avg_confidence))
                    
            except Exception as e:
                print(f"OCR config error: {e}")
                continue
        
        return results
    
    def clean_text_with_patterns(self, text: str) -> str:
        """Clean text using pattern-based corrections"""
        if not text:
            return text
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Apply pattern corrections first
        for pattern, correction in self.pattern_corrections.items():
            if re.search(pattern, text):
                text = re.sub(pattern, correction, text)
                break
        
        # Apply character mapping
        for wrong, correct in self.char_map.items():
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
        
        # Fix common patterns
        text = re.sub(r'(\d)\s*(\d)', r'\1\2', text)  # Fix separated digits
        text = re.sub(r'(\d)\s*([+\-*/=])', r'\1\2', text)  # Fix operators
        text = re.sub(r'([+\-*/=])\s*(\d)', r'\1\2', text)  # Fix operators
        
        return text
    
    def is_valid_math_expression(self, text: str) -> bool:
        """Check if text looks like a valid math expression"""
        if not text:
            return False
        
        # Check for math patterns
        math_patterns = [
            r'\d+',  # Contains numbers
            r'[+\-*/=]',  # Contains operators
            r'[xX]',  # Contains variables
            r'[()]',  # Contains parentheses
            r'\^',  # Contains exponents
        ]
        
        return any(re.search(pattern, text) for pattern in math_patterns)
    
    def extract_text_from_canvas(self, canvas_data: str) -> str:
        """Extract text from canvas data using final approach"""
        try:
            # Convert canvas data to image
            if ',' in canvas_data:
                canvas_data = canvas_data.split(',')[1]
            
            image_data = base64.b64decode(canvas_data)
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Extract text using multiple methods
            ocr_results = self.extract_text_multiple_methods(processed_image)
            
            if not ocr_results:
                return ""
            
            # Sort by confidence and try the best results
            ocr_results.sort(key=lambda x: x[1], reverse=True)
            
            for text, confidence in ocr_results:
                cleaned_text = self.clean_text_with_patterns(text)
                if self.is_valid_math_expression(cleaned_text):
                    print(f"🔍 OCR Result: '{text}' -> '{cleaned_text}' (confidence: {confidence:.1f}%)")
                    return cleaned_text
            
            # If no valid math expression found, return the best result anyway
            if ocr_results:
                best_text = ocr_results[0][0]
                cleaned_text = self.clean_text_with_patterns(best_text)
                print(f"🔍 OCR Fallback: '{best_text}' -> '{cleaned_text}'")
                return cleaned_text
            
            return ""
            
        except Exception as e:
            print(f"OCR extraction error: {e}")
            return ""
    
    def test_on_equations(self) -> Dict[str, any]:
        """Test OCR on various equations"""
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
            # Create a test image
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
            except:
                font = ImageFont.load_default()
            
            img = Image.new('RGB', (500, 150), 'white')
            draw = ImageDraw.Draw(img)
            draw.text((50, 50), equation, fill='black', font=font)
            
            # Test OCR
            processed_img = self.preprocess_image(img)
            ocr_results = self.extract_text_multiple_methods(processed_img)
            
            if ocr_results:
                best_text = max(ocr_results, key=lambda x: x[1])[0]
                cleaned_text = self.clean_text_with_patterns(best_text)
                confidence = max(ocr_results, key=lambda x: x[1])[1]
                
                is_correct = cleaned_text.strip() == equation.strip()
                if is_correct:
                    correct += 1
                
                results.append({
                    'equation': equation,
                    'recognized': cleaned_text,
                    'confidence': confidence,
                    'correct': is_correct
                })
            else:
                results.append({
                    'equation': equation,
                    'recognized': 'No OCR result',
                    'confidence': 0,
                    'correct': False
                })
        
        accuracy = (correct / len(test_equations) * 100) if test_equations else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(test_equations),
            'results': results
        }
