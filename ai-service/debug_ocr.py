#!/usr/bin/env python3

import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
import numpy as np
import pytesseract

def create_better_test_image():
    """Create a better test image with clearer text"""
    # Create a larger white image
    img = Image.new('RGB', (600, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Use a larger, clearer font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        font = ImageFont.load_default()
    
    # Draw text with more spacing
    text = "2x + 3 = 7"
    draw.text((100, 120), text, fill='black', font=font)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def test_tesseract_directly():
    """Test Tesseract directly on the image"""
    # Create test image
    image_data = create_better_test_image()
    
    # Decode image
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Test different configurations
    configs = [
        '--oem 3 --psm 6',
        '--oem 3 --psm 8', 
        '--oem 3 --psm 13',
        '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/=()[]{}^xXyYzZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ]
    
    print("Testing Tesseract directly:")
    for i, config in enumerate(configs):
        try:
            text = pytesseract.image_to_string(gray, config=config)
            data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            print(f"Config {i+1}: '{text.strip()}' (confidence: {avg_confidence:.1f})")
        except Exception as e:
            print(f"Config {i+1}: Error - {e}")

def test_ai_service():
    """Test the AI service"""
    image_data = create_better_test_image()
    
    response = requests.post(
        "http://localhost:8001/solve",
        json={"image": image_data},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ AI Service Test:")
        print(f"Recognized: '{result['input']}'")
        print(f"Normalized: '{result['normalized']}'")
        print(f"Solution: {result['solution']}")
        print(f"Confidence: {result['confidence']}")
    else:
        print(f"❌ AI Service Test Failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_tesseract_directly()
    test_ai_service()
