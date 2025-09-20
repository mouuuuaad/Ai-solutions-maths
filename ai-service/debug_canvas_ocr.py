#!/usr/bin/env python3

import base64
import io
import cv2
import numpy as np
import pytesseract
from PIL import Image
import requests

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
    except:
        font = ImageFont.load_default()
    
    # Draw "2 + 1 =" clearly
    draw.text((150, 180), "2", fill='black', font=font)
    draw.text((250, 180), "+", fill='black', font=font)
    draw.text((350, 180), "1", fill='black', font=font)
    draw.text((450, 180), "=", fill='black', font=font)
    
    return img

def test_ocr_directly():
    """Test OCR directly on the image"""
    print("🔍 Testing OCR directly...")
    
    img = create_test_image()
    
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
    # Resize
    height, width = gray.shape
    if height < 400 or width < 400:
        scale_factor = max(400/height, 400/width)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to PIL
    pil_image = Image.fromarray(thresh)
    
    # Test different configurations
    configs = [
        '--oem 3 --psm 6',
        '--oem 3 --psm 8',
        '--oem 3 --psm 13',
        '--oem 3 --psm 7',
        '--oem 3 --psm 10',
    ]
    
    for i, config in enumerate(configs):
        try:
            text = pytesseract.image_to_string(pil_image, config=config)
            data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            print(f"Config {i+1} ({config}): '{text.strip()}' (confidence: {avg_confidence:.1f})")
        except Exception as e:
            print(f"Config {i+1} error: {e}")

def test_ai_service():
    """Test the AI service"""
    print("\n🤖 Testing AI Service...")
    
    img = create_test_image()
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    image_data = f"data:image/png;base64,{img_str}"
    
    response = requests.post(
        "http://localhost:8001/solve",
        json={"image": image_data},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ AI Service Result: '{result['input']}'")
        print(f"📊 Solution: {result['solution']}")
    else:
        print(f"❌ AI Service Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    from PIL import ImageDraw, ImageFont
    test_ocr_directly()
    test_ai_service()
