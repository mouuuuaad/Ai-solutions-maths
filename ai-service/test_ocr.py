#!/usr/bin/env python3

import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import io

def create_test_image():
    """Create a test image with handwritten-looking text"""
    # Create a white image
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a font that looks more handwritten
    try:
        # Try to use a system font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Draw some text that looks like a math equation
    text = "2x + 3 = 7"
    draw.text((50, 80), text, fill='black', font=font)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def test_ocr():
    """Test the OCR service"""
    # Create test image
    image_data = create_test_image()
    
    # Send to AI service
    response = requests.post(
        "http://localhost:8001/solve",
        json={"image": image_data},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ OCR Test Successful!")
        print(f"Recognized: {result['input']}")
        print(f"Normalized: {result['normalized']}")
        print(f"Solution: {result['solution']}")
        print(f"Confidence: {result['confidence']}")
    else:
        print(f"❌ OCR Test Failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_ocr()
