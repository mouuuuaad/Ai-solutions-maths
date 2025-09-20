#!/usr/bin/env python3

import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import io

def create_simple_equation_image():
    """Create a test image that simulates a hand-drawn equation"""
    # Create a larger white image
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Use a font that looks more like handwriting
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
    except:
        font = ImageFont.load_default()
    
    # Draw "1 + 1" in a more handwritten style
    # Position each character separately to simulate handwriting
    draw.text((100, 150), "1", fill='black', font=font)
    draw.text((200, 150), "+", fill='black', font=font)
    draw.text((300, 150), "1", fill='black', font=font)
    draw.text((400, 150), "=", fill='black', font=font)
    draw.text((500, 150), "?", fill='black', font=font)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def test_real_ocr():
    """Test the OCR with a real equation"""
    print("🧪 Testing Real OCR with '1 + 1 = ?'")
    
    # Create test image
    image_data = create_simple_equation_image()
    
    # Send to AI service
    response = requests.post(
        "http://localhost:8001/solve",
        json={"image": image_data},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ OCR Test Results:")
        print(f"📝 Recognized: '{result['input']}'")
        print(f"🔧 Normalized: '{result['normalized']}'")
        print(f"📊 Solution: {result['solution']}")
        print(f"🎯 Confidence: {result['confidence']}")
        
        # Check if it actually read "1 + 1"
        if "1" in result['input'] and "+" in result['input']:
            print("✅ SUCCESS: OCR correctly read the equation!")
        else:
            print("❌ FAILED: OCR did not read the equation correctly")
            
    else:
        print(f"❌ Test Failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_real_ocr()
