#!/usr/bin/env python3

import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import io

def create_2x_plus_3_image():
    """Create an image that simulates '2x + 3 =' as drawn on canvas"""
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
    except:
        font = ImageFont.load_default()
    
    # Draw "2x + 3 =" clearly
    draw.text((100, 180), "2", fill='black', font=font)
    draw.text((150, 180), "x", fill='black', font=font)
    draw.text((200, 180), "+", fill='black', font=font)
    draw.text((250, 180), "3", fill='black', font=font)
    draw.text((300, 180), "=", fill='black', font=font)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def test_2x_plus_3():
    """Test with the equation '2x + 3 ='"""
    print("🧮 Testing with Equation: '2x + 3 ='")
    
    # Create test image
    image_data = create_2x_plus_3_image()
    
    # Send to AI service
    response = requests.post(
        "http://localhost:8001/solve",
        json={"image": image_data},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ AI Service Test Results:")
        print(f"📝 Recognized: '{result['input']}'")
        print(f"🔧 Normalized: '{result['normalized']}'")
        print(f"📊 Solution: {result['solution']}")
        print(f"🎯 Confidence: {result['confidence']}")
        
        # Check if it read the equation correctly
        if "2" in result['input'] and "x" in result['input'] and "3" in result['input']:
            print("✅ SUCCESS: OCR correctly read '2x + 3 ='!")
        else:
            print("❌ FAILED: OCR did not read '2x + 3 =' correctly")
            print(f"Expected: '2x + 3 ='")
            print(f"Got: '{result['input']}'")
            
    else:
        print(f"❌ Test Failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_2x_plus_3()
