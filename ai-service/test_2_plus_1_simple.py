#!/usr/bin/env python3

import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import io

def create_2_plus_1_simple():
    """Create an image that simulates '2 + 1 =' as drawn on canvas"""
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
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def test_ai_service_direct():
    """Test the AI service directly"""
    print("🤖 Testing AI Service directly with '2 + 1 ='")
    
    # Create test image
    image_data = create_2_plus_1_simple()
    
    # Send to AI service
    response = requests.post(
        "http://localhost:8001/solve",
        json={"image": image_data},
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ AI Service Test Results:")
        print(f"📝 Recognized: '{result['input']}'")
        print(f"🔧 Normalized: '{result['normalized']}'")
        print(f"📊 Solution: {result['solution']}")
        print(f"🎯 Confidence: {result['confidence']}")
    else:
        print(f"❌ AI Service Error: {response.status_code}")
        print(response.text)

def test_frontend_api():
    """Test the frontend API"""
    print("\n🌐 Testing Frontend API with '2 + 1 ='")
    
    # Create test image
    image_data = create_2_plus_1_simple()
    
    # Send to frontend API
    response = requests.post(
        "http://localhost:3000/api/solve",
        json={"image": image_data},
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Frontend API Test Results:")
        print(f"📝 Recognized: '{result['input']}'")
        print(f"🔧 Normalized: '{result['normalized']}'")
        print(f"📊 Solution: {result['solution']}")
        print(f"🎯 Confidence: {result['confidence']}")
    else:
        print(f"❌ Frontend API Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_ai_service_direct()
    test_frontend_api()
