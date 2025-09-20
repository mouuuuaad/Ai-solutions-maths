#!/usr/bin/env python3

import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import io

def create_2_plus_1_image():
    """Create an image that simulates '2 + 1 =' as drawn on canvas"""
    # Create a white image similar to the canvas
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Use a font that looks more like handwriting
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
    except:
        font = ImageFont.load_default()
    
    # Draw "2 + 1 =" exactly as you drew it
    # Position each character to simulate handwriting
    draw.text((150, 180), "2", fill='black', font=font)
    draw.text((250, 180), "+", fill='black', font=font)
    draw.text((350, 180), "1", fill='black', font=font)
    draw.text((450, 180), "=", fill='black', font=font)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def test_2_plus_1():
    """Test with the actual drawing '2 + 1 ='"""
    print("🎨 Testing with Canvas Drawing: '2 + 1 ='")
    
    # Create test image
    image_data = create_2_plus_1_image()
    
    # Send to AI service
    response = requests.post(
        "http://localhost:8001/solve",
        json={"image": image_data},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Canvas Drawing Test Results:")
        print(f"📝 Recognized: '{result['input']}'")
        print(f"🔧 Normalized: '{result['normalized']}'")
        print(f"📊 Solution: {result['solution']}")
        print(f"🎯 Confidence: {result['confidence']}")
        
        # Check if it read the actual drawing
        if "2" in result['input'] and "+" in result['input'] and "1" in result['input']:
            print("✅ SUCCESS: OCR correctly read '2 + 1 ='!")
        else:
            print("❌ FAILED: OCR did not read '2 + 1 =' correctly")
            print(f"Expected: '2 + 1 ='")
            print(f"Got: '{result['input']}'")
            
    else:
        print(f"❌ Test Failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_2_plus_1()
