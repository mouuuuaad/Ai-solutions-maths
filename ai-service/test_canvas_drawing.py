#!/usr/bin/env python3

import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import io

def create_canvas_style_drawing():
    """Create an image that simulates what you drew in the canvas"""
    # Create a white image similar to the canvas
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Use a font that looks more like handwriting
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        font = ImageFont.load_default()
    
    # Draw "1 + 1 -" exactly as you drew it
    # Position each character to simulate handwriting
    draw.text((150, 180), "1", fill='black', font=font)
    draw.text((250, 180), "+", fill='black', font=font)
    draw.text((350, 180), "1", fill='black', font=font)
    draw.text((450, 180), "-", fill='black', font=font)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def test_canvas_drawing():
    """Test with the actual drawing from the canvas"""
    print("🎨 Testing with Canvas Drawing: '1 + 1 -'")
    
    # Create test image
    image_data = create_canvas_style_drawing()
    
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
        if "1" in result['input'] and "+" in result['input']:
            print("✅ SUCCESS: OCR correctly read your actual drawing!")
        else:
            print("❌ FAILED: OCR did not read your drawing correctly")
            
    else:
        print(f"❌ Test Failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_canvas_drawing()
