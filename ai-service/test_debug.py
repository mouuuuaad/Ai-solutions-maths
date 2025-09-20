#!/usr/bin/env python3

import asyncio
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from app.services.simple_canvas_ocr import SimpleCanvasOCR

async def test_canvas_ocr():
    """Test the canvas OCR service directly"""
    print("🔍 Testing Canvas OCR Service directly...")
    
    # Create test image
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
    image_data = f"data:image/png;base64,{img_str}"
    
    # Test the OCR service
    ocr_service = SimpleCanvasOCR()
    result = await ocr_service.extract_text_from_image(image_data)
    
    print(f"✅ Canvas OCR Result: '{result}'")

if __name__ == "__main__":
    asyncio.run(test_canvas_ocr())
