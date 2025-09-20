#!/usr/bin/env python3

import asyncio
from app.services.hybrid_math_ocr import HybridMathOCR

async def test_hybrid_ocr():
    """Test the hybrid math OCR service"""
    print("🧪 Testing Hybrid Math OCR Service")
    print("=" * 50)
    
    ocr_service = HybridMathOCR()
    
    # Test on equations
    print("\n📊 Testing on Equations...")
    test_results = ocr_service.test_on_equations()
    
    print(f"🎯 OCR Accuracy: {test_results['accuracy']:.1f}%")
    print(f"✅ Correct: {test_results['correct']}/{test_results['total']}")
    
    print("\n📝 Detailed Results:")
    for i, result in enumerate(test_results['results']):
        status = "✅" if result['correct'] else "❌"
        print(f"  {i+1:2d}. {status} '{result['equation']}' -> '{result['recognized']}' (confidence: {result['confidence']:.1f}%)")
    
    # Test on canvas data
    print("\n🎨 Testing on Canvas Data...")
    
    # Create test canvas data
    from PIL import Image, ImageDraw, ImageFont
    import base64
    import io
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        font = ImageFont.load_default()
    
    # Test equations
    test_equations = ["2x + 3 =", "1 + 1 =", "5 * 7 =", "10 - 4 ="]
    
    for equation in test_equations:
        # Create image
        img = Image.new('RGB', (500, 150), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), equation, fill='black', font=font)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Test OCR
        result = ocr_service.extract_text_from_canvas(img_str)
        print(f"  🎯 '{equation}' -> '{result}'")
    
    print("\n🎉 Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_hybrid_ocr())
