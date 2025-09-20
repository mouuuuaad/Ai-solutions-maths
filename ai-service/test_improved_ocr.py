#!/usr/bin/env python3

import asyncio
from improved_ocr_service import ImprovedOCRService

async def test_improved_ocr():
    """Test the improved OCR service"""
    print("🧪 Testing Improved OCR Service")
    print("=" * 40)
    
    ocr_service = ImprovedOCRService()
    
    # Test on simple equations
    print("\n📊 Testing on Simple Equations...")
    test_results = ocr_service.test_on_simple_equations()
    
    print(f"🎯 OCR Accuracy: {test_results['accuracy']:.1f}%")
    print(f"✅ Correct: {test_results['correct']}/{test_results['total']}")
    
    print("\n📝 Detailed Results:")
    for i, result in enumerate(test_results['results']):
        status = "✅" if result['correct'] else "❌"
        print(f"  {i+1:2d}. {status} '{result['equation']}' -> '{result['recognized']}' (confidence: {result['confidence']:.1f}%)")
    
    # Test on canvas data
    print("\n🎨 Testing on Canvas Data...")
    
    # Create a simple test canvas data
    from PIL import Image, ImageDraw, ImageFont
    import base64
    import io
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        font = ImageFont.load_default()
    
    # Test equations
    test_equations = ["2x + 3 =", "1 + 1 =", "5 * 7 ="]
    
    for equation in test_equations:
        # Create image
        img = Image.new('RGB', (400, 100), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), equation, fill='black', font=font)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Test OCR
        result = ocr_service.extract_text_from_canvas(img_str)
        print(f"  🎯 '{equation}' -> '{result}'")
    
    print("\n🎉 Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_improved_ocr())
