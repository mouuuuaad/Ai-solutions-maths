#!/usr/bin/env python3

import asyncio
import json
from training_data_generator import TrainingDataGenerator
from app.services.trained_ocr_service import TrainedOCRService

async def train_and_test_ocr():
    """Train and test the OCR system with 100 equations"""
    print("🎯 AI Math Solver OCR Training System")
    print("=" * 50)
    
    # Step 1: Generate training data
    print("\n📊 Step 1: Generating 100 Equation Training Dataset...")
    generator = TrainingDataGenerator()
    training_data = generator.generate_training_images()
    generator.save_training_metadata(training_data)
    
    print(f"✅ Generated {len(training_data)} training images")
    print(f"📁 Images saved to: training_images/")
    
    # Step 2: Test OCR accuracy
    print("\n🧪 Step 2: Testing OCR Accuracy...")
    ocr_service = TrainedOCRService()
    
    # Test on training data
    test_results = ocr_service.test_on_training_data()
    
    if 'error' in test_results:
        print(f"❌ Error: {test_results['error']}")
        return
    
    print(f"📈 OCR Accuracy: {test_results['accuracy']:.1f}%")
    print(f"✅ Correct: {test_results['correct']}/{test_results['total']}")
    
    # Show some examples
    print("\n📝 Sample Results:")
    for i, result in enumerate(test_results['results'][:10]):
        status = "✅" if result['correct'] else "❌"
        print(f"  {i+1:2d}. {status} Expected: '{result['expected']}' | Got: '{result['recognized']}'")
    
    # Step 3: Test on real equations
    print("\n🔬 Step 3: Testing on Real Equations...")
    
    test_equations = [
        "2x + 3 =",
        "1 + 1 =",
        "5 * 7 =",
        "x² + 2x + 1 = 0",
        "3/4 + 1/2 =",
        "2.5 + 3.7 =",
        "10 - 4 =",
        "x + 5 = 12",
        "2x - 7 = 3",
        "x² - 4 = 0"
    ]
    
    print("Testing on real equations:")
    for equation in test_equations:
        # Create a simple test image
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        except:
            font = ImageFont.load_default()
        
        img = Image.new('RGB', (400, 100), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), equation, fill='black', font=font)
        
        # Test OCR
        processed_img = ocr_service.preprocess_image(img)
        ocr_results = ocr_service.extract_text_with_multiple_configs(processed_img)
        
        if ocr_results:
            best_text = max(ocr_results, key=lambda x: x[1])[0]
            cleaned_text = ocr_service.clean_math_text(best_text)
            confidence = max(ocr_results, key=lambda x: x[1])[1]
            
            status = "✅" if cleaned_text.strip() == equation.strip() else "❌"
            print(f"  {status} '{equation}' -> '{cleaned_text}' (confidence: {confidence:.1f}%)")
        else:
            print(f"  ❌ '{equation}' -> No OCR result")
    
    # Step 4: Save training results
    print("\n💾 Step 4: Saving Training Results...")
    
    training_results = {
        'training_data_count': len(training_data),
        'test_accuracy': test_results['accuracy'],
        'test_correct': test_results['correct'],
        'test_total': test_results['total'],
        'number_corrections': len(ocr_service.number_corrections),
        'tesseract_configs': len(ocr_service.tesseract_configs)
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print(f"✅ Training results saved to training_results.json")
    
    # Step 5: Summary
    print("\n🎉 Training Complete!")
    print("=" * 50)
    print(f"📊 Training Dataset: {len(training_data)} images")
    print(f"🎯 OCR Accuracy: {test_results['accuracy']:.1f}%")
    print(f"🔧 Number Corrections: {len(ocr_service.number_corrections)}")
    print(f"⚙️  Tesseract Configs: {len(ocr_service.tesseract_configs)}")
    
    if test_results['accuracy'] > 80:
        print("🎉 Excellent! OCR is performing well.")
    elif test_results['accuracy'] > 60:
        print("👍 Good! OCR is performing reasonably well.")
    else:
        print("⚠️  OCR needs improvement. Consider more training data.")

if __name__ == "__main__":
    asyncio.run(train_and_test_ocr())
