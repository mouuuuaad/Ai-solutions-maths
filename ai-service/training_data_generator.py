#!/usr/bin/env python3

import os
import random
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class TrainingDataGenerator:
    def __init__(self):
        self.equations = []
        self.fonts = []
        self.setup_fonts()
        self.generate_equations()
    
    def setup_fonts(self):
        """Setup different fonts for variety"""
        try:
            # Try to use system fonts
            self.fonts = [
                ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48),
                ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 48),
                ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 48),
            ]
        except:
            # Fallback to default font
            self.fonts = [ImageFont.load_default()]
    
    def generate_equations(self):
        """Generate 100 diverse mathematical equations"""
        equations = []
        
        # Simple arithmetic (20 equations)
        for i in range(20):
            a = random.randint(1, 99)
            b = random.randint(1, 99)
            op = random.choice(['+', '-', '*', '/'])
            if op == '/':
                # Ensure clean division
                b = random.randint(2, 12)
                a = b * random.randint(1, 10)
            equations.append(f"{a} {op} {b} =")
        
        # Linear equations (30 equations)
        for i in range(30):
            a = random.randint(1, 9)
            b = random.randint(1, 99)
            c = random.randint(1, 99)
            equations.append(f"{a}x + {b} = {c}")
        
        # Quadratic equations (15 equations)
        for i in range(15):
            a = random.randint(1, 5)
            b = random.randint(1, 20)
            c = random.randint(1, 20)
            equations.append(f"{a}x² + {b}x + {c} = 0")
        
        # Fractions (15 equations)
        for i in range(15):
            num1 = random.randint(1, 20)
            den1 = random.randint(2, 10)
            num2 = random.randint(1, 20)
            den2 = random.randint(2, 10)
            op = random.choice(['+', '-', '*', '/'])
            equations.append(f"{num1}/{den1} {op} {num2}/{den2} =")
        
        # Decimals (10 equations)
        for i in range(10):
            a = round(random.uniform(0.1, 99.9), 1)
            b = round(random.uniform(0.1, 99.9), 1)
            op = random.choice(['+', '-', '*', '/'])
            equations.append(f"{a} {op} {b} =")
        
        # Mixed numbers (10 equations)
        for i in range(10):
            whole = random.randint(1, 10)
            num = random.randint(1, 9)
            den = random.randint(2, 10)
            equations.append(f"{whole} {num}/{den} + 2 =")
        
        self.equations = equations[:100]  # Ensure exactly 100
    
    def create_equation_image(self, equation, font, noise_level=0.1):
        """Create an image of the equation with realistic handwriting simulation"""
        # Create image with some padding
        width, height = 800, 200
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Get text size
        bbox = draw.textbbox((0, 0), equation, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Add some random offset to simulate handwriting
        x += random.randint(-10, 10)
        y += random.randint(-5, 5)
        
        # Draw the text
        draw.text((x, y), equation, fill='black', font=font)
        
        # Convert to numpy for processing
        img_array = np.array(img)
        
        # Add noise to simulate real handwriting
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Add some rotation to simulate handwriting angle
        angle = random.uniform(-2, 2)  # Small rotation
        if angle != 0:
            img = Image.fromarray(img_array)
            img = img.rotate(angle, fillcolor='white')
            img_array = np.array(img)
        
        return Image.fromarray(img_array)
    
    def generate_training_images(self, output_dir="training_images"):
        """Generate training images for all equations"""
        os.makedirs(output_dir, exist_ok=True)
        
        training_data = []
        
        for i, equation in enumerate(self.equations):
            # Generate multiple variations of each equation
            for variation in range(3):  # 3 variations per equation
                font = random.choice(self.fonts)
                noise_level = random.uniform(0.05, 0.2)
                
                img = self.create_equation_image(equation, font, noise_level)
                
                filename = f"equation_{i:03d}_var_{variation}.png"
                filepath = os.path.join(output_dir, filename)
                img.save(filepath)
                
                training_data.append({
                    'filename': filename,
                    'equation': equation,
                    'expected': equation,
                    'variation': variation
                })
        
        return training_data
    
    def save_training_metadata(self, training_data, output_file="training_metadata.json"):
        """Save training metadata"""
        import json
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"✅ Saved training metadata to {output_file}")

def main():
    print("🎯 Generating 100 Equation Training Dataset...")
    
    generator = TrainingDataGenerator()
    print(f"📊 Generated {len(generator.equations)} equations")
    
    # Show some examples
    print("\n📝 Sample equations:")
    for i, eq in enumerate(generator.equations[:10]):
        print(f"  {i+1:2d}. {eq}")
    
    print("\n🖼️  Generating training images...")
    training_data = generator.generate_training_images()
    
    print(f"✅ Generated {len(training_data)} training images")
    print(f"📁 Images saved to: training_images/")
    
    # Save metadata
    generator.save_training_metadata(training_data)
    
    print("\n🎉 Training dataset ready!")

if __name__ == "__main__":
    main()
