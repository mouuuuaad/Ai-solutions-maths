#!/usr/bin/env python3

import asyncio
import requests
import json

async def test_text_input_system():
    """Test the new text input system"""
    print("🧪 Testing Text Input Math Solver")
    print("=" * 50)
    
    # Test equations
    test_equations = [
        "2x + 3 = 7",
        "x² - 4 = 0",
        "3x + 2 = 14",
        "x + 5 = 12",
        "2x - 7 = 3",
        "x² + 2x + 1 = 0",
        "3/4 + 1/2 = ?",
        "2.5 + 3.7 = ?",
        "2x + 3 =",  # Incomplete equation
        "x² - 4 = 0"  # Quadratic
    ]
    
    # Test AI service directly
    print("\n🤖 Testing AI Service Directly...")
    ai_service_url = "http://localhost:8001"
    
    for i, equation in enumerate(test_equations):
        try:
            response = requests.post(
                f"{ai_service_url}/solve",
                json={"equation_text": equation},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {i+1:2d}. '{equation}' -> '{data['solution'][0] if data['solution'] else 'No solution'}'")
            else:
                print(f"❌ {i+1:2d}. '{equation}' -> Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {i+1:2d}. '{equation}' -> Exception: {e}")
    
    # Test frontend API
    print("\n🌐 Testing Frontend API...")
    frontend_url = "http://localhost:3000"
    
    for i, equation in enumerate(test_equations[:5]):  # Test first 5
        try:
            response = requests.post(
                f"{frontend_url}/api/solve",
                json={"equation_text": equation},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {i+1:2d}. '{equation}' -> '{data['solution'][0] if data['solution'] else 'No solution'}'")
            else:
                print(f"❌ {i+1:2d}. '{equation}' -> Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {i+1:2d}. '{equation}' -> Exception: {e}")
    
    print("\n🎉 Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_text_input_system())
