#!/usr/bin/env python3

import asyncio
from app.services.math_solver import MathSolverService

async def test_equation_solving():
    """Test equation solving directly"""
    print("🧮 Testing equation solving directly...")
    
    math_solver = MathSolverService()
    
    # Test the equation "2*x+3 = 0"
    equation = "2*x+3 = 0"
    print(f"Testing equation: '{equation}'")
    
    try:
        result = await math_solver.solve_equation(equation)
        print("✅ Math Solver Result:")
        print(f"Normalized: {result['normalized']}")
        print(f"Steps: {result['steps']}")
        print(f"Solution: {result['solution']}")
        print(f"Confidence: {result['confidence']}")
    except Exception as e:
        print(f"❌ Math Solver Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_equation_solving())
