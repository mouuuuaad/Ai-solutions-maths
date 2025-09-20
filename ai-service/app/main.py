from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.models.schemas import SolveRequest, SolveResponse
# OCR removed - using direct text input instead
from app.services.ai_math_solver import AIMathSolverService
from app.services.gemini_math_solver import GeminiMathSolverService

app = FastAPI(title="AI Math Solver Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI services
ai_math_solver = AIMathSolverService()
gemini_math_solver = GeminiMathSolverService(api_key="AIzaSyDeh9mTorPUvXqsDxo4NcfGNq2A4vCVVWg")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/solve", response_model=SolveResponse)
async def solve_equation(request: SolveRequest):
    try:
        # Get equation text directly from request
        equation_text = request.equation_text
        
        print(f"🔍 Input equation: '{equation_text}'")
        
        if not equation_text:
            raise HTTPException(status_code=400, detail="Please enter a mathematical equation.")
        
        # Use Gemini AI for enhanced equation solving
        solution = await gemini_math_solver.solve_with_sympy_fallback(equation_text)
        
        return SolveResponse(
            input=equation_text,
            normalized=solution["normalized"],
            steps=solution["steps"],
            solution=solution["solution"],
            confidence=solution["confidence"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing equation: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
