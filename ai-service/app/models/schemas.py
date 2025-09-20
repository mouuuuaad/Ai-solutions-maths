from pydantic import BaseModel
from typing import List

class SolveRequest(BaseModel):
    equation_text: str

class SolveResponse(BaseModel):
    input: str
    normalized: str
    steps: List[str]
    solution: List[str]
    confidence: float
