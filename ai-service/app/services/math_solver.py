import sympy as sp
import re
from typing import Dict, List

class MathSolverService:
    def __init__(self):
        self.x = sp.Symbol('x')
        self.y = sp.Symbol('y')
        self.z = sp.Symbol('z')
    
    async def solve_equation(self, equation_text: str) -> Dict:
        """Solve mathematical equation and return step-by-step solution"""
        try:
            # Normalize the equation
            normalized = self._normalize_equation(equation_text)
            
            # Parse and solve
            steps, solution, confidence = self._solve_equation_steps(normalized)
            
            return {
                "normalized": normalized,
                "steps": steps,
                "solution": solution,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"Math solving error: {e}")
            return {
                "normalized": equation_text,
                "steps": ["Error: Could not solve equation"],
                "solution": ["No solution found"],
                "confidence": 0.0
            }
    
    def _normalize_equation(self, equation: str) -> str:
        """Normalize equation for SymPy parsing"""
        # Remove extra whitespace
        equation = re.sub(r'\s+', '', equation)
        
        # Handle incomplete equations (with ? or ending with operators)
        if '?' in equation:
            # For simple arithmetic like "1 + 1 = ?", just evaluate the left side
            if '=' in equation:
                left, right = equation.split('=', 1)
                if right.strip() == '?':
                    # This is just arithmetic, evaluate the left side
                    equation = left.strip()
                else:
                    equation = equation.replace('?', 'x')
            else:
                equation = equation.replace('?', 'x')
        elif equation.endswith(('-', '+', '*', '/')):
            # Only remove trailing operators if they're at the very end
            equation = equation.rstrip('+-*/')
            if not equation:
                equation = "0"
        elif equation.endswith('='):
            # For equations ending with =, treat as equation = 0
            equation = equation.rstrip('=')
            if not equation:
                equation = "0"
            else:
                # Add = 0 to make it a proper equation
                equation = equation + " = 0"
        
        # Common normalizations
        equation = equation.replace('^', '**')
        equation = equation.replace('×', '*')
        equation = equation.replace('÷', '/')
        
        # Handle implicit multiplication (e.g., 2x -> 2*x)
        equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)
        equation = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', equation)
        equation = re.sub(r'\)(\d)', r')*\1', equation)
        equation = re.sub(r'(\d)\(', r'\1*(', equation)
        
        return equation
    
    def _solve_equation_steps(self, equation: str) -> tuple:
        """Solve equation and generate step-by-step solution"""
        steps = []
        solution = []
        confidence = 0.8
        
        try:
            # Try to parse as equation with equals sign
            if '=' in equation:
                left, right = equation.split('=', 1)
                left_expr = sp.sympify(left)
                right_expr = sp.sympify(right)
                
                steps.append(f"Given: {left} = {right}")
                
                # Check if it's a simple arithmetic expression
                if self.x not in left_expr.free_symbols and self.x not in right_expr.free_symbols:
                    # It's just arithmetic, evaluate it
                    try:
                        result = float(left_expr.evalf())
                        steps.append(f"Calculate: {left} = {result}")
                        solution.append(f"Answer: {result}")
                    except:
                        # Move everything to left side and solve
                        equation_expr = left_expr - right_expr
                        steps.append(f"Rearrange: {left} - {right} = 0")
                        
                        # Solve
                        solutions = sp.solve(equation_expr, self.x)
                        if solutions:
                            if len(solutions) == 1:
                                solution.append(f"x = {solutions[0]}")
                                steps.append(f"Solution: x = {solutions[0]}")
                            else:
                                for i, sol in enumerate(solutions):
                                    solution.append(f"x{i+1} = {sol}")
                                steps.append(f"Solutions: {', '.join(solution)}")
                        else:
                            solution.append("No real solutions")
                            steps.append("No real solutions found")
                else:
                    # Move everything to left side
                    equation_expr = left_expr - right_expr
                    steps.append(f"Move to left side: {sp.latex(equation_expr)} = 0")
                    
                    # Solve
                    solutions = sp.solve(equation_expr, self.x)
                    if solutions:
                        if len(solutions) == 1:
                            solution.append(f"x = {solutions[0]}")
                            steps.append(f"Solution: x = {solutions[0]}")
                        else:
                            for i, sol in enumerate(solutions):
                                solution.append(f"x{i+1} = {sol}")
                            steps.append(f"Solutions: {', '.join(solution)}")
                    else:
                        solution.append("No real solutions")
                        steps.append("No real solutions found")
            
            # Try to parse as expression (evaluation)
            else:
                try:
                    expr = sp.sympify(equation)
                    simplified = sp.simplify(expr)
                    steps.append(f"Original: {equation}")
                    steps.append(f"Simplified: {sp.latex(simplified)}")
                    
                    # Try to evaluate if it's a number
                    if expr.is_number:
                        result = float(expr.evalf())
                        solution.append(str(result))
                        steps.append(f"Result: {result}")
                    else:
                        # Check if it's a linear expression that can be solved
                        if self.x in expr.free_symbols:
                            # It's an expression with variables, try to solve for x
                            try:
                                # For expressions like "2*x + 3", we can't solve without an equation
                                # But we can show that it's a linear expression
                                solution.append(f"Linear expression: {sp.latex(simplified)}")
                                steps.append(f"This is a linear expression in x")
                                steps.append(f"To solve for x, we need an equation (e.g., 2x + 3 = 7)")
                            except:
                                solution.append(sp.latex(simplified))
                                steps.append(f"Simplified form: {sp.latex(simplified)}")
                        else:
                            # It's an expression with variables, just return the simplified form
                            solution.append(sp.latex(simplified))
                            steps.append(f"Simplified form: {sp.latex(simplified)}")
                        
                except:
                    # Try derivative
                    try:
                        expr = sp.sympify(equation)
                        derivative = sp.diff(expr, self.x)
                        steps.append(f"Original: {equation}")
                        steps.append(f"Derivative: {sp.latex(derivative)}")
                        solution.append(sp.latex(derivative))
                    except:
                        # Try integral
                        try:
                            expr = sp.sympify(equation)
                            integral = sp.integrate(expr, self.x)
                            steps.append(f"Original: {equation}")
                            steps.append(f"Integral: {sp.latex(integral)}")
                            solution.append(sp.latex(integral))
                        except:
                            steps.append("Could not process this mathematical expression")
                            solution.append("Unable to solve")
                            confidence = 0.1
            
        except Exception as e:
            steps.append(f"Error: {str(e)}")
            solution.append("Error in solving")
            confidence = 0.1
        
        return steps, solution, confidence
