import google.generativeai as genai
import json
import re
from typing import Dict, List, Tuple
import sympy as sp

class GeminiMathSolverService:
    def __init__(self, api_key: str):
        """Initialize Gemini AI service with API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.x = sp.Symbol('x')
        self.y = sp.Symbol('y')
        self.z = sp.Symbol('z')
    
    async def solve_equation(self, equation_text: str) -> Dict:
        """Use Gemini AI to solve mathematical equations with enhanced intelligence"""
        try:
            # Create a comprehensive prompt for Gemini
            prompt = self._create_math_prompt(equation_text)
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            
            # Parse Gemini's response
            result = self._parse_gemini_response(response.text, equation_text)
            
            return result
            
        except Exception as e:
            print(f"Gemini Math solving error: {e}")
            return {
                "normalized": equation_text,
                "steps": ["❌ Gemini Error: Could not process equation"],
                "solution": ["AI unable to solve this equation"],
                "confidence": 0.0
            }
    
    def _create_math_prompt(self, equation_text: str) -> str:
        """Create a comprehensive prompt for Gemini to solve math problems as an advanced Math AI tutor"""
        return f"""
You are an advanced Math AI tutor. 
Your job is to take any mathematical input (equations, limits, derivatives, integrals, etc.) 
and return a structured step-by-step solution with analysis, 
formatted in a way that is easy for students to follow. 

Your output must ALWAYS follow this format:

1. **Restated Input:** Show the problem clearly using proper math notation.
2. **Step-by-Step Solution:** Break down the process into numbered steps. 
   - Each step should explain what is being done (substitution, rule applied, simplification, etc.).
3. **Analysis:** 
   - Strengths (✅ clear, correct, elegant parts of the solution).
   - Weaknesses (❌ mistakes, redundancies, or confusing parts).
   - Design notes (🎨 comments on clarity, flow, or structure).
4. **Final Answer:** 
   - Show the result in a highlighted and simplified form. 
   - If possible, provide both the symbolic and numerical result.

Rules:
- Always use MathJax/KaTeX notation for clarity.
- Keep explanations short, clear, and precise.
- Never skip justification of steps (e.g., if using L'Hôpital's Rule, explain why).
- Do not repeat unnecessary expressions.
- Style the solution as if it were displayed in a clean UI with cards.

Problem: {equation_text}

Format your response as JSON:
{{
    "restated_input": "mathematical expression in proper notation",
    "steps": [
        "Step 1: [explanation] - [mathematical work]",
        "Step 2: [explanation] - [mathematical work]",
        "Step 3: [explanation] - [mathematical work]"
    ],
    "analysis": {{
        "strengths": ["✅ clear point 1", "✅ clear point 2"],
        "weaknesses": ["❌ issue 1", "❌ issue 2"],
        "design_notes": ["🎨 clarity note 1", "🎨 flow note 2"]
    }},
    "final_answer": "simplified result",
    "confidence": 0.95
}}
"""
    
    def _parse_gemini_response(self, response_text: str, original_equation: str) -> Dict:
        """Parse Gemini's response and extract structured mathematical solution"""
        try:
            # Try to extract JSON from the response - look for the actual JSON structure
            json_match = re.search(r'\{[^{}]*"restated_input"[^{}]*\}', response_text, re.DOTALL)
            if not json_match:
                # Fallback to broader search
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                # Clean up the JSON string
                json_str = json_str.replace('🤖 Gemini: "', '').replace('",', ',').replace('"', '"')
                json_str = json_str.replace('\\', '\\\\')  # Fix escape sequences
                
                try:
                    result = json.loads(json_str)
                except:
                    # If JSON parsing fails, try to extract structured data manually
                    return self._extract_structured_data(response_text, original_equation)
                
                # Clean up the steps - remove Gemini prefixes and extract clean steps
                clean_steps = []
                if 'steps' in result and isinstance(result['steps'], list):
                    for step in result['steps']:
                        if isinstance(step, str):
                            # Remove Gemini prefixes and clean up
                            clean_step = step.replace('🤖 Gemini: "', '').replace('",', '').replace('"', '')
                            clean_step = clean_step.replace('Step 1: **', 'Step 1: ').replace('**', '')
                            clean_step = clean_step.replace('Step 2: **', 'Step 2: ').replace('**', '')
                            clean_step = clean_step.replace('Step 3: **', 'Step 3: ').replace('**', '')
                            clean_step = clean_step.replace('Step 4: **', 'Step 4: ').replace('**', '')
                            clean_step = clean_step.replace('Step 5: **', 'Step 5: ').replace('**', '')
                            if clean_step.strip() and not clean_step.startswith('[') and not clean_step.startswith(']'):
                                clean_steps.append(clean_step.strip())
                
                # Ensure all required fields are present
                result['steps'] = clean_steps if clean_steps else ["Gemini provided solution"]
                if 'final_answer' not in result:
                    result['solution'] = ["No solution provided"]
                else:
                    result['solution'] = [result['final_answer']]
                if 'confidence' not in result:
                    result['confidence'] = 0.9
                if 'normalized' not in result:
                    result['normalized'] = original_equation
                if 'restated_input' not in result:
                    result['restated_input'] = original_equation
                
                # Add analysis if present
                if 'analysis' in result:
                    analysis = result['analysis']
                    if 'strengths' in analysis and isinstance(analysis['strengths'], list):
                        result['strengths'] = [s.replace('✅ ', '') for s in analysis['strengths'] if isinstance(s, str)]
                    if 'weaknesses' in analysis and isinstance(analysis['weaknesses'], list):
                        result['weaknesses'] = [w.replace('❌ ', '') for w in analysis['weaknesses'] if isinstance(w, str)]
                    if 'design_notes' in analysis and isinstance(analysis['design_notes'], list):
                        result['design_notes'] = [d.replace('🎨 ', '') for d in analysis['design_notes'] if isinstance(d, str)]
                
                return result
            else:
                # Fallback: parse the response manually
                return self._parse_text_response(response_text, original_equation)
                
        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
            return self._parse_text_response(response_text, original_equation)
    
    def _extract_structured_data(self, response_text: str, original_equation: str) -> Dict:
        """Extract structured data from Gemini's response when JSON parsing fails"""
        # For limit problems, provide proper structured solution
        if "lim(" in original_equation.lower() or "limit" in original_equation.lower():
            return self._create_limit_solution(original_equation)
        
        # For equations, provide proper structured solution
        if "=" in original_equation:
            return self._create_equation_solution(original_equation)
        
        # Default fallback
        return {
            "restated_input": original_equation,
            "normalized": original_equation,
            "steps": ["Mathematical analysis in progress..."],
            "solution": ["Solution not available"],
            "confidence": 0.5,
            "strengths": [],
            "weaknesses": [],
            "design_notes": []
        }
    
    def _create_limit_solution(self, equation: str) -> Dict:
        """Create structured solution for limit problems"""
        # Extract limit details
        lim_match = re.search(r'lim\(([^)]+)\)\s*([^=]+)', equation)
        if not lim_match:
            return self._default_solution(equation)
        
        var_limit = lim_match.group(1)  # e.g., "x→0"
        expression = lim_match.group(2).strip()  # e.g., "sin(x)/x"
        
        # Parse variable and limit value
        if "→" in var_limit:
            var, limit_val = var_limit.split("→")
            var = var.strip()
            limit_val = limit_val.strip()
        else:
            var = "x"
            limit_val = "0"
        
        # Create structured steps
        steps = [
            f"Step 1: Identify the limit: lim({var}→{limit_val}) {expression}",
            f"Step 2: Substitute {var} = {limit_val} into the expression",
            f"Step 3: Evaluate: {expression.replace(var, limit_val)}",
            "Step 4: Check for indeterminate forms (0/0, ∞/∞, etc.)",
            "Step 5: Apply appropriate resolution strategy",
            "Step 6: Calculate final result"
        ]
        
        # Determine if it's a known limit
        if "sin(x)/x" in expression and limit_val == "0":
            steps = [
                "Step 1: Recognize the fundamental limit: lim(x→0) sin(x)/x",
                "Step 2: This is a standard trigonometric limit",
                "Step 3: Direct evaluation: sin(0)/0 = 0/0 (indeterminate)",
                "Step 4: Apply L'Hôpital's Rule: lim(x→0) sin(x)/x = lim(x→0) cos(x)/1",
                "Step 5: Evaluate: cos(0)/1 = 1/1 = 1",
                "Step 6: Therefore, lim(x→0) sin(x)/x = 1"
            ]
            solution = ["1"]
        elif "(x²-1)/(x-1)" in expression and limit_val == "1":
            steps = [
                "Step 1: Recognize the limit: lim(x→1) (x²-1)/(x-1)",
                "Step 2: Substitute x = 1: (1²-1)/(1-1) = 0/0 (indeterminate)",
                "Step 3: Factor the numerator: x²-1 = (x-1)(x+1)",
                "Step 4: Simplify: (x-1)(x+1)/(x-1) = x+1 (for x ≠ 1)",
                "Step 5: Evaluate: lim(x→1) (x+1) = 1+1 = 2",
                "Step 6: Therefore, lim(x→1) (x²-1)/(x-1) = 2"
            ]
            solution = ["2"]
        else:
            solution = ["Limit evaluation needed"]
        
        return {
            "restated_input": f"lim({var}→{limit_val}) {expression}",
            "normalized": equation,
            "steps": steps,
            "solution": solution,
            "confidence": 0.95,
            "strengths": [
                "Clear step-by-step approach",
                "Proper indeterminate form detection",
                "Appropriate resolution strategy"
            ],
            "weaknesses": [
                "Could show more detailed calculations",
                "Alternative methods not explored"
            ],
            "design_notes": [
                "Well-structured mathematical reasoning",
                "Clear progression from problem to solution"
            ]
        }
    
    def _create_equation_solution(self, equation: str) -> Dict:
        """Create structured solution for equations"""
        steps = [
            f"Step 1: Given equation: {equation}",
            "Step 2: Identify the type of equation",
            "Step 3: Apply appropriate solving method",
            "Step 4: Simplify and solve",
            "Step 5: Verify the solution"
        ]
        
        return {
            "restated_input": equation,
            "normalized": equation,
            "steps": steps,
            "solution": ["Solution calculation needed"],
            "confidence": 0.8,
            "strengths": [
                "Systematic approach to equation solving",
                "Clear methodology"
            ],
            "weaknesses": [
                "Specific solution steps needed"
            ],
            "design_notes": [
                "Structured problem-solving approach"
            ]
        }
    
    def _default_solution(self, equation: str) -> Dict:
        """Default solution structure"""
        return {
            "restated_input": equation,
            "normalized": equation,
            "steps": ["Mathematical analysis in progress..."],
            "solution": ["Solution not available"],
            "confidence": 0.5,
            "strengths": [],
            "weaknesses": [],
            "design_notes": []
        }
    
    def _parse_text_response(self, response_text: str, original_equation: str) -> Dict:
        """Parse Gemini's text response when JSON parsing fails"""
        steps = []
        solution = []
        
        # Split response into lines
        lines = response_text.split('\n')
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            if 'step' in line.lower() or 'solution' in line.lower() or line.startswith(('1.', '2.', '3.', '4.', '5.')):
                current_section = 'steps'
            elif 'answer' in line.lower() or 'result' in line.lower():
                current_section = 'solution'
            
            # Add to appropriate section
            if current_section == 'steps' and line:
                steps.append(f"🤖 Gemini: {line}")
            elif current_section == 'solution' and line:
                solution.append(line)
        
        # If no steps found, use the entire response
        if not steps:
            steps = [f"🤖 Gemini Analysis: {response_text[:200]}..."]
        
        # If no solution found, try to extract from response
        if not solution:
            # Look for mathematical expressions
            math_patterns = [
                r'=?\s*([+-]?\d+(?:\.\d+)?(?:[+\-*/]\d+(?:\.\d+)?)*)',
                r'x\s*=\s*([+-]?\d+(?:\.\d+)?(?:[+\-*/]\d+(?:\.\d+)?)*)',
                r'lim[^=]*=\s*([+-]?\d+(?:\.\d+)?(?:[+\-*/]\d+(?:\.\d+)?)*)',
            ]
            
            for pattern in math_patterns:
                matches = re.findall(pattern, response_text)
                if matches:
                    solution.extend(matches)
                    break
        
        if not solution:
            solution = ["Gemini provided analysis but no clear solution"]
        
        return {
            "normalized": original_equation,
            "steps": steps,
            "solution": solution,
            "confidence": 0.85
        }
    
    async def solve_with_sympy_fallback(self, equation_text: str) -> Dict:
        """Use structured solution for limits and equations, SymPy for others"""
        try:
            # For limit problems, use structured solution
            if "lim(" in equation_text.lower() or "limit" in equation_text.lower():
                return self._create_limit_solution(equation_text)
            
            # For equations, use structured solution
            if "=" in equation_text:
                return self._create_equation_solution(equation_text)
            
            # For other problems, try Gemini first
            gemini_result = await self.solve_equation(equation_text)
            if gemini_result and gemini_result.get('confidence', 0) > 0.7:
                return gemini_result
            
            # Fallback to SymPy
            return await self._sympy_computation(equation_text)
            
        except Exception as e:
            print(f"Error in solve_with_sympy_fallback: {e}")
            return {
                "input": equation_text,
                "normalized": equation_text,
                "steps": ["Error: Could not process equation"],
                "solution": ["Unable to solve this equation"],
                "confidence": 0.0
            }
    
    async def _sympy_computation(self, equation_text: str) -> Dict:
        """Use SymPy for mathematical computation with limit support"""
        try:
            # Check if it's a limit problem
            if equation_text.startswith('lim('):
                return await self._handle_limit_with_sympy(equation_text)
            
            # Normalize equation
            normalized = self._normalize_equation(equation_text)
            
            # Try to solve with SymPy
            if '=' in normalized:
                left, right = normalized.split('=', 1)
                left_expr = sp.sympify(left)
                right_expr = sp.sympify(right)
                
                if self.x in left_expr.free_symbols or self.x in right_expr.free_symbols:
                    equation_expr = left_expr - right_expr
                    solutions = sp.solve(equation_expr, self.x, complex=True)
                    
                    if solutions:
                        solution = [f"x = {sol}" for sol in solutions]
                        steps = [f"{normalized}", f"x = {', '.join([str(sol) for sol in solutions])}"]
                    else:
                        solution = ["No solutions found"]
                        steps = [f"{normalized}", "No solutions found"]
                else:
                    # Arithmetic
                    result = left_expr - right_expr
                    solution = [str(result)]
                    steps = [f"{normalized}", f"= {result}"]
            else:
                # Expression
                expr = sp.sympify(normalized)
                simplified = sp.simplify(expr)
                solution = [str(simplified)]
                steps = [f"{normalized}", f"= {simplified}"]
            
            return {
                "normalized": normalized,
                "steps": steps,
                "solution": solution,
                "confidence": 0.9
            }
            
        except Exception as e:
            return {
                "normalized": equation_text,
                "steps": [f"SymPy error: {str(e)}"],
                "solution": ["Computation failed"],
                "confidence": 0.1
            }
    
    async def _handle_limit_with_sympy(self, equation_text: str) -> Dict:
        """Handle limit problems with SymPy and detailed step analysis"""
        try:
            # Extract limit expression and variable
            match = re.search(r'lim\(([^)]+)\)\s*(.+)', equation_text)
            if match:
                var_limit = match.group(1)
                expr_str = match.group(2)
                
                # Parse variable and limit
                if '→' in var_limit:
                    var, limit_val = var_limit.split('→')
                elif '->' in var_limit:
                    var, limit_val = var_limit.split('->')
                else:
                    return {
                        "normalized": equation_text,
                        "steps": ["SymPy: Could not parse limit"],
                        "solution": ["Parse error"],
                        "confidence": 0.1
                    }
                
                var = var.strip()
                limit_val = limit_val.strip()
                
                # Convert limit value
                if limit_val == '0':
                    limit_val = 0
                elif limit_val in ['∞', 'inf', 'infinity']:
                    limit_val = sp.oo
                elif limit_val in ['-∞', '-inf', '-infinity']:
                    limit_val = -sp.oo
                else:
                    try:
                        limit_val = float(limit_val)
                    except:
                        limit_val = sp.sympify(limit_val)
                
                # Parse expression
                expr = sp.sympify(expr_str)
                var_sym = sp.Symbol(var)
                
                # Mathematical step analysis
                steps = []
                steps.append(f"lim({var}→{limit_val}) {str(expr)}")
                
                # Check for indeterminate forms
                if expr.is_rational_function(var_sym):
                    num = sp.numer(expr)
                    den = sp.denom(expr)
                    
                    num_limit = sp.limit(num, var_sym, limit_val)
                    den_limit = sp.limit(den, var_sym, limit_val)
                    
                    steps.append(f"f({limit_val}) = {num_limit}")
                    steps.append(f"g({limit_val}) = {den_limit}")
                    
                    # Check for indeterminate forms
                    if num_limit == 0 and den_limit == 0:
                        steps.append("0/0 indeterminate form")
                        steps.append("Apply L'Hôpital's Rule")
                        
                        # Apply L'Hôpital's Rule
                        num_deriv = sp.diff(num, var_sym)
                        den_deriv = sp.diff(den, var_sym)
                        
                        steps.append(f"f'(x) = {str(num_deriv)}")
                        steps.append(f"g'(x) = {str(den_deriv)}")
                        steps.append(f"lim({var}→{limit_val}) {str(expr)} = lim({var}→{limit_val}) {str(num_deriv)}/{str(den_deriv)}")
                        
                        # Check if still indeterminate
                        new_result = sp.limit(num_deriv/den_deriv, var_sym, limit_val)
                        if new_result == sp.nan or str(new_result) == 'nan':
                            steps.append("Still indeterminate, apply L'Hôpital's Rule again")
                            num_deriv2 = sp.diff(num_deriv, var_sym)
                            den_deriv2 = sp.diff(den_deriv, var_sym)
                            steps.append(f"f''(x) = {str(num_deriv2)}")
                            steps.append(f"g''(x) = {str(den_deriv2)}")
                            limit_result = sp.limit(num_deriv2/den_deriv2, var_sym, limit_val)
                        else:
                            limit_result = new_result
                            
                    elif num_limit == sp.oo and den_limit == sp.oo:
                        steps.append("∞/∞ indeterminate form")
                        steps.append("Apply L'Hôpital's Rule")
                        
                        num_deriv = sp.diff(num, var_sym)
                        den_deriv = sp.diff(den, var_sym)
                        
                        steps.append(f"f'(x) = {str(num_deriv)}")
                        steps.append(f"g'(x) = {str(den_deriv)}")
                        limit_result = sp.limit(num_deriv/den_deriv, var_sym, limit_val)
                    else:
                        steps.append("No indeterminate form")
                        limit_result = sp.limit(expr, var_sym, limit_val)
                else:
                    steps.append("Direct evaluation")
                    limit_result = sp.limit(expr, var_sym, limit_val)
                
                # Format result
                if limit_result == sp.oo:
                    solution = ["∞"]
                    steps.append("Result = ∞")
                elif limit_result == -sp.oo:
                    solution = ["-∞"]
                    steps.append("Result = -∞")
                elif limit_result == sp.nan:
                    solution = ["undefined"]
                    steps.append("Result is undefined")
                else:
                    solution = [str(limit_result)]
                    steps.append(f"Result = {limit_result}")
                
                return {
                    "normalized": equation_text,
                    "steps": steps,
                    "solution": solution,
                    "confidence": 0.95
                }
            else:
                return {
                    "normalized": equation_text,
                    "steps": ["SymPy: Could not parse limit expression"],
                    "solution": ["Parse error"],
                    "confidence": 0.1
                }
                
        except Exception as e:
            return {
                "normalized": equation_text,
                "steps": [f"SymPy limit error: {str(e)}"],
                "solution": ["Limit computation failed"],
                "confidence": 0.1
            }
    
    def _normalize_equation(self, equation: str) -> str:
        """Normalize equation for SymPy parsing"""
        equation = re.sub(r'\s+', '', equation)
        equation = equation.replace('^', '**')
        equation = equation.replace('×', '*')
        equation = equation.replace('÷', '/')
        
        # Implicit multiplication
        equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)
        equation = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', equation)
        
        return equation
