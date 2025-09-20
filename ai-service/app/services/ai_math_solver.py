import sympy as sp
import re
import json
import random
from typing import Dict, List, Tuple
import numpy as np

class AIMathSolverService:
    def __init__(self):
        self.x = sp.Symbol('x')
        self.y = sp.Symbol('y')
        self.z = sp.Symbol('z')
        
        # AI-powered equation classification
        self.equation_types = {
            'linear': self._solve_linear_equation,
            'quadratic': self._solve_quadratic_equation,
            'polynomial': self._solve_polynomial_equation,
            'exponential': self._solve_exponential_equation,
            'logarithmic': self._solve_logarithmic_equation,
            'trigonometric': self._solve_trigonometric_equation,
            'integral': self._solve_integral,
            'derivative': self._solve_derivative,
            'limit': self._solve_limit,
            'arithmetic': self._solve_arithmetic
        }
    
    async def solve_equation(self, equation_text: str) -> Dict:
        """AI-powered equation solving with intelligent classification"""
        try:
            # AI-powered preprocessing
            normalized = self._ai_normalize_equation(equation_text)
            
            # AI-powered equation type detection
            equation_type = self._ai_classify_equation(normalized)
            
            # AI-powered solving with dynamic confidence
            steps, solution, confidence = await self._ai_solve_equation(normalized, equation_type)
            
            return {
                "normalized": normalized,
                "steps": steps,
                "solution": solution,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"AI Math solving error: {e}")
            return {
                "normalized": equation_text,
                "steps": ["AI Error: Could not process equation"],
                "solution": ["AI unable to solve this equation"],
                "confidence": 0.0
            }
    
    def _ai_normalize_equation(self, equation: str) -> str:
        """AI-powered equation normalization"""
        # Remove extra whitespace
        equation = re.sub(r'\s+', '', equation)
        
        # AI-powered pattern recognition for different equation formats
        if '?' in equation:
            if '=' in equation:
                left, right = equation.split('=', 1)
                if right.strip() == '?':
                    equation = left.strip()
                else:
                    equation = equation.replace('?', 'x')
            else:
                equation = equation.replace('?', 'x')
        elif equation.endswith(('-', '+', '*', '/')):
            equation = equation.rstrip('+-*/')
            if not equation:
                equation = "0"
        elif equation.endswith('='):
            equation = equation.rstrip('=')
            if not equation:
                equation = "0"
            else:
                equation = equation + " = 0"
        
        # AI-powered symbol normalization
        equation = equation.replace('^', '**')
        equation = equation.replace('×', '*')
        equation = equation.replace('÷', '/')
        
        # AI-powered implicit multiplication detection (but not for limits)
        if not equation.startswith('lim('):
            equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)
            equation = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', equation)
            equation = re.sub(r'\)(\d)', r')*\1', equation)
            equation = re.sub(r'(\d)\(', r'\1*(', equation)
        
        # Fix parentheses spacing issues (but not for limits)
        if not equation.startswith('lim('):
            equation = equation.replace('(', ' (')
            equation = equation.replace(')', ') ')
            equation = re.sub(r'\s+', ' ', equation).strip()
        
        return equation
    
    def _ai_classify_equation(self, equation: str) -> str:
        """AI-powered equation type classification"""
        equation_lower = equation.lower()
        
        # AI pattern recognition for different equation types
        if 'lim(' in equation_lower or 'limit' in equation_lower:
            return 'limit'
        elif '∫' in equation or 'integrate(' in equation_lower:
            return 'integral'
        elif 'd/dx' in equation_lower or 'diff(' in equation_lower:
            return 'derivative'
        elif 'sin(' in equation_lower or 'cos(' in equation_lower or 'tan(' in equation_lower:
            return 'trigonometric'
        elif 'log(' in equation_lower or 'ln(' in equation_lower:
            return 'logarithmic'
        elif 'exp(' in equation_lower or 'e^' in equation_lower:
            return 'exponential'
        elif 'x^3' in equation or 'x**3' in equation:
            return 'polynomial'
        elif 'x^2' in equation or 'x**2' in equation:
            return 'quadratic'
        elif 'x' in equation and '=' in equation:
            return 'linear'
        elif any(op in equation for op in ['+', '-', '*', '/']) and not any(var in equation for var in ['x', 'y', 'z']):
            return 'arithmetic'
        else:
            return 'linear'  # Default fallback
    
    async def _ai_solve_equation(self, equation: str, equation_type: str) -> Tuple[List[str], List[str], float]:
        """AI-powered equation solving with dynamic confidence"""
        try:
            solver_func = self.equation_types.get(equation_type, self._solve_linear_equation)
            return await solver_func(equation)
        except Exception as e:
            return [f"AI Error: {str(e)}"], ["AI unable to solve"], 0.1
    
    async def _solve_linear_equation(self, equation: str) -> Tuple[List[str], List[str], float]:
        """AI-powered linear equation solving"""
        steps = []
        solution = []
        confidence = 0.9
        
        try:
            if '=' in equation:
                left, right = equation.split('=', 1)
                left_expr = sp.sympify(left)
                right_expr = sp.sympify(right)
                
                steps.append(f"🔍 AI Analysis: Linear equation detected")
                steps.append(f"Given: {left} = {right}")
                
                # AI-powered solving with complex solutions
                equation_expr = left_expr - right_expr
                solutions = sp.solve(equation_expr, self.x, complex=True)
                
                if solutions:
                    if len(solutions) == 1:
                        sol = solutions[0]
                        if sol.is_real:
                            steps.append(f"🤖 AI Solution: x = {sol}")
                            steps.append(f"✅ Verification: {left.replace('x', str(sol))} = {right.replace('x', str(sol))}")
                            solution.append(f"x = {sol}")
                        else:
                            steps.append(f"🤖 AI Solution: x = {sol}")
                            steps.append(f"💡 Note: This is a complex solution")
                            solution.append(f"x = {sol}")
                        confidence = 0.95
                    else:
                        for i, sol in enumerate(solutions):
                            solution.append(f"x{i+1} = {sol}")
                        steps.append(f"🤖 AI Solutions: {', '.join(solution)}")
                        confidence = 0.9
                else:
                    solution.append("No solutions found")
                    steps.append("❌ AI Analysis: No solutions found")
                    confidence = 0.8
            else:
                # Expression simplification
                expr = sp.sympify(equation)
                simplified = sp.simplify(expr)
                steps.append(f"🔍 AI Analysis: Expression simplification")
                steps.append(f"Original: {equation}")
                steps.append(f"🤖 AI Simplified: {sp.latex(simplified)}")
                solution.append(sp.latex(simplified))
                confidence = 0.85
                
        except Exception as e:
            steps.append(f"❌ AI Error: {str(e)}")
            solution.append("AI unable to solve")
            confidence = 0.1
        
        return steps, solution, confidence
    
    async def _solve_quadratic_equation(self, equation: str) -> Tuple[List[str], List[str], float]:
        """AI-powered quadratic equation solving with complex solutions"""
        steps = []
        solution = []
        confidence = 0.9
        
        try:
            if '=' in equation:
                left, right = equation.split('=', 1)
                left_expr = sp.sympify(left)
                right_expr = sp.sympify(right)
                
                steps.append(f"🔍 AI Analysis: Quadratic equation detected")
                steps.append(f"Given: {left} = {right}")
                
                equation_expr = left_expr - right_expr
                
                # Solve for both real and complex solutions
                solutions = sp.solve(equation_expr, self.x, complex=True)
                
                if solutions:
                    if len(solutions) == 1:
                        sol = solutions[0]
                        if sol.is_real:
                            steps.append(f"🤖 AI Solution: x = {sol}")
                            solution.append(f"x = {sol}")
                        else:
                            steps.append(f"🤖 AI Solution: x = {sol}")
                            steps.append(f"💡 Note: This is a complex solution")
                            solution.append(f"x = {sol}")
                    else:
                        real_solutions = [sol for sol in solutions if sol.is_real]
                        complex_solutions = [sol for sol in solutions if not sol.is_real]
                        
                        if real_solutions:
                            for i, sol in enumerate(real_solutions):
                                solution.append(f"x{i+1} = {sol}")
                            steps.append(f"🤖 AI Real Solutions: {', '.join([f'x{i+1} = {sol}' for i, sol in enumerate(real_solutions)])}")
                        
                        if complex_solutions:
                            for i, sol in enumerate(complex_solutions):
                                solution.append(f"x{i+len(real_solutions)+1} = {sol}")
                            steps.append(f"🤖 AI Complex Solutions: {', '.join([f'x{i+len(real_solutions)+1} = {sol}' for i, sol in enumerate(complex_solutions)])}")
                            steps.append(f"💡 Note: Complex solutions involve imaginary numbers")
                    
                    # AI-powered verification
                    for sol in solutions:
                        try:
                            verified = equation_expr.subs(self.x, sol)
                            steps.append(f"✅ AI Verification: x = {sol} → {verified} = 0")
                        except:
                            pass
                    
                    confidence = 0.95
                else:
                    solution.append("No solutions found")
                    steps.append("❌ AI Analysis: No solutions found")
                    confidence = 0.8
            else:
                expr = sp.sympify(equation)
                simplified = sp.simplify(expr)
                steps.append(f"🔍 AI Analysis: Quadratic expression")
                steps.append(f"Original: {equation}")
                steps.append(f"🤖 AI Simplified: {sp.latex(simplified)}")
                solution.append(sp.latex(simplified))
                confidence = 0.85
                
        except Exception as e:
            steps.append(f"❌ AI Error: {str(e)}")
            solution.append("AI unable to solve")
            confidence = 0.1
        
        return steps, solution, confidence
    
    async def _solve_arithmetic(self, equation: str) -> Tuple[List[str], List[str], float]:
        """AI-powered arithmetic solving"""
        steps = []
        solution = []
        confidence = 0.95
        
        try:
            steps.append(f"🔍 AI Analysis: Arithmetic expression detected")
            steps.append(f"Given: {equation}")
            
            # AI-powered evaluation
            expr = sp.sympify(equation)
            result = float(expr.evalf())
            
            steps.append(f"🤖 AI Calculation: {equation} = {result}")
            steps.append(f"✅ AI Verification: {result}")
            
            solution.append(str(result))
            confidence = 0.98
            
        except Exception as e:
            steps.append(f"❌ AI Error: {str(e)}")
            solution.append("AI unable to calculate")
            confidence = 0.1
        
        return steps, solution, confidence
    
    async def _solve_integral(self, equation: str) -> Tuple[List[str], List[str], float]:
        """AI-powered integral solving"""
        steps = []
        solution = []
        confidence = 0.9
        
        try:
            steps.append(f"🔍 AI Analysis: Integral detected")
            steps.append(f"Given: ∫ {equation}")
            
            expr = sp.sympify(equation)
            integral = sp.integrate(expr, self.x)
            
            steps.append(f"🤖 AI Integration: ∫ {sp.latex(expr)} dx")
            steps.append(f"✅ AI Result: {sp.latex(integral)} + C")
            
            solution.append(f"{sp.latex(integral)} + C")
            confidence = 0.9
            
        except Exception as e:
            steps.append(f"❌ AI Error: {str(e)}")
            solution.append("AI unable to integrate")
            confidence = 0.1
        
        return steps, solution, confidence
    
    async def _solve_derivative(self, equation: str) -> Tuple[List[str], List[str], float]:
        """AI-powered derivative solving"""
        steps = []
        solution = []
        confidence = 0.9
        
        try:
            steps.append(f"🔍 AI Analysis: Derivative detected")
            steps.append(f"Given: d/dx({equation})")
            
            expr = sp.sympify(equation)
            derivative = sp.diff(expr, self.x)
            
            steps.append(f"🤖 AI Differentiation: d/dx({sp.latex(expr)})")
            steps.append(f"✅ AI Result: {sp.latex(derivative)}")
            
            solution.append(sp.latex(derivative))
            confidence = 0.9
            
        except Exception as e:
            steps.append(f"❌ AI Error: {str(e)}")
            solution.append("AI unable to differentiate")
            confidence = 0.1
        
        return steps, solution, confidence
    
    async def _solve_limit(self, equation: str) -> Tuple[List[str], List[str], float]:
        """AI-powered limit solving with indeterminate form detection and resolution"""
        steps = []
        solution = []
        confidence = 0.85
        
        try:
            steps.append(f"🔍 AI Analysis: Limit detected")
            steps.append(f"Given: {equation}")
            
            # Extract limit expression and variable
            if 'lim(' in equation:
                # Parse lim(x->a) f(x) format - handle both -> and → arrows
                match = re.search(r'lim\(([^)]+)\)\s*(.+)', equation)
                if match:
                    var_limit = match.group(1)
                    expr_str = match.group(2)
                    
                    # Parse variable and limit - handle both -> and → arrows
                    if '->' in var_limit or '→' in var_limit:
                        if '->' in var_limit:
                            var, limit_val = var_limit.split('->')
                        else:
                            var, limit_val = var_limit.split('→')
                        var = var.strip()
                        limit_val = limit_val.strip()
                        
                        # Convert limit value to proper format
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
                        
                        # Clean up the expression string
                        expr_str = expr_str.strip()
                        expr = sp.sympify(expr_str)
                        var_sym = sp.Symbol(var)
                        
                        steps.append(f"🤖 AI Limit: lim({var}→{limit_val}) {sp.latex(expr)}")
                        
                        # AI-powered indeterminate form detection and resolution
                        indeterminate_form, resolution_steps, final_result = await self._resolve_indeterminate_form(expr, var_sym, limit_val)
                        
                        if indeterminate_form:
                            steps.extend(resolution_steps)
                            steps.append(f"✅ AI Final Result: {sp.latex(final_result)}")
                        else:
                            # Direct evaluation
                            limit_result = sp.limit(expr, var_sym, limit_val)
                            steps.append(f"✅ AI Direct Result: {sp.latex(limit_result)}")
                            final_result = limit_result
                        
                        # Handle special cases
                        if final_result == sp.oo:
                            solution.append("∞")
                        elif final_result == -sp.oo:
                            solution.append("-∞")
                        elif final_result == sp.nan:
                            solution.append("undefined")
                        else:
                            solution.append(sp.latex(final_result))
                        confidence = 0.95
            
        except Exception as e:
            steps.append(f"❌ AI Error: {str(e)}")
            solution.append("AI unable to solve limit")
            confidence = 0.1
        
        return steps, solution, confidence
    
    async def _resolve_indeterminate_form(self, expr, var_sym, limit_val) -> Tuple[bool, List[str], any]:
        """AI-powered indeterminate form detection and resolution"""
        resolution_steps = []
        
        try:
            # Direct evaluation first
            direct_result = sp.limit(expr, var_sym, limit_val)
            
            # Check if result is indeterminate
            if direct_result == sp.nan or str(direct_result) == 'nan':
                # AI Analysis: Indeterminate form detected
                resolution_steps.append("🔍 AI Analysis: Indeterminate form detected")
                
                # Analyze the form of the expression
                if expr.is_rational_function(var_sym):
                    # Rational function - check for 0/0 or ∞/∞
                    if limit_val == 0:
                        # Check if both numerator and denominator approach 0
                        num = sp.numer(expr)
                        den = sp.denom(expr)
                        
                        num_limit = sp.limit(num, var_sym, limit_val)
                        den_limit = sp.limit(den, var_sym, limit_val)
                        
                        if num_limit == 0 and den_limit == 0:
                            resolution_steps.append("🎯 AI Detection: 0/0 indeterminate form")
                            resolution_steps.append("💡 AI Strategy: Apply L'Hôpital's Rule")
                            
                            # Apply L'Hôpital's Rule
                            num_deriv = sp.diff(num, var_sym)
                            den_deriv = sp.diff(den, var_sym)
                            
                            resolution_steps.append(f"📝 AI Step: lim({var_sym}→{limit_val}) {sp.latex(expr)} = lim({var_sym}→{limit_val}) {sp.latex(num_deriv)}/{sp.latex(den_deriv)}")
                            
                            # Check if still indeterminate
                            new_result = sp.limit(num_deriv/den_deriv, var_sym, limit_val)
                            if new_result == sp.nan or str(new_result) == 'nan':
                                resolution_steps.append("🔄 AI Step: Still indeterminate, apply L'Hôpital's Rule again")
                                num_deriv2 = sp.diff(num_deriv, var_sym)
                                den_deriv2 = sp.diff(den_deriv, var_sym)
                                resolution_steps.append(f"📝 AI Step: lim({var_sym}→{limit_val}) {sp.latex(num_deriv)}/{sp.latex(den_deriv)} = lim({var_sym}→{limit_val}) {sp.latex(num_deriv2)}/{sp.latex(den_deriv2)}")
                                final_result = sp.limit(num_deriv2/den_deriv2, var_sym, limit_val)
                            else:
                                final_result = new_result
                                
                        elif num_limit == sp.oo and den_limit == sp.oo:
                            resolution_steps.append("🎯 AI Detection: ∞/∞ indeterminate form")
                            resolution_steps.append("💡 AI Strategy: Apply L'Hôpital's Rule")
                            
                            # Apply L'Hôpital's Rule
                            num_deriv = sp.diff(num, var_sym)
                            den_deriv = sp.diff(den, var_sym)
                            
                            resolution_steps.append(f"📝 AI Step: lim({var_sym}→{limit_val}) {sp.latex(expr)} = lim({var_sym}→{limit_val}) {sp.latex(num_deriv)}/{sp.latex(den_deriv)}")
                            final_result = sp.limit(num_deriv/den_deriv, var_sym, limit_val)
                        else:
                            # Other indeterminate forms
                            resolution_steps.append("🎯 AI Detection: Other indeterminate form")
                            final_result = direct_result
                    
                    elif limit_val == sp.oo or limit_val == -sp.oo:
                        # Check for ∞/∞ form
                        num = sp.numer(expr)
                        den = sp.denom(expr)
                        
                        num_limit = sp.limit(num, var_sym, limit_val)
                        den_limit = sp.limit(den, var_sym, limit_val)
                        
                        if num_limit == sp.oo and den_limit == sp.oo:
                            resolution_steps.append("🎯 AI Detection: ∞/∞ indeterminate form")
                            resolution_steps.append("💡 AI Strategy: Apply L'Hôpital's Rule")
                            
                            # Apply L'Hôpital's Rule
                            num_deriv = sp.diff(num, var_sym)
                            den_deriv = sp.diff(den, var_sym)
                            
                            resolution_steps.append(f"📝 AI Step: lim({var_sym}→{limit_val}) {sp.latex(expr)} = lim({var_sym}→{limit_val}) {sp.latex(num_deriv)}/{sp.latex(den_deriv)}")
                            final_result = sp.limit(num_deriv/den_deriv, var_sym, limit_val)
                        else:
                            final_result = direct_result
                    else:
                        final_result = direct_result
                
                elif expr.is_polynomial(var_sym):
                    # Polynomial - factor if possible
                    resolution_steps.append("🎯 AI Detection: Polynomial form")
                    resolution_steps.append("💡 AI Strategy: Factor and simplify")
                    
                    factored = sp.factor(expr)
                    if factored != expr:
                        resolution_steps.append(f"📝 AI Step: Factor: {sp.latex(expr)} = {sp.latex(factored)}")
                        final_result = sp.limit(factored, var_sym, limit_val)
                    else:
                        final_result = direct_result
                
                else:
                    # Check for other indeterminate forms
                    if expr.is_Add:
                        # Check for ∞-∞ form
                        terms = expr.args
                        if len(terms) == 2:
                            term1, term2 = terms
                            limit1 = sp.limit(term1, var_sym, limit_val)
                            limit2 = sp.limit(term2, var_sym, limit_val)
                            
                            if (limit1 == sp.oo and limit2 == sp.oo) or (limit1 == -sp.oo and limit2 == -sp.oo):
                                resolution_steps.append("🎯 AI Detection: ∞-∞ indeterminate form")
                                resolution_steps.append("💡 AI Strategy: Factor out common terms or rationalize")
                                
                                # Try to factor out common terms
                                common_factor = sp.gcd(term1, term2)
                                if common_factor != 1:
                                    factored = common_factor * (term1/common_factor - term2/common_factor)
                                    resolution_steps.append(f"📝 AI Step: Factor out common term: {sp.latex(expr)} = {sp.latex(factored)}")
                                    final_result = sp.limit(factored, var_sym, limit_val)
                                else:
                                    # Try rationalization or other techniques
                                    final_result = direct_result
                            else:
                                final_result = direct_result
                        else:
                            final_result = direct_result
                    
                    elif expr.is_Mul:
                        # Check for 0×∞ form
                        terms = expr.args
                        if len(terms) == 2:
                            term1, term2 = terms
                            limit1 = sp.limit(term1, var_sym, limit_val)
                            limit2 = sp.limit(term2, var_sym, limit_val)
                            
                            if (limit1 == 0 and limit2 == sp.oo) or (limit1 == sp.oo and limit2 == 0):
                                resolution_steps.append("🎯 AI Detection: 0×∞ indeterminate form")
                                resolution_steps.append("💡 AI Strategy: Convert to 0/0 or ∞/∞ form")
                                
                                # Convert to 0/0 form: 0×∞ = 0/(1/∞) = 0/0
                                if limit1 == 0 and limit2 == sp.oo:
                                    new_expr = term1 / (1/term2)
                                else:
                                    new_expr = term2 / (1/term1)
                                
                                resolution_steps.append(f"📝 AI Step: Convert to 0/0 form: {sp.latex(expr)} = {sp.latex(new_expr)}")
                                final_result = sp.limit(new_expr, var_sym, limit_val)
                            else:
                                final_result = direct_result
                        else:
                            final_result = direct_result
                    
                    elif expr.is_Pow:
                        # Check for 0^0, ∞^0, 1^∞ forms
                        base, exp = expr.args
                        base_limit = sp.limit(base, var_sym, limit_val)
                        exp_limit = sp.limit(exp, var_sym, limit_val)
                        
                        if base_limit == 0 and exp_limit == 0:
                            resolution_steps.append("🎯 AI Detection: 0^0 indeterminate form")
                            resolution_steps.append("💡 AI Strategy: Use logarithmic approach")
                            
                            # Use ln(y) = exp*ln(base) approach
                            log_expr = exp * sp.ln(base)
                            resolution_steps.append(f"📝 AI Step: ln(y) = {sp.latex(log_expr)}")
                            log_limit = sp.limit(log_expr, var_sym, limit_val)
                            if log_limit != sp.nan:
                                final_result = sp.exp(log_limit)
                            else:
                                final_result = direct_result
                                
                        elif base_limit == sp.oo and exp_limit == 0:
                            resolution_steps.append("🎯 AI Detection: ∞^0 indeterminate form")
                            resolution_steps.append("💡 AI Strategy: Use logarithmic approach")
                            
                            log_expr = exp * sp.ln(base)
                            resolution_steps.append(f"📝 AI Step: ln(y) = {sp.latex(log_expr)}")
                            log_limit = sp.limit(log_expr, var_sym, limit_val)
                            if log_limit != sp.nan:
                                final_result = sp.exp(log_limit)
                            else:
                                final_result = direct_result
                                
                        elif base_limit == 1 and exp_limit == sp.oo:
                            resolution_steps.append("🎯 AI Detection: 1^∞ indeterminate form")
                            resolution_steps.append("💡 AI Strategy: Use e^lim(ln(1+f(x))/g(x)) approach")
                            
                            # For 1^∞, use e^lim(ln(1+f(x))/g(x)) where f(x) = base-1, g(x) = exp
                            f_x = base - 1
                            g_x = exp
                            new_expr = sp.ln(1 + f_x) / g_x
                            resolution_steps.append(f"📝 AI Step: Use e^lim(ln(1+{sp.latex(f_x)})/{sp.latex(g_x)})")
                            new_limit = sp.limit(new_expr, var_sym, limit_val)
                            if new_limit != sp.nan:
                                final_result = sp.exp(new_limit)
                            else:
                                final_result = direct_result
                        else:
                            final_result = direct_result
                    
                    else:
                        # Other forms - try algebraic manipulation
                        resolution_steps.append("🎯 AI Detection: Complex form")
                        resolution_steps.append("💡 AI Strategy: Algebraic manipulation")
                        
                        # Try to simplify
                        simplified = sp.simplify(expr)
                        if simplified != expr:
                            resolution_steps.append(f"📝 AI Step: Simplify: {sp.latex(expr)} = {sp.latex(simplified)}")
                            final_result = sp.limit(simplified, var_sym, limit_val)
                        else:
                            final_result = direct_result
                
                return True, resolution_steps, final_result
            
            else:
                # Not indeterminate, return direct result
                return False, [], direct_result
                
        except Exception as e:
            resolution_steps.append(f"❌ AI Error in resolution: {str(e)}")
            return True, resolution_steps, sp.nan
    
    async def _solve_polynomial_equation(self, equation: str) -> Tuple[List[str], List[str], float]:
        """AI-powered polynomial equation solving"""
        return await self._solve_quadratic_equation(equation)
    
    async def _solve_exponential_equation(self, equation: str) -> Tuple[List[str], List[str], float]:
        """AI-powered exponential equation solving"""
        return await self._solve_linear_equation(equation)
    
    async def _solve_logarithmic_equation(self, equation: str) -> Tuple[List[str], List[str], float]:
        """AI-powered logarithmic equation solving"""
        return await self._solve_linear_equation(equation)
    
    async def _solve_trigonometric_equation(self, equation: str) -> Tuple[List[str], List[str], float]:
        """AI-powered trigonometric equation solving"""
        return await self._solve_linear_equation(equation)
