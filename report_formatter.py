import sympy as sp

__all__ = ["ReportFormatter"]

sp.init_printing(use_unicode=True)

class ReportFormatter:
    def __init__(self, delta, epsilon): #ratio_name="ratio"):
        self.Delta = sp.sympify(delta)
        self.epsilon = sp.sympify(epsilon)
        self.r2 = sp.sympify((self.Delta / self.epsilon) ** 2)  # (Δ/ε)^2

#        self.ratio_name = sp.Symbol(ratio_name, real=True, positive=True)

    """The intention of this method is to take a SymPy expression and return a normalized version of it, 
    where the normalization depends on the specified mode. The modes allow for different ways of 
    grouping and simplifying the expression, particularly in terms of how the noise parameters 
    Δ and ε are represented. This is useful for making the output more interpretable and suitable 
    for presentation in reports or papers.
    OBS!!! It is NOT working good enough yet!!!"""
    def normalize(self, expression: sp.Expr, mode="grouped", simplify_level=2) -> sp.Expr:
        if expression is None: 
            return expression # nothing to normalize

        expr = sp.sympify(expression)

        if mode == "expanded":
            # expands the expression
            return sp.expand(expr) 
        
        elif mode == "noise_scale":
            beta = sp.Symbol("beta", real=True, positive=True)  # define a new symbol beta for noise scale
#            sp.Symbol('beta', locals={'beta': sp.Symbol('beta', real=True, positive=True)}) # define a new symbol beta for noise scale
            expr = sp.powsimp(expr, force=True) # simplifies powers,
            # substitute Delta/epsilon with noise scale \Beta
            expr = expr.subs(self.Delta / self.epsilon, beta)
            expr = sp.together(expr)  # combine into a single fraction
            if simplify_level >= 1:
                expr = sp.cancel(expr) # cancels common factors in numerator and denominator
            if simplify_level >= 2:
                expr = sp.factor(expr)  # factor common terms
            return sp.collect(expr, beta**2) # groups by noise scale beta^2

        elif mode == "grouped":
            expr = sp.powsimp(expr, force=True) # simplifies powers

            expr = expr.replace(
                lambda e: e.is_Pow and e.base == self.Delta / self.epsilon and e.exp == 2,
                lambda e: self.r2
            )
            expr = sp.powsimp(expr, force=True)
            if simplify_level >= 1:
                expr = sp.simplify(expr) # simplifies the expression
            
            if simplify_level >= 2:
                expr = sp.factor_terms(expr)  # factor common terms

            expr = sp.collect(expr, self.r2) # groups by (Delta/epsilon)^2
            expr = sp.factor(expr) 
            return expr

        elif mode == "normalized":
            inv_eps2 = sp.Symbol("inv_eps2", positive=True, real=True)  # placeholder
            expr = sp.powsimp(expr, force=True)
            expr = expr.subs(1 / self.epsilon**2, inv_eps2)
            expr = sp.together(expr)
            if simplify_level >= 1:
                expr = sp.cancel(expr)
                expr = sp.simplify(expr)
            if simplify_level >= 2:
                expr = sp.factor(expr)
            # collect in inv_eps2; Δ stays as a factor
            return sp.collect(expr, inv_eps2)

        else: 
            raise ValueError(f"Unknown mode: {mode}, must be either: \n 'grouped' for grouping by epsilon and Delta, \n 'noise_scale' for expressed as beta = Delta/epsilon, \n 'normalized' for normalizing by writing 1/epsilon, \n 'expanded' for fully expanding the expression.")
            
    def _indent_block(text: str, spaces: int) -> str:
        pad = " " * spaces
        return "\n".join(pad + line if line.strip() != "" else line for line in text.splitlines())

    def format_leaf(self, value, *, latex=False):
        if value is None:
            return "None"
        if isinstance(value, str):
            return value

        # SymPy expression?
        try:
            expr = sp.sympify(value)
            return sp.latex(expr) if latex else sp.pretty(expr, use_unicode=True)
        except (sp.SympifyError, TypeError):
            # fallback for numbers / objects
            return str(value)
        
    # Instead of recursively normalizing the entire report and print the dictionary, 
    # we "pretty-print" the report and print it without the dictionary structure. 
    # This is more readable. We thus return it as sp.pretty() string instead of a dictionary.
    def render(self, obj, *, mode="grouped", latex=False, simplify_level=2, indent=0) -> str:
        pad = "  " * indent
        out = []

        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k)

                if isinstance(v, (dict, list, tuple)):
                    out.append(f"{pad}{key}:")
                    out.append(self.render(v, mode=mode, latex=latex,
                                           simplify_level=simplify_level, indent=indent+1))
                else:
                    nv = self.normalize(v, mode=mode, simplify_level=simplify_level)
                    leaf = self.format_leaf(nv, latex=latex)
                    if "\n" not in leaf and len(leaf) <= 60:
                        out.append(f"{pad}{key}: {leaf}")
                    else:
                        # IMPORTANT: print leaf as a block, not inline
                        out.append(f"{pad}{key}:")
                        out.append(_indent_block(leaf, spaces=2 * (indent + 1)))

            return "\n".join(out)

        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                if isinstance(v, (dict, list, tuple)):
                    out.append(f"{pad}- [{i}]")
                    out.append(self.render(v, mode=mode, latex=latex,
                                           simplify_level=simplify_level, indent=indent+1))
                else:
                    nv = self.normalize(v, mode=mode, simplify_level=simplify_level)
                    leaf = self.format_leaf(nv, latex=latex)
                    out.append(f"{pad}-")
                    out.append(_indent_block(leaf, spaces=2 * (indent + 1)))
            return "\n".join(out)

        # leaf
        nv = self.normalize(obj, mode=mode, simplify_level=simplify_level)
        return self._indent_block(self.format_leaf(nv, latex=latex), spaces=2 * indent)


    """    
    def pretty_report(self, report: dict, mode="grouped", latex=False) -> str:
        pretty_report = {}
        for key, value in report.items():
            if isinstance(value, dict):
                pretty_report[key] = self.pretty_report(value, mode=mode, latex=latex)
            else:
                normalized_value = self.normalize(value, mode=mode)
                pretty_report[key] = sp.latex(normalized_value) if latex else normalized_value
        return pretty_report
    """
    """ Converts a SymPy expression to a LaTeX string. This is used for pretty-printing the report in LaTeX format."""
    #def latex_report(self, expression: sp.Expr) -> str:
    #    return sp.latex(expression)
        

