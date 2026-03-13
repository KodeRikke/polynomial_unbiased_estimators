import sympy as sp

__all__ = ["ReportFormatter"]

sp.init_printing(use_unicode=True)

class ReportFormatter:
    def __init__(self, delta, epsilon): #ratio_name="ratio"):
        self.Delta = sp.sympify(delta)
        self.epsilon = sp.sympify(epsilon)
        self.beta = sp.Symbol("beta", real=True, positive=True)  # define a new symbol beta for noise scale

    """The intention of this method is to take a SymPy expression and return a normalized version of it, 
    where the normalization depends on the specified notation. The notations allow for different ways of 
    grouping and simplifying the expression, particularly in terms of how the noise parameters 
    Δ and ε are represented. This is useful for making the output more interpretable and suitable 
    for presentation in reports or papers.
    OBS!!! It is NOT working good enough yet!!!"""
    def normalize(self, expression: sp.Expr) -> sp.Expr:
        if expression is None: 
            return expression # nothing to normalize

        expr = sp.sympify(expression)
        expr_beta = expr.subs({self.Delta: self.beta * self.epsilon}) # substitute Delta with beta*epsilon
        expr_beta = sp.expand(expr_beta) # expand the expression after substitution
        expr_beta = sp.collect(expr_beta, self.beta) # collect by beta (both even and odd powers)
        expr_beta = sp.factor_terms(expr_beta)  # factor common terms
        return expr_beta
    
    def compact(self, expression: sp.Expr) -> sp.Expr:
        expr = self.normalize(expression)

        # Shorten for display purposes
        expr = sp.factor_terms(expr)  # factor common terms
        expr= sp.collect(expr, self.beta) # collect by beta (both even and odd powers)
        expr = sp.cancel(expr) if expr.is_rational_function else expr # cancel common factors if it's a rational function
        return expr
    
#------------------------------------ For printing summary in terminal -----------------------------------------------
    def _line(self, label, expr):
        return f"  {label:<10}= {expr}"

    def _format_inline(self, expr, notation="beta", compact=False):

        # For display we sometimes want to keep the 
        # expressions more compact. 
        if compact:
            expr = self.compact(expr)
        else:
            expr = self.normalize(expr)

        if notation == "grouped":
            # In the grouped notation, we want to substitute beta back with Delta/epsilon for better readability in the terminal
            expr = expr.subs(self.beta, self.Delta / self.epsilon)
        return sp.sstr(expr)

    def render_summary(self, report, notation="beta", compact=False):
        f = self._format_inline(report["polynomial"], notation=notation, compact=compact)

        n_est = self._format_inline(report["naive"]["estimator"], notation=notation, compact=compact)
        n_mean = self._format_inline(report["naive"]["mean"], notation=notation, compact=compact)
        n_var = self._format_inline(report["naive"]["variance"], notation=notation, compact=compact)
        n_mse = self._format_inline(report["naive"]["mse"], notation=notation, compact=compact)

        u_est = self._format_inline(report["unbiased"]["estimator"], notation=notation, compact=compact)
        u_mean = self._format_inline(report["unbiased"]["mean"], notation=notation, compact=compact)
        u_var = self._format_inline(report["unbiased"]["variance"], notation=notation, compact=compact)
        u_mse = self._format_inline(report["unbiased"]["mse"], notation=notation, compact=compact)

        mean_gap = self._format_inline(report["mean_gap"], notation=notation, compact=compact)
        var_gap = self._format_inline(report["variance_gap"], notation=notation, compact=compact)
        mse_gap = self._format_inline(report["mse_gap"], notation=notation, compact=compact)

        lines = [
            f"f(q) = {f}",
            "",
            "Naive:",
            self._line("estimator", n_est),
            self._line("mean", n_mean),
            self._line("variance", n_var),
            self._line("mse", n_mse),
            "",
            "Unbiased:",
            self._line("estimator", u_est),
            self._line("mean", u_mean),
            self._line("variance", u_var),
            self._line("mse", u_mse),
            "",
            "Comparison:",
            self._line("mean_gap", mean_gap),
            self._line("var_gap", var_gap),
            self._line("mse_gap", mse_gap),
        ]

        if notation == "beta":
            lines.insert(0, "Notation: beta = Delta/epsilon")
            lines.insert(1, "")

        return "\n".join(lines)
    
# -------------------------------------------------- For rendering to LaTeX --------------------------------------------------------
    def _latex_expr(self, expr, notation="grouped", compact=False):
        if compact:
            expr = self.compact(expr)
        else:
            expr = self.normalize(expr) # always normalize to noise_scale for LaTeX rendering, since we want to substitute beta with Delta/epsilon in the grouped notation
        if notation == "grouped":
            return sp.latex(
                expr,
                symbol_names={
                    self.beta: r"{\frac{\Delta}{\epsilon}}"
                }
            )
        return sp.latex(expr)

    def render_latex(self, report, notation="ratio", compact=False):
        f = self._latex_expr(report["polynomial"], notation=notation, compact=compact)

        n_est = self._latex_expr(report["naive"]["estimator"], notation=notation, compact=compact)
        n_mean = self._latex_expr(report["naive"]["mean"], notation=notation, compact=compact)
        n_var = self._latex_expr(report["naive"]["variance"], notation=notation, compact=compact)
        n_mse = self._latex_expr(report["naive"]["mse"], notation=notation, compact=compact)

        u_est = self._latex_expr(report["unbiased"]["estimator"], notation=notation, compact=compact)
        u_mean = self._latex_expr(report["unbiased"]["mean"], notation=notation, compact=compact)
        u_var = self._latex_expr(report["unbiased"]["variance"], notation=notation, compact=compact)
        u_mse = self._latex_expr(report["unbiased"]["mse"], notation=notation, compact=compact)

        mean_gap = self._latex_expr(report["mean_gap"], notation=notation, compact=compact)
        var_gap = self._latex_expr(report["variance_gap"], notation=notation, compact=compact)
        mse_gap = self._latex_expr(report["mse_gap"], notation=notation, compact=compact)

        lines = []

        if notation == "beta":
            lines += [
                r"\[\beta = \frac{\Delta}{\epsilon}\]",
                ""
            ]

        lines += [
            r"\section*{Polynomial}",
            r"\begin{dmath*}",
            rf"f(q) = {f}",
            r"\end{dmath*}",

            r"\section*{Naive}",
            r"\begin{dmath*}",
            rf"g_{{\mathrm{{naive}}}}(X) = {n_est}",
            r"\end{dmath*}",
            r"\begin{dmath*}",
            rf"\mathbb{{E}}[g_{{\mathrm{{naive}}}}(X)] = {n_mean}",
            r"\end{dmath*}",
            r"\begin{dmath*}",
            rf"\operatorname{{Var}}(g_{{\mathrm{{naive}}}}(X)) = {n_var}",
            r"\end{dmath*}",
            r"\begin{dmath*}",
            rf"\operatorname{{MSE}}(g_{{\mathrm{{naive}}}}) = {n_mse}",
            r"\end{dmath*}",

            r"\section*{Unbiased}",
            r"\begin{dmath*}",
            rf"g_{{\mathrm{{unb}}}}(X) = {u_est}",
            r"\end{dmath*}",
            r"\begin{dmath*}",
            rf"\mathbb{{E}}[g_{{\mathrm{{unb}}}}(X)] = {u_mean}",
            r"\end{dmath*}",
            r"\begin{dmath*}",
            rf"\operatorname{{Var}}(g_{{\mathrm{{unb}}}}(X)) = {u_var}",
            r"\end{dmath*}",
            r"\begin{dmath*}",
            rf"\operatorname{{MSE}}(g_{{\mathrm{{unb}}}}) = {u_mse}",
            r"\end{dmath*}",

            r"\section*{Comparison}",
            r"\begin{dmath*}",
            rf"\text{{mean gap}} = {mean_gap}",
            r"\end{dmath*}",
            r"\begin{dmath*}",
            rf"\text{{variance gap}} = {var_gap}",
            r"\end{dmath*}",
            r"\begin{dmath*}",
            rf"\text{{MSE gap}} = {mse_gap}",
            r"\end{dmath*}",
        ]

        return "\n".join(lines)