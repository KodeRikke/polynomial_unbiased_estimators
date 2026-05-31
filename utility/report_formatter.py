import sympy as sp

__all__ = ["ReportFormatter"]

sp.init_printing(use_unicode=True)

class ReportFormatter:
    """
    The ReportFormatter class is responsible for formatting the results of the estimation system for presentation in both terminal and LaTeX formats.
    It takes the noise parameters Delta (the sensitivity of the query) and epsilon (the privacy budget / noise scale) as input,
    and defines methods to normalize expressions, render summaries in the terminal, and render LaTeX reports.
    The normalization process involves substituting Delta with a new symbol beta, which represents the ratio of Delta to epsilon, 
    to make the expressions more interpretable and suitable for presentation. The class also includes methods to format expressions 
    for inline display and to render the summary of the estimation results in a structured format for both terminal and LaTeX outputs.
    """
    def __init__(self, Delta=None, epsilon=None, sigma=None):
        self.beta = sp.Symbol("beta", real=True, positive=True)  # define a new symbol beta for noise scale

        if sigma is not None:
            if Delta is not None or epsilon is not None:
                raise ValueError("Use either (Delta, epsilon) for Laplace or sigma for Gaussian, not both.")
            self.noise_family = "gaussian"
            self.sigma = sp.sympify(sigma)
            self.Delta = None
            self.epsilon = None
        elif Delta is not None and epsilon is not None:
            self.noise_family = "laplace"
            self.Delta = sp.sympify(Delta)
            self.epsilon = sp.sympify(epsilon)
            self.sigma = None
        else:
            raise ValueError("ReportFormatter requires either (Delta, epsilon) or sigma.")

    """
    The normalize method takes a SymPy expression and substitutes Delta with beta*epsilon, where beta is the ratio of Delta to epsilon.
    This normalization process simplifies the expressions and makes them more interpretable, especially when presenting results in the terminal or in LaTeX.
    The method also expands and collects terms by beta to further simplify the expression.

    Input: expression is a SymPy expression that may contain Delta and epsilon.
    Output: a normalized SymPy expression where Delta is replaced by beta*epsilon, and the expression is expanded and collected by beta for better readability.
    """
    def normalize(self, expression: sp.Expr) -> sp.Expr:
        if expression is None: 
            return expression # nothing to normalize

        expr = sp.sympify(expression)
        if self.noise_family == "laplace":
            expr = expr.subs({self.Delta: self.beta * self.epsilon}) # substitute Delta with beta*epsilon
            expr = sp.expand(expr) # expand the expression after substitution
            expr = sp.collect(expr, self.beta) # collect by beta (both even and odd powers)
        expr = sp.factor_terms(expr)  # factor common terms
        expr = sp.simplify(expr)
        return expr
    
    """
    The compact method takes a SymPy expression and applies a series of transformations to make it more concise and suitable for display, especially in LaTeX reports.
    It first normalizes the expression using the normalize method, and then applies additional simplification techniques such as 
    factoring common terms, collecting by beta, and canceling common factors if the expression is a rational function.
    Whether to appy the compat method is optional, deending on the context. For terminal display, this might not be necessary. 

    Input: expression is a SymPy expression that may contain Delta and epsilon.
    Output: a compacted SymPy expression that is normalized and simplified for better readability in presentations, especially in LaTeX reports.
    """
    def compact(self, expression: sp.Expr) -> sp.Expr:
        expr = self.normalize(expression)

        # Shorten for display purposes
        expr = sp.factor_terms(expr)  # factor common terms
        if self.noise_family == "laplace":
            expr = sp.collect(expr, self.beta) # collect by beta (both even and odd powers)
        expr = sp.cancel(expr) if expr.is_rational_function else expr # cancel common factors if it's a rational function
        return expr
    
#------------------------------------ For printing summary in terminal -----------------------------------------------
    """
    The _line method is a helper function that formats a single line of output for the terminal display, aligning the label and the expression for better readability.
    
    Input: 
    label: a string representing the label for the line (e.g., "estimator", "mean", "variance", etc.)
    expr: a string representation of the expression to be displayed next to the label.
    Output: a formatted string that aligns the label and the expression in a structured way for terminal display.
    """
    def _line(self, label, expr):
        return f"  {label:<10}= {expr}"

    """
    The _format_inline method takes a SymPy expression and formats it as a string for inline display in the terminal, with options for notation and compact formatting.
    It first normalizes the expression using the normalize method (or applies compact formatting if specified), and then substitutes beta back with Delta/epsilon if the "grouped" notation is selected for better readability in the terminal.
    Finally, it converts the expression to a string using sp.sstr for a more compact representation suitable for terminal display.

    Input:
    expr: a SymPy expression to be formatted for inline display.
    notation: a string indicating the notation to use for formatting the expression, either "beta" (where beta is the ratio of Delta to epsilon) or "grouped" (where expressions are grouped by beta but beta is not substituted back to Delta/epsilon).
    compact: a boolean indicating whether to apply compact formatting to the expression for a more concise display.
    Output: a string representation of the expression formatted for inline display in the terminal.
    """
    def _format_inline(self, expr, notation="beta", compact=False):

        # For display we sometimes want to keep the 
        # expressions more compact. 
        if compact:
            expr = self.compact(expr)
        else:
            expr = self.normalize(expr)

        if self.noise_family == "laplace" and notation == "grouped":
            # In the grouped notation, we want to substitute beta back with Delta/epsilon for better readability in the terminal
            expr = expr.subs(self.beta, self.Delta / self.epsilon)
        return sp.sstr(expr)

    """
    The render_summary method takes a report dictionary containing the results of the estimation system and formats it into a structured string for terminal display.
    It uses the _format_inline method to format each expression in the report according to the specified notation (either "beta" or "grouped") and whether to apply compact formatting.
    The method organizes the output into sections for the polynomial, the naive estimator, the unbiased estimator, and a comparison of the mean, variance, and MSE gaps between the two estimators. 
    The output is designed to be clear and easy to read in the terminal, with aligned labels and values.

    Input:
    report: a dictionary containing the results of the estimation system, including the polynomial, the naive estimator, the unbiased estimator, and the gaps in mean, variance, and MSE.
    notation: a string indicating the notation to use for formatting the expressions, either "beta" (where beta is the ratio of Delta to epsilon) or "grouped" (where expressions are grouped by beta but beta is not substituted back to Delta/epsilon).
    compact: a boolean indicating whether to apply compact formatting to the expressions for a more concise display.
    Output: a formatted string that summarizes the results of the estimation system in a structured and readable format for terminal display.
    """
    def render_summary(self, report, notation="beta", compact=False):
        f = self._format_inline(report["polynomial"], notation=notation, compact=compact)

        n_est = self._format_inline(report["naive"]["estimator"], notation=notation, compact=compact)
        n_mean = self._format_inline(report["naive"]["mean"], notation=notation, compact=compact)
        n_var = self._format_inline(report["naive"]["variance"], notation=notation, compact=compact)
        n_mse = self._format_inline(report["naive"]["mse"], notation=notation, compact=compact)
        n_bias = self._format_inline(report["naive"]["bias"], notation=notation, compact=compact)

        u_est = self._format_inline(report["unbiased"]["estimator"], notation=notation, compact=compact)
        u_mean = self._format_inline(report["unbiased"]["mean"], notation=notation, compact=compact)
        u_var = self._format_inline(report["unbiased"]["variance"], notation=notation, compact=compact)
        u_mse = self._format_inline(report["unbiased"]["mse"], notation=notation, compact=compact)

        mean_gap = self._format_inline(report["mean_gap"], notation=notation, compact=compact)
        var_gap = self._format_inline(report["variance_gap"], notation=notation, compact=compact)
        mse_gap = self._format_inline(report["mse_gap"], notation=notation, compact=compact)
        n_bias_squared = self._format_inline(report["bias_naive_squared"], notation=notation, compact=compact)

        lines = [
            f"f(q) = {f}",
            "",
            "Naive:",
            self._line("estimator", n_est),
            self._line("mean", n_mean),
            self._line("variance", n_var),
            self._line("mse", n_mse),
            self._line("bias", n_bias),
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
            self._line("bias_naive_squared", n_bias_squared),
        ]

        if self.noise_family == "laplace" and notation == "beta":
            lines.insert(0, "Notation: beta = Delta/epsilon")
            lines.insert(1, "")

        return "\n".join(lines)
    
# -------------------------------------------------- For rendering to LaTeX --------------------------------------------------------

    """
    The _latex_expr method is a helper function that takes a SymPy expression and formats it as a LaTeX string for rendering in the LaTeX report.
    It first normalizes the expression using the normalize method (or applies compact formatting if specified),
    and then substitutes beta back with Delta/epsilon if the "grouped" notation is selected. 
    Finally, it converts the expression to a LaTeX string using sp.latex, with an option to specify custom symbol names for better readability in the LaTeX output.

    Input:
    expr: a SymPy expression to be formatted for LaTeX rendering.
    notation: a string indicating the notation to use for formatting the expression, either "beta" (where beta is the ratio of Delta to epsilon) or "grouped" (where expressions are grouped by beta but beta is not substituted back to Delta/epsilon).
    compact: a boolean indicating whether to apply compact formatting to the expression for a more concise display in the LaTeX report.
    Output: a string representation of the expression formatted for LaTeX rendering in the report.
    """
    def _latex_expr(self, expr, notation="grouped", compact=False):
        if compact:
            expr = self.compact(expr)
        else:
            expr = self.normalize(expr) # always normalize to noise_scale for LaTeX rendering, since we want to substitute beta with Delta/epsilon in the grouped notation
        if self.noise_family == "laplace" and notation == "grouped":
            return sp.latex(
                expr,
                symbol_names={
                    self.beta: r"{\frac{\Delta}{\epsilon}}"
                }
            )
        if self.noise_family == "gaussian":
            return sp.latex(expr)
        return sp.latex(expr)

    """
    The render_latex method takes a report dictionary containing the results of the estimation system and formats it into a structured LaTeX string for rendering in a LaTeX report.
    It uses the _latex_expr method to format each expression in the report according to the specified notation (either "beta" or "grouped") and whether to apply compact formatting.
    The method organizes the output into sections for the polynomial, the naive estimator, the unbiased estimator, and a comparison of the mean, variance, and MSE gaps between the two estimators. 
    The output is designed to be clear and well-structured for rendering in a LaTeX document, with appropriate use of math environments and formatting for mathematical expressions.

    Input:
    report: a dictionary containing the results of the estimation system, including the polynomial, the naive estimator, the unbiased estimator, and the gaps in mean, variance, and MSE.
    notation: a string indicating the notation to use for formatting the expressions, either "beta" (where beta is the ratio of Delta to epsilon) or "grouped" (where expressions are grouped by beta but beta is not substituted back to Delta/epsilon).
    compact: a boolean indicating whether to apply compact formatting to the expressions for a more concise display in the LaTeX report.
    Output: a formatted string that summarizes the results of the estimation system in a structured and well-formatted way for rendering in a LaTeX document. 
    """
    def render_latex(self, report, notation="ratio", compact=False):
        f = self._latex_expr(report["polynomial"], notation=notation, compact=compact)

        n_est = self._latex_expr(report["naive"]["estimator"], notation=notation, compact=compact)
        n_mean = self._latex_expr(report["naive"]["mean"], notation=notation, compact=compact)
        n_var = self._latex_expr(report["naive"]["variance"], notation=notation, compact=compact)
        n_mse = self._latex_expr(report["naive"]["mse"], notation=notation, compact=compact)
        n_bias = self._latex_expr(report["naive"]["bias"], notation=notation, compact=compact)

        u_est = self._latex_expr(report["unbiased"]["estimator"], notation=notation, compact=compact)
        u_mean = self._latex_expr(report["unbiased"]["mean"], notation=notation, compact=compact)
        u_var = self._latex_expr(report["unbiased"]["variance"], notation=notation, compact=compact)
        u_mse = self._latex_expr(report["unbiased"]["mse"], notation=notation, compact=compact)

        mean_gap = self._latex_expr(report["mean_gap"], notation=notation, compact=compact)
        var_gap = self._latex_expr(report["variance_gap"], notation=notation, compact=compact)
        mse_gap = self._latex_expr(report["mse_gap"], notation=notation, compact=compact)
        n_bias_squared = self._latex_expr(report["bias_naive_squared"], notation=notation, compact=compact)

        lines = []

        if self.noise_family == "laplace" and notation == "beta":
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
            rf"g_{{\mathrm{{naive}}}}(Q) = {n_est}",
            r"\end{dmath*}",
            r"\begin{dmath*}",
            rf"\mathbb{{E}}[g_{{\mathrm{{naive}}}}(Q)] = {n_mean}",
            r"\end{dmath*}",
            r"\begin{dmath*}",
            rf"\operatorname{{Var}}(g_{{\mathrm{{naive}}}}(Q)) = {n_var}",
            r"\end{dmath*}",
            r"\begin{dmath*}",
            rf"\operatorname{{MSE}}(g_{{\mathrm{{naive}}}}) = {n_mse}",
            r"\end{dmath*}",
            r"\begin{dmath*}",
            rf"\operatorname{{Bias}}(g_{{\mathrm{{naive}}}}) = {n_bias}",
            r"\end{dmath*}",

            r"\section*{Unbiased}",
            r"\begin{dmath*}",
            rf"g_{{\mathrm{{unb}}}}(Q) = {u_est}",
            r"\end{dmath*}",
            r"\begin{dmath*}",
            rf"\mathbb{{E}}[g_{{\mathrm{{unb}}}}(Q)] = {u_mean}",
            r"\end{dmath*}",
            r"\begin{dmath*}",
            rf"\operatorname{{Var}}(g_{{\mathrm{{unb}}}}(Q)) = {u_var}",
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
            r"\begin{dmath*}",
            rf"\operatorname{{Bias}}(g_{{\mathrm{{naive}}}})^2 = {n_bias_squared}",
            r"\end{dmath*}",
        ]

        return "\n".join(lines)