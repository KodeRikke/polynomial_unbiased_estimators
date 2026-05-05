```mermaid
classDiagram
direction TD

%% The following is an UML diagram, and it can be plugged into Mermaid to se the finished diagram, in case there are some changes. 
%% This is the second (or third??) draft, before adding any Gaussian Dilly Dally.

class ReportFormatter {
  +Delta: sympy.Symbol
  +epsilon: sympy.Symbol
  +beta: sympy.Symbol
  +normalize(expression) sympy.Expr
  +compact(expression) sympy.Expr
  -_line(label, expr)
  -_format_inline(expr, notation="beta", compact=False)
  +render_summary(report, notation="beta", compact=False)
  -_latex_expr(expr, notation="grouped", compact=False)
  +render_latex(report, notation="ratio", compact=False)
}

class EstimatorSystem {
  +context: EstimatorContext
  +analyzer: EstimatorAnalyzer
  +estimator(f, biasedness="naive")
  +compare(f, simplify=False)
  +compare_more(fs, simplify=False, name_fn=None)
  +summary_report(f, notation="beta", compact="False")
  +latex_compare(f, notation="grouped", compact="False")
  +pdf_report(f, output_stem, notation="grouped", title=None, compact=False)
}

class ComparisonReport {
  +system: EstimatorSystem
  +f: sympy.Expr
  +report()
}

class EstimatorContext {
  +noise_model: NoiseModel
  +q: sympy.Symbol
  +x: sympy.Symbol
  +naive(f) sympy.Expr
  +unbiased(f) sympy.Expr
}

class EstimatorAnalyzer {
  +noise_model: NoiseModel
  +q: sympy.Symbol
  +x: sympy.Symbol
  +mean(estimator) sympy.Expr
  +variance(estimator) sympy.Expr
  +mse(estimator, target_statistic) sympy.Expr
  +bias(estimator, target_statistic) sympy.Expr
}

class NoiseModel {
  <<abstract>>
  +moment(n, q)
  +unbiased_transform(f, x)
  +clear_cache()
  +cache_info()
}

class LaplaceNoiseModel {
  +Delta: sympy.Symbol
  +epsilon: sympy.Symbol
  -_moment_about_zero(k) sympy.Expr
  +moment(i, q) sympy.Expr
  +unbiased_transform(f, x) sympy.Expr
  +clear_cache()
  +cache_info()
}

class GaussianNoiseModel {
  +sigma: sympy.Expr
}

%% Inheritance (Strategy pattern)
NoiseModel <|-- LaplaceNoiseModel : inherits
NoiseModel <|-- GaussianNoiseModel : inherits

%% Composition / aggregation
EstimatorSystem *-- EstimatorContext : owns
EstimatorSystem *-- EstimatorAnalyzer : owns
EstimatorSystem *-- ReportFormatter : owns

ComparisonReport o-- EstimatorSystem : references

%% "Uses" relationships
EstimatorContext ..> NoiseModel : calls unbiased_transform()
EstimatorAnalyzer ..> NoiseModel : calls moment()
ComparisonReport ..> EstimatorSystem : calls estimator(), analyzer.mean(), analyzer.variance(), analyzer.mse(), analyzer.bias()

%% Indirect 
%%ComparisonReport ..> EstimatorAnalyzer : uses mean()/variance()/mse()/bias()
%%ComparisonReport ..> EstimatorContext : uses naive()/unbiased()
%%EstimatorSystem ..> ReportFormatter : uses render()
