```mermaid
classDiagram
direction LR

%% The following is an UML diagram, and it can be plugged into Mermaid to se the finished diagram,
%% in case there are some changes. This is the first draft, before adding any Gaussian Dilly Dally.

class EstimatorSystem {
  +context: EstimatorContext
  +analyzer: EstimatorAnalyzer
  +estimator(f, biasedness="naive")
  +compare(f, simplify=False)
  +compare_more(fs, simplify=False, name_fn=None)
  +pretty_print(comparison_report, latex=False)
}

class ComparisonReport {
  +system: EstimatorSystem
  +f: sympy.Expr
  +compute(simplify=False) dict
  +_simplify_result(result) dict
}

class NoiseModel {
  <<abstract>>
  +moment(n, q)
  +unbiased_transform(f, x)
  +clear_cache()
  +cache_info()
}

class LaplaceNoiseModel {
  +delta: sympy.Symbol
  +epsilon: sympy.Symbol
  +moment(i, q) sympy.Expr
  +unbiased_transform(f, x) sympy.Expr
  +clear_cache()
  +cache_info()
  -_central_moment(k) sympy.Expr
}

class GaussianNoiseModel {
  +sigma: sympy.Expr
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
}

%% Inheritance (Strategy pattern)
NoiseModel <|-- LaplaceNoiseModel : inherits
NoiseModel <|-- GaussianNoiseModel : inherits

%% Composition / aggregation
EstimatorSystem *-- EstimatorContext : owns
EstimatorSystem *-- EstimatorAnalyzer : owns

ComparisonReport o-- EstimatorSystem : references

%% "Uses" relationships
EstimatorContext ..> NoiseModel : calls unbiased_transform()
EstimatorAnalyzer ..> NoiseModel : calls moment()
ComparisonReport ..> EstimatorAnalyzer : uses mean()/variance()
ComparisonReport ..> EstimatorContext : uses naive()/unbiased()
