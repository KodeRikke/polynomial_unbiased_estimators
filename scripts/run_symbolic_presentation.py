from pathlib import Path
import sys
from plotting.plot_symbolic_presentation import render_figure_polynomial_overlay, render_figure_epsilon_fixed
from plotting.plot_symbolic_presentation import build_reports, parse_args

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    # count produced figures to confirm all 8 are generated as expected
    count = 0
    args = parse_args()
    reports = build_reports()
    out_dir = Path(args.output_dir)

    # Set 1: Fixed polynomial per plot (normal-degree and higher-degree)
    normal_variance_path = out_dir / "symbolic_presentation_normal_polynomial_overlay_variance.png"
    normal_mse_path = out_dir / "symbolic_presentation_normal_polynomial_overlay_mse.png"
#    higher_variance_path = out_dir / "symbolic_presentation_higher_polynomial_overlay_variance.png"
#    higher_mse_path = out_dir / "symbolic_presentation_higher_polynomial_overlay_mse.png"

    render_figure_polynomial_overlay(
        reports=reports,
        metric="variance_ratio",
        q_range=(args.q_min, args.q_max),
        epsilons=args.epsilons,
        delta_value=args.delta,
        Delta_value=args.Delta,
        out_path=normal_variance_path,
        is_higher_degree=False,
    )
    render_figure_polynomial_overlay(
        reports=reports,
        metric="mse_ratio",
        q_range=(args.q_min, args.q_max),
        epsilons=args.epsilons,
        delta_value=args.delta,
        Delta_value=args.Delta,
        out_path=normal_mse_path,
        is_higher_degree=False,
    )
#    render_figure_polynomial_overlay(
#        reports=reports,
#        metric="variance_ratio",
#        q_range=(args.q_min, args.q_max),
#        epsilons=args.epsilons,
#        delta_value=args.delta,
#        Delta_value=args.Delta,
#        out_path=higher_variance_path,
#        is_higher_degree=True,
#    )
#    render_figure_polynomial_overlay(
#        reports=reports,
#        metric="mse_ratio",
#        q_range=(args.q_min, args.q_max),
#        epsilons=args.epsilons,
#        delta_value=args.delta,
#        Delta_value=args.Delta,
#        out_path=higher_mse_path,
#        is_higher_degree=True,
#    )

    # Set 2: Fixed epsilon per plot (normal-degree and higher-degree)
    normal_eps_variance_path = out_dir / "symbolic_presentation_normal_polynomial_epsilon_variance.png"
    normal_eps_mse_path = out_dir / "symbolic_presentation_normal_polynomial_epsilon_mse.png"
#    higher_eps_variance_path = out_dir / "symbolic_presentation_higher_polynomial_epsilon_variance.png"
#    higher_eps_mse_path = out_dir / "symbolic_presentation_higher_polynomial_epsilon_mse.png"

    render_figure_epsilon_fixed(
        reports=reports,
        metric="variance_ratio",
        q_range=(args.q_min, args.q_max),
        epsilons=args.epsilons,
        delta_value=args.delta,
        Delta_value=args.Delta,
        out_path=normal_eps_variance_path,
        is_higher_degree=False,
    )
    render_figure_epsilon_fixed(
        reports=reports,
        metric="mse_ratio",
        q_range=(args.q_min, args.q_max),
        epsilons=args.epsilons,
        delta_value=args.delta,
        Delta_value=args.Delta,
        out_path=normal_eps_mse_path,
        is_higher_degree=False,
    )
#    render_figure_epsilon_fixed(
#        reports=reports,
#        metric="variance_ratio",
#        q_range=(args.q_min, args.q_max),
#        epsilons=args.epsilons,
#        delta_value=args.delta,
#        Delta_value=args.Delta,
#        out_path=higher_eps_variance_path,
#        is_higher_degree=True,
#    )
#    render_figure_epsilon_fixed(
#        reports=reports,
#        metric="mse_ratio",
#        q_range=(args.q_min, args.q_max),
#        epsilons=args.epsilons,
#        delta_value=args.delta,
#        Delta_value=args.Delta,
#        out_path=higher_eps_mse_path,
#        is_higher_degree=True,
#    )

    print(f"Saved 8 presentation figures in: {out_dir}")


if __name__ == "__main__":
    main()