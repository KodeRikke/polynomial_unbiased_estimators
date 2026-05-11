import argparse
import csv
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in [
            "degree",
            "q",
            "epsilon",
            "sweep_value",
            "emp_var_ratio_unbiased_over_naive",
            "emp_mse_ratio_unbiased_over_naive",
            "err_unbiased_var_rel",
            "err_unbiased_mse_rel",
        ]:
            # Use get to avoid KeyError when CSV missing a column; missing/invalid values become NaN
            r[k] = _to_float(r.get(k, ""))
        r["noise"] = str(r.get("noise", ""))
        r["family"] = str(r.get("family", ""))
        r["sweep_parameter"] = str(r.get("sweep_parameter", ""))
    return rows


def _format_ratio(value: float) -> str:
    if not np.isfinite(value):
        return "∞"
    return f"{value:.3f}"


def _format_pct(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.1f}%"


def summarize_degree_phase(rows: list[dict], noise: str) -> list[dict]:
    grouped = defaultdict(list)
    for r in rows:
        if r["noise"] != noise:
            continue
        deg = int(r["degree"])
        eps = float(r["epsilon"])
        val = r["emp_mse_ratio_unbiased_over_naive"]
        if np.isfinite(val):
            grouped[(deg, eps)].append(val)

    summary_rows: list[dict] = []
    for (deg, eps), values in sorted(grouped.items()):
        if not values:
            continue
        naive_wins = int(sum(v > 1.0 for v in values))
        total_q = int(len(values))
        summary_rows.append(
            {
                "degree": deg,
                "epsilon": eps,
                "median_ratio": float(np.median(values)),
                "naive_wins": naive_wins,
                "total_q": total_q,
                "pct_naive_wins": 100.0 * naive_wins / total_q if total_q else np.nan,
            }
        )
    return summary_rows


def _parse_coefficient_token(token: str) -> float:
    if token == "baseline":
        return np.nan
    negative = token.startswith("m")
    if negative:
        token = token[1:]
    token = token.replace("_", "/")
    value = float(Fraction(token))
    return -value if negative else value


def _parse_coefficient_poly_name(poly_name: str):
    parts = poly_name.split("_")
    if not parts:
        return None

    shape = parts[0]
    if shape not in {"cubic", "quartic"}:
        return None

    if poly_name.endswith("baseline"):
        return shape, "baseline", np.nan

    if len(parts) < 4 or parts[1] not in {"a", "b", "c", "d"} or parts[2] != "only":
        return None

    axis = parts[1]
    # Pattern is shape_a_only_<encoded value>.
    encoded_value = "_".join(parts[3:])
    return shape, axis, _parse_coefficient_token(encoded_value)


def _parse_coefficient_pairwise_poly_name(poly_name: str):
    """
    Parse pairwise interaction poly names like:
      cubic_ab_pair_1_2
      quartic_cd_pair_m1_0
    Returns (shape, pair, coeff1, coeff2) or None if not a pairwise poly.
    """
    parts = poly_name.split("_")
    if not parts or len(parts) < 5:
        return None

    shape = parts[0]
    if shape not in {"cubic", "quartic"}:
        return None

    if parts[2] != "pair":
        return None

    pair = f"{parts[1][0]}{parts[1][1]}"
    if not all(c in "abcd" for c in pair):
        return None

    # Remaining parts form the two encoded coefficient values
    # Pattern: shape_XY_pair_<encoded1>_<encoded2> or with more parts
    encoded1 = parts[3]
    encoded2 = "_".join(parts[4:])  # In case encoded value itself has underscores

    coeff1 = _parse_coefficient_token(encoded1)
    coeff2 = _parse_coefficient_token(encoded2)

    if not np.isfinite(coeff1) or not np.isfinite(coeff2):
        return None

    return shape, pair, coeff1, coeff2


def summarize_coefficient_pairwise_phase(rows: list[dict], noise: str) -> list[dict]:
    grouped = defaultdict(list)
    for r in rows:
        if r["noise"] != noise:
            continue
        parsed = _parse_coefficient_pairwise_poly_name(r["poly_name"])
        if parsed is None:
            continue
        shape, pair, coeff1, coeff2 = parsed
        eps = float(r["epsilon"])
        val = r["emp_mse_ratio_unbiased_over_naive"]
        if np.isfinite(val):
            grouped[(shape, pair, coeff1, coeff2, eps)].append(val)

    summary_rows: list[dict] = []
    for (shape, pair, coeff1, coeff2, eps), values in sorted(grouped.items()):
        if not values:
            continue
        naive_wins = int(sum(v > 1.0 for v in values))
        total_q = int(len(values))
        summary_rows.append(
            {
                "shape": shape,
                "pair": pair,
                "coeff1": coeff1,
                "coeff2": coeff2,
                "epsilon": eps,
                "median_ratio": float(np.median(values)),
                "naive_wins": naive_wins,
                "total_q": total_q,
                "pct_naive_wins": 100.0 * naive_wins / total_q if total_q else np.nan,
            }
        )
    return summary_rows


def plot_coefficient_pairwise_summary(rows: list[dict], out_dir: Path, source_csv: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for noise in sorted({r["noise"] for r in rows if r["noise"]}):
        noise_rows = summarize_coefficient_pairwise_phase(rows, noise=noise)
        if not noise_rows:
            continue

        for shape in sorted({r["shape"] for r in noise_rows}):
            shape_rows = [r for r in noise_rows if r["shape"] == shape]
            if not shape_rows:
                continue

            pairs = sorted({r["pair"] for r in shape_rows})
            for pair in pairs:
                pair_rows = [r for r in shape_rows if r["pair"] == pair]
                if not pair_rows:
                    continue

                coeff_axis1_values = sorted({r["coeff1"] for r in pair_rows})
                coeff_axis2_values = sorted({r["coeff2"] for r in pair_rows})
                eps_values = sorted({r["epsilon"] for r in pair_rows})

                pair_dir = out_dir / noise / shape / pair
                pair_dir.mkdir(parents=True, exist_ok=True)

                # Generate heatmaps for each epsilon
                fig_rows = len(eps_values)
                fig, axes_grid = plt.subplots(fig_rows, 2, figsize=(14, 5 * max(1, fig_rows)), squeeze=False)
                
                for eps_idx, eps in enumerate(eps_values):
                    eps_rows = [r for r in pair_rows if abs(r["epsilon"] - eps) < 1e-12]
                    
                    ratio_grid = np.full((len(coeff_axis1_values), len(coeff_axis2_values)), np.nan)
                    win_grid = np.full((len(coeff_axis1_values), len(coeff_axis2_values)), np.nan)
                    
                    for r in eps_rows:
                        c1_idx = coeff_axis1_values.index(r["coeff1"])
                        c2_idx = coeff_axis2_values.index(r["coeff2"])
                        ratio_grid[c1_idx, c2_idx] = r["median_ratio"]
                        win_grid[c1_idx, c2_idx] = r["pct_naive_wins"]

                    ratio_ax = axes_grid[eps_idx, 0]
                    win_ax = axes_grid[eps_idx, 1]

                    ratio_im = ratio_ax.imshow(ratio_grid, aspect="auto", origin="lower", cmap="RdYlGn_r")
                    ratio_ax.set_title(f"{shape} {pair} at ε={eps:g}: median MSE ratio")
                    ratio_ax.set_xlabel(f"Coefficient {pair[1]}")
                    ratio_ax.set_ylabel(f"Coefficient {pair[0]}")
                    ratio_ax.set_xticks(range(len(coeff_axis2_values)))
                    ratio_ax.set_xticklabels([f"{c:g}" for c in coeff_axis2_values], rotation=45)
                    ratio_ax.set_yticks(range(len(coeff_axis1_values)))
                    ratio_ax.set_yticklabels([f"{c:g}" for c in coeff_axis1_values])
                    fig.colorbar(ratio_im, ax=ratio_ax, fraction=0.046, pad=0.04, label="MSE ratio")

                    win_im = win_ax.imshow(win_grid, aspect="auto", origin="lower", cmap="YlOrRd", vmin=0, vmax=100)
                    win_ax.set_title(f"{shape} {pair} at ε={eps:g}: naive win %")
                    win_ax.set_xlabel(f"Coefficient {pair[1]}")
                    win_ax.set_ylabel(f"Coefficient {pair[0]}")
                    win_ax.set_xticks(range(len(coeff_axis2_values)))
                    win_ax.set_xticklabels([f"{c:g}" for c in coeff_axis2_values], rotation=45)
                    win_ax.set_yticks(range(len(coeff_axis1_values)))
                    win_ax.set_yticklabels([f"{c:g}" for c in coeff_axis1_values])
                    fig.colorbar(win_im, ax=win_ax, fraction=0.046, pad=0.04, label="% naive wins")

                plt.tight_layout()
                plt.savefig(pair_dir / "pairwise_heatmaps.png", dpi=150)
                plt.close()

                # Generate markdown summary
                lines = [
                    f"# {shape.title()} {pair.upper()} Pairwise Summary",
                    "",
                    f"**Source:** `{source_csv}`  ",
                    f"**Noise model:** `{noise}`  ",
                    f"**Family:** `{shape}`  ",
                    f"**Pair:** `{pair}`  ",
                    "",
                    "## Key Finding",
                    "",
                    f"The tables below show median MSE ratio behavior for the {pair.upper()} pairwise interaction across epsilon values.",
                    "",
                    "---",
                    "",
                ]

                for eps in eps_values:
                    eps_rows_local = [r for r in pair_rows if abs(r["epsilon"] - eps) < 1e-12]
                    if not eps_rows_local:
                        continue

                    lines.extend([
                        f"## Epsilon = {eps:g}",
                        "",
                        f"### Median MSE Ratio",
                        "",
                    ])
                    lines.append(f"| {pair[0]} \\ {pair[1]} |" + "".join(f" {c2:>6g} |" for c2 in coeff_axis2_values))
                    lines.append("|" + "".join("-" * 10 for _ in range(len(coeff_axis2_values) + 1)) + "|")
                    
                    for c1 in coeff_axis1_values:
                        row = [f"| {c1:>6g} |"]
                        for c2 in coeff_axis2_values:
                            r = next((r for r in eps_rows_local if abs(r["coeff1"] - c1) < 1e-12 and abs(r["coeff2"] - c2) < 1e-12), None)
                            val = r["median_ratio"] if r is not None else np.nan
                            row.append(f" {_format_ratio(val):>6} |")
                        lines.append("".join(row))
                    lines.append("")

                lines.extend([
                    "---",
                    "",
                    "## Interpretation",
                    "",
                    f"1. Each table shows median ratio behavior for the {pair} pair at a fixed epsilon.",
                    "2. Rows represent coefficient {0}, columns represent coefficient {1}.".format(pair[0], pair[1]),
                    "3. Values < 1 mean unbiased is better; values > 1 mean naive wins in the median.",
                    "",
                ])

                summary_path = pair_dir / "pairwise_summary.md"
                summary_path.write_text("\n".join(lines))


def summarize_coefficient_phase(rows: list[dict], noise: str) -> list[dict]:
    grouped = defaultdict(list)
    for r in rows:
        if r["noise"] != noise:
            continue
        parsed = _parse_coefficient_poly_name(r["poly_name"])
        if parsed is None:
            continue
        shape, axis, coeff_value = parsed
        if axis == "baseline" or not np.isfinite(coeff_value):
            continue
        eps = float(r["epsilon"])
        val = r["emp_mse_ratio_unbiased_over_naive"]
        if np.isfinite(val):
            grouped[(shape, axis, coeff_value, eps)].append(val)

    summary_rows: list[dict] = []
    for (shape, axis, coeff_value, eps), values in sorted(grouped.items()):
        if not values:
            continue
        naive_wins = int(sum(v > 1.0 for v in values))
        total_q = int(len(values))
        summary_rows.append(
            {
                "shape": shape,
                "axis": axis,
                "coefficient_value": coeff_value,
                "epsilon": eps,
                "median_ratio": float(np.median(values)),
                "naive_wins": naive_wins,
                "total_q": total_q,
                "pct_naive_wins": 100.0 * naive_wins / total_q if total_q else np.nan,
            }
        )
    return summary_rows


def write_coefficient_summary_markdown(rows: list[dict], out_path: Path, source_csv: Path, noise: str, shape: str):
    if not rows:
        return

    shape_rows = [r for r in rows if r["shape"] == shape]
    if not shape_rows:
        return

    axes = sorted({r["axis"] for r in shape_rows})
    eps_values = sorted({r["epsilon"] for r in shape_rows})
    coeff_values = sorted({r["coefficient_value"] for r in shape_rows})

    def lookup(metric: str, axis: str, coeff_value: float, epsilon: float) -> float:
        item = next(
            (
                r
                for r in shape_rows
                if r["axis"] == axis and abs(r["coefficient_value"] - coeff_value) < 1e-12 and abs(r["epsilon"] - epsilon) < 1e-12
            ),
            None,
        )
        return float(item[metric]) if item is not None else np.nan

    lines = [
        f"# {shape.title()} Coefficient Sweep Summary",
        "",
        f"**Source:** `{source_csv}`  ",
        f"**Noise model:** `{noise}`  ",
        f"**Family:** `{shape}`  ",
        "",
        "## Key Finding",
        "",
        "The tables below aggregate MSE ratio behavior across q for the one-factor-at-a-time coefficient sweeps.",
        "",
        "---",
    ]

    for axis in axes:
        axis_rows = [r for r in shape_rows if r["axis"] == axis]
        if not axis_rows:
            continue

        lines.extend([
            "",
            f"## Axis `{axis}`",
            "",
            "### Median MSE Ratio by Coefficient Value and Epsilon",
            "",
        ])
        lines.append("| Coefficient |" + "".join(f" ε={eps:g} |" for eps in eps_values) + " Crossover? |")
        lines.append("|------------|" + "-------|" * len(eps_values) + "-----------|")
        for coeff_value in coeff_values:
            has_crossover = False
            row = [f"| {coeff_value:g} |"]
            for eps in eps_values:
                value = lookup("median_ratio", axis, coeff_value, eps)
                row.append(f" {_format_ratio(value):>6} |")
                has_crossover = has_crossover or (np.isfinite(value) and value > 1.0)
            row.append(f" {'✓' if has_crossover else 'No':^11} |")
            lines.append("".join(row))

        lines.extend([
            "",
            "### Percentage of Rows Where Unbiased Is Better",
            "",
        ])
        lines.append("| Coefficient |" + "".join(f" ε={eps:g} |" for eps in eps_values) + "|")
        lines.append("|------------|" + "-------|" * len(eps_values))
        for coeff_value in coeff_values:
            row = [f"| {coeff_value:g} |"]
            for eps in eps_values:
                value = lookup("pct_naive_wins", axis, coeff_value, eps)
                pct_better = 100.0 - value if np.isfinite(value) else np.nan
                row.append(f" {_format_pct(pct_better):>6} |")
            lines.append("".join(row))

        lines.append("")

    lines.extend([
        "---",
        "",
        "## Interpretation",
        "",
        "1. Each table collapses the q dimension and shows the median estimator ratio for a fixed coefficient value.",
        "2. If a coefficient setting crosses above 1, the naive estimator wins somewhere in q for that setting.",
        "3. The percentage table shows how often naive wins across the q grid.",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def plot_coefficient_summary(rows: list[dict], out_dir: Path, source_csv: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for noise in sorted({r["noise"] for r in rows if r["noise"]}):
        noise_rows = summarize_coefficient_phase(rows, noise=noise)
        if not noise_rows:
            continue

        for shape in sorted({r["shape"] for r in noise_rows}):
            shape_rows = [r for r in noise_rows if r["shape"] == shape]
            if not shape_rows:
                continue

            shape_dir = out_dir / noise / shape
            shape_dir.mkdir(parents=True, exist_ok=True)
            write_coefficient_summary_markdown(shape_rows, shape_dir / "crossover_summary.md", source_csv, noise=noise, shape=shape)

            axes = sorted({r["axis"] for r in shape_rows})
            eps_values = sorted({r["epsilon"] for r in shape_rows})
            coeff_values = sorted({r["coefficient_value"] for r in shape_rows})

            fig, axes_grid = plt.subplots(len(axes), 2, figsize=(14, 4 * max(1, len(axes))), squeeze=False)
            for row_idx, axis in enumerate(axes):
                axis_rows = [r for r in shape_rows if r["axis"] == axis]
                ratio_grid = np.full((len(coeff_values), len(eps_values)), np.nan)
                win_grid = np.full((len(coeff_values), len(eps_values)), np.nan)
                for r in axis_rows:
                    ci = coeff_values.index(r["coefficient_value"])
                    ei = eps_values.index(r["epsilon"])
                    ratio_grid[ci, ei] = r["median_ratio"]
                    win_grid[ci, ei] = r["pct_naive_wins"]

                ratio_ax = axes_grid[row_idx, 0]
                win_ax = axes_grid[row_idx, 1]

                ratio_im = ratio_ax.imshow(ratio_grid, aspect="auto", origin="lower", cmap="RdYlGn_r")
                ratio_ax.set_title(f"{shape} {axis}: median MSE ratio")
                ratio_ax.set_xlabel("Epsilon")
                ratio_ax.set_ylabel("Coefficient value")
                ratio_ax.set_xticks(range(len(eps_values)))
                ratio_ax.set_xticklabels([f"{eps:g}" for eps in eps_values], rotation=45)
                ratio_ax.set_yticks(range(len(coeff_values)))
                ratio_ax.set_yticklabels([f"{c:g}" for c in coeff_values])
                fig.colorbar(ratio_im, ax=ratio_ax, fraction=0.046, pad=0.04, label="MSE ratio")

                win_im = win_ax.imshow(win_grid, aspect="auto", origin="lower", cmap="YlOrRd", vmin=0, vmax=100)
                win_ax.set_title(f"{shape} {axis}: naive win %")
                win_ax.set_xlabel("Epsilon")
                win_ax.set_ylabel("Coefficient value")
                win_ax.set_xticks(range(len(eps_values)))
                win_ax.set_xticklabels([f"{eps:g}" for eps in eps_values], rotation=45)
                win_ax.set_yticks(range(len(coeff_values)))
                win_ax.set_yticklabels([f"{c:g}" for c in coeff_values])
                fig.colorbar(win_im, ax=win_ax, fraction=0.046, pad=0.04, label="% naive wins")

            plt.tight_layout()
            plt.savefig(shape_dir / "coefficient_phase_heatmaps.png", dpi=150)
            plt.close()

            fig, axes_grid = plt.subplots(len(axes), 1, figsize=(11, 3.6 * max(1, len(axes))), squeeze=False)
            for row_idx, axis in enumerate(axes):
                axis_rows = [r for r in shape_rows if r["axis"] == axis]
                ax = axes_grid[row_idx, 0]
                for eps in eps_values:
                    xs = [r["coefficient_value"] for r in axis_rows if r["epsilon"] == eps]
                    ys = [r["median_ratio"] for r in axis_rows if r["epsilon"] == eps]
                    if xs:
                        ax.plot(xs, ys, marker="o", label=f"ε={eps:g}")
                ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
                ax.set_yscale("log")
                ax.set_title(f"{shape} {axis}: median MSE ratio vs coefficient value")
                ax.set_xlabel("Coefficient value")
                ax.set_ylabel("Median MSE ratio")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(shape_dir / "coefficient_trends.png", dpi=150)
            plt.close()


def write_degree_summary_markdown(rows: list[dict], out_path: Path, source_csv: Path):
    if not rows:
        return

    degrees = sorted({r["degree"] for r in rows})
    eps_values = sorted({r["epsilon"] for r in rows})
    ratio_lookup = {(r["degree"], r["epsilon"]): r["median_ratio"] for r in rows}
    pct_lookup = {(r["degree"], r["epsilon"]): 100.0 - r["pct_naive_wins"] for r in rows}
    any_crossover_lookup = {(r["degree"], r["epsilon"]): r["median_ratio"] > 1.0 for r in rows}

    lines = [
        "# Systematic Degree Sweep Summary",
        "",
        f"**Source:** `{source_csv}` ({len(rows)} aggregated rows for this noise model)  ",
        f"**Coverage:** Degrees {degrees}, Epsilons {eps_values}, q aggregated across the sweep grid  ",
        "",
        "## Key Finding",
        "",
        "The unbiased estimator remains the better median choice across the tested degree/epsilon grid once the q sweep is aggregated.",
        "",
        "---",
        "",
        "## Median MSE Ratio by Degree and Epsilon",
        "",
    ]

    lines.append("| Degree |" + "".join(f" ε={eps:g} |" for eps in eps_values) + " Crossover? |")
    lines.append("|--------|" + "-------|" * len(eps_values) + "-----------|")
    for deg in degrees:
        row = [f"| {deg:2d}     |"]
        has_crossover = False
        for eps in eps_values:
            value = ratio_lookup.get((deg, eps), np.nan)
            row.append(f" {_format_ratio(value):>6} |")
            has_crossover = has_crossover or any_crossover_lookup.get((deg, eps), False)
        row.append(f" {'✓' if has_crossover else 'No':^11} |")
        lines.append("".join(row))

    lines.extend([
        "",
        "**Legend:** Ratio = MSE_unbiased / MSE_naive. Values < 1 mean unbiased is better.",
        "",
        "---",
        "",
        "## Percentage of Rows Where Unbiased Is Better",
        "",
    ])

    lines.append("| Degree |" + "".join(f" ε={eps:g} |" for eps in eps_values))
    lines.append("|--------|" + "-------|" * len(eps_values))
    for deg in degrees:
        row = [f"| {deg:2d}     |"]
        for eps in eps_values:
            value = pct_lookup.get((deg, eps), np.nan)
            row.append(f" {_format_pct(value):>6} |")
        lines.append("".join(row))

    lines.extend([
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "1. The ratio table summarizes the median q-aggregated behavior.",
        "2. A crossover marker means the median ratio exceeds 1 for that degree/epsilon pair.",
        "3. The percentage table shows how frequently naive wins across q values.",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def plot_degree_summary(rows: list[dict], out_dir: Path, source_csv: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for noise in sorted({r["noise"] for r in rows if r["noise"]}):
        noise_rows = summarize_degree_phase(rows, noise=noise)
        if not noise_rows:
            continue

        noise_dir = out_dir / noise
        noise_dir.mkdir(parents=True, exist_ok=True)
        write_degree_summary_markdown(noise_rows, noise_dir / "crossover_summary.md", source_csv)

        degrees = sorted({r["degree"] for r in noise_rows})
        eps_values = sorted({r["epsilon"] for r in noise_rows})
        ratio_grid = np.full((len(degrees), len(eps_values)), np.nan)
        win_grid = np.full((len(degrees), len(eps_values)), np.nan)
        for r in noise_rows:
            di = degrees.index(r["degree"])
            ei = eps_values.index(r["epsilon"])
            ratio_grid[di, ei] = r["median_ratio"]
            win_grid[di, ei] = r["pct_naive_wins"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ratio_im = axes[0].imshow(ratio_grid, aspect="auto", origin="lower", cmap="RdYlGn_r")
        axes[0].set_title(f"Median MSE ratio ({noise})")
        axes[0].set_xlabel("Epsilon")
        axes[0].set_ylabel("Degree")
        axes[0].set_xticks(range(len(eps_values)))
        axes[0].set_xticklabels([f"{eps:g}" for eps in eps_values], rotation=45)
        axes[0].set_yticks(range(len(degrees)))
        axes[0].set_yticklabels([str(d) for d in degrees])
        fig.colorbar(ratio_im, ax=axes[0], fraction=0.046, pad=0.04, label="MSE ratio")

        win_im = axes[1].imshow(win_grid, aspect="auto", origin="lower", cmap="YlOrRd", vmin=0, vmax=100)
        axes[1].set_title(f"Naive win % ({noise})")
        axes[1].set_xlabel("Epsilon")
        axes[1].set_ylabel("Degree")
        axes[1].set_xticks(range(len(eps_values)))
        axes[1].set_xticklabels([f"{eps:g}" for eps in eps_values], rotation=45)
        axes[1].set_yticks(range(len(degrees)))
        axes[1].set_yticklabels([str(d) for d in degrees])
        fig.colorbar(win_im, ax=axes[1], fraction=0.046, pad=0.04, label="% naive wins")

        plt.tight_layout()
        plt.savefig(noise_dir / "degree_phase_heatmaps.png", dpi=150)
        plt.close()

        plt.figure(figsize=(11, 6))
        for eps in eps_values:
            xs = [r["degree"] for r in noise_rows if r["epsilon"] == eps]
            ys = [r["median_ratio"] for r in noise_rows if r["epsilon"] == eps]
            if xs:
                plt.plot(xs, ys, marker="o", label=f"ε={eps:g}")
        plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
        plt.yscale("log")
        plt.xlabel("Polynomial degree")
        plt.ylabel("Median MSE ratio (unbiased / naive)")
        plt.title(f"Degree trend summary ({noise})")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(noise_dir / "degree_trends.png", dpi=150)
        plt.close()


def _median_by_degree(rows: list[dict], value_key: str, noise: str, epsilon: float, q: float):
    grouped = defaultdict(list)
    for r in rows:
        if r["noise"] != noise:
            continue
        if abs(r["epsilon"] - epsilon) > 1e-12:
            continue
        if abs(r["q"] - q) > 1e-12:
            continue
        deg = int(r["degree"])
        val = r[value_key]
        if np.isfinite(val):
            grouped[deg].append(val)

    xs = sorted(grouped.keys())
    ys = [float(np.median(grouped[d])) for d in xs]
    return xs, ys


def _median_by_sweep(rows: list[dict], value_key: str, family: str, noise: str, epsilon: float, q: float):
    grouped = defaultdict(list)
    for r in rows:
        if r["family"] != family:
            continue
        if r["noise"] != noise:
            continue
        if abs(r["epsilon"] - epsilon) > 1e-12:
            continue
        if abs(r["q"] - q) > 1e-12:
            continue
        sweep_value = r["sweep_value"]
        if not np.isfinite(sweep_value):
            continue
        val = r[value_key]
        if np.isfinite(val):
            grouped[sweep_value].append(val)

    xs = sorted(grouped.keys())
    ys = [float(np.median(grouped[d])) for d in xs]
    return xs, ys


def plot_ratio_by_degree(rows: list[dict], out_dir: Path, q_focus: float):
    out_dir.mkdir(parents=True, exist_ok=True)

    eps_values = sorted({r["epsilon"] for r in rows if np.isfinite(r["epsilon"])})
    for metric_key, title, fname in [
        ("emp_var_ratio_unbiased_over_naive", "Empirical Variance Ratio (unbiased / naive)", "variance_ratio_by_degree.png"),
        ("emp_mse_ratio_unbiased_over_naive", "Empirical MSE Ratio (unbiased / naive)", "mse_ratio_by_degree.png"),
    ]:
        plt.figure(figsize=(10, 6))
        for noise in ["laplace", "gaussian"]:
            for eps in eps_values:
                xs, ys = _median_by_degree(rows, metric_key, noise=noise, epsilon=eps, q=q_focus)
                if not xs:
                    continue
                plt.plot(xs, ys, marker="o", label=f"{noise}, eps={eps:g}")

        plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Polynomial degree")
        plt.ylabel("Ratio")
        plt.title(f"{title} at q={q_focus:g}")
        plt.grid(True)
        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()


def plot_symbolic_match(rows: list[dict], out_dir: Path, q_focus: float):
    out_dir.mkdir(parents=True, exist_ok=True)

    eps_values = sorted({r["epsilon"] for r in rows if np.isfinite(r["epsilon"])})
    for metric_key, title, fname in [
        ("err_unbiased_var_rel", "Relative Error vs Symbolic (Unbiased Variance)", "error_unbiased_variance_by_degree.png"),
        ("err_unbiased_mse_rel", "Relative Error vs Symbolic (Unbiased MSE)", "error_unbiased_mse_by_degree.png"),
    ]:
        plt.figure(figsize=(10, 6))
        for noise in ["laplace", "gaussian"]:
            for eps in eps_values:
                xs, ys = _median_by_degree(rows, metric_key, noise=noise, epsilon=eps, q=q_focus)
                if not xs:
                    continue
                plt.plot(xs, ys, marker="o", label=f"{noise}, eps={eps:g}")

        plt.yscale("log")
        plt.xlabel("Polynomial degree")
        plt.ylabel("Relative error (log scale)")
        plt.title(f"{title} at q={q_focus:g}")
        plt.grid(True)
        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()


def plot_coefficient_sweeps(rows: list[dict], out_dir: Path, q_focus: float):
    families = sorted({r["family"] for r in rows if r["family"] not in {"", "baseline", "edgecase"} and np.isfinite(r["sweep_value"])})
    if not families:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    eps_values = sorted({r["epsilon"] for r in rows if np.isfinite(r["epsilon"])})

    plot_specs = [
        ("emp_var_ratio_unbiased_over_naive", "Empirical Variance Ratio", "Variance Ratio"),
        ("emp_mse_ratio_unbiased_over_naive", "Empirical MSE Ratio", "MSE Ratio"),
        ("err_unbiased_var_rel", "Relative Error vs Symbolic (Variance)", "Relative Error"),
        ("err_unbiased_mse_rel", "Relative Error vs Symbolic (MSE)", "Relative Error"),
    ]

    for family in families:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        axes = np.asarray(axes)
        sweep_param = next((r["sweep_parameter"] for r in rows if r["family"] == family), "sweep")
        fig.suptitle(f"Coefficient sweep family: {family} at q={q_focus:g}")

        for ax, (metric_key, title, ylabel) in zip(axes.flat, plot_specs):
            for noise in ["laplace", "gaussian"]:
                for eps in eps_values:
                    xs, ys = _median_by_sweep(rows, metric_key, family=family, noise=noise, epsilon=eps, q=q_focus)
                    if not xs:
                        continue
                    ax.plot(xs, ys, marker="o", label=f"{noise}, eps={eps:g}")

            if "Ratio" in title:
                ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(fontsize=8)

        for ax in axes[-1, :]:
            ax.set_xlabel(f"Sweep value for {sweep_param}")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(out_dir / f"coefficient_sweep_{family}.png", dpi=150)
        plt.close()


def plot_compact_coefficient_sweeps(rows: list[dict], out_dir: Path, q_focus: float):
    families = ["quadratic_leading", "cubic_leading"]
    out_dir.mkdir(parents=True, exist_ok=True)
    eps_values = sorted({r["epsilon"] for r in rows if np.isfinite(r["epsilon"])})

    fig, axes = plt.subplots(len(families), 2, figsize=(12, 7), sharex=True, sharey=False)
    if len(families) == 1:
        axes = np.asarray([axes])

    metric_specs = [
        ("emp_var_ratio_unbiased_over_naive", "Variance ratio (unbiased / naive)"),
        ("emp_mse_ratio_unbiased_over_naive", "MSE ratio (unbiased / naive)"),
    ]

    for row_idx, family in enumerate(families):
        sweep_param = next((r["sweep_parameter"] for r in rows if r["family"] == family), "sweep")
        for col_idx, (metric_key, title) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            for noise in ["laplace", "gaussian"]:
                for eps in eps_values:
                    xs, ys = _median_by_sweep(rows, metric_key, family=family, noise=noise, epsilon=eps, q=q_focus)
                    if not xs:
                        continue
                    ax.plot(xs, ys, marker="o", label=f"{noise}, eps={eps:g}")

            ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"{family}: {title}")
            ax.set_xlabel(f"{sweep_param}")
            ax.set_ylabel("Ratio")
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(fontsize=8)

    plt.suptitle(f"Compact coefficient sweep summary at q={q_focus:g}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / "coefficient_sweeps_compact.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Monte Carlo study CSV summaries")
    parser.add_argument(
        "--input",
        type=str,
        default="reports/monte_carlo/monte_carlo_results.csv",
        help="Input CSV file generated by scripts/run_monte_carlo_study.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/monte_carlo",
        help="Directory to save generated figures",
    )
    parser.add_argument("--q-focus", type=float, default=1.0, help="q value used for degree-trend summaries")
    parser.add_argument(
        "--plot-mode",
        choices=["all", "compact", "degree-summary", "coefficient-summary", "coefficient-criss-cross"],
        default="all",
        help="Choose between the full plot set and a compact coefficient-focused summary",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_rows(Path(args.input))

    out_dir = Path(args.output_dir)
    if args.plot_mode == "compact":
        plot_compact_coefficient_sweeps(rows, out_dir, q_focus=args.q_focus)
    elif args.plot_mode == "degree-summary":
        plot_degree_summary(rows, out_dir, source_csv=Path(args.input))
    elif args.plot_mode == "coefficient-summary":
        plot_coefficient_summary(rows, out_dir, source_csv=Path(args.input))
    elif args.plot_mode == "coefficient-criss-cross":
        plot_coefficient_pairwise_summary(rows, out_dir, source_csv=Path(args.input))
    else:
        plot_ratio_by_degree(rows, out_dir, q_focus=args.q_focus)
        plot_symbolic_match(rows, out_dir, q_focus=args.q_focus)
        plot_coefficient_sweeps(rows, out_dir, q_focus=args.q_focus)

    print(f"Saved plots in: {out_dir}")


if __name__ == "__main__":
    main()
