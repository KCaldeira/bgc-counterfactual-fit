"""
Find h(T) such that:
    pct_growth_gpp_bgc - h(tas_bgc) + h(tas) ≈ pct_growth_gpp

Rearranged, we minimize over all (region, year) pairs:
    residual = [pct_growth_gpp - pct_growth_gpp_bgc] - [h(tas) - h(tas_bgc)]

For polynomial h(T) = h1*T + h2*T^2 [+ h3*T^3 + ...], this is linear in
the coefficients, so we use ordinary least squares.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(model, start_year=None, end_year=None):
    """Load and merge historical and hist-bgc data for a given model."""
    hist_file = f"data/input/{model}_historical.csv"
    bgc_file = f"data/input/{model}_hist-bgc.csv"

    df_hist = pd.read_csv(hist_file)
    df_bgc = pd.read_csv(bgc_file)

    # Merge on model, region, year
    df = df_hist.merge(
        df_bgc[["model", "region", "year", "tas", "pct_growth_gpp"]],
        on=["model", "region", "year"],
        suffixes=("", "_bgc"),
    )

    # Drop rows where pct_growth_gpp is missing (first year per region)
    df = df.dropna(subset=["pct_growth_gpp", "pct_growth_gpp_bgc"])

    if start_year is not None:
        df = df[df["year"] >= start_year]
    if end_year is not None:
        df = df[df["year"] <= end_year]

    return df


def fit_polynomial_h(df, degree=2):
    """
    Fit h(T) = h1*T + h2*T^2 [+ h3*T^3 + ...] by least squares.

    We want: h(tas) - h(tas_bgc) = pct_growth_gpp - pct_growth_gpp_bgc

    For each power k, the predictor is tas^k - tas_bgc^k.
    """
    y = (df["pct_growth_gpp"] - df["pct_growth_gpp_bgc"]).values

    # Build design matrix: columns are tas^k - tas_bgc^k for k=1..degree
    tas = df["tas"].values
    tas_bgc = df["tas_bgc"].values
    X = np.column_stack([tas**k - tas_bgc**k for k in range(1, degree + 1)])

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Compute R^2 and standard errors
    n, p = X.shape
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    mse = ss_res / (n - p)
    cov = mse * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    return coeffs, r_squared, se


def eval_h(T, coeffs):
    """Evaluate h(T) = h1*T + h2*T^2 + ... for array T."""
    return sum(coeffs[k] * T ** (k + 1) for k in range(len(coeffs)))


def main():
    parser = argparse.ArgumentParser(
        description="Fit polynomial h(T) for BGC counterfactual correction"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["ACCESS-ESM1-5", "CNRM-ESM2-1", "MIROC-ES2L"],
    )
    parser.add_argument("--start_year", type=int, default=None)
    parser.add_argument("--end_year", type=int, default=None)
    parser.add_argument(
        "--degree", type=int, default=2,
        help="Polynomial degree for h(T) (default: 2)",
    )
    args = parser.parse_args()

    df = load_data(args.model, args.start_year, args.end_year)
    print(f"Model: {args.model}")
    print(f"Years: {df['year'].min()}-{df['year'].max()}")
    print(f"Data points: {len(df)} (regions x years)")
    print(f"Degree: {args.degree}")

    coeffs, r2, se = fit_polynomial_h(df, degree=args.degree)

    # Print h(T) formula
    terms = [f"{coeffs[k-1]:.6f} * T^{k}" for k in range(1, args.degree + 1)]
    print(f"\nh(T) = {' + '.join(terms)}")
    print(f"R^2 = {r2:.6f}")

    if args.degree == 2:
        t_opt = -coeffs[0] / (2 * coeffs[1])
        print(f"T_opt = {t_opt:.2f}")

    # Coefficient table
    print(f"\n{'Coeff':<8} {'Value':>12} {'Std Error':>12} {'t-stat':>10}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*10}")
    for k in range(args.degree):
        name = f"h{k+1}"
        val = coeffs[k]
        stderr = se[k]
        t_stat = val / stderr
        print(f"{name:<8} {val:>12.6f} {stderr:>12.6f} {t_stat:>10.2f}")

    # --- Write results for degrees 1-5 to xlsx and plot to pdf ---
    os.makedirs("data/output", exist_ok=True)
    year_tag = ""
    if args.start_year is not None:
        year_tag += f"_{args.start_year}"
    if args.end_year is not None:
        year_tag += f"-{args.end_year}"

    xlsx_path = f"data/output/{args.model}{year_tag}_results.xlsx"
    pdf_path = f"data/output/{args.model}{year_tag}_h_curves.pdf"

    # Fit degrees 1 through 5
    max_degree = 5
    all_rows = []
    all_fits = {}
    for deg in range(1, max_degree + 1):
        c, r2_d, se_d = fit_polynomial_h(df, degree=deg)
        all_fits[deg] = c
        for k in range(deg):
            all_rows.append({
                "degree": deg,
                "coeff": f"h{k+1}",
                "value": c[k],
                "std_error": se_d[k],
                "t_stat": c[k] / se_d[k],
            })
        all_rows.append({
            "degree": deg,
            "coeff": "R^2",
            "value": r2_d,
            "std_error": np.nan,
            "t_stat": np.nan,
        })

    results_df = pd.DataFrame(all_rows)
    results_df.to_excel(xlsx_path, index=False)
    print(f"\nResults written to {xlsx_path}")

    # Plot h(T) curves for degrees 1-5
    tas_all = np.concatenate([df["tas"].values, df["tas_bgc"].values])
    T_range = np.linspace(tas_all.min(), tas_all.max(), 500)

    fig, ax = plt.subplots(figsize=(8, 5))
    for deg in range(1, max_degree + 1):
        h_vals = eval_h(T_range, all_fits[deg])
        ax.plot(T_range, h_vals, label=f"degree {deg}")

    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("h(T)")
    ax.set_title(f"Best-fit h(T) curves — {args.model}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Plot written to {pdf_path}")


if __name__ == "__main__":
    main()
