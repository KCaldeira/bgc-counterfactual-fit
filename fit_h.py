"""
Find h(T) = h1*T + h2*T^2 such that:
    pct_growth_gpp_bgc - h(tas_bgc) + h(tas) ≈ pct_growth_gpp

Rearranged, we minimize over all (region, year) pairs:
    residual = [pct_growth_gpp - pct_growth_gpp_bgc] - [h(tas) - h(tas_bgc)]

For h(T) = h1*T + h2*T^2 this is linear in (h1, h2), so we use
ordinary least squares.
"""

import argparse
import pandas as pd
import numpy as np


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


def fit_quadratic_h(df):
    """
    Fit h(T) = h1*T + h2*T^2 by least squares.

    We want: pct_growth_gpp_bgc - h(tas_bgc) + h(tas) = pct_growth_gpp
    i.e.:    h(tas) - h(tas_bgc) = pct_growth_gpp - pct_growth_gpp_bgc

    Let y = pct_growth_gpp - pct_growth_gpp_bgc
        x1 = tas - tas_bgc
        x2 = tas^2 - tas_bgc^2

    Then y = h1*x1 + h2*x2, which is a standard linear regression.
    """
    y = df["pct_growth_gpp"] - df["pct_growth_gpp_bgc"]
    x1 = df["tas"] - df["tas_bgc"]
    x2 = df["tas"] ** 2 - df["tas_bgc"] ** 2

    X = np.column_stack([x1, x2])
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    h1, h2 = coeffs

    # Compute R^2 and standard errors
    n, p = X.shape
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    mse = ss_res / (n - p)
    cov = mse * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    return h1, h2, r_squared, se


def main():
    parser = argparse.ArgumentParser(
        description="Fit h(T) = h1*T + h2*T^2 for BGC counterfactual correction"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["ACCESS-ESM1-5", "CNRM-ESM2-1", "MIROC-ES2L"],
    )
    parser.add_argument("--start_year", type=int, default=None)
    parser.add_argument("--end_year", type=int, default=None)
    args = parser.parse_args()

    df = load_data(args.model, args.start_year, args.end_year)
    print(f"Model: {args.model}")
    print(f"Years: {df['year'].min()}-{df['year'].max()}")
    print(f"Data points: {len(df)} (regions x years)")

    h1, h2, r2, se = fit_quadratic_h(df)
    print(f"\nh(T) = {h1:.6f} * T + {h2:.6f} * T^2")
    print(f"R^2 = {r2:.6f}")
    t_opt = -h1 / (2 * h2)
    print(f"T_opt = {t_opt:.2f}")

    print(f"\n{'Coeff':<8} {'Value':>12} {'Std Error':>12} {'t-stat':>10}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*10}")
    for name, val, stderr in [("h1", h1, se[0]), ("h2", h2, se[1])]:
        t_stat = val / stderr
        print(f"{name:<8} {val:>12.6f} {stderr:>12.6f} {t_stat:>10.2f}")


if __name__ == "__main__":
    main()
