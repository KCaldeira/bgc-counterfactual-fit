# BGC Counterfactual Fit

Find a correction function h(T) such that:

```
pct_growth_gpp_bgc - h(tas_bgc) + h(tas) ≈ pct_growth_gpp
```

where `_bgc` variables come from `*hist-bgc.csv` files and non-`_bgc` variables come from `*historical.csv` files.

Equivalently, we minimize the residual:

```
[pct_growth_gpp - pct_growth_gpp_bgc] - [h(tas) - h(tas_bgc)]
```

over all regions and years, fitting the parameters of h(T).

## Data

Input CSVs in `data/input/` for three Earth System Models:
- ACCESS-ESM1-5
- CNRM-ESM2-1
- MIROC-ES2L

Each model has a `*_historical.csv` and a `*_hist-bgc.csv` file with columns:
`model, region, year, area, lai, tas, pr, gpp, pct_growth_gpp`

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python fit_h.py --model MODEL [--degree N] [--start_year YYYY] [--end_year YYYY]
```

### Arguments

| Argument       | Required | Default | Description                          |
|----------------|----------|---------|--------------------------------------|
| `--model`      | Yes      |         | `ACCESS-ESM1-5`, `CNRM-ESM2-1`, or `MIROC-ES2L` |
| `--degree`     | No       | 2       | Polynomial degree for h(T)           |
| `--start_year` | No       | None    | First year to include in fit         |
| `--end_year`   | No       | None    | Last year to include in fit          |

### Examples

Quadratic fit (default):
```bash
python fit_h.py --model ACCESS-ESM1-5
```

Cubic fit restricted to 1960 onward:
```bash
python fit_h.py --model CNRM-ESM2-1 --degree 3 --start_year 1960
```

## Output

Each run produces two files in `data/output/`:

- **`{model}_results.xlsx`** — Coefficients, standard errors, t-statistics, and R² for polynomial degrees 1 through 5.
- **`{model}_h_curves.pdf`** — Plot of the best-fit h(T) curves for degrees 1 through 5.

When `--start_year` or `--end_year` are specified, they appear in the filenames, e.g. `ACCESS-ESM1-5_1960_results.xlsx`.
