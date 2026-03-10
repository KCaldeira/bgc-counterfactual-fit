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

## Usage

```bash
python fit_h.py --model ACCESS-ESM1-5 --start_year 1851 --end_year 2014
```

## Requirements

```bash
pip install -r requirements.txt
```
