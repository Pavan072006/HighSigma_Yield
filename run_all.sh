#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 scripts/01_check_mismatch.py                 # ~5 s    mismatch model vs PDK agauss
python3 scripts/02_brute_mc.py "${1:-1000000}"       # ~10 min brute-force baseline
python3 scripts/03_highsigma_is.py                   # ~2 min  GP surrogate + MPFP + importance sampling
python3 scripts/04_plots.py                          # figures -> results/
python3 scripts/05_table.py                          # results table -> results/results_table.md
