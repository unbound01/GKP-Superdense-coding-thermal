# GKP superdense coding (square vs hexagonal) -- code, matching the corrected main.tex

This is exactly the code that generated every number, table, and figure in the current
paper (periodic model for both lattices; hard-decision rate + soft-decision capacity;
kappa/thermal-loss generalization; corrected energy convention N_S = (10^(s/10)-1)/2).

## Files (code/)
- `sweep_periodic.py`   -- core module: both lattices' rates in the periodic model.
                          * C_square_hard, C_hex_hard      -- hard-decision nearest-lattice-point rate
                          * C_square_soft, C_hex_soft       -- capacity (soft decisions), exact quadrature
                          * C_square_soft_fast, C_hex_soft_fast -- fast spline lookup of the above (used by
                            figures_periodic.py / thermal_analysis.py for speed)
                          * sigma_eff, delta2, mean_photon_number, threshold, sigma_star, eta_min
                          Run directly: `python3 sweep_periodic.py --target 1.8 --etas 0.90 0.95 0.99`
                          Options: --target (bits), --etas (list), --kappa (1 or 2), --nth (thermal occupation)
                          Writes results_periodic/*.csv

- `validation.py`       -- all validation sections (needs sweep_periodic.py alongside):
                          0: lattice geometry, 1: brute-force Monte Carlo (hard decisions),
                          1b: soft-decision MC, 2: circuit-level noise bookkeeping,
                          3: n_th->0 regression, 4: Bell-pair models (kappa=1 vs 2) + energy convention,
                          5/6: benchmark capacity checks (C_EA, C_hol, thermal versions)
                          Run: `python3 validation.py`            (default: ~10 min, full sample sizes)
                          Smoke test: `python3 validation.py --n 100000 --nsoft 100000`  (~1 min)
                          Thermal: add `--nth 0.1`

- `figures_periodic.py` -- regenerates Figs. 3-7 and Tables I-III (pure loss, both kappa).
                          Run: `python3 figures_periodic.py`  -> writes figures_periodic/*.pdf,png and *.tex,csv

- `thermal_analysis.py` -- regenerates Figs. 9-11 and Table IV/V (thermal-loss sweep + benchmark scan).
                          Run: `python3 thermal_analysis.py`  -> writes figures/*.pdf,png and tables/*.tex,csv

- `make_schematics.py`  -- regenerates Fig. 1 (protocol schematic) and Fig. 2 (hexagonal Pauli lattice).
                          Run: `python3 make_schematics.py`   -> writes figures/fig1_protocol.*, fig2_hex_lattice.*

## Requirements
Python 3, numpy, scipy, matplotlib. No other dependencies.

## Quick sanity check
```
cd code
python3 sweep_periodic.py --target 1.8 --etas 0.90 0.95 0.99
```
Expected (matches paper Table I, kappa=2, pure loss):
```
 eta |  HARD sq      hex    adv |  SOFT sq      hex    adv
0.90 |   15.642   14.950  0.692 |   12.403   11.941  0.463
0.95 |    8.56*    8.52*   ...  |    9.686    9.432  0.254   (*hard-decision values shown in Table II)
0.99 |    9.308    9.137  0.171 |    8.312    8.126  0.186
```

## sample_results/
Two CSVs from a verification run, included for reference:
- `thresholds_periodic_C1.8_nth0.0_kappa2.0.csv` -- threshold-squeezing table at the 1.8-bit target
- `capacity_grid_periodic_nth0.0_kappa2.0.csv`   -- full (s, eta) capacity/rate grid, pure loss

## Notes
- All scripts must be run from the same directory (they import each other by relative path).
- `figures_periodic.py` and `thermal_analysis.py` cache a spline lookup table
  (`results_periodic/soft_table.npz`) on first run to speed up repeated capacity evaluations;
  delete it if you change `sweep_periodic.py`'s soft-decision quadrature and want a clean rebuild.
- Every number in `main.tex` traces to one of these five files; `validation.py` is what backs the
  paper's Appendices A-C.
