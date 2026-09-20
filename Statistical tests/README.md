# Statistical tests

## Current workflow: four statistical diagnostic figures

`Statistical tests.ipynb` contains two sections and exactly four figures: independence for standard horizons, independence for cutoff horizons, percentage of arms rejected by mean-equality tests for standard horizons, and the corresponding cutoff-horizon figure. All use 1% significance and display horizons 20 through 120. The identical-distribution section explicitly tests the weaker condition of equal mean vectors. Extra diagnostics are preserved in `archive/before_four_main_plots_20260919_202737.ipynb`. The data are **600 saved Humanoid trajectories, 20 uniformly sampled arms, 150 iterations**, with an expected 30 observations per arm/iteration. All rewards are simulated.

The independence comparison is configured in `setup_independence.json` and implemented in `independence_comparison.py`. It reads `result_20_arms_30_evaluations_quantiles.pkl` and writes a separate `result_independence_comparison.pkl`. It cannot launch simulations or training and does not change the input file. Its progress bar counts trajectory/horizon pairs, and completed tests resume automatically.

```cmd
".venv\Scripts\python.exe" -u -B "Statistical tests\independence_comparison.py"
```

Run all notebook cells to refresh the four saved-result figures; this does not launch analyses or simulations. The comparison retains the same trajectories, initial observations, fixed cost normalization, horizons 20, 30, ..., 150 and significance **1%**. Settings are fixed before analysis, without searching for a desired rejection curve.

| Test | Fixed specification | Calibration |
| --- | --- | --- |
| Ljung-Box | Multivariate, maximum lag 5 | Reuse the saved 4,999 joint-row permutation p-values when settings match |
| AutoHSIC | Sum of U-centered Gaussian HSIC over lags 1-5; median-distance bandwidth rule | 4,999 Rademacher wild-bootstrap samples, independent across lags |
| BDS | Embedding dimension 2; epsilon = 1.5 sample standard deviations, for each coordinate | 4,999 joint-row permutations of the two-sided correlation-integral statistic; Bonferroni across the two coordinates |

For BDS, the omitted standard-error factor is invariant under row permutations, so it does not affect permutation p-values when positive. Constant coordinates receive p=1. Coordinate-wise BDS can miss purely cross-coordinate lag dependence. AutoHSIC follows the U-centered statistic and Algorithm 1 of [Li, Xu and Zhou (2026, preprint)](https://arxiv.org/html/2605.22025v1); its bootstrap justification is asymptotic, including at short prefixes. [BDS definition and implementation reference](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.bds.html).

All calibrations assume an iid null, so changes in marginal distributions can also cause rejection. Curves are unsmoothed empirical proportions; neither monotonicity nor slow growth is imposed. The four figures have no reference bands or horizontal significance lines. Horizons share data, and no simultaneous guarantee is claimed.

### Independence after a cutoff

The notebook also retains a second independence figure for cutoffs p = 20, 30, ..., 120. Each test uses observations at iterations p+1 through 150 of all 600 trajectories, rather than a prefix. The original prefix plot and its results are preserved.

```cmd
".venv\Scripts\python.exe" -u -B "Statistical tests\independence_cutoff.py"
```

`setup_independence_cutoff.json` configures this analysis; `result_independence_cutoff.pkl` stores its resumable results. The runner reuses the original Ljung-Box, AutoHSIC and BDS functions, with lag 5, 4,999 resamples per test and 1% significance. Bandwidth and BDS distance thresholds are recalculated within each suffix using the same rules. No simulation or training is run. The notebook's added section loads the completed results, plotting the percentage of trajectories rejected. Later cutoffs retain fewer observations, so rejection can change with statistical power as well as temporal behavior; the same iid-null limitations apply.

### Equality of mean vectors for standard horizons

`mean_prefix_comparison.py` uses the same MANOVA, Welch-James and MATS functions as the cutoff analysis, on iterations 1 through p. `setup_mean_prefix_comparison.json` sets horizons 20, 30, ..., 120, significance 1%, Holm correction across the 20 arms, and 9,999 MATS bootstrap draws. It reads the existing trajectories and writes `result_mean_prefix_comparison.pkl`, resuming completed arm/horizon pairs automatically.

```cmd
".venv\Scripts\python.exe" -u -B "Statistical tests\mean_prefix_comparison.py"
```

### Equality of mean vectors after a cutoff

`mean_comparison.py` reads the same saved trajectories and writes `result_mean_comparison.pkl`, using `setup_mean_comparison.json`. The notebook loads this analysis without recomputing bootstrap draws.

```cmd
".venv\Scripts\python.exe" -u -B "Statistical tests\mean_comparison.py"
```

Each cutoff p = 20, 30, ..., 120 tests iterations p+1 through 150, independently for each of the 20 arms. The three curves use **1% significance** and **Holm correction across arms within each test/cutoff**: Pillai MANOVA, multivariate Welch-James with Johansen's F approximation, and MATS with 9,999 Gaussian parametric-bootstrap draws. Both coordinates are tested jointly. Constant coordinates are omitted. Welch-James uses a documented exploratory rank adjustment for singular contrast covariance; MATS uses its generalized-inverse statistic. These remain independent-group reference calibrations, not a validated treatment of within-trajectory dependence. The notebook documents the qualifications; rank information is retained in the result files. No simulator or training code is invoked.

### Previous quantile coverage (archived)

`setup.json` and `analysis_quantiles.py` configure the earlier quantile-coverage analysis: significance 1%, minimum coverage 80%, and quantile rectangles [5%,95%], [10%,90%], [20%,80%]. A fixed split uses 300 calibration trajectories and 300 evaluation trajectories. Each evaluation trajectory contributes one uniformly chosen time per horizon. One-sided binomial tests compare coverage against 80%, with Holm correction across the 20 arms within each rectangle/horizon. The 80% descriptive confidence intervals do not determine the 1% test decisions. This criterion is not equality of means or distributions.

The simulation and earlier statistical results are preserved. `setup_7_arms_manova.json` and its result file retain the earlier seven-arm MANOVA experiment described below.

## Previous MANOVA workflow (preserved)

Previous analysis: **Humanoid, seven arms, 350 saved simulated trajectories**, horizons 20, 30, ..., 150. No new simulation or policy training is needed.

| Panel | Test | Significance levels |
| --- | --- | --- |
| Independence diagnostic | Multivariate Ljung-Box, lag 5, 4,999 joint-row permutations | 0.1%, 0.5%, 1% |
| Equality of means | MANOVA, Pillai's trace, 4,999 trajectory-level wild-bootstrap draws | 1%, 2%, 3% |

The first panel includes 95% binomial null reference bands. The second applies Holm correction across the seven arms within each horizon. Both retain the descriptive 20% rejection-rate cutoff.

## Reproduce the previous MANOVA analysis

```cmd
".venv\Scripts\python.exe" -u -B "Statistical tests\script.py" --setup "Statistical tests\setup_7_arms_manova.json" --stage analyze
```

The command reads the completed seven-arm trajectories from `result.pkl`, then saves the new analysis in `result_7_arms_manova.pkl`. It displays progress bars and resumes completed tests when repeated. Use `--reanalyze` only to intentionally restart an analysis after changing its statistical settings. The existing simulations and energy-test results are not overwritten.

These historical MANOVA results are preserved separately; the current notebook compares means for both standard and cutoff horizons using the newer 20-arm data.

## Files

- `setup.json`: current 20-arm simulation and quantile-coverage configuration.
- `setup_independence.json`: the saved-data Ljung-Box/AutoHSIC/BDS comparison.
- `script.py`: command entry point; the MANOVA configuration uses saved data only.
- `analysis_manova.py`: saved-data analysis, MANOVA statistic, and bootstrap calibration.
- `analysis.py`: original Ljung-Box and energy-test functions, preserved for prior analyses.
- `Statistical tests.ipynb`: current rejection-rate plots.
- `setup_20_arms_energy.json` and the earlier result files: preserved previous configuration/results.
- The separate practical-equivalence files are retained but are not invoked by this workflow.

## Previous seven-arm MANOVA design

For each arm and horizon, the null states that the **mean vector of overhead and instability is equal across iteration groups**. Iteration is categorical. The statistic is Pillai's trace, tr[H(H+E)^-1]. Responses use the same fixed normalization as before; Pillai's trace is invariant to nonsingular linear changes of response units.

Trajectories are independent, but observations within one trajectory need not be. Therefore plotted p-values use a wild residual bootstrap: residuals from the unrestricted iteration-means model are HC2 adjusted and multiplied by one Rademacher sign per trajectory, shared by both coordinates and all observed iterations. Bootstrap outcomes have a common null mean (zero, which is sufficient because the statistic is translation invariant). Each bootstrap statistic is compared with the observed Pillai trace using the finite Monte Carlo correction `(1 + exceedances)/(1 + bootstrap draws)`.

This is an **approximate trajectory-level wild-bootstrap MANOVA calibration**, not the classical exact F test. Conventional independent-observation F p-values are saved as diagnostics and are not used in the plot. Numerical tests verify the statistic, response transformation invariance, and basic mean-shift behavior; they do not establish nominal coverage for every finite-sample configuration.

The weaker mean-equality null does not establish identical distributions. Ljung-Box detects serial correlation rather than all dependence. Non-rejection proves neither null. Initialization is retained, all horizons use overlapping prefixes, and no multiple-horizon guarantee or validated maximum horizon is claimed. Changing tests does not guarantee a lower rejection rate.

## References

- [Pillai statistic and F approximation](https://www.statsmodels.org/stable/_modules/statsmodels/multivariate/multivariate_ols.html).
- [Wild-bootstrap methods for repeated measurements](https://www.sciencedirect.com/science/article/pii/S0167947316301530).
- [Multivariate Ljung-Box](https://microsoft.github.io/wpa/reference/LjungBox.html).

Numerical checks:

```cmd
".venv\Scripts\python.exe" -B -m unittest discover -s "Statistical tests" -p "test_*.py"
```
