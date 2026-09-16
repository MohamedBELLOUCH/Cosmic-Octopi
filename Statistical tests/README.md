# Statistical tests

## Current workflow: quantile coverage

`setup.json` selects the quantile-coverage analysis. It reuses the completed seven-arm, 350-trajectory dataset and existing Ljung-Box p-values from `result_7_arms_manova.pkl`. The new output is `result_7_arms_quantiles.pkl`. No trajectories or Ljung-Box tests were rerun.

- Significance is 1% for both current panels.
- `confidence_level = 0.80` controls the Ljung-Box null reference band and descriptive coverage intervals; it does not set the rejection threshold.
- Required rectangle coverage is separately configured as `minimum_coverage = 0.80`.
- Wide, interquartile, and very narrow rectangles use coordinate-wise quantile bounds [0.025, 0.975], [0.25, 0.75], and [0.475, 0.525].

The fixed split uses 175 complete trajectories to calibrate pooled rectangles over all arms and all 150 iterations, and 175 disjoint trajectories to evaluate coverage. At each horizon, one uniformly selected iteration per evaluation trajectory supplies an independent vector; selection is independent of the data values and arm. Consequently, 15-35 vectors per arm/horizon were evaluated in this saved run. Other observations are not treated as independent Bernoulli trials. The time-selection seed and all split/selection indices are saved.

For each arm, an exact lower-tail binomial test assesses H0: coverage >= 80% against H1: coverage < 80%. Holm correction covers the seven arms separately for each rectangle/horizon. The target is coverage at a uniform random iteration in the prefix, not coverage at every individual iteration. Rectangles include their boundaries. No multiplicity correction covers the three exploratory widths or different horizons.

Reproduce or resume from the project root:

```cmd
".venv\Scripts\python.exe" -u -B "Statistical tests\script.py" --stage analyze
```

The current analysis is complete and `Statistical tests.ipynb` contains both plots plus final-horizon coverage intervals. The confidence intervals are 80% two-sided Clopper-Pearson intervals for description; decisions use the separate 1% one-sided tests and Holm correction. Non-rejection does not establish sufficient coverage or validate equal means/distributions. The old MANOVA and energy outputs remain intact; `setup_7_arms_manova.json` preserves the prior configuration.

Implementation: `analysis_quantiles.py`. Reference: [SciPy exact binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

## Previous MANOVA workflow (preserved)

Previous analysis: **Humanoid, seven arms, 350 saved simulated trajectories**, horizons 20, 30, ..., 150. No new simulation or policy training is needed.

| Panel | Test | Significance levels |
| --- | --- | --- |
| Independence diagnostic | Multivariate Ljung-Box, lag 5, 4,999 joint-row permutations | 0.1%, 0.5%, 1% |
| Equality of means | MANOVA, Pillai's trace, 4,999 trajectory-level wild-bootstrap draws | 1%, 2%, 3% |

The first panel includes 95% binomial null reference bands. The second applies Holm correction across the seven arms within each horizon. Both retain the descriptive 20% rejection-rate cutoff.

## Run from the project root in CMD

```cmd
".venv\Scripts\python.exe" -u -B "Statistical tests\script.py" --stage analyze
```

The command reads the completed seven-arm trajectories from `result.pkl`, then saves the new analysis in `result_7_arms_manova.pkl`. It displays progress bars and resumes completed tests when repeated. Use `--reanalyze` only to intentionally restart an analysis after changing its statistical settings. The existing simulations and energy-test results are not overwritten.

After completion, run the cells in `Statistical tests.ipynb` to display both figures. The analysis has been prepared but has not been launched by the assistant.

## Files

- `setup.json`: seven-arm simulation metadata and the two test configurations.
- `script.py`: command entry point; the MANOVA configuration uses saved data only.
- `analysis_manova.py`: saved-data analysis, MANOVA statistic, and bootstrap calibration.
- `analysis.py`: original Ljung-Box and energy-test functions, preserved for prior analyses.
- `Statistical tests.ipynb`: current rejection-rate plots.
- `setup_20_arms_energy.json` and the earlier result files: preserved previous configuration/results.
- The separate practical-equivalence files are retained but are not invoked by this workflow.

## MANOVA design

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
