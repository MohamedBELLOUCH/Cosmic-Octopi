# Practical-equivalence sensitivity analysis

This is a separate, exploratory assessment of the saved simulated Humanoid cost
vectors. The original Ljung-Box and energy tests are preserved. No simulations or
training are performed by `practical_equivalence.py`.

## Run

First finish the 600 trajectories in `result_20_arms_30_observations.pkl`.
The original permutation tests do not need to finish. If generation is incomplete,
resume **generation only** with:

```cmd
".venv\Scripts\python.exe" -u -B "Statistical tests\script.py" --stage generate
```

Then run:

```cmd
".venv\Scripts\python.exe" -u -B "Statistical tests\practical_equivalence.py"
```

Use the project root as the working directory. Do not start a second generation
process while the original one is running. The new analysis reads the existing
trajectories, writes only its own `practical_equivalence_20_arms_30_observations.pkl` checkpoint, and has
progress bars. The same command resumes completed arms after interruption. Open
`Practical equivalence.ipynb` and run its cells after the analysis finishes.

## Configuration and interpretation

`setup_practical_equivalence.json` holds the input/output filenames, horizons,
fixed lag 5, 999 bootstrap replicates, 95% confidence level, and exploratory
tolerances. Correlation tolerances are 0.10, 0.20, 0.30; energy-distance tolerances
are 0.02, 0.05, 0.10. Changing a tolerance does not change the underlying effect.
These thresholds do not have an established domain-specific justification yet.

The plots show effect sizes and uncertainty, not rejection rates. Lower is better
for the specified approximation. At each tolerance:

- **Supported:** the upper confidence bound is strictly below the tolerance.
- **Beyond tolerance:** the lower bound is strictly above the tolerance.
- **Inconclusive:** the interval overlaps or touches the tolerance, or the effect
  cannot be identified.

Non-rejection of an exact-null test is never used to establish equivalence.

## Dependence diagnostic

Within each horizon and lag, pairs of cost vectors are pooled across trajectories
and iteration positions. Pearson correlations include all four coordinate pairs
(overhead/overhead, overhead/instability, instability/overhead, and
instability/instability). The reported effect is the largest absolute correlation
over lags 1 through 5. No initialization is removed, and there is no detrending or
arm-wise residualization. Pooled correlations can reflect distribution drift as
well as temporal dependence and can mask time-specific relationships. Small
pooled correlations do not establish independence, conditional independence per
arm, or absence of nonlinear dependence. Singular cases remain inconclusive.

## Distribution diagnostic

For each arm and every pair of iteration groups within a horizon, compute

`D(F,G) = sqrt(2 E||X-Y|| - E||X-X'|| - E||Y-Y'||)`.

The empirical estimate uses the full Euclidean-distance V-statistic, including
diagonal terms. The plot shows the maximum distance across all iteration pairs
and all arms; per-arm maxima and decisions are also saved. Thus a statement about
all arms is based on the worst pair, not an average that could hide drift. The
fixed original bounds normalize overhead and instability before distances are
computed. The tolerance is in these normalized energy-distance units; 0.05 does
not mean a 5% change in either cost. The square root convention is explicit and
differs from the unnormalized DISCO statistic used by the original energy test.

## Uncertainty and limitations

Independent whole trajectories are resampled with replacement. The same
bootstrap multiplicities apply to every time, coordinate, arm, and horizon,
preserving the serial and cross-arm relationships observed in a trajectory.
The bootstrap does not assume independent observations within a trajectory.

For correlations, the confidence radius is the 95th percentile of the largest
absolute bootstrap estimation error across all coordinate/lag pairs. For energy,
the corresponding maximum is taken across all arm/time pairs on the **squared**
energy scale; interval endpoints are then square-root transformed. Bounds are
simultaneous within each family and horizon in the approximate bootstrap sense,
not across horizons or jointly across both figures. The original Holm correction
is not applied to these bounds: the maximum-error construction addresses the
within-family multiplicity instead.

These are **exploratory approximate bootstrap bounds**, not exact finite-sample
confidence guarantees or the standard TOST procedure. Energy V-statistics are
upward biased at small group sizes, and their behavior near zero is nonregular;
ordinary bootstrap coverage can be inaccurate. Only about 30 observations per
arm/time group may yield wide bounds. Empty bootstrap groups receive the full
possible uncertainty instead of being dropped. Calibration on controlled null
and boundary alternatives is needed before treating nominal coverage as verified
for publication. Numerical unit checks establish formula consistency, not coverage.

The study should be described as practical weak-correlation and distributional
stability diagnostics. It cannot validate exact iid assumptions. Preserve the
original tests and disclose tolerance exploration; confirm any selected operating
regime with fresh independent trajectories.

## Method references

- [Lakens (2017), equivalence testing](https://doi.org/10.1177/1948550617697177).
- [Energy distance definition](https://pages.stat.wisc.edu/~wahba/stat860public/pdf4/Energy/EnergyDistance10.1002-wics.1375.pdf).
- [Cluster bootstrap resampling](https://stat.ethz.ch/CRAN/web/packages/ClusterBootstrap/refman/ClusterBootstrap.html).
