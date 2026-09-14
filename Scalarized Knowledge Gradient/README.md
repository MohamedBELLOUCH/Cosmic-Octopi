# Scalarized Knowledge Gradient experiments

The RTS and STR classes follow Selective Pressure Allocation, Section 6.3.2,
Eqs. 49-52, using the cost length scalarization in Eq. 26. All six approach/sweep
pairs have setup/script/result files. `kg_runner.py` serves the entrypoints and
reuses UCB's online simulator and held-out evaluator. No neural-network training
or Bayesian optimizer is run.

## Current configuration

All settings match UCB except the algorithm-specific exploration and required
initialization. Both use 7 arms, 7 directions, 150 online requests, normalized
cost observations, and the lower reference [-0.01,-0.01]. KG requires two full
arm-direction passes (98 requests), leaving 52 adaptive requests. Holistic arms
span [0,1]; reductionist arms are physical Cheetah margins in [-10,20]. Each
result currently contains one independent run, with simulated OU rewards.

| Approach | Held-out horizon | Upper total-cost HV reference (KiB, instability) |
|---|---:|---|
| Holistic | 100 | (2,200,000, 310) |
| Cheetah reductionist | 50 | (90,497.578125, 85) |

Gravity compares 1 and 5 (sync 1, heterogeneity 0.5); synchronization compares
1 and 0.8 (gravity variance 0.1, heterogeneity 0.5); request heterogeneity compares
0.1 and 0.8 (gravity variance 0.1, sync 1). The horizons and upper HV references
also match the corresponding MORBO/qNParEGO experiments.

## Equations and implementation

`update(arm, direction, cost_vec)` receives costs to minimize. Overhead is divided
by M times the largest class payload (holistic) or the Cheetah payload
(reductionist); instability is divided by 1. The scalarization is
`max_d(max(cost_d-reference_d,0)/weight_d)`.

Welford updates maintain separate means and unbiased sample variances (`ddof=1`)
for every arm-direction pair. With `se=sample_std/sqrt(n)`, the Gaussian KG index
uses the best OTHER arm's empirical mean:

```
b_i = min_{a != i} mean_a
q_i = abs(mean_i - b_i) / se_i
v_i = se_i * (phi(q_i) - q_i * Phi(-q_i))
B_i = (T - k) * K * D * v_i
```

The exploration scaling follows Section 4.2 of
[Yahyaa, Drugan and Manderick (2014)](https://www.scitepress.org/PublishedPapers/2014/47966/47966.pdf),
DOI 10.5220/0004796600740083. K=7, D=2, T=150, and k is the upcoming one-based
online request. The final request has zero bonus. Standard errors follow the
multi-objective derivation and adapted paper; the normal PDF/CDF use their
standard definitions despite inconsistent typography in the original article's
introduction. Zero variance gives zero KG index, without an artificial floor.

- RTS: estimate vector means and componentwise bonuses, then select
  `argmax -s(mean_cost_vector - vector_bonus)` (Eqs. 49-50).
- STR: scalarize each observation first, then select
  `argmax -mean_scalarized_cost + scalar_bonus` (Eqs. 51-52).

The original K*D scaling uses D=2 for both variants. Adaptive directions are
uniform draws from an RNG independent of the simulator. Ties select the smallest
arm index. RTS can produce tied optimistic scores when large bonuses push costs
below the lower reference before positive-part scalarization. This behavior is
recorded, rather than changing the paper's rule to improve apparent performance.

## Results and reproducibility

Recommendations omit exploration bonuses. Every checkpoint evaluates 16 common
Sobol sequences plus all constant sequences over the recommended subset,
deduplicated, with two held-out realizations and 128 HV quadrature directions.
Held-out observations never update KG. Seeds and evaluation designs match UCB.
Results save all 150 checkpoints and the 52 pre-observation adaptive diagnostics
(standard errors, KG indices, bonuses, scores, remaining budget), plus final
statistics and raw observations.

The notebook overlays UCB and KG in twelve RTS/STR comparison figures and
reports KG exploration/tie counts. These are single-run empirical results under
placeholder Xi calibration and a small held-out evaluation budget. They do not
prove an algorithm fails generally. The simulator is history-dependent; the
stationarity assumption is inherited from the paper.

```powershell
.\.venv\Scripts\python.exe "Scalarized Knowledge Gradient/script_holistic_gravity.py"
```

All entrypoints use four evaluation workers, resume compatible checkpoints, and
skip complete results. Inline Xi matrices support the deferred self-contained
10-run suite in `Model-free experiments`; that script has not run. Prior pilot
code, settings, results, and documentation are archived under
`archive/before_7_arms_7_directions_*`; the prior notebook is also archived.
