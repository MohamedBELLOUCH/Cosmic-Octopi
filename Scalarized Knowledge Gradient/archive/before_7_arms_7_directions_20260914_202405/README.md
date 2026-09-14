# Scalarized Knowledge Gradient: holistic gravity pilot

This experiment implements the RTS and STR adaptations in **Selective Pressure
Allocation, Section 6.3.2 (Eqs. 49-52)**. Both minimize the normalized overhead and
instability increments using the cost length scalarization in Eq. 26.

## Files and execution

- `RTS_Scalarized_KG.py`: RTS algorithm, Gaussian KG calculation, shared statistics.
- `STR_Scalarized_KG.py`: STR algorithm with statistics of scalarized observations.
- `setup_holistic_gravity.json`: simulation, arms, directions, KG, and evaluation settings.
- `script_holistic_gravity.py`: runnable experiment, checkpointing, and result validation.
- `result_holistic_gravity.pkl`: online observations, selections, KG diagnostics,
  recommended subsets, held-out costs, and hypervolume histories.
- `test_scalarized_kg.py`: formula, statistics, ordering, and simulator checks.

From the project root:

```powershell
.\.venv\Scripts\python.exe "Scalarized Knowledge Gradient/script_holistic_gravity.py"
.\.venv\Scripts\python.exe "Scalarized Knowledge Gradient/test_scalarized_kg.py"
```

The script resumes matching checkpoints and skips a complete result. Changing
settings, implementation, or Xi matrices requires archiving the old result or
choosing a different output filename. Four evaluation workers isolate the
simulator's global NumPy state. No neural-network training or Bayesian optimizer
is run. The shared UCB online simulator and held-out subset evaluator are reused.

## Paper equations and implementation choices

The input to `update(arm, direction, cost_vec)` is a **cost vector to minimize**.
It is not a reward to maximize. The script divides each observed cost by its
fixed maximum single-request value: overhead by 40 times the largest class
policy payload, and instability by 1. The fixed lower reference is
`eta = [-0.01, -0.01]`, strictly below the nonnegative normalized observations.
The scalarization is

```
s_eta,w(c) = max_d(max(c_d - eta_d, 0) / w_d).
```

For each direction, every arm retains its own sample count, vector mean, and
componentwise sample variance. Welford updates compute the unbiased sample
variance (`ddof=1`). Statistics do not leak between directions.

Given empirical mean `m_i` and standard error `se_i = sample_std_i / sqrt(n_i)`,
the best competitor is `b_i = min_{a != i} m_a`. The Gaussian KG index is

```
q_i = abs(m_i - b_i) / se_i
v_i = se_i * (phi(q_i) - q_i * Phi(-q_i))
B_i = (T - k) * K * D * v_i
```

This exploration scaling follows the scalarized-KG bounds in Section 4.2 of
[Yahyaa, Drugan and Manderick (2014)](https://www.scitepress.org/PublishedPapers/2014/47966/47966.pdf),
DOI 10.5220/0004796600740083. `K=10`, `D=2`, and `k` is the upcoming one-based
global request. The paper's adaptation specifies standard errors; we use
`sample_std / sqrt(n)`, consistent with the original paper's multi-objective
derivation. The introductory original formula contains inconsistent typography
for this denominator and the normal density. The implementation uses the
standard normal density and CDF. Zero sample variance produces zero KG index;
no artificial variance floor is introduced.

- **RTS (Eqs. 49-50):** compute vector means and componentwise KG bonuses, then
  select `argmax -s(mean_cost_vector - vector_bonus)`.
- **STR (Eqs. 51-52):** scalarize each observation first, estimate its scalar mean
  and variance, then select `argmax -mean_scalar_cost + scalar_bonus`.

The factor `K*D` uses the original two-objective problem in both variants, as in
the original scalarized KG derivation. The final request has zero exploration
bonus. During adaptive selection, directions are drawn uniformly with an RNG
independent of the simulator. Ties select the smallest arm index. In RTS, large
bonuses can make several optimistic costs fall below the lower reference and
therefore tie after Eq. 26's positive-part operation; this is recorded in the
saved decision scores, rather than changing the paper's rule.

## Pilot setup

| Setting | Value |
|---|---|
| Independent runs | 1 |
| Rewards | Simulated OU, shared `Model calibration/Xi_matrix.json` |
| Gravity variances | 1 and 5 |
| Synchronization / request heterogeneity | 1 / 0.5 |
| Octopi | 40 |
| Arms | 10 evenly spaced values in [0, 1], mapped to active-class margins [-10, 20] |
| Directions | 5 positive unit vectors at angular midpoints |
| Online training | 150 global requests |
| Initialization | Two complete passes of the Eq. 43 schedule: 100 requests |
| Adaptive KG | 50 requests |
| Held-out sequence horizon | 100 global requests |
| Held-out sequences per subset | 16 Sobol sequences plus all constant sequences, deduplicated |
| Realizations per sequence | 2 |
| Hypervolume quadrature | 128 angular midpoint directions |
| Total-cost hypervolume reference | (2,200,000 KiB, 310) |

Two observations per arm-direction pair are necessary to estimate variance.
Thus, a 100-request budget would contain initialization only. The user's allowed
150-request budget leaves 50 adaptive choices. The evaluation horizon stays at
100: online training and held-out evaluation are separate budgets.

At every checkpoint the recommended subset is the union of the empirical
exploitation maximizers for observed directions, with no exploration bonus.
Before all pairs are initialized, only observed arms are eligible. The possible
100-request sequences over this subset are approximated by the sampled set
above. Evaluations are never fed back to KG. Repeated subsets share a cache,
and all conditions use common held-out seeds and Sobol draws.

The saved result contains the pre-observation scores, standard errors, Gaussian
indices, bonuses, and remaining budget for all 50 adaptive decisions. This
allows checking whether a selected arm came from exploitation, exploration, or
a tie. The notebook `Paper plots/Model-free hypervolume results.ipynb` adds
separate RTS and STR KG gravity figures with all 150 checkpoints and a final
statistics table. The vertical dotted line marks initialization ending at 100.

## Interpretation and comparison

This is a single-run diagnostic pilot using placeholder Xi calibration and a
small held-out simulation budget. It does not establish an algorithm ranking.
The shared hypervolume evaluator measures normalized total-cost sets with the
same reference and horizon as holistic UCB, MORBO, and qNParEGO.

There is a separate optimizer-convention difference: existing UCB selects using
`min_d((1 - normalized_cost_d) / w_d)`, whereas this KG implementation follows
the paper's `-max_d((normalized_cost_d - eta_d)_+ / w_d)`. These are not equivalent
for general directions. Existing UCB results remain unchanged. Matching this
convention, training budgets, and evaluation designs is necessary before making
a controlled comparison. Moreover, online costs remain history-dependent;
the stationarity assumption is inherited from the paper, not established by
this experiment.
