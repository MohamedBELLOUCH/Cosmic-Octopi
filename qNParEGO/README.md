# qNParEGO experiments

These experiments adapt `qLogParEGO_success.ipynb` to the same simulated
RTS/STR objectives and evaluation budgets as the current MORBO experiments.
They cover both the holistic approach and Cheetah reductionist approach.

## Optimizer adaptation and comparison

The toy fits a multi-output GP to mean objective vectors and uses
`qLogNParEGO`, with augmented Chebyshev scalarization, over the global unit
cube. It constructs a batch sequentially through `optimize_acqf_list`, using
one acquisition function throughout the batch.

For consistency with the existing MORBO runner, this version instead fits
`SingleTaskGP` to the empirical RTS or STR length-scalarized scores for a
sampled direction, and uses `qLogNoisyExpectedImprovement`. RTS averages
realizations before length scalarization; STR scalarizes each realization
before averaging. The same direction is used throughout each batch.
This is a **qNParEGO-style RTS/STR adaptation**, not an unchanged execution of
the toy's native mean-vector `qLogNParEGO`. The supplied toy notebook is preserved.

The distinction from the current MORBO adaptation is global acquisition
search and sequential batch construction. MORBO restricts search to its
adaptive single trust region and optimizes a joint batch. Both runners share
candidate evaluation, length scalarization, and hypervolume calculation.
This comparison therefore concerns these toy-derived adaptations, not the
unmodified published algorithms.

`hyperparameters.json` retains the toy's float64 calculations, CPU GP fitting,
128 Monte Carlo samples, batches of 4, 20 optimization steps, 8 acquisition
restarts, 256 raw samples, batch limit 5, and maximum 100 optimizer iterations.
Initialization uses **8 Sobol points**, matching MORBO, instead of the toy's
128. Each condition evaluates 88 candidates with 2 simulated realizations each.
The same initial designs, scenario seeds, and scalarization directions are used
for both algorithms; each algorithm generates its own subsequent candidates.

## Experiments and files

| Experiment | Compared values | Fixed dynamics |
| --- | --- | --- |
| Gravity | `gamma_grav = 1, 5` | `gamma_sync = 1`, `gamma_heter = 0.5` |
| Synchronization | `gamma_sync = 1, 0.8` | `gamma_grav = 0.1`, `gamma_heter = 0.5` |
| Request heterogeneity | `gamma_heter = 0.1, 0.8` | `gamma_sync = 1`, `gamma_grav = 0.1` |

For each experiment name (`gravity`, `synchronization`, or `heterogeneity`):

- Holistic: `setup_holistic_<name>.json`, `script_holistic_<name>.py`, and
  `result_holistic_<name>.pkl`.
- Cheetah: `setup_reductionist_<name>.json`, `script_reductionist_<name>.py`, and
  `results_reductionist_<name>.pkl`.

Each setup points to the shared hyperparameters, Xi matrices, its own result,
and the matching MORBO result. The shared implementation is `runner.py`.
Run or resume any experiment from the repository root, for example:

```powershell
.venv\Scripts\python.exe "qNParEGO/script_holistic_gravity.py"
.venv\Scripts\python.exe "qNParEGO/script_reductionist_gravity.py"
```

Results save raw objective realizations, unit-cube candidates, random seeds,
effective settings, source fingerprints, pending batches, and all 21 hypervolume
checkpoints. Rerunning resumes compatible results. `--pilot` selects a short
one-step profile; a subsequent full run archives that pilot before starting.

## Matched reference points and figures

The holistic horizon is 100 global iterations and its fixed cost reference is
**[2,200,000 KiB, 310]**. The reductionist horizon is 50 Cheetah requests and its
fixed reference is **[90,497.578125 KiB, 85]**. These exactly match MORBO, including
its simulator-scale Cheetah overhead reference. Both use 40 octopi and the same
policy sizes, fitness-margin bounds, and simulated reward coefficients.

Unlike the toy's moving reference, reference costs remain fixed across algorithms,
conditions, and optimization steps. The same 128 angular quadrature directions
estimate normalized RTS/STR hypervolume. `Paper plots/Model-based hypervolume results.ipynb`
overlays the two algorithms for every experiment and formulation, with step 0
denoting initialization and steps 1-20 denoting completed batches.
