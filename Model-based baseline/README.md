# Baseline comparisons

The shared results notebook is `Paper plots/Baselines and Coexistence.ipynb`.
It contains separate holistic and Cheetah reductionist comparisons, each with
RTS and STR bars for MORBO, qNParEGO, and the random-threshold baseline.

## Current baseline evaluation: random thresholds

`setup_random_threshold_evaluation.json` configures 20 sequences per case.
At every iteration, each sequence independently draws a threshold uniformly
with replacement from the distinct thresholds in the saved NSGA-II Pareto set.
The horizon remains 100 global iterations for holistic and 50 Cheetah requests
for reductionist. Each sequence uses the original 10 shared held-out simulator
seeds, yielding 200 baseline simulations per case. No Pareto search or
model-based simulation is repeated.

```cmd
".venv\Scripts\python.exe" -B "Model-based baseline\evaluate_random_threshold_baseline.py" --workers 4
```

The evaluator writes `holistic random threshold baseline.pkl` and
`reductionist random threshold baseline.pkl`. Each stores the sampled threshold
indices, physical and normalized sequences, sampling seed, shared simulator
seeds, all measured overhead/instability pairs, and RTS/STR statistics. It
resumes matching partial caches and skips completed ones. `--case holistic`
or `--case reductionist` selects one case; `--validate-only` checks inputs
without simulating or writing results.

The current notebook uses these sampled sequence sets for its baseline bars.
RTS computes exact hypervolume over their mean costs; STR computes the existing
128-direction expected-length hypervolume across the same 20 sequences. These
are finite sampled-set hypervolumes, not the hypervolume of a single random
policy obtained by averaging over all sequences. Model-based costs, reference
points and plotted hypervolumes remain unchanged. The original constant-margin
evaluation caches described below are preserved for historical comparison.

## Holistic setup

All scripts read `setup_model_free_holistic.json`. It embeds the simulator
parameters, Xi matrices, reference costs, and optimizer settings. The standard
setting is `gamma_sync = gamma_grav = gamma_heter = 1`, simulated rewards, and
a horizon of 100 global iterations.

## Constant-margin baseline

`holistic single fintess margin baseline.py` evaluates 10 fixed physical margins
from -10 to 20, with one realization per margin. Gaussian kernel regression
uses bandwidth 40 in physical margin units. NSGA-II searches these smoothed
costs and writes `single fitness margin Pareto set.pkl`.

The saved `pareto_margins` are scalar thresholds to repeat throughout the
horizon. `pareto_predicted_costs` are surrogate predictions. Raw measurements
and their seeds are stored under `training`.

## MORBO and qNParEGO

Run from CMD, sequentially:

```cmd
cd /d "C:\Users\Mohamed2\Documents\Journal\Cosmic Octopi"
".venv\Scripts\python.exe" -B "Model-based baseline\holistic MORBO.py"
".venv\Scripts\python.exe" -B "Model-based baseline\holistic qNParEGO.py"
```

Each command runs RTS then STR. Each formulation uses 8 initial Sobol points
and 20 steps of 4 candidates: 88 evaluated sequences, with two simulated
realizations each. Training hypervolume is recorded at initialization and every
two steps. Progress bars show completed candidates and optimization steps.

The scripts reuse `MORBO/script_holistic_gravity.py` and `qNParEGO/runner.py`.
They retain the project's existing single-trust-region MORBO and global
sequential-batch qNParEGO adaptations.

Outputs in this folder:

- `holistic RTS MORBO Pareto set.pkl`
- `holistic STR MORBO Pareto set.pkl`
- `holistic RTS qNParEGO Pareto set.pkl`
- `holistic STR qNParEGO Pareto set.pkl`

These files also serve as checkpoints during optimization (`complete: false`).
Rerun the same command after interruption to resume; completed formulations
are skipped. Settings, source hashes and dependency versions must match the
checkpoint. Use `--output-dir "some other folder"` for a separate experiment.
Temporary save files are cleaned after each atomic replacement. Other result,
setup, source and notebook files are not written by these scripts.

Optional flags are `--formulation RTS`, `--formulation STR`, `--setup PATH`,
and `--validate-only`. The last checks configuration without running or writing
an experiment.

Completed model-based files contain:

- `pareto_unit_sequences`: selected sequences in [0, 1].
- `pareto_margin_sequences`: the corresponding physical margins in [-10, 20].
- `pareto_indices`, `pareto_cost_realizations`, and `pareto_scenario_seeds`.
- `pareto_mean_costs`, `pareto_directional_scores`, and `training_hypervolume`.
- Full observations and optimizer state in `runs["RTS"]` or `runs["STR"]`.
- The shared setup, source hashes, dependency versions, and completion status.

RTS selects nondominated mean costs. STR selects nondominated expected length
score profiles on the configured 128 directions. The latter is an empirical
approximation to the STR solution set: filtering by mean costs could discard
an STR-optimal sequence. Both selections are checked to preserve the respective
training hypervolume of all evaluated candidates on those same directions.

## Held-out comparison

`evaluate_holistic_comparison.py` stores all held-out cost realizations in a
separate evaluation cache. The notebook uses these same measurements to report
both RTS and STR hypervolume for the constant-margin baseline. The original
setup, search snapshots, and cache are preserved. Evaluation settings come
from `held_out_evaluation` in the shared setup.

```cmd
".venv\Scripts\python.exe" -B "Model-based baseline\evaluate_holistic_comparison.py" --workers 4
```

This evaluates all five returned solution sets with the same 10 fresh seeds,
excluding seed collisions with training, and the same 100-iteration horizon.
Each constant threshold is repeated over the whole horizon. The separate
`holistic hypervolume comparison.pkl` cache stores all measured costs and
resumes interrupted evaluations. A completed cache is reused without rerunning
simulations. The original five Pareto-set files remain untouched.

Open `Paper plots/Baselines and Coexistence.ipynb` to recompute the six
hypervolumes from these measurements and display the grouped barplot and
results table. Each formulation contains MORBO, qNParEGO, and the constant-margin
baseline. The plot is also exported as `Holistic baseline comparison.png` and
`.svg` beside the notebook. Reopening the notebook does not rerun simulations.

RTS uses exact standard two-dimensional hypervolume of held-out mean costs.
STR uses the existing expected-length hypervolume with 128 directions. Both
definitions are also applied to the same constant-margin baseline set.
All use reference costs (2,200,000 KiB, 310)
and the same normalization, with reference (1, 1) after normalization. The
notebook reports the difference between exact RTS area and the earlier
128-direction approximation. These are single-optimization-run results;
the 10 held-out realizations per solution are not independent optimization runs.

## Verification

```cmd
".venv\Scripts\python.exe" -B "Model-based baseline\test_holistic_model_based_runner.py"
```

The checks exercise both existing optimization loops with cheap stand-ins for
simulation and acquisition, including interruption/resume and an example where
RTS and STR select different solutions. Test files are confined to system Temp.

## Cheetah reductionist comparison

`setup_model_free_reductionist.json` is self-contained. It uses the same standard
setting (`gamma_sync = gamma_grav = gamma_heter = 1`), simulated rewards, and
40 octopi. The horizon is 50 requests of the Cheetah class, matching the existing
reductionist experiments. All three methods use reference costs
(90,497.578125 KiB, 85), normalized to (1, 1).

The baseline retains 10 training margins in [-10, 20], one realization per
margin, Gaussian bandwidth 40, and NSGA-II with population 100 and 200
generations. MORBO and qNParEGO each retain 8 initial points and 20 optimization
steps of 4 candidates per formulation, with two realizations per candidate.
Both reuse the existing reductionist optimizer implementations. Unit search
sequences are converted to physical Cheetah margins before simulation.

Run the following commands from the project root, in this order:

```cmd
".venv\Scripts\python.exe" -B "Model-based baseline\reductionist single fitness margin baseline.py"
".venv\Scripts\python.exe" -B "Model-based baseline\reductionist MORBO.py"
".venv\Scripts\python.exe" -B "Model-based baseline\reductionist qNParEGO.py"
".venv\Scripts\python.exe" -B "Model-based baseline\evaluate_reductionist_comparison.py" --workers 4
```

The search scripts write:

- `reductionist single fitness margin Pareto set.pkl`
- `reductionist RTS MORBO Pareto set.pkl`
- `reductionist STR MORBO Pareto set.pkl`
- `reductionist RTS qNParEGO Pareto set.pkl`
- `reductionist STR qNParEGO Pareto set.pkl`

The evaluator writes `reductionist hypervolume comparison.pkl`. It evaluates
every returned solution on the same 10 fresh seeds and 50-request horizon.
The same constant-margin measurements are scored under both RTS (exact 2D
hypervolume of mean costs) and STR (expected-length hypervolume with 128
directions). Optimization and evaluation support resuming matching checkpoints;
completed files are reused. These are one optimization run per method and
formulation, with 10 held-out realizations per solution.

The notebook's reductionist section reads this evaluation cache and produces
the six-bar comparison and table. It exports `Reductionist baseline comparison.png`
and `.svg` beside the notebook. Re-executing its plotting cells does not run the
simulator. The existing holistic setup, results, and evaluation cache are preserved.

The reductionist checks additionally verify that normalized sequences map to
physical margins consistently in optimization and held-out evaluation:

```cmd
".venv\Scripts\python.exe" -B "Model-based baseline\test_reductionist_model_based_runner.py"
```
