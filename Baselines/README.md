# Holistic baseline comparison

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
".venv\Scripts\python.exe" -B "Baselines\holistic MORBO.py"
".venv\Scripts\python.exe" -B "Baselines\holistic qNParEGO.py"
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

The user selected standard hypervolume for the constant-margin baseline after
the searches finished. `evaluate_holistic_comparison.py` records this choice in
the evaluation cache; the original setup and completed search snapshots are
preserved. Its other evaluation settings come from `held_out_evaluation` in the
shared setup.

```cmd
".venv\Scripts\python.exe" -B "Baselines\evaluate_holistic_comparison.py" --workers 4
```

This evaluates all five returned solution sets with the same 10 fresh seeds,
excluding seed collisions with training, and the same 100-iteration horizon.
Each constant threshold is repeated over the whole horizon. The separate
`holistic hypervolume comparison.pkl` cache stores all measured costs and
resumes interrupted evaluations. A completed cache is reused without rerunning
simulations. The original five Pareto-set files remain untouched.

Open `Paper plots/Holistic baseline comparison.ipynb` to recompute the five
hypervolumes from these measurements and display the barplot and results table.
The plot is also exported to PNG and SVG beside the notebook.

RTS and the constant-margin baseline use exact standard two-dimensional
hypervolume of held-out mean costs. STR uses the existing expected-length
hypervolume with 128 directions. All use reference costs (2,200,000 KiB, 310)
and the same normalization, with reference (1, 1) after normalization. The
notebook reports the difference between exact RTS area and the earlier
128-direction approximation. These are single-optimization-run results;
the 10 held-out realizations per solution are not independent optimization runs.

## Verification

```cmd
".venv\Scripts\python.exe" -B "Baselines\test_holistic_model_based_runner.py"
```

The checks exercise both existing optimization loops with cheap stand-ins for
simulation and acquisition, including interruption/resume and an example where
RTS and STR select different solutions. Test files are confined to system Temp.
