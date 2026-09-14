# MORBO experiments

The experiment runs separate RTS and STR optimizations for gravity variances 1
and 5, with 40 octopi, a common horizon of 100 global iterations, and simulated
rewards from `../Model calibration/Xi_matrix.json`.

- `setup_holistic_gravity.json` contains the simulator setup and file locations.
- `hyperparameters.json` contains the optimizer settings.
- `script_holistic_gravity.py` runs the experiment and saves checkpoints.
- `result_holistic_gravity.pkl` contains the observations and hypervolume histories.

Run or resume the configured experiment from the repository root, using the project's virtual environment:

```powershell
.venv\Scripts\python.exe "MORBO/script_holistic_gravity.py"
```

The optional `--pilot` flag selects 8 initial points and one batch of 4 new points
per condition; use it with a fresh result file after archiving any existing run. Its
result explicitly records `profile: pilot` and the effective settings. The current
configured run uses 8 initial points, 20 optimization steps, and batch size 4:
88 candidates per condition. Both profiles use 2 simulated realizations per
candidate, configurable in the setup. The first pilot has been preserved as
`result_holistic_gravity_pilot.pkl`. The 10-step checkpoint is preserved as
`result_holistic_gravity_10_steps.pkl`; the 20-step run continues its saved states.

`Paper plots/Model-based hypervolume results.ipynb` plots initialization at step 0 and one
hypervolume checkpoint after each completed optimization batch, through step 20.

Rerunning with the same profile, configuration, Xi matrices, and source code
resumes the saved checkpoint, including an interrupted candidate batch. Archive
the existing result before changing an experiment's settings or code.

## Optimizer and objectives

This adapts the **single trust region** in `MORBO_success.ipynb`; it is not the
published MORBO implementation with multiple coordinated regions. Its trust
region sizes, update factors, batch size, GP, Monte Carlo sample count, and
acquisition optimization settings follow that notebook. Historical training
points retain their original unit-cube coordinates; the current trust region
restricts the acquisition search.

Each candidate is a sequence of 100 values in [0, 1]. At each global iteration,
the corresponding value is mapped to the active class's fitness-margin interval.
The intervals are currently [-10, 20], as in the existing Pareto experiment.

The scalar acquisition is `qLogNoisyExpectedImprovement`, replacing the toy's
mean-vector `qLogNoisyExpectedHypervolumeImprovement` to distinguish the two
orders of scalarization. For a cost vector `c`, reference costs `eta`, and a
positive unit direction `lambda`, the score to maximize is:

```text
s(c, lambda) = min_i max(1 - c_i / eta_i, 0) / lambda_i
RTS = s(mean(cost realizations), lambda)
STR = mean(s(each cost realization, lambda))
```

Directions change at each optimization iteration. The reference costs are fixed
at [2200000, 310], and normalization prevents the two objectives' different
units from dominating direction sampling. Conditions share initial Sobol points,
scenario seeds, and direction draws. Their GP targets and subsequent candidate
choices follow their respective formulations.

## Result contents

Each entry of `runs` stores `X_unit`, `cost_realizations`, `scenario_seeds`,
`evaluation_counts`, `hypervolume_history`, and the trust-region state/history.
Raw costs are ordered as **total overhead in KiB** and **total instability**.
Policy payloads use 32-bit weights, independently of the GP's 64-bit arithmetic.

Hypervolumes use the RTS/STR polar indicator, estimated over 128 fixed angular
midpoints. They are recorded in normalized objective units. Multiplying by the
product of the reference costs converts the normalized area to raw objective
units. The result also records the setup, effective optimizer settings, Xi
matrices, source fingerprints, profile, and completion status.

## Reductionist gravity experiment: Cheetah

The independent reductionist experiment uses:

- `setup_reductionist_gravity.json`
- `script_reductionist_gravity.py`
- `results_reductionist_gravity.pkl`

Run it from the repository root:

```powershell
.venv\Scripts\python.exe "MORBO/script_reductionist_gravity.py"
```

It optimizes 50 Cheetah margins over 50 Cheetah requests. Only Cheetah overhead
and instability contribute to its objectives. The class requests still arise
from the four-class system dynamics. The current horizon and margin interval
[-10, 20] follow the existing Cheetah Pareto experiment; simulator settings match
the current holistic run.

RTS and STR each compare gravity variances 1 and 5 with 40 octopi, 8 initial
Sobol points, 20 optimization steps, batches of 4, and 2 realizations per
candidate. The runner shares scalarization, hypervolume, acquisition, and
trust-region functions with `script_holistic_gravity.py`. It reads the same
`hyperparameters.json`, deriving the effective GP dimension of 50 from the
reductionist horizon; the holistic dimension remains 100.

The fixed reference costs are **[90497.578125 KiB, 85]**. The overhead coordinate
is 1.01 times the maximum possible overhead: 40 octopi times 50 requests times
the 44.80078125 KiB Cheetah policy size. This replaces the paper's 4000 KiB
reference, which is below the overhead of the current simulated Cheetah runs.
The instability reference remains 85. Both gravity conditions and both
formulations use this same fixed reference throughout optimization.

`Paper plots/Model-based hypervolume results.ipynb` contains separate Cheetah RTS and STR
figures, plotted against completed optimization steps. The reductionist and
holistic normalized hypervolumes use different reference costs.

## Synchronization experiments

The synchronization experiments compare `gamma_sync = 1` and `gamma_sync = 0.8`,
with `gamma_grav = 0.1` and `gamma_heter = 0.5`. These use the paper's
synchronization comparison with the updated synchronization levels.
The other simulator settings and fixed reference costs match the
corresponding gravity experiments.

The previous comparison is preserved in the result files ending in
`_gamma_sync_0p5.pkl`. The current results reuse the unchanged `gamma_sync = 1`
runs and contain newly optimized `gamma_sync = 0.8` runs.

Holistic files:

- `setup_holistic_synchronization.json`
- `script_holistic_synchronization.py`
- `result_holistic_synchronization.pkl`

Cheetah reductionist files:

- `setup_reductionist_synchronization.json`
- `script_reductionist_synchronization.py`
- `results_reductionist_synchronization.pkl`

Run or resume each approach from the repository root:

```powershell
.venv\Scripts\python.exe "MORBO/script_holistic_synchronization.py"
.venv\Scripts\python.exe "MORBO/script_reductionist_synchronization.py"
```

Both scripts use `synchronization_runner.py`, which shares candidate evaluation,
scalarization, acquisition, and trust-region functions with the gravity scripts.
They read `hyperparameters.json`, use 40 octopi, and run separate RTS/STR
optimizations for each synchronization level: 8 initial points, 20 optimization
steps, batches of 4, and 2 simulated realizations per candidate. The holistic
horizon is 100 global iterations; the reductionist horizon is 50 Cheetah requests.
Effective GP dimensions follow these respective horizons.

The JSON `sweep` block identifies `gamma_sync` and its two values. Each result
records the actual sweep value for every run, along with raw objective samples,
random seeds, optimizer state, effective settings, and 21 hypervolume checkpoints.
Reference points remain fixed across synchronization levels and optimization
steps, including the simulator-scale Cheetah reference already used for gravity.

`Paper plots/Model-based hypervolume results.ipynb` includes four additional synchronization
figures: holistic RTS/STR and Cheetah RTS/STR. Their horizontal axes count
optimization steps, with initialization at step 0.

## Request heterogeneity experiments

Following Section 7.4.3 of the paper, these experiments compare
`gamma_heter = 0.1` and `gamma_heter = 0.8`, with `gamma_sync = 1` and
`gamma_grav = 0.1`. All other simulator settings, shared MORBO hyperparameters,
and fixed reference costs match the corresponding synchronization experiments.

Holistic files:

- `setup_holistic_heterogeneity.json`
- `script_holistic_heterogeneity.py`
- `result_holistic_heterogeneity.pkl`

Cheetah reductionist files:

- `setup_reductionist_heterogeneity.json`
- `script_reductionist_heterogeneity.py`
- `results_reductionist_heterogeneity.pkl`

Run or resume from the repository root:

```powershell
.venv\Scripts\python.exe "MORBO/script_holistic_heterogeneity.py"
.venv\Scripts\python.exe "MORBO/script_reductionist_heterogeneity.py"
```

The scripts use `heterogeneity_runner.py`, sharing the established gravity
candidate evaluators, RTS/STR scalarization, GP acquisition, and trust-region
updates. Each approach runs separate RTS and STR optimizations for both levels,
with 8 initial Sobol points, 20 optimization steps, batches of 4, and 2 simulated
realizations per candidate. Horizons remain 100 global iterations for the
holistic case and 50 Cheetah requests for the reductionist case, with 40 octopi.

The setup `sweep` block identifies `gamma_heter`; each run records its actual
heterogeneity value, raw objectives, random seeds, optimizer state, and all 21
hypervolume checkpoints. `Paper plots/Model-based hypervolume results.ipynb` contains four
request heterogeneity figures, plotted against optimization steps. The previous
gravity and synchronization sections remain available in the same notebook.
