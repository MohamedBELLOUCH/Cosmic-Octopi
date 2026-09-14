# Deferred model-free suite

`setup.json` and `script.py` are ready for the final batch stage. **The script
has not been run and no batch results have been generated.**

The JSON is self-contained: it embeds the simulation settings, UCB/KG options,
Xi matrices, arms, directions, budgets, evaluation horizons, sweep levels, and
references. It does not read the adjacent algorithm setup JSON files or the
calibration JSON when executed. Existing Python experiment scripts remain the
implementations of the algorithms and evaluator.

The suite configures 10 independent runs of all 12 algorithm/approach/sweep
combinations: UCB/KG, holistic/Cheetah reductionist, and gravity/synchronization/
request heterogeneity. Each invocation evaluates both RTS/STR at both sweep
levels: 120 experiment records containing 480 online trajectories in total,
stored together in **`model_free_results.pkl`** beside the script.

Both algorithms use 7 arms, 7 directions, and 150 online requests. UCB initializes
49 pairs; KG makes two passes (98 requests). Held-out horizons remain 100 holistic
and 50 reductionist, with the same respective fixed hypervolume references as
the current single-run results. Both minimize the paper's cost scalarization
with lower reference [-0.01,-0.01]. This is distinct from the upper total-cost
reference used for hypervolume.

`evaluation.checkpoint_every = 5` evaluates and saves hypervolume at online
iterations 5, 10, ..., 150: 30 checkpoints per trajectory. Both bandits still
update at all 150 requests. For reductionist runs these are Cheetah requests.
Unselected checkpoints do not trigger held-out evaluation; repeated recommended
subsets reuse the evaluation cache. Runtime therefore need not fall by exactly
a factor of five. The final request is always evaluated, even for a budget that
is not divisible by the checkpoint interval.

The existing single-run results retain all 150 checkpoints. Their provenance
records the compatible checkpoint-scheduling correction (the interval of 1 has
identical behavior); original sources and results are preserved in this folder's
`archive/before_every_5_*` directory.

Run r uses training seed `42+(r-1)` and evaluation seed `20260914+(r-1)`. Independent
runs restart the simulation, optimization, and evaluation cache; seeds are paired
across algorithms and conditions within a run. The two held-out realizations
per sequence are separate from the ten independent algorithm runs.

When the final batch is authorized, launch:

```cmd
cd /d "C:\Users\Mohamed2\Documents\Journal\Cosmic Octopi"
".venv\Scripts\python.exe" -B "Model-free experiments\script.py"
```

The only persistent project output is `Model-free experiments/model_free_results.pkl`.
The script passes complete setups directly to the existing UCB/KG runner
functions, with four held-out workers. It does not create per-run setup files,
result directories, notebook outputs, or logs in the project. Bytecode writes
are disabled for the parent and workers. Existing source, setup, calibration,
notebook, and individual result files are not modified by this command.

Intermediate checkpoints and atomic-save staging files use the system temporary
directory and are removed after use. It must be on the same volume as the output
for atomic replacement (both are on C: in the current installation).

The combined file checkpoints after each completed experiment. Ctrl+C or a
Python error also saves the latest available checkpoint of the current experiment
before exiting. Rerunning the same command resumes partial work and skips finished
experiments. A forced process termination or power loss retains the last completed
experiment in the combined file. Changed configurations or sources are rejected;
choose a new `output_file` for a different setup.

The pickle dictionary contains `setup`, `source_sha256`, `complete`, and
`experiments`. Experiment keys have the form
`run_01_seed_42/UCB/holistic/gravity`. Each value preserves the existing runner's
complete result dictionary, including its setup, seeds, raw observations,
held-out evaluation cache, and RTS/STR hypervolume histories. Group matching
algorithm/approach/experiment/condition entries across the ten independent runs
to calculate the mean and shaded variability band later.

The notebook currently shows one run per condition. Repeated-run aggregation
and uncertainty intervals can be added after the deferred batch finishes.
