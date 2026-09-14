# Model-based batch

Run these commands in CMD:

```cmd
cd /d "C:\Users\Mohamed2\Documents\Journal\Cosmic Octopi"
".venv\Scripts\python.exe" -B "Model-based experiments\script.py"
```

The sole persistent project output is **`model_based_results.pkl`** in this
folder. The script does not change existing source, setup, calibration,
notebook, or individual result files. Bytecode writes are disabled. Intermediate
checkpoints and atomic-save staging files use the system temporary directory
and are cleaned up. Staging and output must share a volume for atomic replacement;
both are on C: in this installation.

The self-contained setup configures 10 independent runs of MORBO and qNParEGO,
each with holistic and Cheetah reductionist gravity, synchronization, and request
heterogeneity experiments. Each of these 120 experiment records contains both
RTS and STR at two sweep values, for 480 optimization trajectories.

The existing numerical budgets are preserved: 8 initial candidates, 20 BO steps,
batches of 4, and 2 simulated realizations per candidate. Each trajectory has 88
candidates and 21 hypervolume checkpoints, including initialization at step 0.
Holistic and reductionist evaluation horizons remain 100 and 50 respectively,
with their existing fixed reference points. The optimizer adaptations and
simulator are reused from MORBO and qNParEGO without changing their files.

The ASCII CMD bar reports completed experiments out of 120, elapsed time,
estimated time remaining, and the current run/algorithm/setting. Its status
shows initialization progress, then the completed BO step out of 20 for the
active RTS/STR condition. H/R denote holistic/reductionist; grav/sync/hete denote
the three sweeps. The overall bar advances after an experiment is saved to the
combined pickle. The time estimate appears after the first newly completed
experiment and depends on the settings. No log file is written.

The combined file checkpoints after each completed experiment. Ctrl+C or a
Python error also preserves the latest available checkpoint of the active
experiment. The same command resumes partial work and skips complete experiments.
A forced termination or power loss retains the last completed experiment in
the combined file. If settings or sources change, resume is refused; choose a
new `output_file` for a different configuration.

The pickle is a dictionary with `setup`, `source_sha256`, `complete`, and
`experiments`. Keys such as `run_01_seed_42/MORBO/holistic/gravity` identify each
experiment. Values contain the original runner result structure: raw objective
realizations, seeds, candidate sequences, optimizer states, and RTS/STR
hypervolume histories. Matching entries across independent runs can be used
later for mean curves and shaded variability bands.

Validation uses stub optimizers in temporary folders. It does not launch the
real batch or modify the running model-free experiment.
