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
levels: 120 result pickle files containing 480 online trajectories in total.

Both algorithms use 7 arms, 7 directions, and 150 online requests. UCB initializes
49 pairs; KG makes two passes (98 requests). Held-out horizons remain 100 holistic
and 50 reductionist, with the same respective fixed hypervolume references as
the current single-run results. Both minimize the paper's cost scalarization
with lower reference [-0.01,-0.01]. This is distinct from the upper total-cost
reference used for hypervolume.

Run r uses training seed `42+(r-1)` and evaluation seed `20260914+(r-1)`. Independent
runs restart the simulation, optimization, and evaluation cache; seeds are paired
across algorithms and conditions within a run. The two held-out realizations
per sequence are separate from the ten independent algorithm runs.

When the final batch is authorized, launch:

```powershell
.\.venv\Scripts\python.exe "Model-free experiments/script.py"
```

The script creates complete per-run setup JSON files, then calls existing
entrypoints in fresh processes. It runs experiments sequentially, with four
held-out workers per child. Future outputs go under
`results/run_XX_seed_SEED/UCB|KG/`. It does not read or overwrite the algorithm
folders' current single-run results. Compatible checkpoints resume or are
skipped; changed configurations or sources are rejected. Use a new output
directory when changing an existing batch's configuration.

The notebook currently shows one run per condition. Repeated-run aggregation
and uncertainty intervals can be added after the deferred batch finishes.
