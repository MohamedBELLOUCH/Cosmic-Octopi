# Scalarized UCB experiments

The six current experiments cover holistic and Cheetah reductionist gravity,
synchronization, and request-rate heterogeneity. Each pair has a setup JSON,
script, and result pickle. All scripts call `experiment_runner.py`; the separate
RTS and STR classes are in `RTS_Scalarized_UCB.py` and `STR_Scalarized_UCB.py`.

## Current configuration

- One independent run in each current result; simulated OU rewards and 40 octopi.
- 7 arms, 7 positive unit directions, and 150 online requests.
- 49 initialization requests followed by 101 adaptive UCB choices.
- Holistic arms span [0,1] and map to active-class fitness margins [-10,20].
- Reductionist arms span the physical Cheetah margin interval [-10,20].
- Both algorithms use lower normalized-cost reference [-0.01,-0.01].
- Held-out horizons: 100 holistic, 50 reductionist.
- Upper total-cost HV references: holistic (2,200,000 KiB, 310);
  reductionist (90,497.578125 KiB, 85), matching MORBO/qNParEGO.

| Sweep | Levels | Other fixed parameters |
|---|---|---|
| Gravity variance | 1, 5 | synchronization 1; request heterogeneity 0.5 |
| Synchronization | 1, 0.8 | gravity variance 0.1; request heterogeneity 0.5 |
| Request heterogeneity | 0.1, 0.8 | gravity variance 0.1; synchronization 1 |

## Aligned cost convention

Current setups select `observation_mode="cost"`. Observed incremental overhead
is divided by M times the largest class payload (holistic), or M times the
Cheetah payload (reductionist). Instability is divided by 1. These normalized
costs and the lower reference are shared with KG.

The paper's Eq. 26 is `s(c)=max_d(max(c_d-eta_d,0)/w_d)`. RTS uses
`-s(mean cost vector)`; STR averages `-s(c)` over observations. UCB adds
`sqrt(2 log(n_j)/n_ji)` to these maximized exploitation scores. Statistics are
specific to each arm-direction pair. Recommendations omit exploration bonuses
and unseen pairs; ties use the smallest arm index. Scores include the latest
observation. Legacy reward-mode constructor defaults remain available for the
original toy; the current experiments explicitly use cost mode.

## Evaluation and execution

Every checkpoint assesses the currently recommended arm subset using 16 common
Sobol sequences plus each recommended arm's constant sequence, deduplicated.
There are two held-out realizations per sequence and 128 angular directions for
the shared RTS/STR polar hypervolume. Evaluation never updates UCB. Seeds and
designs are paired across algorithms, formulations, and sweep levels.

The notebook `Paper plots/Single run model-free hypervolume.ipynb` has twelve
UCB/KG comparison figures with all 150 checkpoints and the initialization
boundaries (49 for UCB, 98 for KG). Curves can decrease as recommendations
change. These single-run estimates and small evaluation sets are not definitive
algorithm rankings. The simulator is history-dependent; the stationarity
assumption is inherited from the paper.

Run an individual experiment using the project interpreter, for example:

```powershell
.\.venv\Scripts\python.exe "Scalarized UCB/script_holistic_gravity.py"
```

Scripts use four held-out workers, resume compatible checkpoints, and skip
complete results. Changing code/setup/calibration requires a new output filename
or archiving the old result. Inline `Xi_matrices` support the self-contained
10-run suite in `Model-free experiments`, which is now complete. Its mean curves
and bands of one sample standard deviation appear in
`Paper plots/Multiple run model-free hypervolume.ipynb`.

Earlier code, setups, results, and documentation are preserved under
`archive/before_7_arms_7_directions_*`. The previous plotting notebook and hash
manifest are under `Paper plots/archive`. The original `bandits.ipynb` is preserved.
