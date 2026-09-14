# Scalarized UCB experiments

The first experiment runs the **RTS_ScalarizedMultiObjectiveUCB** and
**STR_ScalarizedMultiObjectiveUCB** algorithms from `bandits.ipynb` in the
holistic Cosmic Octopi environment. The original toy notebook is preserved.
Knowledge Gradient, Improved UCB, and Pareto UCB are not included.

## Files and execution

- `RTS_Scalarized_UCB.py`: RTS algorithm and shared selection/bookkeeping helpers.
- `STR_Scalarized_UCB.py`: STR algorithm, sharing the same UCB selection rule.
- `setup_holistic_gravity.json`: simulation, bandit, and evaluation settings.
- `script_holistic_gravity.py`: online training and held-out hypervolume evaluation.
- `result_holistic_gravity.pkl`: trajectories, recommendations, evaluation costs,
  seeds, and hypervolume checkpoints.
- `holistic_environment.py`: online simulator with one margin selection and one
  cost observation per global iteration.
- `test_scalarized_ucb.py`: algorithm ordering and simulator parity checks.

From the repository root:

```powershell
.\.venv\Scripts\python.exe "Scalarized UCB/script_holistic_gravity.py"
.\.venv\Scripts\python.exe "Scalarized UCB/test_scalarized_ucb.py"
```

The script resumes matching checkpoints and skips a completed result. Changing
settings, implementation files, or Xi matrices requires a new output filename or
archiving the previous result. Online training is saved after its short complete
trajectory; expensive evaluation work is checkpointed after each assessed subset.
The initial holistic gravity evaluation uses four worker processes; the other
five setups use two workers each, isolating the simulator's global NumPy RNG. They do not train any neural networks.

## First experiment

| Setting | Value |
|---|---|
| Independent runs | 1 |
| Holistic horizon | 100 global iterations |
| Arms | 10 evenly spaced values in [0, 1] |
| Scalarization directions | 5 positive unit vectors at evenly spaced angular midpoints |
| Initialization | Every arm-direction pair once: 50 iterations |
| Adaptive UCB | The remaining 50 iterations |
| Exploration bonus | sqrt(2 log(n_j) / n_ji), as in the toy |
| Gravity variances | 1 and 5 |
| Synchronization / request heterogeneity | 1 / 0.5 |
| Number of octopi | 40 |
| Fitness margin mapping | x in [0, 1] to [-10, 20] for the active class |
| Rewards | Simulated OU rewards, using `Model calibration/Xi_matrix.json` |
| Total-cost hypervolume reference | (2,200,000 KiB, 310) |

The paper's Section 7.5 specifies 200 global iterations, 10 arms, and 10
directions. This pilot instead uses the requested horizon of 100 and 5 directions
so that initialization leaves time for adaptive decisions. Other simulation
settings, the evaluation horizon, and the total-cost reference match the current
model-based holistic gravity experiments.

## Online feedback and RTS/STR

An arm is **one normalized fitness margin**, not an entire margin sequence.
At each global iteration the algorithm selects an arm, the simulator advances
its existing state, and the algorithm observes the overhead and instability
increments. No state reset occurs between pulls. The online environment mirrors
the shared offline evaluator and is tested to reproduce identical total costs
when given the same fixed sequence and scenario seed.

The overhead increment is bounded by the octopus count times the largest class
policy payload; the instability increment is bounded by 1. Observations are
converted to maximization rewards as `1 - increment / bound`, with reference
`z = [0, 0]`. The fixed bounds do not use future observations. These per-step
bounds are distinct from the reference for **total-cost** hypervolumes.

Both algorithms retain one UCB instance per scalarization direction. STR
scalarizes each observed vector first and averages those scalar values. RTS
averages reward vectors first and then scalarizes. For this experiment RTS uses
the selected scalarizer's observations, matching Eqs. 46-47. The extracted RTS
class also supports `mean_scope="global"` (its default) for compatibility with
the toy's pooled-vector statistics.

Two issues in the toy are addressed in the extracted implementation and runner:

1. RTS previously computed a scalar score before incorporating the latest reward.
   It now includes the new observation. In global pooling mode, cached scores
   for all directions are refreshed when an arm's pooled mean changes.
2. The toy's repeated comparison loop instantiated RTS under its STR label.
   This experiment explicitly instantiates the separate STR class.

The paper's deterministic arm-direction initialization is used instead of drawing
random directions during initialization. After initialization, directions are
drawn uniformly using an RNG independent of the simulator. The legacy `rho`
constructor argument remains available but is unused, as in the toy: length
scalarization has no augmented-Chebyshev term.

## Hypervolume evaluation

After each online update, the recommended subset is the union of one empirical
exploitation maximizer per observed direction. Unobserved arm-direction pairs
are excluded; ties use the lowest arm index. Exploration bonuses are not part
of this recommendation.

Section 7.5 evaluates the set of all horizon-length tuples over that subset.
Enumerating it is exponential. This pilot approximates it using 16 sequences
sampled uniformly through scrambled Sobol draws plus each recommended arm's
constant sequence, with duplicates removed. Every sequence is evaluated on two
held-out scenario realizations of the complete, stateful simulator. Identical
Sobol draws and held-out seeds are used across subsets, algorithms, and gravity
levels. Repeated subsets reuse cached evaluations. None of these costs are
fed back to UCB or used to recommend arms.

The empirical total costs use the **same normalized RTS/STR polar hypervolume
functions** as MORBO and qNParEGO: 128 angular midpoint quadrature directions and
the fixed total-cost reference above. The toy's mean-vector hypervolume monitor
is not used for STR, since it would collapse the distinction between RTS and STR.

`Paper plots/Model-free hypervolume results.ipynb` displays separate RTS and STR
figures, with two gravity curves each and all 100 checkpoints. The dotted line
marks the end of initialization. Hypervolume estimates can decrease as the
currently recommended subset changes; curves are not replaced by cumulative
maxima. These are single-run results with a small Monte Carlo budget, without
confidence intervals. Although horizons and references match the model-based
experiments, one online bandit iteration and one Bayesian optimization step have
different evaluation costs.

The simulator retains history-dependent dynamics. Applying stationary UCB to
these increments follows the paper's stated stationarity assumption; this pilot
does not establish that assumption empirically.


## Complete experiment set

The folder now covers gravity, synchronization, and request-rate heterogeneity
for both holistic and Cheetah reductionist formulations. Each row has the
corresponding `.json`, `.py`, and `.pkl` files:

| Approach | Sweep | Setup | Script | Result |
|---|---|---|---|---|
| Holistic | Gravity | `setup_holistic_gravity.json` | `script_holistic_gravity.py` | `result_holistic_gravity.pkl` |
| Holistic | Synchronization | `setup_holistic_synchronization.json` | `script_holistic_synchronization.py` | `result_holistic_synchronization.pkl` |
| Holistic | Request heterogeneity | `setup_holistic_heterogeneity.json` | `script_holistic_heterogeneity.py` | `result_holistic_heterogeneity.pkl` |
| Reductionist | Gravity | `setup_reductionist_gravity.json` | `script_reductionist_gravity.py` | `result_reductionist_gravity.pkl` |
| Reductionist | Synchronization | `setup_reductionist_synchronization.json` | `script_reductionist_synchronization.py` | `result_reductionist_synchronization.pkl` |
| Reductionist | Request heterogeneity | `setup_reductionist_heterogeneity.json` | `script_reductionist_heterogeneity.py` | `result_reductionist_heterogeneity.pkl` |

The five additional scripts call `experiment_runner.py`, which uses the same
RTS and STR classes and held-out evaluation method as the original holistic
pilot. `online_environment.py` supports both approaches. The original holistic
gravity source and result remain unchanged.

All sweeps match the current MORBO and qNParEGO setups:

| Sweep | Compared levels | Fixed other parameters |
|---|---|---|
| Gravity variance | 1, 5 | synchronization 1; request heterogeneity 0.5 |
| Synchronization | 1, 0.8 | gravity variance 0.1; request heterogeneity 0.5 |
| Request heterogeneity | 0.1, 0.8 | gravity variance 0.1; synchronization 1 |

### Online training versus evaluation horizons

Every condition uses one independent run, **10 arms, 5 directions, and 100 online
requests**. The first 50 requests initialize every arm-direction pair; the next
50 use UCB. For holistic runs these are global requests across classes. For
reductionist runs, these are Cheetah requests only, and the bandit pulls physical
Cheetah fitness margins from 10 evenly spaced values in **[-10, 20]**.

The sequence horizon for held-out hypervolume evaluation remains **100 for
holistic** and **50 for reductionist**, matching MORBO/qNParEGO. It is configured
in `holistic.horizon` or `reductionist.horizon`. The five new setups separately
configure the online budget in `bandit.n_online_iterations`. Keeping the online
budget at 50 would leave no adaptive choices with 10 arms and 5 directions.

Both reference points remain fixed:

- Holistic: (2,200,000 KiB, 310).
- Cheetah reductionist: (90,497.578125 KiB, 85).

The reductionist overhead reference is 1.01 times 40 octopi times 50 evaluation
requests times the Cheetah float32 payload. It must not be recomputed using the
100-request online training budget. Online reward normalization separately uses
the maximum *single-request* Cheetah overhead (40 times its payload) and an
instability bound of 1.

The reductionist online simulator reproduces the existing offline evaluator's
class filtering and state transitions, including asynchronous communication and
request heterogeneity. Tests check exact total-cost agreement at both 50- and
100-request horizons for all three sweeps, ensure that only Cheetah observations
reach UCB, and check physical-margin mapping and sweep parameter routing.

The notebook `Paper plots/Model-free hypervolume results.ipynb` contains **12
figures**: RTS and STR for each of the six experiment pairs, with all 100 online
checkpoints. These results retain the current constant-sequence augmentation
and small held-out evaluation budget. Matching references and horizons alone
does not establish a controlled performance ranking against Bayesian
optimization; the comparison-protocol changes discussed earlier are deferred.

Run any individual script with the project Python interpreter. For example:

```powershell
.\.venv\Scripts\python.exe "Scalarized UCB/script_reductionist_gravity.py"
.\.venv\Scripts\python.exe "Scalarized UCB/script_holistic_synchronization.py"
.\.venv\Scripts\python.exe "Scalarized UCB/test_experiment_runner.py"
```
