# Quantized model uploads

`octopus quantized.py` and `oracle quantized.py` are separate copies of the
original classes. `utils quantized.py` supplies the shared upload codec; other
training and scheduling utilities are imported from `Cosmic Octopi/utils.py`.

The octopus quantizes each float32 policy/value tensor using symmetric int8
quantization just before uploading it. Its local training model stays float32.
The response's model field is a **bytes payload**, containing architecture,
tensor names/shapes, per-tensor scales, and int8 values. The oracle reconstructs
float32 weights from those bytes before the existing episode-weighted federated
average, then broadcasts the averaged floating model as before.

The overhead increment is `sum(len(payload) for payload in uploads) / 1024`, in
**KiB**, consistent with the original `get_model_size_in_kb` units. It includes
the entire quantization header and scales. As in the original model-size metric,
fitness/episode messages, network protocol framing, and downlink broadcasts are
outside this overhead measure.

Two additional oracle histories make the accounting inspectable:

- `uploaded_payload_sizes_history[class_id]`: individual upload byte counts per round.
- `unquantized_overhead_increments_history[class_id]`: float32 model sizes in KiB
  for exactly the same contributors, before compression.

A payload is rejected if its full size is not smaller than its float32 model.
No-upload rounds have zero overhead for both representations. Upload errors are
propagated to the caller, so failed uploads cannot appear as compression savings.

## Loading the copies

Because the requested filenames contain spaces, load them explicitly. Set
`project_root` to the project directory when running from another folder:

```python
from pathlib import Path
import importlib.util

project_root = Path.cwd()
folder = project_root / "Compression coexistence"

def load_quantized(name):
    spec = importlib.util.spec_from_file_location(
        f"compression_{name}", folder / f"{name} quantized.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

Octopus = load_quantized("octopus").Octopus
Oracle = load_quantized("oracle").Oracle
```

Use these two classes together with the same constructor/step arguments as the
original classes. Existing experiment runners continue to use their original
implementations until explicitly wired to these copies.

With simulated rewards, the existing Xi/OU reward model is unchanged: it does
not predict a reward penalty caused by quantization. With real training, the
dequantized aggregate is actually installed in each candidate's policy. Byte
savings below compare matching uploads; a full true-reward run may also change
which agents contribute as the learned policies evolve.

## Verification

Run from the project root in CMD:

```cmd
.venv\Scripts\python.exe -B "Compression coexistence\test_quantized_uploads.py"
```

The checks cover round-trip error, zero tensors, unchanged local weights and
decoder RNG state, malformed uploads, backward propagation, real octopus/oracle
uploads for all four v5 environments, weighted aggregation, broadcasts, exact
byte accounting, no-upload rounds, and propagation of failed uploads. They use
short simulated rewards, run no PPO training, and write no experiment results.

Observed mean sizes per upload in the two-agent smoke test (metadata included):

| Class | Float32 (KiB) | Int8 payload (KiB) | Reduction |
| --- | ---: | ---: | ---: |
| Cheetah | 44.801 | 11.833 | 73.59% |
| Ant | 89.816 | 23.089 | 74.29% |
| Leg/Hopper | 40.277 | 10.701 | 73.43% |
| Humanoid | 1247.137 | 312.439 | 74.95% |

Payload sizes can vary slightly with the serialized per-tensor scales.

## Calibrating Xi for quantized uploads

`calibration quantized.py` reads `setup calibration quantized.json`. The setup
selects all four v5 classes, **2 agents and 10 wet experiments per class**. It
contains the environment settings, gravity distribution, PPO parameters, int8
codec, OU fitting bounds, regularization, seeds and output filenames.

This calibration uses real PPO returns (`simulated_rewards: false`). Generating
the observations from the old Xi would not measure quantization's effect.
No original Xi matrix or offset file is used to manufacture training rewards.

For each wet experiment, agents start from a common freshly initialized model,
receive independently sampled gravities, and train for randomized warmup lengths.
All three nonempty subsets of the two agents are considered. Their model uploads
are quantized and decoded using `utils quantized.py`, and averaged with exactly
the same episode weights as the quantized oracle. Each subset starts an
independent continuation on one uniformly selected contributing agent's gravity.
Its fresh optimizer and rollout state match the oracle's model replacement
behavior. Separate training/evaluation environments keep reward evaluation from
invalidating the training runner's cached observation.

Each continuation produces an initial measured return and **1,500 calibration
reward checkpoints**, retaining the previous calibration length. One checkpoint
means 20 PPO minibatch updates followed by the mean of 5 environment episode
returns; it is not one environment step or one rollout. Thus there are 30
post-aggregation traces per class, in addition to 20 local warmup traces.

The OU fit estimates mean reversion, long-term reward and volatility for each
trace. The existing standardized ridge regression then fits a 4x4 matrix with
rows `[global fitness, contributor count, gravity, 1]` and columns
`[initial reward, log reversion, log long-term reward, log volatility]`. Global
fitness is the mean of the contributors' mean warmup returns at zero freshness.
The initial reward column uses least squares, as in the earlier calibration.
Fit convergence, parameters at bounds, valid rows and design rank are recorded;
the script requires enough valid independent rows before exporting a matrix.
Calibration quality must be assessed after the actual run, especially because
the OU form constrains long-term reward to be positive while measured returns
can be negative.

When launched, the script writes only its own calibration outputs:

- `calibration quantized result.pkl`: raw warmup/continuation traces, features,
  setup snapshot, upload sizes, fitting diagnostics and fitted matrices.
- `Xi_matrix calibrated quantized.json`: the class-name-to-4x4-matrix mapping for
  this setting. The existing `Xi_matrix quantized.json` copy of the original
  coefficients is preserved.

It checkpoints after each complete wet experiment. From the project root in CMD:

```cmd
.venv\Scripts\python.exe -B "Compression coexistence\calibration quantized.py"
```

Append `--resume` to continue from the saved wet experiments, or `--fit-only` to
refit complete saved traces without repeating training. OU fitting settings and
regularization may be adjusted when refitting; changing the wet experiment
configuration requires a separate dataset. A progress bar tracks wet experiments
and calibration checkpoints. An interrupted, partially completed wet experiment
is repeated when resuming.

For validation without creating environments, training or writing results:

```cmd
.venv\Scripts\python.exe -B "Compression coexistence\calibration quantized.py" --check
```

Only this read-only configuration check was run when these files were prepared.

## Quantized model-free coexistence comparisons

The two `setup_model_free_*_quantized.json` files configure the holistic and
Cheetah reductionist comparisons. They load **`Xi_matrix quantized.json`** at
launch, independently of the unrun calibration script. The exact loaded matrices
and their file hash are included in the saved run setup.

Both comparisons retain the standard Model-free baseline settings:

- Scalarized UCB and Scalarized KG under RTS and STR; Pareto UCB learns one arm
  set that is evaluated under both RTS and STR.
- 7 arms, 7 scalarization directions, 150 online learning iterations and one
  independent learning run. Pareto UCB alone retains the delayed
  `(overhead_t, instability_{t+1})` feedback definition.
- Synchronous requests, gravity variance 1 and request heterogeneity 1.
- Final-set evaluation with 20 random sequence draws and 10 shared held-out
  realizations per sequence. These are distinct from independent learning runs.
- Evaluation horizons of 100 for holistic and 50 for reductionist.
- The saved standard reference costs: `[2200000, 310]` for holistic and
  `[90497.578125, 85]` for reductionist (overhead in KiB).

`model_free_quantized.py` measures each complete serialized int8 payload using
the actual codec, including metadata. The lightweight simulated-reward
environment has no evolving neural weights, so it uses a **fixed representative
payload size per class**, measured from a seeded Network. Contributor counts
are multiplied by those sizes in both online learning and final evaluation.
Representative uploads are approximately 73–75% smaller than float32. This is
a simulated communication-size model, not repeated PPO training or measurement
of a different weight-dependent header on every upload.

The `*_runner_quantized.py` and `evaluate_*_comparison_quantized.py` files adapt
the standard comparison orchestration and call the existing UCB, KG, Pareto UCB
and simulated-reward implementations. Each algorithm also has an individual
entry point, such as `holistic Scalarized UCB quantized.py`.

Run **both approaches sequentially** from the project root in CMD:

```cmd
.venv\Scripts\python.exe -B "Compression coexistence\run_model_free_quantized.py"
```

Or run them separately:

```cmd
.venv\Scripts\python.exe -B "Compression coexistence\run_holistic_comparison_quantized.py"
.venv\Scripts\python.exe -B "Compression coexistence\run_reductionist_comparison_quantized.py"
```

The commands show progress bars and default to four workers for held-out
evaluation. Use `--workers 2` to reduce concurrency or `--validate-only` to
check settings and payload sizes without simulations. Repeating a command skips
matching completed training files and resumes a partial held-out evaluation;
incomplete online training itself restarts from its seed.

Outputs stay in this folder, with `quantized.pkl` filenames. Each approach
produces five learned arm-set files and one comparison file:

- `holistic hypervolume comparison quantized.pkl`
- `reductionist hypervolume comparison quantized.pkl`

Each comparison stores sequence-level cost realizations, six RTS/STR hypervolume
rows, payload measurements, and `standard_comparison` rows with the saved
standard value, quantized value, difference and percentage change. It reads the
standard results without modifying or rerunning them. Their settings and source
file hashes are checked to keep the comparison traceable.

At fixed sequences and unchanged instability, reducing overhead cannot decrease
these hypervolumes with a common reference. New learning runs and a different Xi
can change the selected sequences and instability, so the script reports the
observed change without enforcing a higher hypervolume.

Validation uses `test_model_free_quantized.py`: actual codec byte measurements,
mocked training/evaluation dispatch, feedback alignment, reference consistency
and spawned-worker input checks. It produces no experiment results.

## Quantized model-based coexistence comparisons

`setup_model_based_holistic_quantized.json` and
`setup_model_based_reductionist_quantized.json` prepare the remaining comparison
for MORBO, qNParEGO and the Gaussian/NSGA-II threshold baseline. These scripts
use simulated rewards and **`Xi_matrix quantized.json`**, with the same complete
representative int8 payload sizes as the quantized model-free comparisons.
The calibration script is not invoked.

The standard model-based budgets are retained: one independent optimization per
algorithm/formulation, 8 initial points, 20 optimization steps, batch size 4,
and 10 realizations per candidate. Hypervolume checkpoints are recorded every
two optimization steps. The holistic horizon is 100; the reductionist horizon
is 50 for Cheetah. Reference costs, scenario settings and hyperparameters remain
matched to the saved standard results.

The baseline first evaluates 10 constant margins, fits Gaussian kernel regressions
with bandwidth 40 and searches the interpolated objectives with NSGA-II. For the
final comparison, it draws 20 sequences, independently sampling one threshold
from its learned Pareto set at each iteration. This retains the latest baseline
definition used in `Baselines and Coexistence`, rather than evaluating only
constant sequences. The baseline is scored under both RTS and STR.

Run both approaches sequentially from the project root in CMD:

```cmd
.venv\Scripts\python.exe -B "Compression coexistence\run_model_based_quantized.py"
```

Use `--approach holistic` or `--approach reductionist` to run just one approach.
`--workers 4` controls held-out evaluation concurrency (the default is four).
`--validate-only` checks both setups, dependencies and codec sizes without running
optimization or reward simulations. Each approach runs these stages:

1. The single-fitness-margin search.
2. MORBO RTS and STR optimization.
3. qNParEGO RTS and STR optimization.
4. Common held-out evaluation of all solution sets and the 20 baseline sequences.

Progress bars report stages, baseline grid/NSGA-II progress, optimizer candidates
and steps, and held-out evaluations. Rerunning the command skips matching completed
searches and resumes optimizer/evaluation checkpoints. An interrupted baseline
search restarts; it saves its Pareto set after completing its search.

Each approach writes five search result files and a separate comparison cache:

- `holistic model-based hypervolume comparison quantized.pkl`
- `reductionist model-based hypervolume comparison quantized.pkl`

The comparison caches contain all cost realizations, the sampled baseline
sequences, six RTS/STR hypervolume rows, payload measurements and a
`standard_comparison` table showing the saved standard hypervolume, measured
quantized hypervolume, difference and percentage change. The ten actual saved
standard held-out seeds are reused exactly. Standard inputs remain read-only,
and none of these filenames overlaps the quantized model-free results.

The helper `model_based_quantized.py` handles input snapshots and comparison
checks. Separate `*_model_based_runner_quantized.py` adapters call the existing
MORBO/qNParEGO implementations; the existing Gaussian/NSGA-II search is retained
in the quantized baseline scripts. Source and input hashes prevent silent reuse
of results after settings, Xi or dependencies change.

Validation consists of setup/dependency checks and
`test_model_based_quantized.py`, with stand-ins for simulations, GP optimization
and result writes. Tests cover optimizer and baseline payload routing, random
threshold sequence construction, RTS/STR comparison rows, standard references
and Windows worker initialization. No model-based experiment results were
generated while preparing these files.
