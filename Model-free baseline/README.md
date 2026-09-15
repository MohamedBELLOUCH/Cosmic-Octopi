# Holistic model-free baseline comparison

All settings are embedded in `setup_model_free_holistic.json`: the standard
gamma_sync = gamma_grav = gamma_heter = 1 environment, Xi matrices, payload
shapes, 7 arms, 7 scalarization directions, 150 online iterations, and the
100-iteration held-out horizon. Total-cost reference: (2,200,000 KiB, 310).

Run from the project root, in order:

```cmd
".venv\Scripts\python.exe" -B "Model-free baseline\holistic Pareto UCB.py"
".venv\Scripts\python.exe" -B "Model-free baseline\holistic Scalarized UCB.py"
".venv\Scripts\python.exe" -B "Model-free baseline\holistic Scalarized KG.py"
".venv\Scripts\python.exe" -B "Model-free baseline\evaluate_holistic_comparison.py" --workers 4
```

The three training scripts produce these learned recommendation sets:

- `holistic Pareto UCB Pareto set.pkl`
- `holistic RTS Scalarized UCB Pareto set.pkl`
- `holistic STR Scalarized UCB Pareto set.pkl`
- `holistic RTS Scalarized KG Pareto set.pkl`
- `holistic STR Scalarized KG Pareto set.pkl`

Scalarized UCB and KG reuse their existing training implementations and
current-iteration cost definitions. Their final arm sets are the union of
empirical exploitation recommendations over the seven directions. These are
learned recommendations, not certificates of true Pareto optimality.

`pareto_online.py` adapts the unchanged `Pareto UCB/Pareto_UCB.py` class to
costs: reward = negative normalized cost, with the original mean-plus-UCB
confidence and uniform selection among nondominated UCB vectors. Its learned
set contains the nondominated empirical mean-cost arms. A local random
generator separates bandit choices from simulator randomness.

Only Pareto UCB observes (overhead_t, instability_t+1), credited to arm_t
after the next **global** iteration, even if its class differs. Before then,
the pending observation cannot influence selection. The final action remains
pending: 150 decisions and 149 complete observations. Initialization selects
each arm, then waits for the pending arm through one further choice before
using fully observed confidence vectors. No extra environment step is added.
This feedback difference is intentional and selected by the user; the
comparison is not an isolation of algorithm choice under identical feedback.

The evaluator samples 20 common independent uniform sequence draws from each
final arm set, removes duplicates, and evaluates each unique sequence on the
same 10 fresh simulator seeds. Identical sequences across sets are evaluated
once. All complete-rollout costs use the existing simulator. This evaluates
the final recommended sets; it does not evaluate adaptive training trajectories
or plot online checkpoints.

`holistic hypervolume comparison.pkl` stores all draws, sequences, costs,
source hashes, and six summary rows. RTS uses exact normalized 2D hypervolume
of per-sequence mean costs. STR uses the existing 128-direction expected-length
hypervolume. Pareto UCB has one trained set scored under both definitions.

Open `Paper plots/Model-free baseline comparison.ipynb` for the six-bar plot,
table, and delayed-feedback verification. Executing it only reads saved costs
and exports PNG/SVG figures; it never trains or simulates.

Training skips matching completed output files. Evaluation checkpoints after
every ten completed simulations, resumes matching partial caches, and skips
completed ones. Mismatched setup or source hashes raise an error instead of
silently replacing results. Use `--validate-only` on a training entrypoint to
check the setup without running it. Scalarized entrypoints also accept
`--formulation RTS` or `--formulation STR`.

Focused timing and minimization checks:

```cmd
".venv\Scripts\python.exe" -B "Model-free baseline\test_holistic_runner.py"
```

Existing model-based results, previous UCB/KG experiment results, and the
original extracted Pareto UCB file are not modified by this comparison.

## Cheetah reductionist comparison

`setup_model_free_reductionist.json` embeds the corresponding setup: all three
gamma parameters equal 1, 7 arms, 7 scalarization directions, 150 Cheetah
training requests, and a 50-request held-out horizon. Reference costs are
(90,497.578125 KiB, 85). The normalized arm grid is mapped to physical Cheetah
margins [-10, 20] before both online training and held-out simulation.

Only Pareto UCB uses next-request instability feedback. Here "next" refers
to the next modeled Cheetah request. The final pending action is excluded
from feedback updates: 150 decisions and 149 complete observations. UCB/KG
preserve their existing reductionist SPA definitions. All final sets use
20 common random sequence draws and 10 common held-out seeds, with duplicates
evaluated only once.

Run all three algorithms and their final comparison with one CMD command:

```cmd
".venv\Scripts\python.exe" -B "Model-free baseline\run_reductionist_comparison.py" --workers 4
```

This command shows progress and invokes the reductionist training/evaluation
code only. Matching complete files are skipped; interrupted held-out evaluation
resumes from its checkpoints. It writes these files in this folder:

- `reductionist Pareto UCB Pareto set.pkl`
- `reductionist RTS Scalarized UCB Pareto set.pkl`
- `reductionist STR Scalarized UCB Pareto set.pkl`
- `reductionist RTS Scalarized KG Pareto set.pkl`
- `reductionist STR Scalarized KG Pareto set.pkl`
- `reductionist hypervolume comparison.pkl`

The individual entrypoints are `reductionist Pareto UCB.py`,
`reductionist Scalarized UCB.py`, `reductionist Scalarized KG.py`, and
`evaluate_reductionist_comparison.py`. Use `--validate-only` on the combined
command to validate configuration without creating result files.

After completion, run the reductionist section in
`Paper plots/Model-free baseline comparison.ipynb`. It displays the six bars
and table, and exports `Reductionist model-free baseline comparison.png` and
`.svg`. Until results exist, this section reports a pending status. The command
does not execute or modify the notebook automatically.

The reductionist experiment was prepared without running it. Its checks use
test doubles for the simulator and file writer; they verify physical margins,
Cheetah-specific overhead normalization, delayed feedback, and delegation to
the original scalarized implementations:

```cmd
".venv\Scripts\python.exe" -B "Model-free baseline\test_reductionist_runner.py"
```
