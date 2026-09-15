# Pareto UCB

`Pareto_UCB.py` contains `EmpiricalParetoUCB` and its required helpers
(`dominates`, `get_pareto_indices`, and `hypervolume_2d`), extracted unchanged
from `Scalarized UCB/bandits.ipynb`. It depends only on NumPy. Importing the
module does not execute the notebook or launch an experiment.

The algorithm initializes by playing every arm once, adds the original UCB
confidence radius to each empirical reward vector, and samples uniformly
from the nondominated UCB arms.

The extracted version maximizes rewards and uses the notebook's synchronous
`run` interface. The future cost-based environment must pair the overhead
increment at iteration t with the instability increment at iteration t+1
and attribute that pair to the arm chosen at t. Cost-to-reward conversion
and delayed feedback will be integrated in `Model-free baseline` during
the next implementation steps. No experiment has been run here.
