<div align="center">

# ?? Cosmic Octopi

### Smarter model sharing. Less communication. More room for cooperation.

**Selective Pressure Allocation (SPA)** for communication-efficient federated learning.

[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-PPO%20%26%20surrogates-EE4C2C?logo=pytorch&logoColor=white)
![Experiments](https://img.shields.io/badge/Experiments-Reproducible-8250df)

[Explore the results](#-explore-the-results) ? [Get started](#-get-started) ? [Run experiments](#-run-experiments) ? [Repository map](#-repository-map)

</div>

---

## ?? The idea

Not every locally trained model needs to be uploaded.

Cosmic Octopi studies **when agents should communicate** with a central aggregator. SPA controls fitness margins?the thresholds that determine whether a local policy contributes?to balance **communication overhead** and **learning instability**. The project explores this trade-off across heterogeneous requests, synchronization, gravity and agent populations.

Two complementary formulations, **RTS** and **STR**, are studied under both a **holistic** system-wide approach and a **reductionist** class-specific approach. The repository brings together Bayesian optimization, multiobjective bandits, calibration, baselines, saved measurements and paper-ready figures.

### What is inside?

| Research component | Implementations |
| --- | --- |
| Model-based SPA | MORBO and qNParEGO, each with RTS and STR |
| Model-free SPA | Scalarized UCB and Scalarized Knowledge Gradient, each with RTS and STR |
| Model-based baseline | Gaussian kernel regression + NSGA-II; thresholds sampled from the learned Pareto set |
| Model-free baseline | Pareto UCB with delayed instability feedback |
| Compression coexistence | SPA and baselines combined with int8 model uploads |
| Environment classes | Cheetah, Ant, Leg/Hopper and Humanoid; Gymnasium MuJoCo v5 |

## ?? Explore the results

Start with **[Baselines and Coexistence.ipynb](Paper%20plots/Baselines%20and%20Coexistence.ipynb)**. Its final section compares standard and quantized communication for all four cases below.

**Solid bars:** standard hypervolume. **Hatched additions:** the observed quantization gain. **Total height:** quantized hypervolume. Each figure groups algorithms under RTS and STR.

<table>
<tr>
<td align="center"><b>Model-based ? Holistic</b><br><img src="Paper%20plots/Final%20results%20model-based%20holistic.png" width="460" alt="Holistic model-based hypervolumes, standard case plus quantization gains"></td>
<td align="center"><b>Model-based ? Reductionist</b><br><img src="Paper%20plots/Final%20results%20model-based%20reductionist.png" width="460" alt="Cheetah reductionist model-based hypervolumes, standard case plus quantization gains"></td>
</tr>
<tr>
<td align="center"><b>Model-free ? Holistic</b><br><img src="Paper%20plots/Final%20results%20model-free%20holistic.png" width="460" alt="Holistic model-free hypervolumes, standard case plus quantization gains"></td>
<td align="center"><b>Model-free ? Reductionist</b><br><img src="Paper%20plots/Final%20results%20model-free%20reductionist.png" width="460" alt="Cheetah reductionist model-free hypervolumes, standard case plus quantization gains"></td>
</tr>
</table>

The saved coexistence comparisons show positive gains for every displayed algorithm/formulation. Standard and quantized values use the **same reference costs and held-out seeds within each case**. Reference costs differ between holistic and reductionist cases, so bar heights should be compared within a case.

| Notebook | What you will find |
| --- | --- |
| [Preliminaries and ablations](Paper%20plots/Preliminaries%20and%20ablations.ipynb) | Parameter studies, Pareto fronts and model calibration |
| [Model-based hypervolume results](Paper%20plots/Model-based%20hypervolume%20results.ipynb) | MORBO and qNParEGO optimization histories |
| [Single run model-free hypervolume](Paper%20plots/Single%20run%20model-free%20hypervolume.ipynb) | Individual UCB/KG experiment trajectories |
| [Multiple run model-free hypervolume](Paper%20plots/Multiple%20run%20model-free%20hypervolume.ipynb) | Mean curves and variability across independent runs |
| [Baselines and Coexistence](Paper%20plots/Baselines%20and%20Coexistence.ipynb) | Baselines, SPA and quantization comparisons |

PNG and SVG figure exports are available alongside the notebooks. Saved outputs let you inspect the results before launching expensive experiments.

## ?? Get started

The project was developed with **Python 3.14**. Exact dependency versions are recorded in [`requirements.txt`](requirements.txt).

The large model-free result file uses [Git LFS](https://git-lfs.com/). Install Git LFS before cloning to download the actual data rather than a pointer file.

```bash
git lfs install
git clone https://github.com/MohamedBELLOUCH/Cosmic-Octopi.git
cd Cosmic-Octopi
python -m venv .venv
```

**Windows CMD**

```cmd
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Linux / macOS**

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Open a notebook in VS Code and select the `.venv` Python environment as its kernel. Alternatively, use your preferred Jupyter installation with this environment registered as a kernel:

```bash
python -m ipykernel install --user --name cosmic-octopi --display-name "Cosmic Octopi"
```

Run that last command with the virtual environment activated. For CUDA-enabled Windows installations, use [`requirements-cuda.txt`](requirements-cuda.txt) **instead of** `requirements.txt`; it selects the CUDA 13.0 PyTorch build. The simulated-reward and optimizer workflows do not require GPU-based policy training.

## ?? Run experiments

The usual experiment layout is deliberately simple:

```text
Experiment/
??? setup.json     # Parameters and seeds
??? script.py      # Reproducible entry point
??? result.pkl     # Saved measurements
```

Multi-algorithm comparisons use descriptive setup, script and result names. Consult each folder's README for its exact output and resume behavior.

### Quantization coexistence

From the project root in **Windows CMD**:

```cmd
REM Model-free: Scalarized UCB, Scalarized KG and Pareto UCB
.venv\Scripts\python.exe -B "Compression coexistence\run_model_free_quantized.py"

REM Model-based: MORBO, qNParEGO and the random-threshold baseline
.venv\Scripts\python.exe -B "Compression coexistence\run_model_based_quantized.py"
```

Each command runs holistic and Cheetah reductionist comparisons sequentially, shows progress bars and saves separate quantized results. Add `--approach holistic` or `--approach reductionist` to select one case; `--workers 2` reduces evaluation concurrency. Add **`--validate-only`** to check configuration without launching simulations.

Matching completed results are reused; optimizer and evaluation checkpoints can be resumed. Changing source files, calibration or settings can invalidate a saved checkpoint. See the [compression workflow](Compression%20coexistence/README.md) for details.

### Independent-run suites

```cmd
.venv\Scripts\python.exe -B "Model-free experiments\script.py"
.venv\Scripts\python.exe -B "Model-based experiments\script.py"
```

**Saved batch status at publication:** the model-free suite is complete (120/120 experiment records). The model-based suite is a resumable partial checkpoint (62/120 records, `complete=false`). The final baseline/coexistence comparisons shown above are complete.

These suites configure **10 independent runs** across gravity, synchronization and request-rate heterogeneity for both approaches. Their combined outputs are `model_free_results.pkl` and `model_based_results.pkl`. See the [model-free suite](Model-free%20experiments/README.md) and [model-based suite](Model-based%20experiments/README.md) for checkpoint conventions and budgets.

## ?? Reading the experiments correctly

- **Reward model.** The main experiment workflows use simulated rewards. A class-specific 4?4 Xi matrix maps `[global policy fitness, contributor count, gravity, 1]` to initial reward and the logarithms of three Ornstein?Uhlenbeck parameters. This supports controlled simulation studies; it is not, by itself, an empirical validation of MuJoCo training performance.
- **Compression model.** The quantized Octopus/Oracle copies serialize int8 tensors before upload and decode them before aggregation. Lightweight simulations use fixed per-class sizes measured from complete representative payloads, including metadata?approximately **73?75% smaller** than float32 for these architectures.
- **Calibration.** [`Model calibration`](Model%20calibration) contains the calibration workflow and coefficients. A separate true-reward quantized calibration script is prepared in [`Compression coexistence`](Compression%20coexistence); the coexistence experiments use the supplied `Xi_matrix quantized.json`. Running the plots does not run calibration.
- **Hypervolume.** Overhead and instability are minimized. RTS comparisons use hypervolume of mean costs; STR uses the expected-length formulation. Check each setup for the fixed reference point and direction grid.
- **Replications.** Independent algorithm runs and held-out realizations are different. The coexistence comparison uses one independent optimization/learning run per variant, with multiple held-out realizations; the batch suites configure ten independent runs.
- **Units.** Model overhead is recorded in KiB. Plot-specific rescaling is documented in the corresponding notebook.

## ?? Repository map

| Folder | Purpose |
| --- | --- |
| [`Cosmic Octopi`](Cosmic%20Octopi) | Agents, oracle, PPO components, reward simulation and Pareto utilities |
| [`Synchronization and request rates`](Synchronization%20and%20request%20rates), [`Fitness margins`](Fitness%20margins), [`Temperature`](Temperature) | Scheduling and threshold ablations |
| [`Planet gravity`](Planet%20gravity), [`Octopi count`](Octopi%20count) | Environmental and population studies |
| [`Pareto fronts`](Pareto%20fronts) | Sobol space-filling experiments |
| [`Model calibration`](Model%20calibration) | Xi fitting and calibration diagnostics |
| [`MORBO`](MORBO), [`qNParEGO`](qNParEGO) | Model-based algorithms and individual experiments |
| [`Scalarized UCB`](Scalarized%20UCB), [`Scalarized Knowledge Gradient`](Scalarized%20Knowledge%20Gradient), [`Pareto UCB`](Pareto%20UCB) | Bandit algorithms and experiment runners |
| [`Model-based baseline`](Model-based%20baseline), [`Model-free baseline`](Model-free%20baseline) | Standard-case baseline comparisons |
| [`Model-based experiments`](Model-based%20experiments), [`Model-free experiments`](Model-free%20experiments) | Independent-run batch suites |
| [`Compression coexistence`](Compression%20coexistence) | Int8 transport, quantized experiments and paired comparisons |
| [`Paper plots`](Paper%20plots) | Notebooks and publication figures |

## ?? Research & contributions

Created by **Mohamed Bellouch**. Questions, reproducibility reports and contributions are welcome through [GitHub issues](https://github.com/MohamedBELLOUCH/Cosmic-Octopi/issues) and pull requests. Include the setup, random seed and relevant software versions when reporting an experiment issue.

The project uses PyTorch, Gymnasium/MuJoCo, BoTorch, GPyTorch, PyMLE and pymoo. Algorithm-specific folders retain implementation notes and references. Third-party dependencies retain their respective licenses.

## ?? License

This project's original code is released under the **[MIT License](LICENSE)**.

<div align="center">

*Explore the trade-off. Share selectively. Compress cooperatively.* ??

</div>
