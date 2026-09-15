<div align="center">

# 🐙 Cosmic Octopi

### Selective Pressure Allocation:
**A Paradigm for Communication Efficiency in Federated Deep Reinforcement Learning**

Implementation and experiments accompanying the paper by **Mohamed Bellouch**.

[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-PPO-EE4C2C?logo=pytorch&logoColor=white)
![MuJoCo](https://img.shields.io/badge/Environments-MuJoCo-8250df)

[Abstract](#abstract) · [Contributions](#contributions) · [Installation](#installation) · [Run experiments](#run-experiments)

</div>

---

## Abstract

Federated Deep Reinforcement Learning (FDRL) enables agents to collaboratively learn shared global policies without exchanging sensitive or costly raw trajectory data. However, it can still incur significant uplink overhead due to (1) a large number of agents, (2) large exchanged updates, i.e., model parameters or gradient vectors, (3) limited aggregator-agent communication capacity, or (4) frequent transmission of updates. In response, existing work mainly relies on update compression and agent selection across aggregation rounds. In this work, we introduce a complementary paradigm termed Selective Pressure Allocation. To highlight its adaptivity, we formalize the paradigm within a framework in which (1) agents interact with multiple heterogeneous environment classes, (2) the number of agents fluctuates, and (3) aggregator-agent communication is not necessarily synchronized. In a nutshell, the paradigm relies on two quantities: fitness and freshness. Fitness measures the relevance of updates, while freshness measures how up-to-date local updates are. At each global iteration, a local update is uploaded only if its fitness exceeds that of the corresponding global update by a selected fitness margin. To trade off communication overhead against learning performance, we formulate the selection of optimal fitness margins as a stochastic multi-objective Bayesian optimization problem. We then propose several reformulations to address the resulting challenges, together with two classes of solution approaches: model-based and model-free. We conduct several experiments on a hand-crafted MuJoCo case study that matches the formalized framework. The results demonstrate the effectiveness of the proposed paradigm and highlight the relative strengths of the reformulations and solution approaches across different scenarios. The source code is publicly available in our [GitHub repo](https://github.com/MohamedBELLOUCH/Cosmic-Octopi).


## Contributions

- We introduce a complementary paradigm termed **Selective Pressure Allocation**. Fitness margins are dynamically adjusted across global iterations to account for the framework's stochastic and time-varying nature. The paradigm acts only during the agents' uploading phase, enabling its coexistence with other communication-efficiency techniques.
- We define two metrics to quantify **overhead and learning-performance instability**, and provide two problem formulations: **reductionist and holistic**. The reductionist formulation considers each class, while the holistic formulation considers all classes.
- We propose two solution approaches: **model-based and model-free**. The model-based approach casts the formulations as multi-objective Bayesian optimization problems; the model-free approach casts them as multi-objective multi-armed bandit (MOMAB) problems.
- We conduct several experiments on a **hand-crafted MuJoCo case study** that matches the formalized framework, demonstrating the effectiveness of the proposed paradigm and highlighting the relative strengths of the reformulations and solution approaches across different scenarios.

## Installation

Install Git LFS, then run from Windows CMD:

```bat
git lfs install
git clone https://github.com/MohamedBELLOUCH/Cosmic-Octopi.git
cd Cosmic-Octopi
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the `.venv` interpreter for the notebooks. CUDA dependencies are provided in [requirements-cuda.txt](requirements-cuda.txt).

## Run experiments

From the repository root:

```bat
.venv\Scripts\python.exe -B "Model-based experiments\script.py"
.venv\Scripts\python.exe -B "Model-free experiments\script.py"
```

Each suite uses its `setup.json`. Individual experiments provide setup files, scripts, and saved results.

## Repository structure

| Folder | Contents |
|---|---|
| [Cosmic Octopi](Cosmic%20Octopi/) | Cosmic Octopi case study and simulator |
| [Model calibration](Model%20calibration/) | Ornstein-Uhlenbeck model calibration |
| [MORBO](MORBO/) / [qNParEGO](qNParEGO/) | Model-based solution approaches |
| [Scalarized UCB](Scalarized%20UCB/) / [Scalarized Knowledge Gradient](Scalarized%20Knowledge%20Gradient/) | Model-free solution approaches |
| [Model-based baseline](Model-based%20baseline/) / [Model-free baseline](Model-free%20baseline/) | Baseline comparisons |
| [Compression coexistence](Compression%20coexistence/) | Quantization and Selective Pressure Allocation coexistence experiments |
| [Paper plots](Paper%20plots/) | Experiment notebooks and figures |

## License

[MIT License](LICENSE). Copyright (c) 2026 Mohamed Bellouch.
