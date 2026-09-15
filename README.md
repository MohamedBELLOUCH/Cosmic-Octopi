<div align="center">

# 🐙 Cosmic Octopi

### Selective Pressure Allocation:
**A Paradigm for Communication Efficiency in Federated Deep Reinforcement Learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-PPO-EE4C2C?logo=pytorch&logoColor=white)
![MuJoCo](https://img.shields.io/badge/Environments-MuJoCo-8250df)

[Paper abstract and contributions](#paper-abstract-and-contributions) · [Cosmic Octopi](#cosmic-octopi-case-study) · [Installation](#installation) · [Run experiments](#run-experiments)

</div>

---

## Paper abstract and contributions

Federated Deep Reinforcement Learning (FDRL) enables agents to collaboratively learn shared global policies without exchanging sensitive or costly raw trajectory data. However, it can still incur significant uplink overhead due to (1) a large number of agents, (2) large exchanged updates, i.e., model parameters or gradient vectors, (3) limited aggregator-agent communication capacity, or (4) frequent transmission of updates. In response, existing work mainly relies on update compression and agent selection across aggregation rounds. In this work, we introduce a complementary paradigm termed Selective Pressure Allocation. To highlight its adaptivity, we formalize the paradigm within a framework in which (1) agents interact with multiple heterogeneous environment classes, (2) the number of agents fluctuates, and (3) aggregator-agent communication is not necessarily synchronized. In a nutshell, the paradigm relies on two quantities: fitness and freshness. Fitness measures the relevance of updates, while freshness measures how up-to-date local updates are. At each global iteration, a local update is uploaded only if its fitness exceeds that of the corresponding global update by a selected fitness margin. To trade off communication overhead against learning performance, we formulate the selection of optimal fitness margins as a stochastic multi-objective Bayesian optimization problem. We then propose several reformulations to address the resulting challenges, together with two classes of solution approaches: model-based and model-free. We conduct several experiments on a hand-crafted MuJoCo case study that matches the formalized framework. The results demonstrate the effectiveness of the proposed paradigm and highlight the relative strengths of the reformulations and solution approaches across different scenarios.


### Contributions

- A complementary paradigm termed **Selective Pressure Allocation** is introduced. Fitness margins are dynamically adjusted across global iterations to account for the framework's stochastic and time-varying nature. Regulation is restricted to the agents' uploading phase, enabling coexistence with other communication-efficiency techniques.
- Two metrics are defined to quantify **overhead and learning-performance instability**, and two problem formulations are provided: **reductionist and holistic**. Each class is considered in the reductionist formulation, while all classes are considered in the holistic formulation.
- Two solution approaches are proposed: **model-based and model-free**. The formulations are cast as multi-objective Bayesian optimization problems in the model-based approach, and as multi-objective multi-armed bandit (MOMAB) problems in the model-free approach.
- Several experiments are conducted on a **hand-crafted MuJoCo case study** that matches the formalized framework. The effectiveness of the proposed paradigm is demonstrated, and the relative strengths of the reformulations and solution approaches across different scenarios are highlighted.

<a name="cosmic-octopi-case-study"></a>

## Cosmic Octopi

Consider $M > 0$ octopi, each living on a distinct planet and manipulates four marionettes: a **Humanoid**, an **Ant**, a **Leg**, and a **Cheetah**. The octopus alternates between learning and break periods, which are exponentially distributed with means $\gamma_{\text{learn}} > 0$ and $\gamma_{\text{break}} > 0$, respectively. During the learning periods, through trial and error, it learns to make the cheetah run, the ant and humanoid walk, and the leg jump.

### Planet gravity
The surface gravities of the planets (in m/s$^2$) are i.i.d. and sampled from the Gamma distribution

$$
\text{Gamma}\left(\frac{g^2}{\gamma_{\text{grav}}}, \frac{\gamma_{\text{grav}}}{g}\right),
$$

where $g = 9.81$ m/s$^2$ is the gravity of Earth, and $\gamma_{\text{grav}} > 0$ is a constant. The distribution is centered at $g$ and its variance is $\gamma_{\text{grav}}$. 

### Request rates
To cooperate, the octopi exchange learning "knowledge" with a Galactic Oracle. Specifically, the Galactic Oracle regularly broadcasts requests to the octopi to gather their local "knowledge", construct a global one, and return it to them. These requests follow a marked Poisson point process (PPP). The request instants form a standard PPP with intensity $\gamma_{\text{req}} > 0$, and each request carries a mark indicating the corresponding marionette type.

At a given request instant, the probability that the request is associated with the marionette $`\mathcal{M} \in \mathscr{M} \triangleq \left\lbrace\text{"Humanoid", "Ant", "Cheetah", "Leg"}\right\rbrace`$ is

$$
\frac{\exp\left(\frac{w_\mathcal{M}}{\gamma_{\text{heter}}}\right)}{\sum\limits_{\mathcal{M}' \in \mathscr{M}} \exp\left(\frac{w_{\mathcal{M}'}}{\gamma_{\text{heter}}}\right)}.
$$

Here, $\gamma_{\text{heter}} \geq 0$ is a parameter, and for any $\mathcal{M}' \in \mathscr{M}$, $w_{\mathcal{M}'} > 0$ is a constant chosen arbitrarily subject to the ordering constraint $w_{\text{"Leg"}} \leq w_{\text{"Cheetah"}} \leq w_{\text{"Ant"}} \leq w_{\text{"Humanoid"}}$.

### Synchronization
Upon receiving a request, an octopus responds with probability $\gamma_{\text{sync}} \in (0,1]$. Sending its local "knowledge" to the Galactic Oracle requires opening a wormhole, which consumes immense energy.

### Objective

The objective is to apply Selective Pressure Allocation to minimize wormhole openings without hindering the octopi's exchange of "knowledge" for cooperation.

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

The mapping follows the subsections of **Section 7: Experiments: Cosmic Octopi**. Run the linked Python files from the repository root, using their corresponding setup JSON files. For example:

```bat
.venv\Scripts\python.exe -B "Temperature\script.py"
```

**7.1 Cosmic Octopi** and **7.2 Implementation details** describe the shared simulator implemented in [octopus.py](Cosmic%20Octopi/octopus.py) and [oracle.py](Cosmic%20Octopi/oracle.py); they have no standalone experiment runners.

### 7.3 Preliminaries and ablations

| Paper subsection | Experiment runner |
|---|---|
| 7.3.1 Synchronization and request heterogeneity | [Synchronization and request rates/script.py](Synchronization%20and%20request%20rates/script.py) |
| 7.3.2 Planet gravity and number of octopi | [Planet gravity/script.py](Planet%20gravity/script.py) and [Octopi count/script.py](Octopi%20count/script.py) |
| 7.3.3 Fitness margins | [Fitness margins/script.py](Fitness%20margins/script.py) |
| 7.3.4 Temperature | [Temperature/script.py](Temperature/script.py) |
| 7.3.5 Robust Pareto fronts | [Pareto fronts/script.py](Pareto%20fronts/script.py) |
| 7.3.6 Model calibration | [Model calibration/script.py](Model%20calibration/script.py) |

### 7.4 Model-based approach

Each algorithm link opens the corresponding runner for both RTS and STR.

| Paper subsection | Holistic runners | Reductionist runners |
|---|---|---|
| 7.4.1 Gravity | [MORBO](MORBO/script_holistic_gravity.py), [qNParEGO](qNParEGO/script_holistic_gravity.py) | [MORBO](MORBO/script_reductionist_gravity.py), [qNParEGO](qNParEGO/script_reductionist_gravity.py) |
| 7.4.2 Synchronization | [MORBO](MORBO/script_holistic_synchronization.py), [qNParEGO](qNParEGO/script_holistic_synchronization.py) | [MORBO](MORBO/script_reductionist_synchronization.py), [qNParEGO](qNParEGO/script_reductionist_synchronization.py) |
| 7.4.3 Request rates | [MORBO](MORBO/script_holistic_heterogeneity.py), [qNParEGO](qNParEGO/script_holistic_heterogeneity.py) | [MORBO](MORBO/script_reductionist_heterogeneity.py), [qNParEGO](qNParEGO/script_reductionist_heterogeneity.py) |

### 7.5 Model-free approach

Each algorithm link opens the corresponding runner for both RTS and STR.

| Paper subsection | Holistic runners | Reductionist runners |
|---|---|---|
| 7.5.1 Gravity | [Scalarized UCB](Scalarized%20UCB/script_holistic_gravity.py), [Scalarized Knowledge Gradient](Scalarized%20Knowledge%20Gradient/script_holistic_gravity.py) | [Scalarized UCB](Scalarized%20UCB/script_reductionist_gravity.py), [Scalarized Knowledge Gradient](Scalarized%20Knowledge%20Gradient/script_reductionist_gravity.py) |
| 7.5.2 Synchronization | [Scalarized UCB](Scalarized%20UCB/script_holistic_synchronization.py), [Scalarized Knowledge Gradient](Scalarized%20Knowledge%20Gradient/script_holistic_synchronization.py) | [Scalarized UCB](Scalarized%20UCB/script_reductionist_synchronization.py), [Scalarized Knowledge Gradient](Scalarized%20Knowledge%20Gradient/script_reductionist_synchronization.py) |
| 7.5.3 Request rates | [Scalarized UCB](Scalarized%20UCB/script_holistic_heterogeneity.py), [Scalarized Knowledge Gradient](Scalarized%20Knowledge%20Gradient/script_holistic_heterogeneity.py) | [Scalarized UCB](Scalarized%20UCB/script_reductionist_heterogeneity.py), [Scalarized Knowledge Gradient](Scalarized%20Knowledge%20Gradient/script_reductionist_heterogeneity.py) |

### 7.6 Summary of hypervolume results

This subsection summarizes the experiments in Sections 7.4 and 7.5 and requires no separate simulation. The corresponding results can be inspected in [Model-based hypervolume results](Paper%20plots/Model-based%20hypervolume%20results.ipynb) and [Multiple run model-free hypervolume](Paper%20plots/Multiple%20run%20model-free%20hypervolume.ipynb).

### Batch runners

The [model-based suite](Model-based%20experiments/script.py) runs all Section 7.4 experiments, and the [model-free suite](Model-free%20experiments/script.py) runs all Section 7.5 experiments. Each uses its self-contained `setup.json`, configured for 10 independent runs.

```bat
.venv\Scripts\python.exe -B "Model-based experiments\script.py"
.venv\Scripts\python.exe -B "Model-free experiments\script.py"
```

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
