# Cosmic Octopi

Implementation and experiments for **Selective Pressure Allocation: A Paradigm for Communication Efficiency in Federated Deep Reinforcement Learning**, by Mohamed Bellouch.

Selective Pressure Allocation (SPA) regulates uplink transmissions through a fitness margin selector, balancing uplink overhead and learning-performance instability.

The holistic and reductionist formulations target **Robustify-then-Scalarize (RTS)** and **Scalarize-then-Robustify (STR)** weak Pareto sets through two solution approaches:

- **Model-based:** MORBO and qNParEGO.
- **Model-free:** Scalarized UCB (Sc-UCB) and Scalarized Knowledge Gradient (Sc-KG).

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
| [Compression coexistence](Compression%20coexistence/) | Quantization and SPA coexistence experiments |
| [Paper plots](Paper%20plots/) | Experiment notebooks and figures |

## License

[MIT License](LICENSE). Copyright (c) 2026 Mohamed Bellouch.
