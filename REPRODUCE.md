# Reproducibility Guide for SAERL Experiments

This guide provides instructions to reproduce the experiments and results reported in the paper **"Safe Adaptive Ensemble Reinforcement Learning (SAERL) for Physics-Consistent Fast Charging of Lithium-Ion Battery Packs"**.

## 1. Environment Setup

The code requires Python 3.8+ and the dependencies listed in `requirements.txt`.

```bash
# Install dependencies
pip install -r requirements.txt
```

## 2. Dataset Availability

The experiments use three public battery datasets. Due to size, they are not included in the repo but must be downloaded to `data/`:

1.  **NASA Randomized Battery Usage Dataset**: [Download Link](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) -> Place in `data/nasa/`
2.  **CALCE Battery Dataset**: [Download Link](https://web.calce.umd.edu/batteries/data.htm) -> Place in `data/calce/`
3.  **MATR Battery Dataset**: [Download Link](https://data.matr.io/1/) -> Place in `data/matr/`

## 3. Scenarios

We evaluate on 15 specific scenarios defined in `scenarios.json`. These capture a range of:
- **Chemistries/Conditions**: NASA (NMC), CALCE (LCO), MATR (LFP/NMC mix).
- **Temperatures**: 0°C, 10°C, 25°C, 40°C.
- **Initial States**: Fresh vs Aged, Low SoC vs High SoC.

## 4. Running Benchmarks

### A. Run Baselines (CCCV & MPC)

The baselines are run using `scripts/run_baseline_benchmarks.py`. This script uses the `scenarios.json` configuration implicitly through the flags.

```bash
# Run MPC and CCCV on all default scenarios
python scripts/run_baseline_benchmarks.py --controllers cccv,mpc --metrics charge_time_min,final_soc,safety_event_count,reached_80_soc
```

**Note on MPC Implementation**:
The MPC controller uses a "perfect model" (a deepcopy of the simulation environment) but uses a coarse finite-horizon shooting method (7 action points, 8 steps). This design choice highlights the trade-off between model accuracy and optimization complexity.

### B. Run SAERL (Proposed Method)

To train and evaluate the SAERL agent:

```bash
# Train the residual policy
python scripts/train_saerl_policy.py --config configs/saerl_default.yaml

# Evaluate on the test scenarios
python scripts/eval_saerl_vs_baselines.py --load-model models/saerl_best.pt
```

## 5. Metric Definitions

-   **Time to 80%**: Time in minutes to reach 80% Pack SoC. If 80% is not reached, this metric is censored (returns NaN or max duration).
-   **Reached 80% SoC**: Binary flag (1.0 or 0.0) indicating if the target was met.
-   **Safety Events**: Count of time steps where voltage, temperature, or current constraints were violated.
    -   *Note*: The shield uses a discrete linear search and is **not** differentiable. It acts as a runtime filter.

## 6. Troubleshooting

-   **"Optimizer failed"**: The underlying scipy optimizer in the parameter identification module might fail on very noisy data. Ensure datasets are unzipped correctly.
-   **Memory Errors**: The MATR dataset is large. Use the `--max-files` flag in data loading scripts if you have limited RAM.
