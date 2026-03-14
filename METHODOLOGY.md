# SAERL Methodology — Detailed Reference

> **Safe Adaptive Ensemble Reinforcement Learning for Physics-Consistent Fast Charging of Lithium-Ion Battery Packs**

This document provides an exhaustive, structured derivation of the SAERL methodology, tracing every algorithmic component from the paper back to its concrete implementation.  It is designed to serve as a self-contained reference for reading, reviewing, reproducing, or extending the work.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Battery & Pack Physics Model](#2-battery--pack-physics-model)
   - 2.1 [Single-Cell Equivalent Circuit Model (ECM)](#21-single-cell-equivalent-circuit-model-ecm)
   - 2.2 [Lumped Thermal Model](#22-lumped-thermal-model)
   - 2.3 [Semi-Empirical Aging / Degradation Model](#23-semi-empirical-aging--degradation-model)
   - 2.4 [Pack-Level Aggregation](#24-pack-level-aggregation)
   - 2.5 [Cell-to-Cell Heterogeneity](#25-cell-to-cell-heterogeneity)
   - 2.6 [Thermal Interconnection](#26-thermal-interconnection)
   - 2.7 [Balancing Logic](#27-balancing-logic)
3. [Constrained MDP Formulation](#3-constrained-mdp-formulation)
   - 3.1 [State Space](#31-state-space)
   - 3.2 [Action Space](#32-action-space)
   - 3.3 [Reward Function](#33-reward-function)
   - 3.4 [Safety Constraints & Termination](#34-safety-constraints--termination)
4. [SAERL Architecture Overview](#4-saerl-architecture-overview)
5. [Predictive Ensemble](#5-predictive-ensemble)
   - 5.1 [GRU Quantile Predictor](#51-gru-quantile-predictor)
   - 5.2 [MLP Quantile Predictor](#52-mlp-quantile-predictor)
   - 5.3 [Random Forest Predictor](#53-random-forest-predictor)
   - 5.4 [Quantile Pinball Loss](#54-quantile-pinball-loss)
6. [Adaptive Gating Network](#6-adaptive-gating-network)
   - 6.1 [Gate Architecture](#61-gate-architecture)
   - 6.2 [Weight Computation](#62-weight-computation)
   - 6.3 [Prediction Fusion & Risk Scoring](#63-prediction-fusion--risk-scoring)
   - 6.4 [Rolling Error Statistics](#64-rolling-error-statistics)
7. [Residual Policy](#7-residual-policy)
   - 7.1 [MPC Baseline](#71-mpc-baseline)
   - 7.2 [Residual Actor Network](#72-residual-actor-network)
   - 7.3 [Action Composition](#73-action-composition)
8. [Safety Shield](#8-safety-shield)
   - 8.1 [Candidate Generation & Scoring](#81-candidate-generation--scoring)
   - 8.2 [Hard-Constraint Check (One-Step Simulation)](#82-hard-constraint-check-one-step-simulation)
   - 8.3 [Least-Stress Fallback](#83-least-stress-fallback)
   - 8.4 [Anti-Stall Mechanism](#84-anti-stall-mechanism)
9. [Training Procedure](#9-training-procedure)
   - 9.1 [Phase 1 — Dataset Generation & Offline Warm-Start](#91-phase-1--dataset-generation--offline-warm-start)
   - 9.2 [Phase 2 — PPO Online Fine-Tuning](#92-phase-2--ppo-online-fine-tuning)
10. [Data Pipeline & Parameter Identification](#10-data-pipeline--parameter-identification)
    - 10.1 [Dataset Loaders](#101-dataset-loaders)
    - 10.2 [Pack Parameter Identification](#102-pack-parameter-identification)
11. [Evaluation Protocol](#11-evaluation-protocol)
    - 11.1 [Scenarios](#111-scenarios)
    - 11.2 [Baselines](#112-baselines)
    - 11.3 [Acceptance Criteria](#113-acceptance-criteria)
12. [Hyperparameter Summary](#12-hyperparameter-summary)
13. [File Map — Paper ↔ Code](#13-file-map--paper--code)

---

## 1. Problem Statement

Pack-level fast charging is formulated as a **constrained Markov decision process (CMDP)** tuple $(\mathcal{S}, \mathcal{A}, P, r, c, \gamma)$.  The objective is to bring a series-connected battery pack of $N$ cells from an initial SoC (10–30 %) to a target SoC (80 %) as quickly as possible while satisfying **per-cell** hard constraints at every time step $t$ and for every cell $i \in \{1, \ldots, N\}$:

$$V_i(t) \in [V_{\min}, V_{\max}], \qquad T_i(t) \in [T_{\min}, T_{\max}]$$

Safety monitoring uses **worst-cell** quantities ($\max_i V_i$, $\max_i T_i$) so that no single cell's violation can be masked by pack-level averaging.

---

## 2. Battery & Pack Physics Model

### 2.1 Single-Cell Equivalent Circuit Model (ECM)

**Paper reference:** Section 4 (Battery and Pack Model), Eq. governing equations  
**Code:** `physics_model.py` → `CellModel`

Each cell is modeled with a **2nd-order Thevenin equivalent circuit** containing:

- Ohmic resistance $R_0$
- Two RC polarization pairs $(R_1, C_1)$ and $(R_2, C_2)$

**State of Charge (Coulomb counting):**

$$\frac{d(\text{SoC})}{dt} = -\frac{\eta \cdot I}{Q_{\text{eff}} \cdot 3600}$$

Implementation (`CellModel.step`, line 110):
```python
soc_dot = -self.params.eta * total_current / (self.Q_effective * 3600)
self.soc = clip(self.soc + soc_dot * dt, 0, 1)
```

**RC dynamics (two polarization voltages):**

$$\frac{dv_k}{dt} = -\frac{v_k}{R_k C_k} + \frac{I}{C_k}, \quad k \in \{1, 2\}$$

**Terminal voltage:**

$$V_{\text{terminal}} = V_{\text{OCV}}(\text{SoC}) - I \cdot R_0^{\text{eff}} - v_1 - v_2$$

where $R_0^{\text{eff}} = R_0 + R_0^{\text{growth}}$ accounts for aged resistance.

**OCV lookup:** linear interpolation over 11 SoC–OCV breakpoints (3.0 V at 0 % to 4.2 V at 100 %).

| Parameter     | Symbol     | Default   |
|---------------|------------|-----------|
| Nominal capacity | $Q_{\text{nom}}$ | 2.5 Ah |
| Ohmic resistance | $R_0$     | 20 mΩ    |
| RC pair 1     | $R_1 / C_1$ | 10 mΩ / 2000 F |
| RC pair 2     | $R_2 / C_2$ | 20 mΩ / 10000 F |
| Coulombic efficiency | $\eta$ | 0.995   |

### 2.2 Lumped Thermal Model

$$\frac{dT}{dt} = \frac{Q_{\text{gen}} - Q_{\text{diss}}}{C_{\text{th}}}$$

where:
- $Q_{\text{gen}} = I^2 (R_0 + R_0^{\text{growth}})$ — resistive heat generation
- $Q_{\text{diss}} = hA \cdot (T - T_{\text{ambient}})$ — convective heat dissipation

| Parameter | Default |
|-----------|---------|
| $C_{\text{th}}$ | 75.0 J/°C |
| $hA$ | 0.5 W/°C |

### 2.3 Semi-Empirical Aging / Degradation Model

**Capacity fade rate** follows a stress-factor approach:

$$\frac{dQ_{\text{loss}}}{dt} = k_1 \cdot S_I \cdot S_T \cdot S_{\text{SoC}}$$

where the three stress factors are:

| Factor | Formula | Parameter |
|--------|---------|-----------|
| Current stress | $S_I = |I| / Q_{\text{nom}}$ | — (C-rate proxy) |
| Temperature stress | $S_T = \exp\left(k_2 (T - T_{\text{ref}})\right)$ | $k_2 = 0.05$, $T_{\text{ref}} = 25\,°\text{C}$ |
| SoC stress | $S_{\text{SoC}} = \exp(k_3 \cdot \text{SoC})$ | $k_3 = 0.1$ |

**Capacity fade & resistance growth coupling:**

$$\text{fade} = \min\!\left(\frac{Q_{\text{loss}}}{0.2 \cdot Q_{\text{nom}}}, 0.8\right), \qquad Q_{\text{eff}} = Q_{\text{nom}} (1 - \text{fade}), \qquad R_0^{\text{growth}} = 2 R_0 \cdot \text{fade}$$

**Code:** `CellModel._update_degradation` (line 136–149).

### 2.4 Pack-Level Aggregation

**Code:** `battery_pack_model.py` → `BatteryPack`

The pack connects $N_s$ cells in series (default 20) with $N_p$ parallel strings (default 1).

- **Pack voltage:** $V_{\text{pack}} = \sum_{i=1}^{N_s} V_i$ (mean of parallel strings)
- **Cell current:** $I_{\text{cell}} = I_{\text{pack}} / N_p$
- **Pack SoC:** capacity-weighted average: $\text{SoC}_{\text{pack}} = \sum_i \text{SoC}_i \cdot Q_{\text{eff},i} \big/ \sum_i Q_{\text{eff},i}$
- **Pack temperature:** $\max_i T_i$ (worst-cell)

Derived metrics tracked at each step:
- `voltage_imbalance` = $\max V_i - \min V_i$
- `soc_imbalance` = $\max \text{SoC}_i - \min \text{SoC}_i$
- `temperature_imbalance` = $\max T_i - \min T_i$

### 2.5 Cell-to-Cell Heterogeneity

**Code:** `BatteryPack._generate_cells` (line 105–143), `CellVariance` dataclass

Manufacturing spread is modeled by sampling cell parameters from Gaussian distributions:

| Parameter | Mean | Std (CoV) |
|-----------|------|-----------|
| $Q_{\text{nom}}$ | 2.5 Ah | 5 % |
| $R_0$ | 20 mΩ | 20 % |
| $C_{\text{th}}$ | 75.0 J/°C | 10 % |

A **negative correlation** ($\rho_{QR} = -0.3$) couples capacity and resistance: cells with lower capacity tend to have higher resistance.  
Initial SoC per cell is offset by $\mathcal{U}(-0.05, +0.05)$ around the pack-wide initial SoC.

### 2.6 Thermal Interconnection

A nearest-neighbor **thermal conductance matrix** models heat transfer between adjacent cells:

```
matrix[i, i±1] = 0.3 / (1 + cell_spacing × 100)
```

After each step, temperatures are updated by: $\mathbf{T}_{\text{new}} = \mathbf{M} \cdot \mathbf{T}_{\text{old}}$ where $\mathbf{M}$ is the row-normalized thermal matrix.

**Code:** `BatteryPack._create_thermal_interaction_matrix` and `_apply_thermal_interactions`.

### 2.7 Balancing Logic

Two strategies are implemented:

1. **Passive:** drain-bleed cells whose SoC exceeds $\overline{\text{SoC}} + \sigma/2$ at rate `balancing_current` (default 0.1 A).
2. **Active:** bidirectional transfer proportional to SoC deviation from pack mean.

Balancing activates only when `soc_imbalance > balancing_threshold` (default 0.01).

---

## 3. Constrained MDP Formulation

**Code:** `hambrl_pack_env.py` → `HAMBRLPackEnvironment`

### 3.1 State Space

The environment observation vector $s_t \in \mathbb{R}^6$ is constructed as:

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | Pack SoC | raw $[0, 1]$ |
| 1 | Pack voltage | $V_{\text{pack}} / 100$ |
| 2 | Pack temperature | $T / 100$ |
| 3 | Voltage imbalance | $\Delta V \times 1000$ |
| 4 | Pack current | $I_{\text{pack}} / 10$ |
| 5 | Time fraction | $t / t_{\max}$ |

The SAERL controller internally extends this to a **6-dimensional feature vector per step** (in `state_to_feature_vector`) with the following normalization:

| Feature | Formula |
|---------|---------|
| SoC | raw |
| Voltage | $V_{\text{pack}} / 100$ |
| Temperature | $T / 60$ |
| Imbalance | $\text{clip}(\Delta V \times 1000 / 200, -3, 3)$ |
| Current (normalized) | $I_{\text{charge}} / I_{\max}$ |
| Current² | $(\text{current norm})^2$ |

A **sliding window** of the last $L = 20$ steps of this feature vector forms the input to ensemble predictors.

### 3.2 Action Space

$a_t \in [-1, 1]$ is a normalized control signal.  Mapping to physical current:

$$I_{\text{pack}} = -0.5 \cdot I_{\max} \cdot (a + 1)$$

where $I_{\max}$ is the maximum charge current in amps (positive discharge convention).  Hence:
- $a = -1 \Rightarrow I_{\text{pack}} = 0$ (no charging)
- $a = +1 \Rightarrow I_{\text{pack}} = -I_{\max}$ (full charge rate)

### 3.3 Reward Function

$$r_t = 10 \cdot \text{SoC}_t - 0.01 - 0.1 \cdot \max(0, T_t - 40)^2 - 0.05 \cdot \Delta V \times 1000 - 1.0 \cdot n_{\text{safety}} + 50 \cdot \mathbf{1}[\text{SoC} \geq \text{target}]$$

| Component | Weight | Purpose |
|-----------|--------|---------|
| SoC progress | +10.0 | Encourage charge accumulation |
| Time penalty | −0.01 | Penalize every step |
| Over-temperature | −0.1 | Quadratic penalty above 40 °C |
| Voltage imbalance | −0.05 | Penalize cell imbalance |
| Safety events | −1.0 per event | Strong penalty for any violation |
| Completion bonus | +50.0 | Reward reaching target SoC |

### 3.4 Safety Constraints & Termination

An episode terminates if **any** of these conditions is met:

1. $t \geq t_{\max}$ (max steps reached)
2. $\text{SoC}_{\text{pack}} \geq \text{target\_soc}$ (goal achieved)
3. **Critical safety violations:**
   - Pack over-voltage ($V_{\text{pack}} > V_{\text{pack\_max}}$)
   - Pack under-voltage ($V_{\text{pack}} < V_{\text{pack\_min}}$)
   - More than 5 individual cells exceeding $T_{\max}$

---

## 4. SAERL Architecture Overview

The full decision pipeline at each time step is:

```
                          ┌──────────────┐
    State Window (L=20)──►│  GRU Quantile │──► ŷ_gru, u_gru
                          │  MLP Quantile │──► ŷ_mlp, u_mlp
                          │  Random Forest│──► ŷ_rf,  u_rf
                          └──────┬───────┘
                                 │  per-expert prediction + uncertainty
                                 ▼
                          ┌──────────────┐
    errors, uncertainties─►│ Adaptive Gate │──► weights w_gru, w_mlp, w_rf
                          └──────┬───────┘
                                 │
                                 ▼
                       ŷ_fused = Σ w_m · ŷ_m   (+ risk score)
                                 │
                                 ▼
                       ┌─────────────────┐
    seq + mpc_action───►│ Residual Policy  │──► δ_action
                       └────────┬────────┘
                                │   a_raw = clip(a_mpc + δ, −1, 1)
                                ▼
                       ┌─────────────────┐
                       │  Safety Shield   │──► a_safe
                       │  (candidate scan │
                       │   + 1-step sim)  │
                       └─────────────────┘
```

---

## 5. Predictive Ensemble

**Code:** `controllers/adaptive_ensemble_rl.py` → `AdaptiveEnsemblePredictor`

All three experts predict **4 target dimensions** at each step:

| Index | Target | Symbol |
|-------|--------|--------|
| 0 | Next pack SoC | `next_soc` |
| 1 | Next min cell voltage | `next_voltage` |
| 2 | Next pack temperature | `next_temp` |
| 3 | Next voltage imbalance | `next_imbalance` |

### 5.1 GRU Quantile Predictor

**Code:** `QuantileGRUModel` (line 91–121)

- **Architecture:** 2-layer GRU with 64 hidden units (code) / 128 (paper), batch-first
- **Input:** sliding window `[batch, L=20, 6]` + scalar action
- **Output head:** `Linear(hidden+1, 128) → ReLU → Linear(128, 3×4 = 12)` producing **three quantiles** (q10, q50, q90) for each of the 4 targets
- **Uncertainty estimate:** $u_{\text{GRU}} = (q_{90} - q_{10}) \times \text{calibration factor}$

### 5.2 MLP Quantile Predictor

**Code:** `QuantileMLPModel` (line 124–140)

- **Architecture:** `Linear(121, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 12)`
  - Input dimension = $L \times 6 + 1 = 121$ (flattened window + action)
- **Output:** same 3-quantile × 4-target structure as GRU
- **Uncertainty:** same quantile-width approach

### 5.3 Random Forest Predictor

**Code:** `AdaptiveEnsemblePredictor.__init__` (line 307–313)

- **Scikit-learn** `RandomForestRegressor`: 200 trees, max depth 16, min samples leaf 5
- **Input:** flattened window + action (same as MLP)
- **Uncertainty:** variance across tree predictions (subsampled to $k = 32$ trees for speed)

```python
# Pseudocode from _rf_tree_stats
tree_preds = [tree.predict(x) for tree in estimators]
mean, std = mean(tree_preds), std(tree_preds)
# Synthetic quantiles: q10 = mean − 1.64·std, q90 = mean + 1.64·std  (Gaussian)
```

### 5.4 Quantile Pinball Loss

Used to train both GRU and MLP:

$$\mathcal{L}_\tau(y, \hat{y}) = \begin{cases} \tau (y - \hat{y}) & \text{if } y \geq \hat{y} \\ (1 - \tau)(\hat{y} - y) & \text{if } y < \hat{y} \end{cases}$$

Total loss = $\mathcal{L}_{0.1} + \mathcal{L}_{0.5} + \mathcal{L}_{0.9}$

**Code:** `quantile_pinball_loss` (line 74–88).

---

## 6. Adaptive Gating Network

### 6.1 Gate Architecture

**Code:** `GateNetwork` (line 143–155)

```
Linear(13, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 3) → Softmax
```

Input dimension = $6 + 1 + 3 + 3 = 13$:
- 6 features from latest state vector
- 1 action value
- 3 uncertainty values (one per expert)
- 3 rolling absolute error values (one per expert)

### 6.2 Weight Computation

Weights are refreshed every `gate_update_interval` = 50 steps (configurable). At each refresh:

1. Forward pass through gate network → raw logits
2. Apply softmax → raw weights $\tilde{w}_m$
3. Zero out disabled experts, re-normalize: $w_m = \tilde{w}_m \cdot \text{mask}_m / \sum_j \tilde{w}_j \cdot \text{mask}_j$
4. Cache weights for reuse between refreshes

**Code:** `AdaptiveEnsemblePredictor._compute_weights` (line 426–448).

### 6.3 Prediction Fusion & Risk Scoring

**Fused prediction:**

$$\hat{y}_t = \sum_{m \in \{\text{GRU, MLP, RF}\}} w_m \hat{y}_{m,t}, \qquad \hat{u}_t = \sum_m w_m u_{m,t}$$

**Risk score** (used by shield and candidate selection):

$$\text{risk} = 8 \cdot (\Delta T^+)^2 + 8 \cdot (\Delta V^+)^2 + 0.02 \cdot |\text{imb} \times 1000| + 0.25 \cdot \bar{u} + 8 \cdot \max(0, \text{SoC}_{\text{prev}} - \hat{\text{SoC}}_{\text{next}})$$

where $\Delta T^+ = \max(0, \hat{T} - T_{\text{soft\_limit}})$ and $\Delta V^+ = \max(0, \hat{V} - (V_{\text{CV}} + V_{\text{margin}}))$.

**Code:** `predict_fused` and `predict_fused_batch` (line 505–678).

### 6.4 Rolling Error Statistics

After each environment step, the predictor updates per-expert rolling MAE with exponential momentum $\alpha = 0.90$:

$$e_{m,t}^{\text{new}} = 0.90 \cdot e_{m,t-1} + 0.10 \cdot |y_t^{\text{true}} - \hat{y}_{m,t}|$$

This feeds back into the gate input for the next step.

**Code:** `update_error_statistics` (line 681–700).

---

## 7. Residual Policy

### 7.1 MPC Baseline

The MPC baseline provides a conservative action $a_{\text{mpc}}$ at each step.  In the full SAERL system, MPC is external; the environment provides it.  The controller also generates a **CCCV anchor** candidate:

- **CC phase:** $I_{\text{target}} = I_{\max}$ when $V_{\text{pack}} < V_{\text{CV}} - 0.05$
- **CV taper:** linear derating as voltage approaches CV threshold

**Code:** `SafeAdaptiveEnsembleController._cccv_anchor_action` (line 1090–1105).

### 7.2 Residual Actor Network

**Code:** `ResidualActorNetwork` (line 158–173) and `ResidualActorPolicy` (line 176–247)

```
Linear(input_dim, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 1)
```

- **Output:** Gaussian policy $\mathcal{N}(\mu(s), \sigma^2)$ where $\sigma = \exp(\log\_\text{std})$ is a state-independent learned parameter
- **Delta clipping:** $\delta = \tanh(\mu) \times \Delta_{\max}$ where $\Delta_{\max} = 0.40$
- **Input construction:** flattened sliding window + MPC action + SoC gap = $L \times 6 + 2$ features

### 7.3 Action Composition

$$a_{\text{raw}} = \text{clip}(a_{\text{mpc}} + \delta, -1, 1)$$

The controller then generates **candidate actions** by varying $\delta$ around the proposed value:

```python
delta_candidates = linspace(δ − 0.35, δ + 0.35, 9)  # ±candidate_radius
```

Plus additional anchors:
- $\delta = 0$ (pure MPC)
- CCCV anchor
- Full charge ($a = 1.0$)

Each candidate is scored using the ensemble's fused prediction and the scoring function below.

---

## 8. Safety Shield

**Code:** `SafeAdaptiveEnsembleController` (line 767–1256)

### 8.1 Candidate Generation & Scoring

For each candidate action, the scoring function computes:

$$\text{score}(a) = -120 \cdot \Delta\text{SoC} + \frac{1}{\max(\text{C-rate}, 0.05)} + 30 (\Delta T^+)^2 + 15 \cdot D_{\text{proxy}} + 0.005 |\Delta V_{\text{imb}}| + 300 \cdot \mathbf{1}_{\text{unsafe}} + \text{risk}$$

| Term | Weight | Purpose |
|------|--------|---------|
| SoC gain | −120 | Prefer more charge delivered |
| Time proxy | 1.0 | Penalize low current |
| Temperature excess² | 30 | Penalize over-temperature |
| Degradation proxy | 15 | Penalize high aging stress |
| Imbalance | 0.005 | Mildly penalize cell imbalance |
| Safety violation | 300 | Heavy penalty for predicted violation |
| Risk score | 1.0 | Uncertainty-aware penalty |

**Degradation proxy:**

$$D = (\text{C-rate})^2 \cdot \exp\!\left(\frac{\max(0, T-25)}{18}\right) \cdot \exp(0.8 \cdot \text{SoC})$$

The candidate with the **lowest score** is selected.

### 8.2 Hard-Constraint Check (One-Step Simulation)

**Code:** `_is_safe_action` (line 907–944) and `_simulate_next_state` (line 946–1003)

For a proposed action, the shield:

1. **Saves the full pack state** (cell SoCs, voltages, temperatures, RC states, aging vars, history lengths)
2. **Steps the real physics model** one time step with the candidate current
3. **Checks hard constraints:** per-cell over/under-voltage, over-temperature, pack-level limits, current limit, imbalance margin
4. **Restores pack state** to its pre-simulation values

This avoids `deepcopy` overhead while still using the full-fidelity physics model.

**Checked limits:**

| Constraint | Limit |
|------------|-------|
| Pack voltage | $\min(V_{\text{pack\_max}}, V_{\text{CV}} + 0.03)$ |
| Pack temperature | $\min(T_{\text{pack\_max}}, 42.0 + 0.5)$ |
| Pack current | $I_{\text{pack\_max}}$ |
| Voltage imbalance | 95 mV |
| Per-cell over/under voltage | via `CellModel` limits |
| Per-cell over temperature | via `CellModel` limits |

### 8.3 Least-Stress Fallback

If neither the selected action nor the MPC fallback satisfies constraints:

```python
for candidate in linspace(-1.0, max(mpc_action, -1.0), 21):
    if is_safe(candidate):
        return candidate
return -1.0  # zero current as ultimate fallback
```

This searches from **low stress** (zero current) toward the MPC action, returning the first safe candidate.

### 8.4 Anti-Stall Mechanism

Prevents the controller from stalling at near-zero current when conditions are actually safe:

1. **Normalize risk:** $\text{risk}_{\text{norm}} = \text{risk} / (\text{risk} + \text{scale})$ where scale = 3.0
2. **Accumulate low-risk duration:** if SoC gap > 0.25 **and** $\text{risk}_{\text{norm}} < 0.35$, increment timer
3. **Trigger:** when low-risk duration ≥ 120 s, propose a floor action corresponding to $\max(0.2\,\text{A}, 0.20 \times I_{\max})$
4. **Validate:** only apply if the floor action passes the safety check

**Code:** `_apply_antistall` (line 1012–1045).

---

## 9. Training Procedure

### 9.1 Phase 1 — Dataset Generation & Offline Warm-Start

**Code:** `scripts/generate_saerl_dataset.py`

1. **Data generation:** Episodes are simulated with three mixed behavior policies (weights configurable):
   - Pure MPC rollouts
   - MPC + Gaussian perturbations
   - CCCV rollouts
2. **CEM labels:** The **Cross-Entropy Method** generates near-optimal short-horizon action sequences as supervision targets for each step. CEM minimizes a cost function over 5-step lookaheads.
3. **Dataset structure:** Each row contains the sliding window features, the action taken, the MPC baseline action, the CEM target action, and the next-state targets (SoC, voltage, temperature, imbalance).

**Code:** `scripts/train_saerl_ensemble.py`

4. **Ensemble training:**
   - GRU and MLP are trained for up to 100 epochs (`gru_epochs`, `mlp_epochs`) with the quantile pinball loss, Adam optimizer (lr = 1e-3, weight decay = 1e-5), and early stopping (patience 5).
   - RF is trained via `sklearn.fit()` on flattened features.
   - Gate network is trained for 20 epochs on MSE between fused prediction and true next-state values.

**Code:** `scripts/train_saerl_policy.py`

5. **Behavioral cloning warm-start:** The residual actor is initialized by supervised learning on the CEM-derived target deltas ($\delta^* = \text{CEM action} - \text{MPC action}$), minimizing MSE or log-likelihood.

### 9.2 Phase 2 — PPO Online Fine-Tuning

The policy is fine-tuned via interaction with the simulation environment using PPO:

**PPO objective:**

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[\min\!\left(\rho_t \hat{A}_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

where $\rho_t = \pi_\theta(a_t | s_t) / \pi_{\theta_{\text{old}}}(a_t | s_t)$.

| Hyperparameter | Value |
|----------------|-------|
| Discount $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| Clip ratio $\epsilon$ | 0.2 |
| Learning rate | $3 \times 10^{-4}$ |
| Minibatch size | 64 |
| PPO epochs per update | 4 |

**Each online step:**

1. Build sliding window from recent history
2. Compute per-expert predictions with uncertainties
3. Update gate weights (forward-only)
4. Fuse predictions via gate
5. Sample residual action from policy
6. Compose raw action: $a_{\text{raw}} = \text{clip}(a_{\text{mpc}} + \delta, -1, 1)$
7. Apply safety shield → $a_{\text{safe}}$
8. Step environment, observe next state and reward
9. Store transition for PPO update
10. After rollout: update $\pi_\theta$ via PPO, update gate via supervised MSE

---

## 10. Data Pipeline & Parameter Identification

### 10.1 Dataset Loaders

**Code:** `data_ingestion.py`

Three public dataset formats are supported:

| Dataset | Loader | Format |
|---------|--------|--------|
| NASA Ames | `load_nasa_csv()` | CSV with impedance parsing |
| CALCE | `load_calce_zip()` | ZIP of XLSX files, sheet 2 |
| MATR.io | `load_matrio_zip()` | ZIP of JSON structures |

All loaders apply **column name normalization** (`normalize_pack_dataframe`) to map dataset-specific column names (e.g., `Voltage_measured`, `Current(A)`) to standard names (`pack_voltage`, `pack_current`, `pack_temperature`, `time`).

### 10.2 Pack Parameter Identification

**Code:** `parameter_identification.py` → `PackParameterIdentifier`

Given experimental data, the identifier extracts:

1. **ECM parameters** (from voltage pulse response):
   - $R_0$ from instantaneous voltage step / current step
   - $R_1 = 0.4 R_0$, $R_2 = 0.6 R_0$ (fixed ratios)
   
2. **Thermal parameters** (from temperature response):
   - $C_{\text{th}}$ and $hA$ via `scipy.optimize.curve_fit` on a lumped model
   
3. **Cell variations** (from individual cell data or pack voltage spread):
   - If per-cell voltages available: estimate capacity/resistance CoV from voltage spread
   - Otherwise: use typical manufacturing tolerances ($\text{CoV}_Q = 5\%$, $\text{CoV}_{R_0} = 20\%$)
   
4. **Aging characteristics** (from multi-cycle data):
   - Linear regression of voltage range vs. cycle number to estimate capacity fade rate

---

## 11. Evaluation Protocol

### 11.1 Scenarios

**Code:** `scenarios.json` — 15 scenarios organized in 5 groups × 3 objectives:

| Group | Dataset | Ambient Temp | Initial SoC | Objectives |
|-------|---------|-------------|-------------|------------|
| 1 (NASA fresh) | NASA | 25 °C | 20 % | F1 (4C), S1 (3C), L1 (2C) |
| 2 (CALCE aged) | CALCE | 25 °C | 20 % | F2 (4C), S2 (3C), L2 (2C) |
| 3 (MATR cold) | MATR | 0 °C | 20 % | F3 (4C), S3 (3C), L3 (2C) |
| 4 (NASA hot aged) | NASA | 40 °C | 10 % | F4 (4C), S4 (3C), L4 (2C) |
| 5 (MATR variable) | MATR | 10 °C | 30 % | F5 (4C), S5 (3C), L5 (2C) |

Each scenario defines: initial SoC, target SoC (80 %), ambient temperature, max C-rate, and max voltage.

### 11.2 Baselines

| Method | Description |
|--------|-------------|
| **CCCV** | 1C CC until 4.2 V, then CV until 0.05C taper |
| **MPC** | 5-step look-ahead with perfect model knowledge |
| **PPO only** | Model-free PPO, no ensemble/gate/shield |
| **Static ensemble + PPO** | Equal-weight ensemble, no adaptive gate |

### 11.3 Acceptance Criteria

A controller **passes** a scenario if all three conditions are met:

1. ✅ Reaches 80 % SoC within 120 minutes
2. ✅ No voltage or temperature violation exceeds 50 ms duration
3. ✅ Final SoC within ±2 % of target

---

## 12. Hyperparameter Summary

| Category | Parameter | Value |
|----------|-----------|-------|
| **Environment** | Sampling time $\Delta t$ | 1 s |
| | Number of cells $N$ | 10 (paper) / 20 (code default) |
| | $I_{\max}$ | 5C (paper) / configurable |
| | $V_{\max}$ | 4.2 V |
| | $T_{\max}$ | 45 °C (paper) / 60 °C (code) |
| | SoC noise $\sigma_{\text{soc}}$ | 0.01 |
| | Voltage noise $\sigma_V$ | 10 mV |
| | Temperature noise $\sigma_T$ | 0.5 °C |
| **Ensemble** | GRU layers / hidden | 2 / 64 |
| | MLP architecture | [128, 128, 64] |
| | RF trees / min leaf | 200 / 5 |
| | Window length $L$ | 20 |
| | Target dim | 4 |
| **PPO** | $\gamma$ | 0.99 |
| | GAE $\lambda$ | 0.95 |
| | Clip $\epsilon$ | 0.2 |
| | Learning rate | $3 \times 10^{-4}$ |
| | Minibatch | 64 |
| | Epochs/update | 4 |
| **Gate** | Hidden layers | [64, 64] |
| | Activation | ReLU |
| | Update interval | 50 steps |
| | Error momentum | 0.90 |
| **Safety Shield** | Candidate points | 9 |
| | Candidate radius | 0.35 |
| | $\delta_{\max}$ | 0.40 |
| | Temp soft limit | 42 °C |
| | Voltage margin | 30 mV |
| | Temperature margin | 0.5 °C |
| | Imbalance margin | 95 mV |
| **Anti-Stall** | Min charge fraction | 0.20 |
| | SoC gap threshold | 0.25 |
| | Risk threshold | 0.35 |
| | Duration trigger | 120 s |
| | Risk scale | 3.0 |

---

## 13. File Map — Paper ↔ Code

| Paper Section | Primary Code File(s) | Key Classes / Functions |
|---------------|----------------------|------------------------|
| §3 Problem Formulation (CMDP) | `hambrl_pack_env.py` | `HAMBRLPackEnvironment` |
| §4 Battery & Pack Model | `physics_model.py`, `battery_pack_model.py` | `CellModel`, `BatteryPack` |
| §4.1 Predictive Ensemble | `controllers/adaptive_ensemble_rl.py` | `QuantileGRUModel`, `QuantileMLPModel`, `AdaptiveEnsemblePredictor` |
| §4.2 Adaptive Gating | `controllers/adaptive_ensemble_rl.py` | `GateNetwork`, `_compute_weights`, `predict_fused` |
| §4.3 Residual Policy + Shield | `controllers/adaptive_ensemble_rl.py` | `ResidualActorPolicy`, `SafeAdaptiveEnsembleController` |
| §4.4 Training (offline) | `scripts/generate_saerl_dataset.py`, `scripts/train_saerl_ensemble.py` | `main()`, `cem_target_action`, `train_quantile_gru` |
| §4.4 Training (online PPO) | `scripts/train_saerl_policy.py` | `main()`, `evaluate_policy_short` |
| §5 Experimental Setup | `scenarios.json`, `scripts/saerl_common.py` | Scenario definitions |
| Data Ingestion | `data_ingestion.py` | `load_nasa_csv`, `load_calce_zip`, `load_matrio_zip` |
| Parameter Identification | `parameter_identification.py` | `PackParameterIdentifier` |
| Evaluation | `scripts/eval_saerl_vs_baselines.py`, `scripts/eval_all_controllers.py` | Evaluation scripts |
| CCCV Baseline | `controllers/adaptive_ensemble_rl.py` | `_cccv_anchor_action` |
| Residual Controller (v1) | `controllers/residual_hambrl.py` | `ResidualHAMBRLController` |
| Configs | `configs/*.json` | Phase-specific training profiles |

---

