# Codebase Technical Review

Date: 2026-02-28

This document is a technical audit of the current repository implementation. It is based on the code in the workspace and the stored result artifacts under `results/` and `paper/results/`.

## Scope

Reviewed components:

- `physics_model.py`
- `battery_pack_model.py`
- `hambrl_pack_env.py`
- `parameter_identification.py`
- `pack_experiments.py`
- `controllers/adaptive_ensemble_rl.py`
- `controllers/residual_hambrl.py`
- `scripts/run_baseline_benchmarks.py`
- `scripts/generate_saerl_dataset.py`
- `scripts/train_saerl_ensemble.py`
- `scripts/train_saerl_policy.py`
- `scripts/eval_saerl_vs_baselines.py`
- `scripts/aggregate_saerl_results.py`
- `scripts/saerl_common.py`

Primary result artifacts reviewed:

- `results/saerl_phase3h_progress_full/aggregate_allfolds_3family/*`
- `results/saerl_phase3h_progress/aggregate_allfolds_calce_matr/*`
- `paper/results/*`
- `data/training/saerl_phase3h_progress_full_dataset_meta.json`

## Executive Assessment

The repository implements a hybrid safe charging controller, not a pure model-based RL stack.

The actual controller is:

1. a simplified electro-thermal battery simulator,
2. two handcrafted baselines (CCCV and rollout MPC),
3. a learned one-step dynamics ensemble (GRU + MLP + Random Forest),
4. a residual actor that perturbs MPC,
5. a hard safety shield that rejects unsafe actions,
6. heuristic online PPO fine-tuning on top of the residual actor.

The strongest evidence in the current stored artifacts is the 3-family aggregate run in `results/saerl_phase3h_progress_full/aggregate_allfolds_3family`:

- Scenario pass rate: `13 / 27 = 48.15%`
- Failures are dominated by performance, not safety:
  - `pass_safety_zero` failures: `0`
  - `pass_temp` failures: `0`
  - `pass_q_loss` failures: `3`
  - `pass_perf` failures: `11`

So the current implementation is doing what its structure suggests:

- it is strong at staying safe,
- reasonably strong at preventing thermal issues,
- moderately strong at limiting degradation,
- still weak at matching or beating baseline charge completion speed, especially in `safe` and `long_life`.

That is the central technical shortfall.

## 1. Mathematical Model Implemented

### 1.1 Single-cell model

The core cell model in `physics_model.py` is a 2-RC Thevenin equivalent circuit with lumped thermal and a simple aging law.

Implemented state update:

- State of charge:

  `dSoC/dt = -eta * I / (Q_effective * 3600)`

- RC overpotentials:

  `dv1/dt = -v1 / (R1*C1) + I / C1`

  `dv2/dt = -v2 / (R2*C2) + I / C2`

- Terminal voltage:

  `V = OCV(SoC) - I*(R0 + R0_growth) - v1 - v2`

- Thermal model:

  `dT/dt = (Q_gen - Q_diss) / C_th`

  `Q_gen = I^2 * (R0 + R0_growth)`

  `Q_diss = hA * (T - T_ambient)`

- Aging law:

  `dQ_loss/dt = k1 * S_I * S_T * S_SoC`

  `S_I = |I| / Q_nominal`

  `S_T = exp(k2 * (T - T_ref))`

  `S_SoC = exp(k3 * SoC)`

Then:

- `capacity_fade = min(Q_loss / (0.2 * Q_nominal), 0.8)`
- `Q_effective = Q_nominal * (1 - capacity_fade)`
- `R0_growth = 2 * R0 * capacity_fade`

### 1.2 OCV model

OCV is not physics-derived from chemistry-specific equations. It is piecewise linear interpolation over fixed `soc_points` and `ocv_points`.

Implication:

- This is computationally convenient.
- It is not chemistry-identifiable by itself.
- Real chemistry differences are injected later by rescaling OCV curves from dataset quantiles, not by a mechanistic electrode model.

### 1.3 Pack model

The pack model in `battery_pack_model.py` composes many cells and adds balancing and a simple thermal mixing matrix.

Key pack equations:

- For each parallel string:

  `V_string = sum(V_cell in series)`

- Reported pack voltage:

  `V_pack = mean(V_string across parallel strings)`

- Pack SoC:

  `SoC_pack = sum(SoC_i * Q_i) / sum(Q_i)`

- Charge current per cell:

  `I_cell = I_pack / n_parallel`

Balancing:

- Passive balancing drains higher-SoC cells.
- Active balancing redistributes current proportionally around mean SoC.

Thermal interaction:

- A fixed nearest-neighbor mixing matrix smooths cell temperatures:

  `T_new = ThermalMatrix @ T_old`

This is not a heat equation solve. It is a normalized linear smoothing step.

## 2. Parameter Identification and Calibration

### 2.1 What the code actually estimates

`parameter_identification.py` performs pack-level heuristic fitting, not rigorous system identification.

ECM fitting:

- Detects the first large current step.
- Estimates:

  `R0_pack = |Delta V / Delta I|`

- Then sets:

  `R1_pack = 0.4 * R0_pack`

  `R2_pack = 0.6 * R0_pack`

  `C1_pack = 1000`

  `C2_pack = 5000`

So only `R0` is truly inferred. The rest are fixed ratios/constants.

Thermal fitting:

- Uses a one-state thermal fit with:

  `Q_gen = I^2 * R_heat`

- But `R_heat` is hard-coded as `0.02 * 20`.

This means the thermal fit is anchored to an assumed resistance, not independently identified from data.

Cell variation fitting:

- If cell-level data are absent, variation is inferred from voltage spread only.
- If cell-level data are present, the method still inserts placeholder values (`Q_nominal=2.5`, `R0=0.02`, `initial_soc=0.5`), then computes summary statistics.

Aging fitting:

- If cycle index exists, it fits a linear trend on per-cycle voltage range and converts slope into a "capacity fade rate".

### 2.2 Practical conclusion

The code uses parameter identification mainly as calibration scaffolding, not as a high-fidelity identification layer.

Strength:

- It provides stable default values and lets the pipeline run on heterogeneous data.

Weakness:

- The identified parameters are too coarse to support strong claims of electrochemical realism.
- The calibration is sufficient for controller bootstrapping, but not for a serious physics validation argument.

## 3. Environment and Reward Design

`hambrl_pack_env.py` defines a Gym-like environment.

### 3.1 Action mapping

Normalized action `a in [-1, 1]` maps to charge current:

- `a = -1` -> `0 A`
- `a = +1` -> `-max_charge_current_a`

Implemented:

`I_pack = -0.5 * I_max * (a + 1)`

### 3.2 Observation vector

The observation is 6-dimensional:

1. pack SoC
2. normalized pack voltage
3. normalized pack temperature
4. voltage imbalance in mV
5. normalized pack current
6. normalized progress through episode

### 3.3 Environment reward

Base reward:

- `+10 * pack_soc`
- `-0.01` per step
- temperature penalty above 40 C
- voltage imbalance penalty
- `-1` per counted safety event
- `+50` terminal bonus if target SoC reached

Important caveat:

This environment reward is not the main reward used in SAERL policy training. The SAERL trainer uses a different custom reward in `train_saerl_policy.py`.

## 4. Baseline Controllers

### 4.1 CCCV

`scripts/run_baseline_benchmarks.py` implements a PI-based CCCV controller:

- Constant-current mode until near CV threshold
- Then PI regulation on voltage
- Additional SoC taper:

  `charge_current *= clip(soc_gap / soc_taper_window, 0.15, 1.0)`

This is simple, transparent, and easy to reason about.

### 4.2 Rollout MPC

The MPC is not a continuous optimizer. It is a brute-force rollout controller:

- finite action grid,
- fixed horizon,
- constant first action search,
- simple heuristic terminal policy after the first step.

Stage cost:

`J = w_soc * soc_error^2`

`+ w_voltage * over_voltage^2`

`+ w_temp * over_temp^2`

`+ w_imbalance * imbalance^2`

`+ w_current * current_norm^2`

`+ w_smooth * (a_t - a_{t-1})^2`

`+ 500 * safety_count`

`+ hard penalties for pack over-voltage / over-temp`

This is a sensible safety-oriented benchmark, but it is still a very coarse MPC:

- single-action shooting,
- low-dimensional search,
- no explicit state estimator,
- no explicit multi-step control sequence optimization.

## 5. SAERL Model: What It Really Is

### 5.1 Predictor ensemble

`controllers/adaptive_ensemble_rl.py` uses three one-step predictors:

- GRU quantile model
- MLP quantile model
- Random Forest regressor with tree-spread uncertainty

Targets:

- next SoC
- next voltage
- next temperature
- next imbalance

Fusion:

- `y_hat = sum_i w_i * y_hat_i`
- `u_hat = sum_i w_i * u_i`

The gate uses:

- latest normalized state features,
- action,
- each expert's mean uncertainty,
- rolling absolute error by expert.

### 5.2 Risk score

The risk score used for scoring and anti-stall is:

`risk = w_temp * over_temp^2`

`+ w_voltage * over_voltage^2`

`+ w_imbalance * |pred_imbalance_mV|`

`+ w_uncertainty * mean(fused_uncertainty)`

`+ 8 * max(0, prev_soc - pred_soc)`

This is a heuristic scalar risk, not a calibrated probability of violation.

### 5.3 Residual actor

The actor predicts only a residual over MPC:

- input = flattened state window + MPC action + SoC gap
- output = one scalar residual
- final action = `clip(a_mpc + delta, -1, 1)`

This strongly constrains the policy:

- It can improve MPC locally.
- It cannot invent a control law fundamentally different from the anchor.

### 5.4 Candidate scoring

The final controller does not directly execute the actor output.

It scores a candidate set around the residual proposal, optionally including:

- the MPC action,
- a CCCV anchor action,
- full charge action.

Candidate objective:

`score = -w_soc_gain * soc_gain`

`+ w_time * time_proxy`

`+ w_temp * over_temp^2`

`+ w_deg * degradation_proxy`

`+ w_imbalance * |imbalance|`

`+ w_safety * safety_violation`

`+ w_risk * risk`

This is closer to a one-step action selector with learned roll-forward estimates than to a full policy optimization framework.

### 5.5 Safety shield and anti-stall

The hard shield:

- simulates one step with the candidate action,
- rejects actions that violate pack voltage, temperature, current, or imbalance margins,
- falls back to MPC,
- otherwise searches for the least-stress safe action.

The anti-stall logic:

- tracks low normalized risk over time,
- if SoC gap stays large and risk stays low long enough, it forces a minimum charge floor.

This is one of the key reasons the system is robust in safety metrics.

## 6. Data and Training Pipeline

### 6.1 Dataset generation

`scripts/generate_saerl_dataset.py` builds supervised transitions from:

- mixed behavior policies: MPC, perturbed MPC, CCCV
- CEM-labeled target actions
- optional "final SoC gain" relabeling for scenarios where baselines do not hit target

Stored dataset features:

- rolling 20-step normalized state window
- current action traces
- baseline actions
- target action and target residual

Latest stored `saerl_phase3h_progress_full` dataset metadata:

- rows: `8640`
- episodes: `54`
- objectives: `fastest`, `safe`, `long_life`
- families: `nasa`, `calce`, `matr`
- `final_soc_gain_episode_count`: `30`

### 6.2 Important mismatch: code vs stored dataset

Current code now includes:

- adaptive horizon in dataset generation and policy rollouts,
- chemistry-aware MPC anchoring for SAERL residuals.

But the stored dataset used by `saerl_phase3h_progress_full` still shows the old regime:

- `max_steps = 160`
- episode file shows `n_steps = 160` for every episode
- metadata does not include the newer adaptive-horizon config fields

Conclusion:

The current repository code is ahead of the stored training artifacts. The best available stored results do **not** yet reflect the new adaptive-horizon and chemistry-aware-anchor changes.

### 6.3 Ensemble training

`train_saerl_ensemble.py`:

- trains GRU and MLP by quantile pinball loss,
- fits RF on same one-step targets,
- calibrates uncertainty widths by coverage on validation,
- trains the gate to approximate inverse-error weighting.

Observed ensemble calibration quality in `results/saerl_phase3h_progress_full/training/ensemble`:

- CALCE:
  - GRU mean coverage: `0.9635`
  - MLP mean coverage: `0.9393`
  - RF mean coverage: `0.7962`
- MATR:
  - GRU mean coverage: `0.8711`
  - MLP mean coverage: `0.8968`
  - RF mean coverage: `0.7241`
- NASA:
  - GRU mean coverage: `0.7336`
  - MLP mean coverage: `0.6442`
  - RF mean coverage: `0.8099`

This is a major signal:

- CALCE is modeled well.
- MATR is moderate.
- NASA is materially harder for the neural experts.

### 6.4 Policy training

`train_saerl_policy.py` has two phases:

1. Offline behavior cloning on `target_delta_action`
2. Online PPO-like fine-tuning with custom reward shaping

Online reward:

`reward = + w_soc_gain * delta_soc`

`- step_penalty`

`- temp_penalty`

`- imbalance_penalty`

`- q_loss_step_penalty`

`- safety_penalty`

plus optional acceptance-shaped progress and terminal penalties/bonuses relative to baseline references.

Observed offline fit quality from `results/saerl_phase3h_progress_full/training/policy`:

- `safe / calce`: mean offline MSE `0.000677`
- `safe / matr`: mean offline MSE `0.026642`
- `safe / nasa`: mean offline MSE `0.110686`

- `fastest / calce`: mean offline MSE `0.002556`
- `fastest / matr`: mean offline MSE `0.026292`
- `fastest / nasa`: mean offline MSE `0.023679`

- `long_life / calce`: mean offline MSE `0.000735`
- `long_life / matr`: mean offline MSE `0.009071`
- `long_life / nasa`: mean offline MSE `0.006563`

Interpretation:

- CALCE actor fitting is very good.
- MATR is much noisier.
- NASA `safe` is particularly poor, which matches the weaker evaluation pass rate on NASA.

## 7. Evaluation Logic and Acceptance Criteria

`eval_saerl_vs_baselines.py` compares CCCV, MPC, and SAERL on held-out case-test scenarios.

A scenario passes only if all of these hold:

1. `pass_safety_zero`
   - SAERL must have zero safety events.
2. `pass_temp`
   - SAERL peak temperature must be no worse than the better baseline by more than `0.2 C`.
3. `pass_q_loss`
   - SAERL total q-loss must be at most `1.05 * MPC q-loss` when MPC q-loss is finite.
4. `pass_perf`
   - If either baseline reaches 80% SoC, SAERL must also reach 80% and do so within `10%` of the fastest eligible baseline.
   - If neither baseline reaches 80%, SAERL must exceed the best safe baseline final SoC by `0.02`.

This acceptance logic strongly rewards:

- zero hard failures,
- near-baseline speed,
- modest degradation control.

It is strict enough that a safe-but-slower controller fails frequently.

## 8. Current Results Audit

### 8.1 Latest strong aggregate: 3-family progress run

From `results/saerl_phase3h_progress_full/aggregate_allfolds_3family`:

- `27` primary scenarios
- `13` passes
- pass rate `48.15%`

By objective:

- `fastest`: `7 / 9 = 77.78%`
- `long_life`: `3 / 9 = 33.33%`
- `safe`: `3 / 9 = 33.33%`

By family:

- `calce`: `5 / 9 = 55.56%`
- `matr`: `5 / 9 = 55.56%`
- `nasa`: `3 / 9 = 33.33%`

The result pattern is consistent:

- The stack is strongest when the objective permits aggressive charging.
- It is weakest when the objective demands conservative behavior while still matching baseline completion.

### 8.2 Why scenarios fail

Failures by criterion:

- safety failures: `0`
- temperature failures: `0`
- q-loss failures: `3`
- performance failures: `11`

So the model is not failing because it is unsafe. It is failing because it often leaves too much performance on the table.

### 8.3 Typical failure modes in stored results

Common pattern in failed rows:

- SAERL finishes safer than CCCV,
- often with `0` safety events when CCCV has many,
- but SAERL misses the time-to-80 requirement or slightly undershoots final SoC.

Representative failures from the 3-family aggregate:

- `safe / calce / CS2_3`
  - CCCV final SoC: `0.800009`
  - MPC final SoC: `0.786444`
  - SAERL final SoC: `0.793197`
  - fails performance

- `long_life / calce / CS2_9`
  - CCCV final SoC: `0.800008`
  - MPC final SoC: `0.786487`
  - SAERL final SoC: `0.793333`
  - fails performance

- `safe / matr / 2018-09-10_oed_3_CH7`
  - SAERL remains safety-clean while baselines incur many events
  - but SAERL final SoC is below the best baseline and fails performance

There is also a different NASA-specific pattern:

- `nasa / 00006` scenarios fail `pass_q_loss` but pass `pass_perf`
- SAERL gains much more final SoC than baselines, but degradation proxy becomes relatively worse

This suggests the controller can recover underachieving NASA baselines, but sometimes by pushing current harder than the q-loss criterion allows.

### 8.4 Aggregate controller averages

From `overall_summary.csv` in the 3-family aggregate:

- `saerl` and `saerl_family_specific`:
  - final SoC: `0.7713`
  - safety events: `0.0`
  - charge time: `110.16 min`
  - time to 80%: `101.80 min`
  - q-loss total: `0.04233`
  - inference latency: `41.22 ms`

- `mpc`:
  - final SoC: `0.7336`
  - safety events: `80.63`
  - time to 80%: `83.62 min`
  - q-loss total: `0.04126`
  - inference latency: `68.12 ms`

- `cccv`:
  - final SoC: `0.7597`
  - safety events: `620.74`
  - time to 80%: `89.57 min`
  - q-loss total: `0.04179`

Interpretation:

- SAERL is the safest controller by a wide margin.
- SAERL improves average final SoC over MPC and CCCV.
- SAERL is slower than MPC in reaching 80% SoC.
- SAERL q-loss is slightly worse than MPC on average.

This directly explains the acceptance failures.

### 8.5 Paper snapshot vs latest aggregate

`paper/results/acceptance_summary.csv` shows an older snapshot:

- pass rate: `6 / 15 = 40%`

This is weaker than the later 3-family aggregate (`48.15%`), but still consistent with the same story:

- safety is generally controlled,
- performance remains the bottleneck.

## 9. Main Technical Shortcomings

### 9.1 The stored training artifacts still reflect the old short horizon

This is the most important operational issue.

The current code has been updated to support:

- adaptive episode horizons,
- chemistry-aware SAERL MPC anchors.

But the stored `saerl_phase3h_progress_full` dataset and results were generated before those updates:

- dataset metadata still uses fixed `max_steps=160`
- episode CSV shows fixed `n_steps=160`

This means the best current code path and the best stored results are out of sync.

If the paper is judging the model on current stored results, it is still judging the pre-fix regime.

### 9.2 One-step dynamics only

The learned ensemble predicts only one-step transitions.

Consequences:

- long-horizon performance depends on repeated greedy selection,
- compounding model error is not explicitly controlled,
- the actor has no explicit long-horizon value estimate.

This is one reason the controller can stay safe but still become conservative or slightly myopic.

### 9.3 The actor is heavily boxed in by the MPC anchor

Because:

- the actor only predicts a bounded residual,
- the final action is rescored across candidates,
- the shield can overwrite the result,

the learned policy has limited authority.

This is good for safety but reduces the chance of large performance gains, especially when the baseline MPC itself is under-aggressive.

### 9.4 The uncertainty model is heuristic, not probabilistically coherent

Current uncertainty is:

- quantile width for GRU/MLP,
- tree spread for RF,
- linearly fused with gate weights.

This is not a proper joint uncertainty model. It can be useful for ranking but should not be over-claimed as calibrated predictive uncertainty.

### 9.5 Parameter identification is too weak for strong physics claims

The current identification path:

- infers only one true ECM parameter (`R0`) from pulse data,
- hard-codes other ECM values,
- uses a fixed heat resistance for thermal fitting,
- treats cell variation mostly as defaults or voltage-spread heuristics.

This is enough for simulation stabilization, but not enough to justify a strong claim of chemistry-faithful electro-thermal identification.

### 9.6 Safety is strong because the shield is doing real work

The evaluation shows near-zero SAERL safety events.

That is good, but it also means:

- a material share of performance comes from the shield gatekeeping actions,
- the learned actor is not independently demonstrating safe control.

This is still a valid system design, but the paper should describe the controller as a safety-filtered hybrid policy, not as a purely learned safe controller.

### 9.7 Mode comparison is incomplete in the latest aggregate

In `results/saerl_phase3h_progress_full/aggregate_allfolds_3family`:

- `saerl` is the alias for the primary mode,
- `saerl_family_specific` is also present,
- both are numerically identical in the summary.

So the latest stored aggregate is not a true global-vs-family_specific-vs-shared_plus_heads comparison for SAERL. It is effectively a family-specific run plus a duplicated primary alias.

That limits what can be claimed about chemistry-mode ablations.

### 9.8 NASA remains the weakest domain

Evidence:

- lowest pass rate by family (`33.33%`)
- weakest ensemble neural coverage
- highest offline actor error in `safe`

This points to a real domain mismatch:

- fewer usable samples,
- different current/voltage scales,
- weaker surrogate fit,
- poorer transfer from label policy to learned residual.

## 10. What Is Most Likely Holding Back Performance

Based on the current code and stored results, the biggest blockers are:

1. Fixed-horizon training artifacts still used in the best reported runs.
2. SAERL residual still historically trained against too-conservative or mismatched anchors in stored artifacts.
3. One-step predictor errors accumulate, especially outside CALCE.
4. The shield and anti-stall preserve safety but often clip away aggressive progress.
5. The acceptance metric is speed-sensitive, and SAERL is still slower than MPC in many target-reaching cases.

## 11. Highest-Impact Next Steps

If the goal is to improve the paper-acknowledged model performance, the highest-value actions are:

1. Re-run the full `saerl_phase3h_progress_full` pipeline with the new code path.
   - This is required before the paper can claim benefit from adaptive horizon or chemistry-aware SAERL anchoring.

2. Rebuild the training dataset with adaptive horizons enabled.
   - Current stored dataset is still fixed-step.
   - This should directly address undercharging in `safe` and `long_life`.

3. Re-train and re-evaluate with chemistry-aware SAERL MPC anchoring enabled end to end.
   - This should improve alignment between residual learning and family-specific operating envelopes.

4. Separate true SAERL mode ablations.
   - Run explicit `global`, `family_specific`, and `shared_plus_heads` evaluations rather than relying on the primary alias.

5. Prioritize NASA-specific model repair.
   - The metrics show NASA is the weakest domain.
   - If paper space is limited, this is the best place to argue where the model still fails.

## 12. Bottom Line

The codebase is technically coherent and internally consistent as a hybrid safe charging system.

Its strongest claims are:

- robust safety behavior,
- good integration of heuristic control with learned residuals,
- clear performance gains over unsafe baselines in some hard cases,
- a practical chemistry-aware routing structure.

Its weakest claims are:

- strong physics fidelity,
- strong long-horizon optimality,
- consistently superior charge speed,
- up-to-date evidence for the newly implemented adaptive-horizon and chemistry-aware-anchor improvements.

The main paper-safe conclusion is:

The system is best described as a safety-dominant hybrid residual controller that already achieves strong robustness, but whose current stored results are still limited by conservative horizon assumptions, one-step surrogate error, and performance underreach in `safe` and `long_life`. The newest code changes target exactly those gaps, but they still need a full re-run before they can be claimed in the results section.
