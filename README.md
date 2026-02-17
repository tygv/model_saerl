# modelBased_rl

## Baseline Benchmarking (CCCV vs MPC)

Run the two classical charging baselines on the same pack environment:

```bash
python scripts/run_baseline_benchmarks.py --objective safe --target-soc 0.8 --max-steps 1200
```

Supported objective presets come from `pack_experiments.py`:
- `fastest`
- `safe`
- `long_life`

Run all presets in one pass:

```bash
python scripts/run_baseline_benchmarks.py --objective all --target-soc 0.8 --max-steps 1200
```

Outputs are automatically separated into folders:
- `results/baselines/cccv`
- `results/baselines/mpc`
- `results/baselines/comparison`

Each folder includes:
- `trajectory.csv` (step-by-step simulation data)
- `metrics.json` or `metrics_summary.csv` (benchmark KPIs)
- high-resolution paper-style figures in both `.png` and `.pdf`

### Data-Calibrated Baselines (uses your real NASA/CALCE/MATR recordings)

Run CCCV vs MPC with environment settings calibrated from standardized CSVs and fitted parameter JSONs:

```bash
python scripts/run_baseline_benchmarks.py --use-real-data --dataset-families nasa,calce,matr --max-files-per-dataset 1 --objective safe --max-steps 1200
```

By default this mode treats those files as **cell-level behavior references** and scales them to your configured pack topology (`--n-series`, `--n-parallel`) so output figures are pack-focused.

When `--use-real-data` is enabled, outputs are written under:

- `results/baselines/data_calibrated/<objective>/<dataset_family>/<case>/cccv`
- `results/baselines/data_calibrated/<objective>/<dataset_family>/<case>/mpc`
- `results/baselines/data_calibrated/<objective>/<dataset_family>/<case>/comparison`

### Aggregated Comparison Across All Objectives + Datasets

After generating data-calibrated baselines, build one aggregated analysis package:

```bash
python scripts/aggregate_baseline_results.py --input-root results/baselines/data_calibrated --output-root results/baselines/aggregate
```

This writes:
- `results/baselines/aggregate/all_runs_long_metrics.csv`
- `results/baselines/aggregate/scenario_pairwise_deltas.csv`
- `results/baselines/aggregate/group_means.csv`
- `results/baselines/aggregate/group_stds.csv`
- `results/baselines/aggregate/pairwise_delta_means.csv`
- `results/baselines/aggregate/win_rates.csv`
- figures (`.png` + `.pdf`) under `results/baselines/aggregate/figures`

## Phase 1: Residual H-AMBRL Pipeline

1) Generate MPC-guided residual training data:

```bash
python scripts/generate_mpc_dataset.py --objective all --condition all --episodes-per-setting 2 --max-steps 1200
```

2) Train residual policy:

```bash
python scripts/train_residual_policy.py --dataset-csv data/training/residual_mpc_dataset.csv --model-out models/residual_hambrl_policy.json
```

3) Evaluate all controllers (CCCV vs MPC vs Residual):

```bash
python scripts/eval_all_controllers.py --model-path models/residual_hambrl_policy.json --objective all --condition all --max-steps 1200
```

Evaluation outputs are grouped by objective/condition under:
- `results/residual_phase1/evaluation/<objective>/<condition>/cccv`
- `results/residual_phase1/evaluation/<objective>/<condition>/mpc`
- `results/residual_phase1/evaluation/<objective>/<condition>/residual`
- `results/residual_phase1/evaluation/<objective>/<condition>/comparison`

## Phase 2: SAERL (Safe Adaptive Ensemble RL)

1) Generate mixed-behavior dataset with CEM action targets and fold splits:

```bash
python scripts/generate_saerl_dataset.py --objective all --max-files-per-dataset 3 --episodes-per-setting 2
```

2) Train ensemble experts + adaptive gate:

```bash
python scripts/train_saerl_ensemble.py --dataset-csv data/training/saerl_phase2_dataset.csv --split-manifest-json data/training/saerl_phase2_splits.json
```

3) Train SAERL residual actor (offline warm-start + safe online fine-tune):

```bash
python scripts/train_saerl_policy.py --dataset-csv data/training/saerl_phase2_dataset.csv --split-manifest-json data/training/saerl_phase2_splits.json
```

4) Evaluate CCCV vs MPC vs SAERL on held-out fold cases:

```bash
python scripts/eval_saerl_vs_baselines.py --dataset-csv data/training/saerl_phase2_dataset.csv --split-manifest-json data/training/saerl_phase2_splits.json --run-ablations
```

For cross-chemistry fairness (different `dt` and current scales), use adaptive horizon scaling:

```bash
python scripts/eval_saerl_vs_baselines.py --dataset-csv data/training/saerl_phase3d_dataset.csv --split-manifest-json data/training/saerl_phase3d_splits.json --chemistry-mode family_specific --primary-saerl-mode family_specific --min-episode-minutes 120 --feasible-time-slack 1.35 --max-steps-cap 5000 --skip-detailed-figures
```

5) Aggregate SAERL phase-2 outputs:

```bash
python scripts/aggregate_saerl_results.py --input-root results/saerl_phase2/evaluation --output-root results/saerl_phase2/aggregate
```

6) Run SAERL checks (unit + integration + runtime):

```bash
python scripts/run_saerl_tests.py
```
