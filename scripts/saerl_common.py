"""Common helpers for SAERL phase-2 scripts."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

from battery_pack_model import PackConfiguration
from hambrl_pack_env import HAMBRLPackEnvironment
from pack_experiments import ChargingObjective
from scripts.run_baseline_benchmarks import (
    BenchmarkConfig,
    apply_data_profile_to_pack,
    apply_fitted_params_to_pack,
    collect_data_calibrated_scenarios,
    compute_metrics,
    count_safety_events,
    trim_pack_histories,
)


@dataclass
class SAERLScenarioSetting:
    objective_key: str
    objective: ChargingObjective
    scenario: Dict[str, Any]
    pack_config: PackConfiguration
    cv_cell_v: float
    cv_voltage_v: float
    max_charge_current_a: float
    initial_soc: float
    initial_temp_c: float
    ambient_temp_c: float
    dt_s: float
    source_is_cell_level: bool


def split_csv_arg(value: str) -> List[str]:
    return [x.strip().lower() for x in str(value).split(",") if x.strip()]


def load_data_calibrated_scenarios(
    standardized_root: str = "data/standardized",
    params_root: str = "data/standardized_params",
    dataset_families: str = "nasa,calce,matr",
    max_files_per_dataset: int = 3,
    n_series: int = 20,
    n_parallel: int = 1,
    max_charge_current_a: float = 10.0,
    balancing_type: str = "passive",
    initial_soc: float = 0.2,
    target_soc: float = 0.8,
    ambient_temp_c: float = 25.0,
) -> Tuple[BenchmarkConfig, List[Dict[str, Any]]]:
    run_config = BenchmarkConfig(
        objective="all",
        output_root="results/saerl_phase2/tmp",
        max_steps=1200,
        initial_soc=initial_soc,
        target_soc=target_soc,
        ambient_temp_c=ambient_temp_c,
        n_series=n_series,
        n_parallel=n_parallel,
        max_charge_current_a=max_charge_current_a,
        balancing_type=balancing_type,
        use_real_data=True,
        standardized_root=standardized_root,
        params_root=params_root,
        dataset_families=dataset_families,
        max_files_per_dataset=max_files_per_dataset,
        data_is_cell_level=True,
        include_cell_figures=False,
    )
    scenarios = collect_data_calibrated_scenarios(run_config)
    return run_config, scenarios


def build_setting_for_objective(
    run_config: BenchmarkConfig,
    objective_key: str,
    objective: ChargingObjective,
    scenario: Dict[str, Any],
) -> SAERLScenarioSetting:
    profile = scenario["profile"]
    n_series = int(max(1, run_config.n_series))
    n_parallel = int(max(1, run_config.n_parallel))

    data_cv_cell_v = float(profile["voltage_q99_v"] + 0.02)
    cv_cell_v = float(min(objective.v_max, data_cv_cell_v))
    cv_cell_v = float(max(cv_cell_v, min(objective.v_max, profile["voltage_q50_v"])))
    cv_voltage_v = float(cv_cell_v * n_series)
    v_cell_min = float(max(1.8, profile["voltage_q01_v"] - 0.05))
    v_cell_max = float(max(cv_cell_v + 0.08, profile["voltage_q99_v"] + 0.05))

    pack_config = PackConfiguration(
        n_series=n_series,
        n_parallel=n_parallel,
        balancing_type=run_config.balancing_type,
        V_pack_max=float(v_cell_max * n_series),
        V_pack_min=float(v_cell_min * n_series),
    )
    capacity_ah = pack_config.get_total_capacity()
    objective_current_limit_a = objective.i_max_c_rate * capacity_ah
    data_current_limit_a = float(max(0.20, profile["current_abs_q95_a"] * 1.10 * n_parallel))
    max_charge_current_a = float(
        min(run_config.max_charge_current_a, objective_current_limit_a, data_current_limit_a)
    )
    max_charge_current_a = float(max(0.20, max_charge_current_a))
    pack_config.I_pack_max = float(max(2.0, 1.2 * max_charge_current_a))

    return SAERLScenarioSetting(
        objective_key=objective_key,
        objective=objective,
        scenario=scenario,
        pack_config=pack_config,
        cv_cell_v=cv_cell_v,
        cv_voltage_v=cv_voltage_v,
        max_charge_current_a=max_charge_current_a,
        initial_soc=float(run_config.initial_soc),
        initial_temp_c=float(profile["initial_temp_c"]),
        ambient_temp_c=float(profile["ambient_temp_c"]),
        dt_s=float(profile["dt_s"]),
        source_is_cell_level=bool(run_config.data_is_cell_level),
    )


def make_env(setting: SAERLScenarioSetting, max_steps: int, target_soc: float) -> HAMBRLPackEnvironment:
    env = HAMBRLPackEnvironment(
        pack_config=copy.deepcopy(setting.pack_config),
        max_steps=max_steps,
        target_soc=target_soc,
        ambient_temp=setting.ambient_temp_c,
        max_charge_current_a=setting.max_charge_current_a,
        dt=setting.dt_s,
    )
    apply_fitted_params_to_pack(
        env.pack,
        setting.scenario.get("fitted_payload"),
        source_is_cell_level=setting.source_is_cell_level,
    )
    apply_data_profile_to_pack(
        env.pack,
        profile=setting.scenario.get("profile"),
        cv_cell_v=setting.cv_cell_v,
    )
    return env


def initial_state_from_env(env: HAMBRLPackEnvironment) -> Dict[str, Any]:
    return {
        "pack_soc": float(env.pack.pack_soc),
        "pack_voltage": float(env.pack.pack_voltage),
        "pack_temperature": float(env.pack.pack_temperature),
        "voltage_imbalance": float(env.pack.voltage_imbalance),
        "pack_current": float(env.pack.pack_current),
        "safety_events": env.pack.safety_events,
    }


def compute_q_loss_total(env: HAMBRLPackEnvironment) -> float:
    if not env.pack.cells:
        return 0.0
    return float(sum(float(cell.Q_loss) for cell in env.pack.cells))


def compute_extended_metrics(
    results: pd.DataFrame,
    target_soc: float,
    initial_soc: float,
    controller_name: str,
) -> Dict[str, float]:
    metrics = compute_metrics(results=results, target_soc=target_soc)
    if not metrics:
        return {}
    if "q_loss_total" in results.columns:
        q_loss_total = float(results["q_loss_total"].iloc[-1])
    else:
        q_loss_total = float("nan")
    charge_time_min = float(metrics.get("charge_time_min", np.nan))
    soc_gain = float(metrics.get("final_soc", np.nan) - initial_soc)
    shield_rate = (
        float(np.mean(results["shield_used"].astype(float)))
        if "shield_used" in results.columns and len(results)
        else 0.0
    )
    antistall_rate = (
        float(np.mean(results["antistall_used"].astype(float)))
        if "antistall_used" in results.columns and len(results)
        else 0.0
    )
    latency_mean_ms = (
        float(np.mean(results["inference_latency_ms"].to_numpy(dtype=float)))
        if "inference_latency_ms" in results.columns and len(results)
        else float("nan")
    )
    latency_p95_ms = (
        float(np.quantile(results["inference_latency_ms"].to_numpy(dtype=float), 0.95))
        if "inference_latency_ms" in results.columns and len(results)
        else float("nan")
    )
    metrics["q_loss_total"] = q_loss_total
    metrics["q_loss_rate"] = (
        float(q_loss_total / max(charge_time_min, 1e-6)) if np.isfinite(q_loss_total) else float("nan")
    )
    metrics["shield_intervention_rate"] = shield_rate
    metrics["antistall_intervention_rate"] = antistall_rate
    metrics["soc_gain_per_min"] = float(soc_gain / max(charge_time_min, 1e-6))
    metrics["inference_latency_mean_ms"] = latency_mean_ms
    metrics["inference_latency_p95_ms"] = latency_p95_ms
    metrics["controller"] = controller_name
    return metrics


def safety_count_from_state(state: Dict[str, Any]) -> int:
    return int(count_safety_events(state.get("safety_events", {})))


def apply_domain_randomization(env: HAMBRLPackEnvironment, rng: np.random.Generator) -> None:
    """Randomize a subset of thermal/ohmic factors before an episode."""
    scale_r = float(rng.uniform(0.92, 1.10))
    scale_th = float(rng.uniform(0.90, 1.10))
    ambient_shift = float(rng.uniform(-2.0, 2.0))
    env.ambient_temp = float(env.ambient_temp + ambient_shift)
    for cell in env.pack.cells:
        cell.params.R0 = float(max(1e-5, cell.params.R0 * scale_r))
        cell.params.R1 = float(max(1e-5, cell.params.R1 * scale_r))
        cell.params.R2 = float(max(1e-5, cell.params.R2 * scale_r))
        cell.params.C_th = float(max(1.0, cell.params.C_th * scale_th))
        cell.params.hA = float(max(1e-3, cell.params.hA * scale_th))
    env.pack._update_pack_state()


def deterministic_hash_float(key: str) -> float:
    import hashlib

    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 1000000) / 1000000.0


def build_leave_case_out_folds(
    episodes_df: pd.DataFrame,
    n_folds: int = 3,
) -> Dict[str, Any]:
    """Create folds with per-family held-out cases and train/val/internal splits."""
    if episodes_df.empty:
        return {"folds": []}

    family_to_cases: Dict[str, List[str]] = {}
    for family, group in episodes_df.groupby("dataset_family"):
        family_to_cases[str(family)] = sorted(group["dataset_case"].astype(str).unique().tolist())

    folds: List[Dict[str, Any]] = []
    for fold_idx in range(n_folds):
        test_case_by_family: Dict[str, str] = {}
        for family, cases in family_to_cases.items():
            if not cases:
                continue
            test_case_by_family[family] = cases[fold_idx % len(cases)]

        split_rows: Dict[str, List[str]] = {"train": [], "val": [], "internal_test": [], "case_test": []}
        for _, row in episodes_df.iterrows():
            episode_id = str(row["episode_id"])
            family = str(row["dataset_family"])
            case = str(row["dataset_case"])
            if test_case_by_family.get(family) == case:
                split_rows["case_test"].append(episode_id)
                continue

            r = deterministic_hash_float(f"fold={fold_idx}|episode={episode_id}")
            if r < 0.70:
                split_rows["train"].append(episode_id)
            elif r < 0.85:
                split_rows["val"].append(episode_id)
            else:
                split_rows["internal_test"].append(episode_id)

        folds.append(
            {
                "fold_id": fold_idx,
                "test_case_by_family": test_case_by_family,
                "splits": split_rows,
            }
        )
    return {"n_folds": n_folds, "folds": folds}
