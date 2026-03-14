"""Common helpers for SAERL phase-2 scripts."""

from __future__ import annotations

import copy
import json
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
    CCCVConfig,
    MPCConfig,
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


SOURCE_CONTEXT_COLUMNS_V1: Tuple[str, ...] = (
    "ctx_family_nasa",
    "ctx_family_calce",
    "ctx_family_matr",
    "ctx_regime_aging_eis",
    "ctx_regime_cycle_cccv",
    "ctx_regime_fast_charge",
    "ctx_nominal_capacity_ah",
    "ctx_source_dt_median_s",
    "ctx_source_dt_q95_s",
    "ctx_source_current_abs_q95_a",
    "ctx_source_temp_present",
    "ctx_source_temp_missing_frac",
    "ctx_source_cycle_index_max",
    "ctx_source_step_index_max",
    "ctx_source_internal_resistance_present",
    "ctx_source_internal_resistance_median_ohm",
    "ctx_source_internal_resistance_q95_ohm",
    "ctx_source_ac_impedance_present",
    "ctx_source_ac_impedance_median_ohm",
    "ctx_source_ac_impedance_q95_ohm",
    "ctx_source_nasa_impedance_present",
    "ctx_source_nasa_rectified_impedance_median_ohm",
    "ctx_source_nasa_rectified_impedance_q95_ohm",
)


def split_csv_arg(value: str) -> List[str]:
    return [x.strip().lower() for x in str(value).split(",") if x.strip()]


def _context_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _clip_float(value: Any, scale: float, lower: float = 0.0, upper: float = 10.0) -> float:
    raw = _context_float(value, default=0.0)
    if scale <= 0:
        return float(np.clip(raw, lower, upper))
    return float(np.clip(raw / float(scale), lower, upper))


def _log_scaled(value: Any, max_value: float) -> float:
    raw = max(0.0, _context_float(value, default=0.0))
    denom = np.log1p(max(float(max_value), 1.0))
    if denom <= 0:
        return 0.0
    return float(np.clip(np.log1p(raw) / denom, 0.0, 1.0))


def load_family_metadata(path: str = "configs/source_family_metadata_v1.json") -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("family metadata JSON must be a dict")
    return payload


def get_context_columns(context_feature_set: str = "none") -> List[str]:
    mode = str(context_feature_set).strip().lower()
    if mode == "source_v1":
        return list(SOURCE_CONTEXT_COLUMNS_V1)
    return []


def build_source_context(
    family: str,
    profile: Dict[str, Any],
    family_metadata: Optional[Dict[str, Any]] = None,
    context_feature_set: str = "none",
) -> Dict[str, float]:
    mode = str(context_feature_set).strip().lower()
    if mode != "source_v1":
        return {}

    family_key = str(family).strip().lower()
    metadata = family_metadata or {}
    family_node = {}
    if isinstance(metadata.get("families"), dict):
        family_node = metadata["families"].get(family_key, {}) or {}
    regime = str(family_node.get("test_regime", "")).strip().lower()

    out = {
        "ctx_family_nasa": 1.0 if family_key == "nasa" else 0.0,
        "ctx_family_calce": 1.0 if family_key == "calce" else 0.0,
        "ctx_family_matr": 1.0 if family_key == "matr" else 0.0,
        "ctx_regime_aging_eis": 1.0 if regime == "aging_eis" else 0.0,
        "ctx_regime_cycle_cccv": 1.0 if regime == "cycle_cccv" else 0.0,
        "ctx_regime_fast_charge": 1.0 if regime == "fast_charge" else 0.0,
        "ctx_nominal_capacity_ah": _clip_float(family_node.get("nominal_capacity_ah", 0.0), scale=5.0, upper=5.0),
        "ctx_source_dt_median_s": _clip_float(profile.get("source_dt_median_s", profile.get("dt_s", 0.0)), scale=60.0, upper=300.0),
        "ctx_source_dt_q95_s": _clip_float(profile.get("source_dt_q95_s", profile.get("dt_s", 0.0)), scale=60.0, upper=300.0),
        "ctx_source_current_abs_q95_a": _clip_float(profile.get("source_current_abs_q95_a", profile.get("current_abs_q95_a", 0.0)), scale=10.0, upper=10.0),
        "ctx_source_temp_present": 1.0 if _context_float(profile.get("source_temp_present", 0.0)) > 0.5 else 0.0,
        "ctx_source_temp_missing_frac": float(np.clip(_context_float(profile.get("source_temp_missing_frac", 1.0), default=1.0), 0.0, 1.0)),
        "ctx_source_cycle_index_max": _log_scaled(profile.get("source_cycle_index_max", 0.0), max_value=5000.0),
        "ctx_source_step_index_max": _log_scaled(profile.get("source_step_index_max", 0.0), max_value=1000.0),
        "ctx_source_internal_resistance_present": 1.0 if _context_float(profile.get("source_internal_resistance_present", 0.0)) > 0.5 else 0.0,
        "ctx_source_internal_resistance_median_ohm": _clip_float(profile.get("source_internal_resistance_median_ohm", 0.0), scale=0.2, upper=10.0),
        "ctx_source_internal_resistance_q95_ohm": _clip_float(profile.get("source_internal_resistance_q95_ohm", 0.0), scale=0.2, upper=10.0),
        "ctx_source_ac_impedance_present": 1.0 if _context_float(profile.get("source_ac_impedance_present", 0.0)) > 0.5 else 0.0,
        "ctx_source_ac_impedance_median_ohm": _clip_float(profile.get("source_ac_impedance_median_ohm", 0.0), scale=0.2, upper=10.0),
        "ctx_source_ac_impedance_q95_ohm": _clip_float(profile.get("source_ac_impedance_q95_ohm", 0.0), scale=0.2, upper=10.0),
        "ctx_source_nasa_impedance_present": 1.0 if _context_float(profile.get("source_nasa_impedance_present", 0.0)) > 0.5 else 0.0,
        "ctx_source_nasa_rectified_impedance_median_ohm": _clip_float(profile.get("source_nasa_rectified_impedance_median_ohm", 0.0), scale=0.2, upper=10.0),
        "ctx_source_nasa_rectified_impedance_q95_ohm": _clip_float(profile.get("source_nasa_rectified_impedance_q95_ohm", 0.0), scale=0.2, upper=10.0),
    }
    return {key: float(out.get(key, 0.0)) for key in SOURCE_CONTEXT_COLUMNS_V1}


def scenario_context_array(
    scenario: Dict[str, Any],
    context_feature_set: str = "none",
    context_columns: Optional[Iterable[str]] = None,
) -> np.ndarray:
    cols = list(context_columns) if context_columns is not None else get_context_columns(context_feature_set)
    if not cols:
        return np.zeros((0,), dtype=np.float32)
    ctx = scenario.get("source_context", {}) if isinstance(scenario, dict) else {}
    if not isinstance(ctx, dict):
        ctx = {}
    return np.asarray([_context_float(ctx.get(col, 0.0)) for col in cols], dtype=np.float32)


def load_data_calibrated_scenarios(
    standardized_root: str = "data/standardized",
    params_root: str = "data/standardized_params",
    dataset_families: str = "nasa,calce,matr",
    exclude_dataset_cases: str = "",
    max_files_per_dataset: int = 3,
    n_series: int = 20,
    n_parallel: int = 1,
    max_charge_current_a: float = 10.0,
    balancing_type: str = "passive",
    initial_soc: float = 0.2,
    target_soc: float = 0.8,
    ambient_temp_c: float = 25.0,
    context_feature_set: str = "none",
    family_metadata_json: str = "configs/source_family_metadata_v1.json",
    nasa_impedance_root: str = "data/standardized/nasa_impedance",
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
        exclude_dataset_cases=exclude_dataset_cases,
        max_files_per_dataset=max_files_per_dataset,
        data_is_cell_level=True,
        include_cell_figures=False,
        nasa_impedance_root=nasa_impedance_root,
    )
    scenarios = collect_data_calibrated_scenarios(run_config)
    context_columns = get_context_columns(context_feature_set)
    family_metadata: Optional[Dict[str, Any]] = None
    if context_columns:
        family_metadata = load_family_metadata(family_metadata_json)
    for scenario in scenarios:
        scenario["source_context"] = build_source_context(
            family=str(scenario.get("family", "")),
            profile=scenario.get("profile", {}),
            family_metadata=family_metadata,
            context_feature_set=context_feature_set,
        )
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


BASELINE_FAMILY_TUNES: Dict[str, Dict[str, float]] = {
    "nasa": {
        "cccv_kp_mult": 1.12,
        "cccv_ki_mult": 1.08,
        "cccv_taper_mult": 0.92,
        "cccv_hyst_mult": 0.90,
        "mpc_w_soc_mult": 1.12,
        "mpc_w_temp_mult": 1.05,
        "mpc_w_imb_mult": 1.10,
        "mpc_horizon_mult": 1.00,
    },
    "calce": {
        "cccv_kp_mult": 1.18,
        "cccv_ki_mult": 1.12,
        "cccv_taper_mult": 0.85,
        "cccv_hyst_mult": 0.85,
        "mpc_w_soc_mult": 1.20,
        "mpc_w_temp_mult": 0.95,
        "mpc_w_imb_mult": 1.05,
        "mpc_horizon_mult": 0.95,
    },
    "matr": {
        "cccv_kp_mult": 0.92,
        "cccv_ki_mult": 0.90,
        "cccv_taper_mult": 1.10,
        "cccv_hyst_mult": 1.10,
        "mpc_w_soc_mult": 1.00,
        "mpc_w_temp_mult": 1.20,
        "mpc_w_imb_mult": 1.20,
        "mpc_horizon_mult": 1.15,
    },
}


BASELINE_OBJECTIVE_TUNES: Dict[str, Dict[str, float]] = {
    "fastest": {
        "cccv_kp_mult": 1.08,
        "cccv_ki_mult": 1.05,
        "cccv_taper_mult": 0.88,
        "cccv_hyst_mult": 0.95,
        "mpc_w_soc_mult": 1.25,
        "mpc_w_temp_mult": 0.90,
        "mpc_w_imb_mult": 0.95,
        "mpc_horizon_mult": 0.90,
    },
    "safe": {
        "cccv_kp_mult": 0.98,
        "cccv_ki_mult": 0.98,
        "cccv_taper_mult": 1.05,
        "cccv_hyst_mult": 1.05,
        "mpc_w_soc_mult": 1.00,
        "mpc_w_temp_mult": 1.15,
        "mpc_w_imb_mult": 1.10,
        "mpc_horizon_mult": 1.05,
    },
    "long_life": {
        "cccv_kp_mult": 0.92,
        "cccv_ki_mult": 0.90,
        "cccv_taper_mult": 1.15,
        "cccv_hyst_mult": 1.12,
        "mpc_w_soc_mult": 0.95,
        "mpc_w_temp_mult": 1.20,
        "mpc_w_imb_mult": 1.25,
        "mpc_horizon_mult": 1.15,
    },
}


def _blend_multiplier(mult: float, alpha: float) -> float:
    return float(1.0 + alpha * (float(mult) - 1.0))


def chemistry_aware_cccv_config(family: str, mode: str, objective_key: str) -> CCCVConfig:
    cfg = CCCVConfig()
    tune = BASELINE_FAMILY_TUNES.get(str(family).lower(), BASELINE_FAMILY_TUNES["nasa"])
    obj_tune = BASELINE_OBJECTIVE_TUNES.get(str(objective_key).lower(), {})
    mode_l = str(mode).strip().lower()
    alpha = 1.0 if mode_l == "family_specific" else (0.5 if mode_l == "shared_plus_heads" else 0.0)
    kp_mult = _blend_multiplier(tune["cccv_kp_mult"], alpha) * float(obj_tune.get("cccv_kp_mult", 1.0))
    ki_mult = _blend_multiplier(tune["cccv_ki_mult"], alpha) * float(obj_tune.get("cccv_ki_mult", 1.0))
    taper_mult = _blend_multiplier(tune["cccv_taper_mult"], alpha) * float(obj_tune.get("cccv_taper_mult", 1.0))
    hyst_mult = _blend_multiplier(tune["cccv_hyst_mult"], alpha) * float(obj_tune.get("cccv_hyst_mult", 1.0))
    cfg.kp_a_per_v *= kp_mult
    cfg.ki_a_per_vs *= ki_mult
    cfg.soc_taper_window *= taper_mult
    cfg.cv_hysteresis_v *= hyst_mult
    cfg.soc_taper_window = float(np.clip(cfg.soc_taper_window, 0.03, 0.20))
    cfg.cv_hysteresis_v = float(np.clip(cfg.cv_hysteresis_v, 0.05, 0.35))
    return cfg


def chemistry_aware_mpc_config(family: str, mode: str, objective_key: str) -> MPCConfig:
    cfg = MPCConfig()
    tune = BASELINE_FAMILY_TUNES.get(str(family).lower(), BASELINE_FAMILY_TUNES["nasa"])
    obj_tune = BASELINE_OBJECTIVE_TUNES.get(str(objective_key).lower(), {})
    mode_l = str(mode).strip().lower()
    alpha = 1.0 if mode_l == "family_specific" else (0.5 if mode_l == "shared_plus_heads" else 0.0)
    w_soc_mult = _blend_multiplier(tune["mpc_w_soc_mult"], alpha) * float(obj_tune.get("mpc_w_soc_mult", 1.0))
    w_temp_mult = _blend_multiplier(tune["mpc_w_temp_mult"], alpha) * float(obj_tune.get("mpc_w_temp_mult", 1.0))
    w_imb_mult = _blend_multiplier(tune["mpc_w_imb_mult"], alpha) * float(obj_tune.get("mpc_w_imb_mult", 1.0))
    horizon_mult = _blend_multiplier(tune["mpc_horizon_mult"], alpha) * float(obj_tune.get("mpc_horizon_mult", 1.0))
    cfg.w_soc *= w_soc_mult
    cfg.w_temp *= w_temp_mult
    cfg.w_imbalance *= w_imb_mult
    cfg.horizon_steps = int(np.clip(round(cfg.horizon_steps * horizon_mult), 4, 16))
    return cfg


def recommend_episode_max_steps(
    setting: SAERLScenarioSetting,
    base_max_steps: int,
    target_soc: float,
    min_episode_minutes: float,
    feasible_time_slack: float,
    max_steps_cap: int,
) -> Tuple[int, Dict[str, float]]:
    base_steps = int(max(1, base_max_steps))
    dt_s = float(max(1e-6, setting.dt_s))
    base_horizon_s = float(base_steps * dt_s)

    capacity_ah = float(max(1e-6, setting.pack_config.get_total_capacity()))
    soc_gap = float(max(0.0, target_soc - setting.initial_soc))
    max_current_a = float(max(1e-6, setting.max_charge_current_a))

    ideal_cc_time_to_target_s = float((soc_gap * capacity_ah * 3600.0) / max_current_a)
    min_horizon_s = float(max(0.0, min_episode_minutes) * 60.0)
    feasible_horizon_s = float(max(0.0, feasible_time_slack) * ideal_cc_time_to_target_s)
    recommended_horizon_s = float(max(base_horizon_s, min_horizon_s, feasible_horizon_s))

    recommended_steps = int(max(base_steps, int(np.ceil(recommended_horizon_s / dt_s))))
    if int(max_steps_cap) > 0:
        recommended_steps = int(min(recommended_steps, int(max_steps_cap)))
        recommended_horizon_s = float(recommended_steps * dt_s)

    required_current_for_base_horizon_a = float(
        (soc_gap * capacity_ah * 3600.0) / max(base_horizon_s, 1e-6)
    )
    feasibility_ratio = float(max_current_a / max(required_current_for_base_horizon_a, 1e-6))

    return recommended_steps, {
        "dt_s": dt_s,
        "base_max_steps": float(base_steps),
        "effective_max_steps": float(recommended_steps),
        "base_horizon_s": base_horizon_s,
        "effective_horizon_s": recommended_horizon_s,
        "ideal_cc_time_to_target_s": ideal_cc_time_to_target_s,
        "required_current_for_base_horizon_a": required_current_for_base_horizon_a,
        "configured_max_charge_current_a": max_current_a,
        "current_feasibility_ratio_vs_base_horizon": feasibility_ratio,
    }


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
