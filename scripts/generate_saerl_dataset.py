"""Generate SAERL phase-2 dataset with mixed behavior and CEM labels."""

from __future__ import annotations

import copy
import json
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from controllers.adaptive_ensemble_rl import SAERLConfig, window_to_sequence
from pack_experiments import build_default_objectives
from scripts.run_baseline_benchmarks import CCCVConfig, CCCVController, RolloutMPCController, count_safety_events, trim_pack_histories
from scripts.saerl_common import (
    build_leave_case_out_folds,
    build_setting_for_objective,
    chemistry_aware_mpc_config,
    get_context_columns,
    initial_state_from_env,
    load_data_calibrated_scenarios,
    make_env,
    recommend_episode_max_steps,
)


@dataclass
class DatasetConfig:
    output_csv: str = "data/training/saerl_phase2_dataset.csv"
    output_meta_json: str = "data/training/saerl_phase2_dataset_meta.json"
    split_manifest_json: str = "data/training/saerl_phase2_splits.json"
    objective: str = "all"
    standardized_root: str = "data/standardized"
    params_root: str = "data/standardized_params"
    dataset_families: str = "nasa,calce,matr"
    exclude_dataset_cases: str = ""
    max_files_per_dataset: int = 3
    episodes_per_setting: int = 2
    max_steps: int = 500
    n_series: int = 20
    n_parallel: int = 1
    max_charge_current_a: float = 10.0
    balancing_type: str = "passive"
    initial_soc: float = 0.2
    target_soc: float = 0.8
    ambient_temp_c: float = 25.0
    mix_mpc: float = 0.50
    mix_mpc_perturb: float = 0.30
    mix_cccv: float = 0.20
    perturb_radius: float = 0.25
    cem_horizon: int = 5
    cem_iterations: int = 3
    cem_population: int = 48
    cem_elite_frac: float = 0.2
    cem_soc_gap_weight: float = 140.0
    cem_voltage_weight: float = 60.0
    cem_temp_weight: float = 35.0
    cem_imbalance_weight: float = 0.004
    cem_current_weight: float = 0.01
    cem_q_loss_weight: float = 12.0
    cem_safety_weight: float = 500.0
    cem_hard_violation_weight: float = 1200.0
    cem_progress_bonus_weight: float = 0.0
    cem_current_floor_fraction: float = 0.0
    cem_current_floor_penalty_weight: float = 0.0
    cem_profile_json: str = ""
    cem_label_interval: int = 1
    baseline_results_root: str = "results/baselines/data_calibrated"
    saerl_mpc_anchor_mode: str = "family_specific"
    context_feature_set: str = "none"
    family_metadata_json: str = "configs/source_family_metadata_v1.json"
    nasa_impedance_root: str = "data/standardized/nasa_impedance"
    adaptive_horizon: bool = True
    min_episode_minutes: float = 120.0
    feasible_time_slack: float = 1.35
    max_steps_cap: int = 5000
    n_folds: int = 3
    random_seed: int = 123
    window_len: int = 20


def parse_args() -> DatasetConfig:
    import argparse

    parser = argparse.ArgumentParser(description="Generate SAERL phase-2 training dataset.")
    parser.add_argument("--output-csv", type=str, default="data/training/saerl_phase2_dataset.csv")
    parser.add_argument(
        "--output-meta-json",
        type=str,
        default="data/training/saerl_phase2_dataset_meta.json",
    )
    parser.add_argument(
        "--split-manifest-json",
        type=str,
        default="data/training/saerl_phase2_splits.json",
    )
    parser.add_argument("--objective", type=str, default="all")
    parser.add_argument("--standardized-root", type=str, default="data/standardized")
    parser.add_argument("--params-root", type=str, default="data/standardized_params")
    parser.add_argument("--dataset-families", type=str, default="nasa,calce,matr")
    parser.add_argument(
        "--exclude-dataset-cases",
        type=str,
        default="",
        help="Optional comma list of family/case_id pairs to skip during scenario selection.",
    )
    parser.add_argument("--max-files-per-dataset", type=int, default=3)
    parser.add_argument("--episodes-per-setting", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--n-series", type=int, default=20)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--max-charge-current-a", type=float, default=10.0)
    parser.add_argument("--balancing-type", type=str, default="passive")
    parser.add_argument("--initial-soc", type=float, default=0.2)
    parser.add_argument("--target-soc", type=float, default=0.8)
    parser.add_argument("--ambient-temp-c", type=float, default=25.0)
    parser.add_argument("--mix-mpc", type=float, default=0.50)
    parser.add_argument("--mix-mpc-perturb", type=float, default=0.30)
    parser.add_argument("--mix-cccv", type=float, default=0.20)
    parser.add_argument("--perturb-radius", type=float, default=0.25)
    parser.add_argument("--cem-horizon", type=int, default=5)
    parser.add_argument("--cem-iterations", type=int, default=3)
    parser.add_argument("--cem-population", type=int, default=48)
    parser.add_argument("--cem-elite-frac", type=float, default=0.2)
    parser.add_argument("--cem-soc-gap-weight", type=float, default=140.0)
    parser.add_argument("--cem-voltage-weight", type=float, default=60.0)
    parser.add_argument("--cem-temp-weight", type=float, default=35.0)
    parser.add_argument("--cem-imbalance-weight", type=float, default=0.004)
    parser.add_argument("--cem-current-weight", type=float, default=0.01)
    parser.add_argument("--cem-q-loss-weight", type=float, default=12.0)
    parser.add_argument("--cem-safety-weight", type=float, default=500.0)
    parser.add_argument("--cem-hard-violation-weight", type=float, default=1200.0)
    parser.add_argument("--cem-progress-bonus-weight", type=float, default=0.0)
    parser.add_argument("--cem-current-floor-fraction", type=float, default=0.0)
    parser.add_argument("--cem-current-floor-penalty-weight", type=float, default=0.0)
    parser.add_argument(
        "--cem-profile-json",
        type=str,
        default="",
        help=(
            "Optional profile JSON to override CEM settings by objective/family. "
            "Supported sections: global, objectives, families, family_objectives."
        ),
    )
    parser.add_argument(
        "--cem-label-interval",
        type=int,
        default=1,
        help="Run CEM every N steps and reuse the last CEM target in between (>=1).",
    )
    parser.add_argument(
        "--baseline-results-root",
        type=str,
        default="results/baselines/data_calibrated",
        help="Baseline metrics root used to detect final_soc_gain-mode scenarios.",
    )
    parser.add_argument(
        "--saerl-mpc-anchor-mode",
        type=str,
        default="family_specific",
        choices=["global", "family_specific", "shared_plus_heads"],
        help="Chemistry-aware MPC anchor mode used by SAERL data generation.",
    )
    parser.add_argument(
        "--context-feature-set",
        type=str,
        default="none",
        choices=["none", "source_v1"],
        help="Optional static context feature set appended as ctx_* columns.",
    )
    parser.add_argument(
        "--family-metadata-json",
        type=str,
        default="configs/source_family_metadata_v1.json",
        help="Family metadata JSON used to build source_v1 context features.",
    )
    parser.add_argument(
        "--nasa-impedance-root",
        type=str,
        default="data/standardized/nasa_impedance",
        help="Optional NASA impedance sidecar root used by source_v1 context features.",
    )
    parser.add_argument(
        "--disable-adaptive-horizon",
        action="store_true",
        help=(
            "Disable horizon scaling based on dt/current feasibility. "
            "By default adaptive horizon is enabled for training data generation."
        ),
    )
    parser.add_argument(
        "--min-episode-minutes",
        type=float,
        default=120.0,
        help="Minimum physical episode duration used by adaptive horizon scaling.",
    )
    parser.add_argument(
        "--feasible-time-slack",
        type=float,
        default=1.35,
        help=(
            "Multiplier on ideal CC time-to-target when computing adaptive episode horizon "
            "(>1 adds taper/degradation slack)."
        ),
    )
    parser.add_argument(
        "--max-steps-cap",
        type=int,
        default=5000,
        help="Upper cap on adaptive max steps per episode (<=0 disables cap).",
    )
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--window-len", type=int, default=20)
    args = parser.parse_args()

    return DatasetConfig(
        output_csv=args.output_csv,
        output_meta_json=args.output_meta_json,
        split_manifest_json=args.split_manifest_json,
        objective=args.objective,
        standardized_root=args.standardized_root,
        params_root=args.params_root,
        dataset_families=args.dataset_families,
        exclude_dataset_cases=args.exclude_dataset_cases,
        max_files_per_dataset=args.max_files_per_dataset,
        episodes_per_setting=args.episodes_per_setting,
        max_steps=args.max_steps,
        n_series=args.n_series,
        n_parallel=args.n_parallel,
        max_charge_current_a=args.max_charge_current_a,
        balancing_type=args.balancing_type,
        initial_soc=args.initial_soc,
        target_soc=args.target_soc,
        ambient_temp_c=args.ambient_temp_c,
        mix_mpc=args.mix_mpc,
        mix_mpc_perturb=args.mix_mpc_perturb,
        mix_cccv=args.mix_cccv,
        perturb_radius=args.perturb_radius,
        cem_horizon=args.cem_horizon,
        cem_iterations=args.cem_iterations,
        cem_population=args.cem_population,
        cem_elite_frac=args.cem_elite_frac,
        cem_soc_gap_weight=args.cem_soc_gap_weight,
        cem_voltage_weight=args.cem_voltage_weight,
        cem_temp_weight=args.cem_temp_weight,
        cem_imbalance_weight=args.cem_imbalance_weight,
        cem_current_weight=args.cem_current_weight,
        cem_q_loss_weight=args.cem_q_loss_weight,
        cem_safety_weight=args.cem_safety_weight,
        cem_hard_violation_weight=args.cem_hard_violation_weight,
        cem_progress_bonus_weight=args.cem_progress_bonus_weight,
        cem_current_floor_fraction=args.cem_current_floor_fraction,
        cem_current_floor_penalty_weight=args.cem_current_floor_penalty_weight,
        cem_profile_json=args.cem_profile_json,
        cem_label_interval=args.cem_label_interval,
        baseline_results_root=args.baseline_results_root,
        saerl_mpc_anchor_mode=args.saerl_mpc_anchor_mode,
        context_feature_set=args.context_feature_set,
        family_metadata_json=args.family_metadata_json,
        nasa_impedance_root=args.nasa_impedance_root,
        adaptive_horizon=not bool(args.disable_adaptive_horizon),
        min_episode_minutes=float(args.min_episode_minutes),
        feasible_time_slack=float(args.feasible_time_slack),
        max_steps_cap=int(args.max_steps_cap),
        n_folds=args.n_folds,
        random_seed=args.random_seed,
        window_len=args.window_len,
    )


def load_optional_json(path: str) -> Dict[str, Any]:
    raw = str(path).strip()
    if not raw:
        return {}
    profile_path = Path(raw)
    if not profile_path.exists():
        raise FileNotFoundError(f"CEM profile JSON not found: {profile_path}")
    with profile_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("CEM profile JSON must be a dict at top level.")
    return payload


def _apply_numeric_overrides(base: Dict[str, float], overrides: Dict[str, Any]) -> None:
    if not isinstance(overrides, dict):
        return
    for key in list(base.keys()):
        if key not in overrides:
            continue
        try:
            value = float(overrides[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            base[key] = value


def resolve_cem_profile(
    config: DatasetConfig,
    profile_payload: Dict[str, Any],
    objective_key: str,
    family: str,
    final_soc_gain_mode: bool = False,
) -> Dict[str, float]:
    profile = {
        "horizon": float(config.cem_horizon),
        "iterations": float(config.cem_iterations),
        "population": float(config.cem_population),
        "elite_frac": float(config.cem_elite_frac),
        "soc_gap_weight": float(config.cem_soc_gap_weight),
        "voltage_weight": float(config.cem_voltage_weight),
        "temp_weight": float(config.cem_temp_weight),
        "imbalance_weight": float(config.cem_imbalance_weight),
        "current_weight": float(config.cem_current_weight),
        "q_loss_weight": float(config.cem_q_loss_weight),
        "safety_weight": float(config.cem_safety_weight),
        "hard_violation_weight": float(config.cem_hard_violation_weight),
        "progress_bonus_weight": float(config.cem_progress_bonus_weight),
        "current_floor_fraction": float(config.cem_current_floor_fraction),
        "current_floor_penalty_weight": float(config.cem_current_floor_penalty_weight),
        "final_soc_gain_soc_gap_mult": 1.0,
        "final_soc_gain_progress_bonus_mult": 1.0,
        "final_soc_gain_current_floor_fraction_add": 0.0,
        "final_soc_gain_current_floor_penalty_mult": 1.0,
        "final_soc_gain_horizon_add": 0.0,
        "final_soc_gain_min_residual": 0.0,
        "final_soc_gain_label_blend": 0.0,
        "final_soc_gain_floor_fraction": 0.0,
        "final_soc_gain_soc_gap_threshold": 0.08,
        "final_soc_gain_low_risk_threshold": 0.45,
        "final_soc_gain_min_residual_low_risk_boost": 0.08,
        "final_soc_gain_label_blend_low_risk_boost": 0.20,
        "final_soc_gain_floor_fraction_low_risk_boost": 0.10,
    }

    fam = str(family).strip().lower()
    obj = str(objective_key).strip().lower()
    _apply_numeric_overrides(profile, profile_payload.get("global", {}))
    objectives = profile_payload.get("objectives", {})
    if isinstance(objectives, dict):
        _apply_numeric_overrides(profile, objectives.get(obj, {}))
        _apply_numeric_overrides(profile, objectives.get(objective_key, {}))
    families = profile_payload.get("families", {})
    if isinstance(families, dict):
        _apply_numeric_overrides(profile, families.get(fam, {}))
        _apply_numeric_overrides(profile, families.get(family, {}))
    fam_obj = profile_payload.get("family_objectives", {})
    if isinstance(fam_obj, dict):
        fam_node = fam_obj.get(fam, fam_obj.get(family, {}))
        if isinstance(fam_node, dict):
            _apply_numeric_overrides(profile, fam_node.get(obj, {}))
            _apply_numeric_overrides(profile, fam_node.get(objective_key, {}))

    profile["horizon"] = float(max(1, int(round(profile["horizon"]))))
    profile["iterations"] = float(max(1, int(round(profile["iterations"]))))
    profile["population"] = float(max(8, int(round(profile["population"]))))
    profile["elite_frac"] = float(np.clip(profile["elite_frac"], 0.05, 0.8))
    non_negative_keys = [
        "soc_gap_weight",
        "voltage_weight",
        "temp_weight",
        "imbalance_weight",
        "current_weight",
        "q_loss_weight",
        "safety_weight",
        "hard_violation_weight",
        "progress_bonus_weight",
        "current_floor_penalty_weight",
        "final_soc_gain_soc_gap_mult",
        "final_soc_gain_progress_bonus_mult",
        "final_soc_gain_current_floor_penalty_mult",
        "final_soc_gain_min_residual_low_risk_boost",
        "final_soc_gain_label_blend_low_risk_boost",
        "final_soc_gain_floor_fraction_low_risk_boost",
    ]
    for key in non_negative_keys:
        profile[key] = float(max(0.0, profile[key]))
    profile["current_floor_fraction"] = float(np.clip(profile["current_floor_fraction"], 0.0, 1.0))

    profile["final_soc_gain_current_floor_fraction_add"] = float(
        np.clip(profile["final_soc_gain_current_floor_fraction_add"], 0.0, 1.0)
    )
    profile["final_soc_gain_horizon_add"] = float(np.clip(profile["final_soc_gain_horizon_add"], -4.0, 8.0))
    profile["final_soc_gain_min_residual"] = float(np.clip(profile["final_soc_gain_min_residual"], 0.0, 1.0))
    profile["final_soc_gain_label_blend"] = float(np.clip(profile["final_soc_gain_label_blend"], 0.0, 1.0))
    profile["final_soc_gain_floor_fraction"] = float(np.clip(profile["final_soc_gain_floor_fraction"], 0.0, 1.0))
    profile["final_soc_gain_soc_gap_threshold"] = float(np.clip(profile["final_soc_gain_soc_gap_threshold"], 0.0, 1.0))
    profile["final_soc_gain_low_risk_threshold"] = float(
        np.clip(profile["final_soc_gain_low_risk_threshold"], 0.0, 1.0)
    )
    profile["final_soc_gain_min_residual_low_risk_boost"] = float(
        np.clip(profile["final_soc_gain_min_residual_low_risk_boost"], 0.0, 1.0)
    )
    profile["final_soc_gain_label_blend_low_risk_boost"] = float(
        np.clip(profile["final_soc_gain_label_blend_low_risk_boost"], 0.0, 1.0)
    )
    profile["final_soc_gain_floor_fraction_low_risk_boost"] = float(
        np.clip(profile["final_soc_gain_floor_fraction_low_risk_boost"], 0.0, 1.0)
    )

    if final_soc_gain_mode:
        profile["soc_gap_weight"] = float(profile["soc_gap_weight"] * profile["final_soc_gain_soc_gap_mult"])
        profile["progress_bonus_weight"] = float(
            profile["progress_bonus_weight"] * profile["final_soc_gain_progress_bonus_mult"]
        )
        profile["current_floor_penalty_weight"] = float(
            profile["current_floor_penalty_weight"] * profile["final_soc_gain_current_floor_penalty_mult"]
        )
        profile["current_floor_fraction"] = float(
            np.clip(
                profile["current_floor_fraction"] + profile["final_soc_gain_current_floor_fraction_add"],
                0.0,
                1.0,
            )
        )
        profile["horizon"] = float(
            max(1, int(round(profile["horizon"] + profile["final_soc_gain_horizon_add"])))
        )
    return profile


def load_final_soc_gain_mode_map(
    baseline_root: str,
    target_soc: float,
) -> Dict[Tuple[str, str, str], bool]:
    root = Path(str(baseline_root))
    mode_map: Dict[Tuple[str, str, str], bool] = {}
    if not root.exists():
        return mode_map

    grouped: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for metrics_path in sorted(root.glob("*/*/*/*/metrics.json")):
        objective = str(metrics_path.parents[3].name).strip().lower()
        family = str(metrics_path.parents[2].name).strip().lower()
        case = str(metrics_path.parents[1].name).strip()
        controller = str(metrics_path.parents[0].name).strip().lower()
        if controller not in {"cccv", "mpc"}:
            continue
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        metrics = payload.get("metrics", payload)
        try:
            final_soc = float(metrics.get("final_soc", np.nan))
        except Exception:
            final_soc = float("nan")
        if np.isfinite(final_soc):
            grouped.setdefault((objective, family, case), {})[controller] = final_soc

    for key, record in grouped.items():
        cccv_soc = float(record.get("cccv", np.nan))
        mpc_soc = float(record.get("mpc", np.nan))
        hits_target = bool(
            (np.isfinite(cccv_soc) and cccv_soc >= target_soc)
            or (np.isfinite(mpc_soc) and mpc_soc >= target_soc)
        )
        mode_map[key] = bool(not hits_target)
    return mode_map


def apply_final_soc_gain_label_adjustment(
    env,
    target_action: float,
    mpc_action: float,
    current_soc: float,
    target_soc: float,
    cem_profile: Dict[str, float],
    final_soc_gain_mode: bool,
    risk_hint: float | None = None,
) -> float:
    if not final_soc_gain_mode:
        return float(np.clip(target_action, -1.0, 1.0))

    soc_gap = max(0.0, float(target_soc) - float(current_soc))
    if soc_gap < float(cem_profile["final_soc_gain_soc_gap_threshold"]):
        return float(np.clip(target_action, -1.0, 1.0))

    adjusted = float(target_action)
    risk_val = float(1.0 if risk_hint is None else np.clip(risk_hint, 0.0, 1.0))
    low_risk = bool(risk_val <= float(cem_profile["final_soc_gain_low_risk_threshold"]))

    floor_fraction = float(cem_profile["final_soc_gain_floor_fraction"])
    if low_risk:
        floor_fraction += float(cem_profile["final_soc_gain_floor_fraction_low_risk_boost"])
    if floor_fraction > 0.0:
        floor_current = -float(np.clip(floor_fraction, 0.0, 1.0) * env.max_charge_current_a)
        floor_action = float(env.pack_current_to_action(floor_current))
        guided = max(adjusted, floor_action)
        blend = float(np.clip(cem_profile["final_soc_gain_label_blend"], 0.0, 1.0))
        if low_risk:
            blend += float(cem_profile["final_soc_gain_label_blend_low_risk_boost"])
        blend = float(np.clip(blend, 0.0, 1.0))
        adjusted = float((1.0 - blend) * adjusted + blend * guided)

    min_residual = float(np.clip(cem_profile["final_soc_gain_min_residual"], 0.0, 1.0))
    if low_risk:
        min_residual += float(cem_profile["final_soc_gain_min_residual_low_risk_boost"])
    min_residual = float(np.clip(min_residual, 0.0, 1.0))
    if min_residual > 0.0:
        adjusted = max(adjusted, float(mpc_action) + min_residual)
    return float(np.clip(adjusted, -1.0, 1.0))


def estimate_final_soc_gain_risk(
    state: Dict[str, Any],
    cv_voltage_v: float,
    temp_soft_limit_c: float = 42.0,
    imbalance_margin_v: float = 0.095,
) -> float:
    pack_v = float(state.get("pack_voltage", 0.0))
    pack_t = float(state.get("pack_temperature", 25.0))
    imbalance_v = abs(float(state.get("voltage_imbalance", 0.0)))
    events = state.get("safety_events", {})

    v_ratio = max(0.0, (pack_v - (float(cv_voltage_v) - 0.12)) / 0.25)
    t_ratio = max(0.0, (pack_t - (float(temp_soft_limit_c) - 2.0)) / 2.5)
    i_ratio = max(0.0, (imbalance_v - 0.7 * float(imbalance_margin_v)) / max(1e-6, 0.4 * float(imbalance_margin_v)))
    event_flag = 0.0
    if isinstance(events, dict):
        if bool(events.get("pack_over_voltage", False)) or bool(events.get("pack_under_voltage", False)):
            event_flag = 1.0
        elif bool(events.get("current_limit", False)):
            event_flag = 1.0
        elif len(events.get("over_temperature_cells", [])) > 0:
            event_flag = 1.0

    risk = 0.45 * v_ratio + 0.35 * t_ratio + 0.20 * i_ratio + event_flag
    return float(np.clip(risk, 0.0, 1.0))


def normalize_mix_weights(config: DatasetConfig) -> Dict[str, float]:
    weights = {
        "mpc": max(0.0, float(config.mix_mpc)),
        "mpc_perturb": max(0.0, float(config.mix_mpc_perturb)),
        "cccv": max(0.0, float(config.mix_cccv)),
    }
    s = sum(weights.values())
    if s <= 1e-8:
        return {"mpc": 1.0, "mpc_perturb": 0.0, "cccv": 0.0}
    return {k: float(v / s) for k, v in weights.items()}


def choose_behavior_policy(rng: np.random.Generator, mix: Dict[str, float]) -> str:
    r = float(rng.uniform(0.0, 1.0))
    if r < mix["mpc"]:
        return "mpc"
    if r < mix["mpc"] + mix["mpc_perturb"]:
        return "mpc_perturb"
    return "cccv"


def is_safe_action(env, action: float, cv_voltage_v: float, temp_soft_limit_c: float = 42.0) -> bool:
    test_pack = copy.deepcopy(env.pack)
    pack_current = env.action_to_pack_current(action)
    next_state = test_pack.step(pack_current, ambient_temp=env.ambient_temp)
    if float(next_state["pack_voltage"]) > min(env.pack.config.V_pack_max, cv_voltage_v + 0.03):
        return False
    if float(next_state["pack_temperature"]) > min(env.pack.config.T_pack_max, temp_soft_limit_c + 0.5):
        return False
    if abs(float(next_state["pack_current"])) > float(env.pack.config.I_pack_max):
        return False
    events = next_state.get("safety_events", {})
    if isinstance(events, dict):
        if events.get("pack_over_voltage", False) or events.get("pack_under_voltage", False):
            return False
        if events.get("current_limit", False):
            return False
        if len(events.get("over_temperature_cells", [])) > 0:
            return False
    return True


def sequence_cost(
    env,
    action_seq: np.ndarray,
    cv_voltage_v: float,
    target_soc: float,
    start_soc: float,
    cem_profile: Dict[str, float],
) -> float:
    test_pack = copy.deepcopy(env.pack)
    total_cost = 0.0
    prev_soc = float(np.clip(start_soc, 0.0, 1.0))
    current_floor_a = float(np.clip(cem_profile["current_floor_fraction"], 0.0, 1.0) * env.max_charge_current_a)
    for action in action_seq:
        pack_current = env.action_to_pack_current(float(action))
        next_state = test_pack.step(pack_current, ambient_temp=env.ambient_temp)
        next_soc = float(next_state["pack_soc"])
        soc_gap = max(0.0, target_soc - next_soc)
        soc_progress = max(0.0, next_soc - prev_soc)
        over_v = max(0.0, float(next_state["pack_voltage"]) - cv_voltage_v)
        over_t = max(0.0, float(next_state["pack_temperature"]) - 42.0)
        imbalance_mv = float(next_state["voltage_imbalance"]) * 1000.0
        current_a = -float(next_state["pack_current"])
        low_current_penalty = max(0.0, current_floor_a - current_a) if soc_gap > 0.05 else 0.0
        safety = int(count_safety_events(next_state.get("safety_events", {})))
        q_loss = sum(float(cell.Q_loss) for cell in test_pack.cells)
        hard_violation = 1.0 if (over_v > 0.0 or over_t > 0.0 or safety > 0) else 0.0
        total_cost += (
            cem_profile["soc_gap_weight"] * (soc_gap**2)
            + cem_profile["voltage_weight"] * (over_v**2)
            + cem_profile["temp_weight"] * (over_t**2)
            + cem_profile["imbalance_weight"] * imbalance_mv
            + cem_profile["current_weight"] * current_a
            + cem_profile["q_loss_weight"] * q_loss
            + cem_profile["safety_weight"] * safety
            + cem_profile["hard_violation_weight"] * hard_violation
            + cem_profile["current_floor_penalty_weight"] * low_current_penalty
            - cem_profile["progress_bonus_weight"] * soc_progress
        )
        prev_soc = next_soc
    return float(total_cost)


def cem_target_action(
    env,
    mpc_action: float,
    cv_voltage_v: float,
    target_soc: float,
    current_soc: float,
    cem_profile: Dict[str, float],
    rng: np.random.Generator,
) -> float:
    horizon = int(max(1, round(cem_profile["horizon"])))
    iterations = int(max(1, round(cem_profile["iterations"])))
    population = int(max(8, round(cem_profile["population"])))
    elite_frac = float(np.clip(cem_profile["elite_frac"], 0.05, 0.8))
    mean = np.ones(horizon, dtype=np.float32) * float(np.clip(mpc_action, -1.0, 1.0))
    std = np.ones(horizon, dtype=np.float32) * 0.25
    elite_count = max(4, int(population * elite_frac))
    for _ in range(iterations):
        samples = rng.normal(loc=mean, scale=std, size=(population, horizon)).astype(np.float32)
        samples = np.clip(samples, -1.0, 1.0)
        costs = np.array(
            [
                sequence_cost(
                    env=env,
                    action_seq=samples[i],
                    cv_voltage_v=cv_voltage_v,
                    target_soc=target_soc,
                    start_soc=current_soc,
                    cem_profile=cem_profile,
                )
                for i in range(population)
            ],
            dtype=np.float64,
        )
        elite_idx = np.argsort(costs)[:elite_count]
        elite = samples[elite_idx]
        mean = np.mean(elite, axis=0).astype(np.float32)
        std = np.std(elite, axis=0).astype(np.float32)
        std = np.clip(std, 0.03, 0.40)
    return float(np.clip(mean[0], -1.0, 1.0))


def main() -> None:
    config = parse_args()
    rng = np.random.default_rng(config.random_seed)
    saerl_cfg = SAERLConfig(window_len=config.window_len)
    cem_profile_payload = load_optional_json(config.cem_profile_json)
    final_soc_gain_mode_map = load_final_soc_gain_mode_map(
        baseline_root=config.baseline_results_root,
        target_soc=config.target_soc,
    )
    mix = normalize_mix_weights(config)

    objectives = build_default_objectives()
    objective_keys = list(objectives.keys()) if config.objective == "all" else [config.objective]

    run_config, scenarios = load_data_calibrated_scenarios(
        standardized_root=config.standardized_root,
        params_root=config.params_root,
        dataset_families=config.dataset_families,
        exclude_dataset_cases=config.exclude_dataset_cases,
        max_files_per_dataset=config.max_files_per_dataset,
        n_series=config.n_series,
        n_parallel=config.n_parallel,
        max_charge_current_a=config.max_charge_current_a,
        balancing_type=config.balancing_type,
        initial_soc=config.initial_soc,
        target_soc=config.target_soc,
        ambient_temp_c=config.ambient_temp_c,
        context_feature_set=config.context_feature_set,
        family_metadata_json=config.family_metadata_json,
        nasa_impedance_root=config.nasa_impedance_root,
    )
    if not scenarios:
        raise SystemExit("No data-calibrated scenarios were found.")
    context_columns = get_context_columns(config.context_feature_set)

    rows: List[Dict[str, Any]] = []
    episodes_meta: List[Dict[str, Any]] = []
    episode_counter = 0

    for objective_key in objective_keys:
        objective = objectives[objective_key]
        for scenario in scenarios:
            setting = build_setting_for_objective(
                run_config=run_config,
                objective_key=objective_key,
                objective=objective,
                scenario=scenario,
            )
            family = str(scenario["family"])
            case_id = str(scenario["case_id"])
            final_soc_gain_mode = bool(
                final_soc_gain_mode_map.get(
                    (str(objective_key).strip().lower(), str(family).strip().lower(), str(case_id)),
                    False,
                )
            )
            cem_profile = resolve_cem_profile(
                config=config,
                profile_payload=cem_profile_payload,
                objective_key=objective_key,
                family=family,
                final_soc_gain_mode=final_soc_gain_mode,
            )
            for ep in range(config.episodes_per_setting):
                episode_id = f"ep_{episode_counter:05d}"
                episode_counter += 1
                behavior_policy = choose_behavior_policy(rng=rng, mix=mix)

                if config.adaptive_horizon:
                    episode_max_steps, horizon_info = recommend_episode_max_steps(
                        setting=setting,
                        base_max_steps=config.max_steps,
                        target_soc=config.target_soc,
                        min_episode_minutes=config.min_episode_minutes,
                        feasible_time_slack=config.feasible_time_slack,
                        max_steps_cap=config.max_steps_cap,
                    )
                else:
                    episode_max_steps = int(max(1, config.max_steps))
                    dt_s = float(max(1e-6, setting.dt_s))
                    horizon_info = {
                        "dt_s": dt_s,
                        "base_max_steps": float(config.max_steps),
                        "effective_max_steps": float(episode_max_steps),
                        "base_horizon_s": float(config.max_steps * dt_s),
                        "effective_horizon_s": float(episode_max_steps * dt_s),
                        "ideal_cc_time_to_target_s": float("nan"),
                        "required_current_for_base_horizon_a": float("nan"),
                        "configured_max_charge_current_a": float(setting.max_charge_current_a),
                        "current_feasibility_ratio_vs_base_horizon": float("nan"),
                    }

                env = make_env(setting=setting, max_steps=episode_max_steps, target_soc=config.target_soc)
                env.reset(initial_soc=setting.initial_soc, temperature=setting.initial_temp_c)
                trim_pack_histories(env.pack)
                mpc = RolloutMPCController(
                    config=chemistry_aware_mpc_config(
                        family=family,
                        mode=config.saerl_mpc_anchor_mode,
                        objective_key=objective_key,
                    ),
                    cv_voltage_v=setting.cv_voltage_v,
                    max_charge_current_a=setting.max_charge_current_a,
                    target_soc=config.target_soc,
                )
                cccv = CCCVController(
                    config=CCCVConfig(),
                    cv_voltage_v=setting.cv_voltage_v,
                    max_charge_current_a=setting.max_charge_current_a,
                    target_soc=config.target_soc,
                )
                mpc.reset()
                cccv.reset()

                state = initial_state_from_env(env)
                state_window: Deque[Dict[str, Any]] = deque(maxlen=saerl_cfg.window_len)
                for _ in range(saerl_cfg.window_len):
                    state_window.append(dict(state))

                cem_interval = max(1, int(config.cem_label_interval))
                last_cem_target = 0.0
                for step_idx in range(episode_max_steps):
                    mpc_action, mpc_info = mpc.act(state, env)
                    cccv_action, cccv_info = cccv.act(state, env)

                    if behavior_policy == "mpc":
                        behavior_action = float(mpc_action)
                    elif behavior_policy == "mpc_perturb":
                        noise = float(rng.uniform(-config.perturb_radius, config.perturb_radius))
                        behavior_action = float(np.clip(float(mpc_action) + noise, -1.0, 1.0))
                    else:
                        behavior_action = float(cccv_action)

                    if step_idx % cem_interval == 0:
                        target_action = cem_target_action(
                            env=env,
                            mpc_action=float(mpc_action),
                            cv_voltage_v=setting.cv_voltage_v,
                            target_soc=config.target_soc,
                            current_soc=float(state["pack_soc"]),
                            cem_profile=cem_profile,
                            rng=rng,
                        )
                        last_cem_target = float(target_action)
                    else:
                        target_action = float(last_cem_target)

                    risk_hint = estimate_final_soc_gain_risk(
                        state=state,
                        cv_voltage_v=setting.cv_voltage_v,
                        temp_soft_limit_c=42.0,
                        imbalance_margin_v=0.095,
                    )
                    target_action = apply_final_soc_gain_label_adjustment(
                        env=env,
                        target_action=float(target_action),
                        mpc_action=float(mpc_action),
                        current_soc=float(state["pack_soc"]),
                        target_soc=float(config.target_soc),
                        cem_profile=cem_profile,
                        final_soc_gain_mode=final_soc_gain_mode,
                        risk_hint=risk_hint,
                    )
                    if not is_safe_action(env=env, action=target_action, cv_voltage_v=setting.cv_voltage_v):
                        if is_safe_action(env=env, action=float(mpc_action), cv_voltage_v=setting.cv_voltage_v):
                            target_action = float(mpc_action)
                        elif is_safe_action(env=env, action=float(cccv_action), cv_voltage_v=setting.cv_voltage_v):
                            target_action = float(cccv_action)
                        else:
                            target_action = -1.0
                    last_cem_target = float(target_action)

                    seq = window_to_sequence(
                        state_window=list(state_window),
                        window_len=saerl_cfg.window_len,
                        max_charge_current_a=setting.max_charge_current_a,
                    )
                    _, reward, done, next_state = env.step(behavior_action)
                    trim_pack_histories(env.pack)
                    q_loss_total = float(sum(float(cell.Q_loss) for cell in env.pack.cells))

                    row = {
                        "episode_id": episode_id,
                        "objective": objective_key,
                        "dataset_family": family,
                        "dataset_case": case_id,
                        "behavior_policy": behavior_policy,
                        "label_mode_final_soc_gain": int(final_soc_gain_mode),
                        "step_idx": int(step_idx),
                        "time_s": float(next_state["time"]),
                        "dt_s": float(setting.dt_s),
                        "reward_env": float(reward),
                        "pack_soc": float(state["pack_soc"]),
                        "pack_voltage": float(state["pack_voltage"]),
                        "pack_temperature": float(state["pack_temperature"]),
                        "voltage_imbalance": float(state["voltage_imbalance"]),
                        "pack_current": float(state["pack_current"]),
                        "action_behavior": float(behavior_action),
                        "action_mpc": float(mpc_action),
                        "action_cccv": float(cccv_action),
                        "target_action": float(target_action),
                        "target_delta_action": float(np.clip(target_action - float(mpc_action), -1.0, 1.0)),
                        "next_soc": float(next_state["pack_soc"]),
                        "next_voltage": float(next_state["pack_voltage"]),
                        "next_temp": float(next_state["pack_temperature"]),
                        "next_imbalance": float(next_state["voltage_imbalance"]),
                        "next_q_loss_total": q_loss_total,
                        "safety_event_count": int(count_safety_events(next_state.get("safety_events", {}))),
                        "cv_voltage_v": float(setting.cv_voltage_v),
                        "max_charge_current_a": float(setting.max_charge_current_a),
                        "episode_max_steps": int(episode_max_steps),
                        "horizon_dt_s": float(horizon_info["dt_s"]),
                        "horizon_base_horizon_s": float(horizon_info["base_horizon_s"]),
                        "horizon_effective_horizon_s": float(horizon_info["effective_horizon_s"]),
                        "horizon_ideal_cc_time_to_target_s": float(horizon_info["ideal_cc_time_to_target_s"]),
                        "horizon_feasibility_ratio_vs_base": float(horizon_info["current_feasibility_ratio_vs_base_horizon"]),
                    }
                    if context_columns:
                        source_context = scenario.get("source_context", {}) if isinstance(scenario, dict) else {}
                        if not isinstance(source_context, dict):
                            source_context = {}
                        for col in context_columns:
                            row[col] = float(source_context.get(col, 0.0))
                    for t in range(seq.shape[0]):
                        for f in range(seq.shape[1]):
                            row[f"window_t{t:02d}_f{f:02d}"] = float(seq[t, f])
                    rows.append(row)

                    state = {
                        "pack_soc": float(next_state["pack_soc"]),
                        "pack_voltage": float(next_state["pack_voltage"]),
                        "pack_temperature": float(next_state["pack_temperature"]),
                        "voltage_imbalance": float(next_state["voltage_imbalance"]),
                        "pack_current": float(next_state["pack_current"]),
                        "safety_events": next_state.get("safety_events", {}),
                    }
                    state_window.append(dict(state))
                    if done:
                        break

                episodes_meta.append(
                    {
                        "episode_id": episode_id,
                        "objective": objective_key,
                        "dataset_family": family,
                        "dataset_case": case_id,
                        "behavior_policy": behavior_policy,
                        "label_mode_final_soc_gain": int(final_soc_gain_mode),
                        "n_steps": int(step_idx + 1),
                        "cv_voltage_v": float(setting.cv_voltage_v),
                        "max_charge_current_a": float(setting.max_charge_current_a),
                        "episode_max_steps": int(episode_max_steps),
                        "horizon_dt_s": float(horizon_info["dt_s"]),
                        "horizon_base_horizon_s": float(horizon_info["base_horizon_s"]),
                        "horizon_effective_horizon_s": float(horizon_info["effective_horizon_s"]),
                        "horizon_ideal_cc_time_to_target_s": float(horizon_info["ideal_cc_time_to_target_s"]),
                        "horizon_feasibility_ratio_vs_base": float(horizon_info["current_feasibility_ratio_vs_base_horizon"]),
                    }
                )
                if context_columns:
                    for col in context_columns:
                        episodes_meta[-1][col] = float(
                            scenario.get("source_context", {}).get(col, 0.0)
                            if isinstance(scenario.get("source_context", {}), dict)
                            else 0.0
                        )

    if not rows:
        raise SystemExit("No rows generated.")

    dataset_df = pd.DataFrame(rows)
    episodes_df = pd.DataFrame(episodes_meta)
    split_manifest = build_leave_case_out_folds(episodes_df=episodes_df, n_folds=config.n_folds)

    fold_split_map: Dict[Tuple[str, int], str] = {}
    for fold in split_manifest.get("folds", []):
        fold_id = int(fold["fold_id"])
        split_rows = fold["splits"]
        for split_name, ids in split_rows.items():
            for ep_id in ids:
                fold_split_map[(str(ep_id), fold_id)] = str(split_name)

    for fold in split_manifest.get("folds", []):
        fold_id = int(fold["fold_id"])
        col = f"fold_{fold_id}_split"
        dataset_df[col] = [
            fold_split_map.get((str(ep), fold_id), "train") for ep in dataset_df["episode_id"].tolist()
        ]
        episodes_df[col] = [
            fold_split_map.get((str(ep), fold_id), "train") for ep in episodes_df["episode_id"].tolist()
        ]

    output_csv = Path(config.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset_df.to_csv(output_csv, index=False)
    episodes_csv = output_csv.with_name(output_csv.stem + "_episodes.csv")
    episodes_df.to_csv(episodes_csv, index=False)

    meta = {
        "config": asdict(config),
        "cem_profile_payload": cem_profile_payload,
        "final_soc_gain_mode_map_size": int(len(final_soc_gain_mode_map)),
        "final_soc_gain_episode_count": int(episodes_df["label_mode_final_soc_gain"].astype(int).sum()),
        "n_rows": int(len(dataset_df)),
        "n_episodes": int(len(episodes_df)),
        "objectives": objective_keys,
        "dataset_families": sorted(episodes_df["dataset_family"].astype(str).unique().tolist()),
        "feature_dim": int(saerl_cfg.feature_dim),
        "window_len": int(saerl_cfg.window_len),
        "window_columns": [c for c in dataset_df.columns if c.startswith("window_t")],
        "context_feature_set": str(config.context_feature_set),
        "context_columns": context_columns,
        "context_dim": int(len(context_columns)),
        "target_columns": ["next_soc", "next_voltage", "next_temp", "next_imbalance"],
        "episodes_csv": str(episodes_csv),
    }
    output_meta = Path(config.output_meta_json)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    with output_meta.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=True)

    split_manifest["episodes_csv"] = str(episodes_csv)
    split_manifest["dataset_csv"] = str(output_csv)
    output_split = Path(config.split_manifest_json)
    output_split.parent.mkdir(parents=True, exist_ok=True)
    with output_split.open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2, ensure_ascii=True)

    print("Generated SAERL phase-2 dataset.")
    print(f"Rows: {len(dataset_df)}")
    print(f"Episodes: {len(episodes_df)}")
    print(f"Dataset CSV: {output_csv.resolve()}")
    print(f"Episodes CSV: {episodes_csv.resolve()}")
    print(f"Metadata JSON: {output_meta.resolve()}")
    print(f"Split manifest JSON: {output_split.resolve()}")


if __name__ == "__main__":
    main()
