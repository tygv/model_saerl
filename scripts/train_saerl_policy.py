"""Train SAERL residual actor policy (offline warm-start + safe online fine-tune)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from controllers.adaptive_ensemble_rl import (
    AdaptiveEnsemblePredictor,
    ResidualActorPolicy,
    SAERLConfig,
    SafeAdaptiveEnsembleController,
    window_to_sequence,
)
from pack_experiments import build_default_objectives
from scripts.run_baseline_benchmarks import RolloutMPCController, count_safety_events, trim_pack_histories
from scripts.saerl_common import (
    apply_domain_randomization,
    build_setting_for_objective,
    chemistry_aware_mpc_config,
    get_context_columns,
    initial_state_from_env,
    load_data_calibrated_scenarios,
    make_env,
    recommend_episode_max_steps,
    scenario_context_array,
)


@dataclass
class TrainPolicyConfig:
    dataset_csv: str = "data/training/saerl_phase2_dataset.csv"
    split_manifest_json: str = "data/training/saerl_phase2_splits.json"
    ensemble_root: str = "models/saerl_phase2/ensemble"
    output_root: str = "models/saerl_phase2/policy"
    reports_root: str = "results/saerl_phase2/training/policy"
    fold: str = "all"
    objective: str = "all"
    standardized_root: str = "data/standardized"
    params_root: str = "data/standardized_params"
    dataset_families: str = "nasa,calce,matr"
    exclude_dataset_cases: str = ""
    max_files_per_dataset: int = 3
    random_seed: int = 123
    offline_epochs: int = 20
    online_epochs: int = 12
    ppo_epochs: int = 4
    batch_size: int = 256
    learning_rate: float = 8e-4
    weight_decay: float = 1e-5
    gamma: float = 0.98
    clip_eps: float = 0.2
    entropy_coef: float = 1e-3
    online_rollout_episodes: int = 4
    online_max_steps: int = 220
    adaptive_online_horizon: bool = True
    min_episode_minutes: float = 120.0
    feasible_time_slack: float = 1.35
    max_steps_cap: int = 5000
    target_soc: float = 0.8
    n_series: int = 20
    n_parallel: int = 1
    max_charge_current_a: float = 10.0
    balancing_type: str = "passive"
    initial_soc: float = 0.2
    ambient_temp_c: float = 25.0
    early_stop_patience: int = 4
    window_len: int = 20
    feature_dim: int = 6
    reward_soc_gain_weight: float = 18.0
    reward_step_penalty: float = 0.01
    reward_temp_penalty_weight: float = 0.04
    reward_imbalance_penalty_weight: float = 0.001
    reward_q_loss_penalty_weight: float = 2.0
    reward_safety_penalty_weight: float = 3.0
    eval_score_soc_weight: float = 120.0
    eval_score_time_weight: float = 0.35
    eval_score_safety_weight: float = 2.0
    saerl_score_soc_gain_weight: float = 120.0
    saerl_anti_stall_soc_gap: float = 0.25
    saerl_anti_stall_low_risk_threshold: float = 0.35
    saerl_anti_stall_duration_s: float = 120.0
    saerl_anti_stall_risk_scale: float = 3.0
    saerl_min_safe_charge_fraction: float = 0.20
    saerl_enable_antistall: bool = True
    saerl_score_time_weight: float = 1.0
    saerl_score_temp_weight: float = 30.0
    saerl_score_degradation_weight: float = 15.0
    saerl_score_imbalance_weight: float = 0.005
    saerl_score_safety_weight: float = 300.0
    saerl_score_risk_weight: float = 1.0
    saerl_imbalance_margin_v: float = 0.095
    baseline_results_root: str = "results/baselines/data_calibrated"
    reward_acceptance_progress_weight: float = 0.0
    reward_acceptance_terminal_weight: float = 0.0
    reward_acceptance_success_bonus: float = 0.0
    reward_acceptance_time_slack: float = 1.10
    reward_acceptance_soc_margin: float = 0.02
    reward_acceptance_start_fraction: float = 0.35
    reward_acceptance_final_soc_progress_mult: float = 1.0
    reward_acceptance_final_soc_terminal_mult: float = 1.0
    reward_acceptance_final_soc_success_bonus_mult: float = 1.0
    chemistry_mode: str = "global"
    chemistry_families: str = ""
    saerl_mpc_anchor_mode: str = "family_specific"
    context_feature_set: str = "none"
    family_metadata_json: str = "configs/source_family_metadata_v1.json"
    nasa_impedance_root: str = "data/standardized/nasa_impedance"
    init_actor_root: str = ""
    saerl_family_profile_json: str = ""
    objective_specific_heads: bool = True
    allow_global_actor_init: bool = False
    antistall_calibration_risk_quantile: float = 0.75
    antistall_calibration_floor_quantile: float = 0.65
    antistall_calibration_min_samples: int = 48
    antistall_calibration_max_rows: int = 2500
    antistall_calibration_soc_gap_mult: float = 0.75


def parse_args() -> TrainPolicyConfig:
    import argparse

    parser = argparse.ArgumentParser(description="Train SAERL actor policy.")
    parser.add_argument("--dataset-csv", type=str, default="data/training/saerl_phase2_dataset.csv")
    parser.add_argument("--split-manifest-json", type=str, default="data/training/saerl_phase2_splits.json")
    parser.add_argument("--ensemble-root", type=str, default="models/saerl_phase2/ensemble")
    parser.add_argument("--output-root", type=str, default="models/saerl_phase2/policy")
    parser.add_argument("--reports-root", type=str, default="results/saerl_phase2/training/policy")
    parser.add_argument("--fold", type=str, default="all")
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
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--offline-epochs", type=int, default=20)
    parser.add_argument("--online-epochs", type=int, default=12)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--online-rollout-episodes", type=int, default=4)
    parser.add_argument("--online-max-steps", type=int, default=220)
    parser.add_argument(
        "--disable-adaptive-online-horizon",
        action="store_true",
        help=(
            "Disable horizon scaling based on dt/current feasibility for online SAERL training. "
            "By default adaptive online horizon is enabled."
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
        help="Upper cap on adaptive max steps per scenario (<=0 disables cap).",
    )
    parser.add_argument("--target-soc", type=float, default=0.8)
    parser.add_argument("--n-series", type=int, default=20)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--max-charge-current-a", type=float, default=10.0)
    parser.add_argument("--balancing-type", type=str, default="passive")
    parser.add_argument("--initial-soc", type=float, default=0.2)
    parser.add_argument("--ambient-temp-c", type=float, default=25.0)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--window-len", type=int, default=20)
    parser.add_argument("--feature-dim", type=int, default=6)
    parser.add_argument("--reward-soc-gain-weight", type=float, default=18.0)
    parser.add_argument("--reward-step-penalty", type=float, default=0.01)
    parser.add_argument("--reward-temp-penalty-weight", type=float, default=0.04)
    parser.add_argument("--reward-imbalance-penalty-weight", type=float, default=0.001)
    parser.add_argument("--reward-q-loss-penalty-weight", type=float, default=2.0)
    parser.add_argument("--reward-safety-penalty-weight", type=float, default=3.0)
    parser.add_argument("--eval-score-soc-weight", type=float, default=120.0)
    parser.add_argument("--eval-score-time-weight", type=float, default=0.35)
    parser.add_argument("--eval-score-safety-weight", type=float, default=2.0)
    parser.add_argument("--saerl-score-soc-gain-weight", type=float, default=120.0)
    parser.add_argument("--saerl-anti-stall-soc-gap", type=float, default=0.25)
    parser.add_argument("--saerl-anti-stall-low-risk-threshold", type=float, default=0.35)
    parser.add_argument("--saerl-anti-stall-duration-s", type=float, default=120.0)
    parser.add_argument("--saerl-anti-stall-risk-scale", type=float, default=3.0)
    parser.add_argument("--saerl-min-safe-charge-fraction", type=float, default=0.20)
    parser.add_argument("--saerl-enable-antistall", type=int, choices=[0, 1], default=1)
    parser.add_argument("--saerl-score-time-weight", type=float, default=1.0)
    parser.add_argument("--saerl-score-temp-weight", type=float, default=30.0)
    parser.add_argument("--saerl-score-degradation-weight", type=float, default=15.0)
    parser.add_argument("--saerl-score-imbalance-weight", type=float, default=0.005)
    parser.add_argument("--saerl-score-safety-weight", type=float, default=300.0)
    parser.add_argument("--saerl-score-risk-weight", type=float, default=1.0)
    parser.add_argument("--saerl-imbalance-margin-v", type=float, default=0.095)
    parser.add_argument("--baseline-results-root", type=str, default="results/baselines/data_calibrated")
    parser.add_argument("--reward-acceptance-progress-weight", type=float, default=0.0)
    parser.add_argument("--reward-acceptance-terminal-weight", type=float, default=0.0)
    parser.add_argument("--reward-acceptance-success-bonus", type=float, default=0.0)
    parser.add_argument("--reward-acceptance-time-slack", type=float, default=1.10)
    parser.add_argument("--reward-acceptance-soc-margin", type=float, default=0.02)
    parser.add_argument("--reward-acceptance-start-fraction", type=float, default=0.35)
    parser.add_argument("--reward-acceptance-final-soc-progress-mult", type=float, default=1.0)
    parser.add_argument("--reward-acceptance-final-soc-terminal-mult", type=float, default=1.0)
    parser.add_argument("--reward-acceptance-final-soc-success-bonus-mult", type=float, default=1.0)
    parser.add_argument(
        "--chemistry-mode",
        type=str,
        default="global",
        choices=["global", "family_specific", "shared_plus_heads"],
        help="Checkpoint layout mode for chemistry-aware policy training.",
    )
    parser.add_argument(
        "--chemistry-families",
        type=str,
        default="",
        help="Optional comma list of chemistry families (defaults to dataset families in CSV).",
    )
    parser.add_argument(
        "--saerl-mpc-anchor-mode",
        type=str,
        default="family_specific",
        choices=["global", "family_specific", "shared_plus_heads"],
        help="Chemistry-aware MPC anchor mode used by SAERL residual training.",
    )
    parser.add_argument(
        "--context-feature-set",
        type=str,
        default="none",
        choices=["none", "source_v1"],
        help="Optional static context feature set sourced from ctx_* columns.",
    )
    parser.add_argument(
        "--family-metadata-json",
        type=str,
        default="configs/source_family_metadata_v1.json",
        help="Family metadata JSON used to build source_v1 scenario context.",
    )
    parser.add_argument(
        "--nasa-impedance-root",
        type=str,
        default="data/standardized/nasa_impedance",
        help="Optional NASA impedance sidecar root used by source_v1 context.",
    )
    parser.add_argument(
        "--init-actor-root",
        type=str,
        default="",
        help="Optional actor checkpoint root to warm-start from.",
    )
    parser.add_argument(
        "--saerl-family-profile-json",
        type=str,
        default="",
        help=(
            "Optional JSON with family/objective SAERL overrides. "
            "Supported sections: defaults, objectives, families, family_objectives."
        ),
    )
    parser.add_argument(
        "--objective-specific-heads",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "When objective=all and chemistry mode is non-global, train one actor head per objective "
            "under objective-namespaced output/report roots."
        ),
    )
    parser.add_argument(
        "--allow-global-actor-init",
        type=int,
        choices=[0, 1],
        default=0,
        help="Allow warm-start from non-chemistry/global actor root for chemistry-aware policy heads.",
    )
    parser.add_argument("--antistall-calibration-risk-quantile", type=float, default=0.75)
    parser.add_argument("--antistall-calibration-floor-quantile", type=float, default=0.65)
    parser.add_argument("--antistall-calibration-min-samples", type=int, default=48)
    parser.add_argument("--antistall-calibration-max-rows", type=int, default=2500)
    parser.add_argument("--antistall-calibration-soc-gap-mult", type=float, default=0.75)
    args = parser.parse_args()

    return TrainPolicyConfig(
        dataset_csv=args.dataset_csv,
        split_manifest_json=args.split_manifest_json,
        ensemble_root=args.ensemble_root,
        output_root=args.output_root,
        reports_root=args.reports_root,
        fold=args.fold,
        objective=args.objective,
        standardized_root=args.standardized_root,
        params_root=args.params_root,
        dataset_families=args.dataset_families,
        exclude_dataset_cases=args.exclude_dataset_cases,
        max_files_per_dataset=args.max_files_per_dataset,
        random_seed=args.random_seed,
        offline_epochs=args.offline_epochs,
        online_epochs=args.online_epochs,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        online_rollout_episodes=args.online_rollout_episodes,
        online_max_steps=args.online_max_steps,
        adaptive_online_horizon=not bool(args.disable_adaptive_online_horizon),
        min_episode_minutes=float(args.min_episode_minutes),
        feasible_time_slack=float(args.feasible_time_slack),
        max_steps_cap=int(args.max_steps_cap),
        target_soc=args.target_soc,
        n_series=args.n_series,
        n_parallel=args.n_parallel,
        max_charge_current_a=args.max_charge_current_a,
        balancing_type=args.balancing_type,
        initial_soc=args.initial_soc,
        ambient_temp_c=args.ambient_temp_c,
        early_stop_patience=args.early_stop_patience,
        window_len=args.window_len,
        feature_dim=args.feature_dim,
        reward_soc_gain_weight=args.reward_soc_gain_weight,
        reward_step_penalty=args.reward_step_penalty,
        reward_temp_penalty_weight=args.reward_temp_penalty_weight,
        reward_imbalance_penalty_weight=args.reward_imbalance_penalty_weight,
        reward_q_loss_penalty_weight=args.reward_q_loss_penalty_weight,
        reward_safety_penalty_weight=args.reward_safety_penalty_weight,
        eval_score_soc_weight=args.eval_score_soc_weight,
        eval_score_time_weight=args.eval_score_time_weight,
        eval_score_safety_weight=args.eval_score_safety_weight,
        saerl_score_soc_gain_weight=args.saerl_score_soc_gain_weight,
        saerl_anti_stall_soc_gap=args.saerl_anti_stall_soc_gap,
        saerl_anti_stall_low_risk_threshold=args.saerl_anti_stall_low_risk_threshold,
        saerl_anti_stall_duration_s=args.saerl_anti_stall_duration_s,
        saerl_anti_stall_risk_scale=args.saerl_anti_stall_risk_scale,
        saerl_min_safe_charge_fraction=args.saerl_min_safe_charge_fraction,
        saerl_enable_antistall=bool(args.saerl_enable_antistall),
        saerl_score_time_weight=args.saerl_score_time_weight,
        saerl_score_temp_weight=args.saerl_score_temp_weight,
        saerl_score_degradation_weight=args.saerl_score_degradation_weight,
        saerl_score_imbalance_weight=args.saerl_score_imbalance_weight,
        saerl_score_safety_weight=args.saerl_score_safety_weight,
        saerl_score_risk_weight=args.saerl_score_risk_weight,
        saerl_imbalance_margin_v=args.saerl_imbalance_margin_v,
        baseline_results_root=args.baseline_results_root,
        reward_acceptance_progress_weight=args.reward_acceptance_progress_weight,
        reward_acceptance_terminal_weight=args.reward_acceptance_terminal_weight,
        reward_acceptance_success_bonus=args.reward_acceptance_success_bonus,
        reward_acceptance_time_slack=args.reward_acceptance_time_slack,
        reward_acceptance_soc_margin=args.reward_acceptance_soc_margin,
        reward_acceptance_start_fraction=args.reward_acceptance_start_fraction,
        reward_acceptance_final_soc_progress_mult=args.reward_acceptance_final_soc_progress_mult,
        reward_acceptance_final_soc_terminal_mult=args.reward_acceptance_final_soc_terminal_mult,
        reward_acceptance_final_soc_success_bonus_mult=args.reward_acceptance_final_soc_success_bonus_mult,
        chemistry_mode=args.chemistry_mode,
        chemistry_families=args.chemistry_families,
        saerl_mpc_anchor_mode=args.saerl_mpc_anchor_mode,
        context_feature_set=args.context_feature_set,
        family_metadata_json=args.family_metadata_json,
        nasa_impedance_root=args.nasa_impedance_root,
        init_actor_root=args.init_actor_root,
        saerl_family_profile_json=args.saerl_family_profile_json,
        objective_specific_heads=bool(args.objective_specific_heads),
        allow_global_actor_init=bool(args.allow_global_actor_init),
        antistall_calibration_risk_quantile=args.antistall_calibration_risk_quantile,
        antistall_calibration_floor_quantile=args.antistall_calibration_floor_quantile,
        antistall_calibration_min_samples=args.antistall_calibration_min_samples,
        antistall_calibration_max_rows=args.antistall_calibration_max_rows,
        antistall_calibration_soc_gap_mult=args.antistall_calibration_soc_gap_mult,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_window_cols(columns: Sequence[str]) -> List[str]:
    cols = [c for c in columns if c.startswith("window_t") and "_f" in c]
    cols.sort()
    return cols


def parse_context_cols(columns: Sequence[str]) -> List[str]:
    cols_set = {str(c) for c in columns if str(c).startswith("ctx_")}
    ordered = [c for c in get_context_columns("source_v1") if c in cols_set]
    extras = sorted([c for c in cols_set if c not in set(ordered)])
    return ordered + extras


def split_csv_arg(value: str) -> List[str]:
    return [x.strip().lower() for x in str(value).split(",") if x.strip()]


PROFILE_OVERRIDE_KEYS: Tuple[str, ...] = (
    "reward_soc_gain_weight",
    "reward_step_penalty",
    "reward_temp_penalty_weight",
    "reward_imbalance_penalty_weight",
    "reward_q_loss_penalty_weight",
    "reward_safety_penalty_weight",
    "eval_score_soc_weight",
    "eval_score_time_weight",
    "eval_score_safety_weight",
    "saerl_score_soc_gain_weight",
    "saerl_anti_stall_soc_gap",
    "saerl_anti_stall_low_risk_threshold",
    "saerl_anti_stall_duration_s",
    "saerl_anti_stall_risk_scale",
    "saerl_min_safe_charge_fraction",
    "saerl_score_time_weight",
    "saerl_score_temp_weight",
    "saerl_score_degradation_weight",
    "saerl_score_imbalance_weight",
    "saerl_score_safety_weight",
    "saerl_score_risk_weight",
    "saerl_imbalance_margin_v",
    "reward_acceptance_progress_weight",
    "reward_acceptance_terminal_weight",
    "reward_acceptance_success_bonus",
    "reward_acceptance_time_slack",
    "reward_acceptance_soc_margin",
    "reward_acceptance_start_fraction",
    "reward_acceptance_final_soc_progress_mult",
    "reward_acceptance_final_soc_terminal_mult",
    "reward_acceptance_final_soc_success_bonus_mult",
)


@dataclass
class BaselineAcceptanceRef:
    best_final_soc: float
    best_hits_target: bool
    fastest_target_time_min: float
    horizon_time_min: float


def load_optional_json(path: str) -> Dict[str, Any]:
    raw = str(path).strip()
    if not raw:
        return {}
    p = Path(raw)
    if not p.exists():
        raise FileNotFoundError(f"Profile JSON not found: {p}")
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Profile JSON must be a dict at top level.")
    return payload


def _apply_profile_overrides(base: Dict[str, float], node: Any) -> None:
    if not isinstance(node, dict):
        return
    for key in PROFILE_OVERRIDE_KEYS:
        if key not in node:
            continue
        try:
            value = float(node[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            base[key] = value


def resolve_profile_overrides(
    profile_payload: Dict[str, Any],
    family: str | None,
    objective: str | None,
) -> Dict[str, float]:
    resolved: Dict[str, float] = {}
    fam = str(family or "").strip().lower()
    obj = str(objective or "").strip().lower()

    _apply_profile_overrides(resolved, profile_payload.get("defaults", {}))
    objectives = profile_payload.get("objectives", {})
    if isinstance(objectives, dict) and obj:
        _apply_profile_overrides(resolved, objectives.get(obj, {}))
        _apply_profile_overrides(resolved, objectives.get(objective, {}))

    families = profile_payload.get("families", {})
    if isinstance(families, dict) and fam:
        fam_node = families.get(fam, families.get(family, {}))
        _apply_profile_overrides(resolved, fam_node)
        if isinstance(fam_node, dict) and obj:
            fam_obj_node = fam_node.get("objectives", {})
            if isinstance(fam_obj_node, dict):
                _apply_profile_overrides(resolved, fam_obj_node.get(obj, {}))
                _apply_profile_overrides(resolved, fam_obj_node.get(objective, {}))

    fam_obj = profile_payload.get("family_objectives", {})
    if isinstance(fam_obj, dict) and fam and obj:
        fam_node = fam_obj.get(fam, fam_obj.get(family, {}))
        if isinstance(fam_node, dict):
            _apply_profile_overrides(resolved, fam_node.get(obj, {}))
            _apply_profile_overrides(resolved, fam_node.get(objective, {}))
    return resolved


def load_baseline_acceptance_refs(
    baseline_root: str,
    target_soc: float,
) -> Dict[Tuple[str, str, str], BaselineAcceptanceRef]:
    root = Path(str(baseline_root))
    refs: Dict[Tuple[str, str, str], BaselineAcceptanceRef] = {}
    if not root.exists():
        return refs

    grouped: Dict[Tuple[str, str, str], Dict[str, Dict[str, float]]] = {}
    for metrics_path in sorted(root.glob("*/*/*/*/metrics.json")):
        objective = str(metrics_path.parents[3].name).lower()
        family = str(metrics_path.parents[2].name).lower()
        case = str(metrics_path.parents[1].name)
        controller = str(metrics_path.parents[0].name).lower()
        if controller not in {"cccv", "mpc"}:
            continue
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        metrics = payload.get("metrics", payload)
        final_soc = float(metrics.get("final_soc", np.nan))
        charge_time = float(metrics.get("charge_time_min", np.nan))
        t80 = float(metrics.get("time_to_80_soc_min", np.nan))
        if not np.isfinite(t80):
            t80 = charge_time
        grouped.setdefault((objective, family, case), {})[controller] = {
            "final_soc": final_soc,
            "charge_time_min": charge_time,
            "time_to_80_soc_min": t80,
        }

    for key, rec in grouped.items():
        cccv = rec.get("cccv", {})
        mpc = rec.get("mpc", {})
        cccv_soc = float(cccv.get("final_soc", np.nan))
        mpc_soc = float(mpc.get("final_soc", np.nan))
        best_final_soc = float(np.nanmax(np.array([cccv_soc, mpc_soc], dtype=float)))
        if not np.isfinite(best_final_soc):
            continue
        cccv_hit = bool(np.isfinite(cccv_soc) and cccv_soc >= target_soc)
        mpc_hit = bool(np.isfinite(mpc_soc) and mpc_soc >= target_soc)
        best_hits_target = bool(cccv_hit or mpc_hit)
        candidate_t = []
        if cccv_hit:
            candidate_t.append(float(cccv.get("time_to_80_soc_min", np.nan)))
        if mpc_hit:
            candidate_t.append(float(mpc.get("time_to_80_soc_min", np.nan)))
        if candidate_t:
            fastest_target_time = float(np.nanmin(np.array(candidate_t, dtype=float)))
        else:
            fastest_target_time = float("inf")
        horizon_time = float(
            np.nanmax(
                np.array(
                    [
                        float(cccv.get("charge_time_min", np.nan)),
                        float(mpc.get("charge_time_min", np.nan)),
                    ],
                    dtype=float,
                )
            )
        )
        if not np.isfinite(horizon_time):
            horizon_time = fastest_target_time if np.isfinite(fastest_target_time) else 0.0
        refs[key] = BaselineAcceptanceRef(
            best_final_soc=best_final_soc,
            best_hits_target=best_hits_target,
            fastest_target_time_min=fastest_target_time,
            horizon_time_min=float(max(0.0, horizon_time)),
        )
    return refs


def resolve_chemistry_families(config: TrainPolicyConfig, df: pd.DataFrame) -> List[str]:
    explicit = split_csv_arg(config.chemistry_families)
    if explicit:
        return explicit
    families = sorted(df["dataset_family"].astype(str).str.lower().unique().tolist())
    return [str(x) for x in families if str(x)]


def filter_family(df: pd.DataFrame, family: str | None) -> pd.DataFrame:
    if family is None:
        return df
    fam = str(family).lower()
    return df[df["dataset_family"].astype(str).str.lower() == fam].copy()


def scope_roots(
    mode: str,
    root: Path,
    family: str | None,
) -> Path:
    if mode == "global":
        return root
    if mode == "family_specific":
        if family is None:
            raise ValueError("family is required for family_specific mode")
        return root / "family_specific" / family
    if mode == "shared_plus_heads":
        key = "shared" if family is None else family
        return root / "shared_plus_heads" / key
    raise ValueError(f"Unsupported chemistry mode: {mode}")


def select_rows_by_split(df: pd.DataFrame, split_ids: Dict[str, List[str]], split_name: str) -> pd.DataFrame:
    ids = set(split_ids.get(split_name, []))
    return df[df["episode_id"].astype(str).isin(ids)].copy()


def build_actor_dataset(
    df: pd.DataFrame,
    window_cols: List[str],
    context_cols: List[str],
    target_soc: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x_window = df[window_cols].to_numpy(dtype=np.float32)
    x_mpc = df["action_mpc"].to_numpy(dtype=np.float32).reshape(-1, 1)
    soc_gap = (target_soc - df["pack_soc"].to_numpy(dtype=np.float32)).reshape(-1, 1)
    if context_cols:
        x_ctx = df[context_cols].to_numpy(dtype=np.float32)
    else:
        x_ctx = np.zeros((len(df), 0), dtype=np.float32)
    x = np.concatenate([x_window, x_mpc, soc_gap, x_ctx], axis=1).astype(np.float32)
    y = np.clip(df["target_delta_action"].to_numpy(dtype=np.float32), -1.0, 1.0).reshape(-1, 1)
    return x, y


def compute_discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i] + gamma * running)
        out[i] = running
    return out


def evaluate_policy_short(
    actor: ResidualActorPolicy,
    predictor: AdaptiveEnsemblePredictor,
    scenario_tuples: List[Tuple[str, str, str]],
    setting_map: Dict[Tuple[str, str, str], Any],
    max_steps: int,
    max_steps_map: Dict[Tuple[str, str, str], int] | None,
    saerl_mpc_anchor_mode: str,
    target_soc: float,
    eval_score_soc_weight: float,
    eval_score_time_weight: float,
    eval_score_safety_weight: float,
) -> float:
    if not scenario_tuples:
        return float("-inf")
    scores: List[float] = []
    for tup in scenario_tuples[: min(3, len(scenario_tuples))]:
        setting = setting_map[tup]
        scenario_steps = int(max(1, max_steps))
        if max_steps_map is not None:
            scenario_steps = int(max(1, max_steps_map.get(tup, scenario_steps)))
        env = make_env(setting=setting, max_steps=scenario_steps, target_soc=target_soc)
        env.reset(initial_soc=setting.initial_soc, temperature=setting.initial_temp_c)
        trim_pack_histories(env.pack)
        objective_key, family, _ = tup
        mpc = RolloutMPCController(
            config=chemistry_aware_mpc_config(
                family=family,
                mode=saerl_mpc_anchor_mode,
                objective_key=objective_key,
            ),
            cv_voltage_v=setting.cv_voltage_v,
            max_charge_current_a=setting.max_charge_current_a,
            target_soc=target_soc,
        )
        mpc.reset()
        shield_helper = SafeAdaptiveEnsembleController(
            predictor=predictor,
            actor=None,
            config=predictor.config,
        )
        shield_helper.reset()
        ctx_vec = scenario_context_array(
            setting.scenario,
            context_feature_set=predictor.config.context_feature_set,
            context_columns=predictor.config.context_columns,
        )
        state = initial_state_from_env(env)
        state_window: Deque[Dict[str, Any]] = deque(maxlen=predictor.config.window_len)
        for _ in range(predictor.config.window_len):
            state_window.append(dict(state))
        done = False
        step_idx = 0
        safety_total = 0
        while not done and step_idx < scenario_steps:
            mpc_action, _ = mpc.act(state, env)
            seq = window_to_sequence(
                state_window=list(state_window),
                window_len=predictor.config.window_len,
                max_charge_current_a=setting.max_charge_current_a,
            )
            delta, _ = actor.predict_delta(
                sequence=seq,
                mpc_action=mpc_action,
                target_soc=target_soc,
                context=ctx_vec,
                stochastic=False,
            )
            action = float(np.clip(float(mpc_action) + delta, -1.0, 1.0))
            if not isfinite_action(action):
                action = float(mpc_action)
            fused = predictor.predict_fused(
                state_window=list(state_window),
                action=action,
                max_charge_current_a=setting.max_charge_current_a,
                cv_voltage_v=setting.cv_voltage_v,
                context=ctx_vec,
            )
            action, _ = shield_helper.validate_and_shield_action(
                state=state,
                env=env,
                proposed_action=action,
                mpc_action=float(mpc_action),
                cv_voltage_v=setting.cv_voltage_v,
                risk_score=float(fused["risk_score"]),
                apply_antistall=True,
                update_counters=False,
            )
            _, _, done, next_state = env.step(action)
            trim_pack_histories(env.pack)
            safety_total += int(count_safety_events(next_state.get("safety_events", {})))
            state = {
                "pack_soc": float(next_state["pack_soc"]),
                "pack_voltage": float(next_state["pack_voltage"]),
                "pack_temperature": float(next_state["pack_temperature"]),
                "voltage_imbalance": float(next_state["voltage_imbalance"]),
                "pack_current": float(next_state["pack_current"]),
                "safety_events": next_state.get("safety_events", {}),
            }
            state_window.append(dict(state))
            step_idx += 1
        final_soc = float(state["pack_soc"])
        charge_time_min = float(env.pack.cells[0].time / 60.0) if env.pack.cells else float(step_idx / 60.0)
        score = (
            eval_score_soc_weight * final_soc
            - eval_score_time_weight * charge_time_min
            - eval_score_safety_weight * safety_total
        )
        scores.append(score)
    return float(np.mean(scores))


def _strip_cli_arg(argv: Sequence[str], arg_name: str) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(argv):
        tok = str(argv[i])
        if tok == arg_name:
            if i + 1 < len(argv) and not str(argv[i + 1]).startswith("--"):
                i += 2
            else:
                i += 1
            continue
        if tok.startswith(f"{arg_name}="):
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def maybe_run_objective_specific_heads(config: TrainPolicyConfig) -> bool:
    if (
        not bool(config.objective_specific_heads)
        or str(config.objective).strip().lower() != "all"
        or str(config.chemistry_mode).strip().lower() == "global"
    ):
        return False

    objective_keys = list(build_default_objectives().keys())
    if not objective_keys:
        return False

    base_args = list(sys.argv[1:])
    for key in ["--objective", "--output-root", "--reports-root", "--objective-specific-heads"]:
        base_args = _strip_cli_arg(base_args, key)

    this_script = str(Path(__file__).resolve())
    for objective_key in objective_keys:
        objective_output_root = str(Path(config.output_root) / objective_key)
        objective_reports_root = str(Path(config.reports_root) / objective_key)
        cmd = [
            sys.executable,
            this_script,
            *base_args,
            "--objective",
            str(objective_key),
            "--output-root",
            objective_output_root,
            "--reports-root",
            objective_reports_root,
            "--objective-specific-heads",
            "0",
        ]
        print(
            f"[objective-heads] Launching objective={objective_key} -> "
            f"output={objective_output_root}",
            flush=True,
        )
        subprocess.run(cmd, check=True)
    print(
        "[objective-heads] Completed objective-specific family-head training.",
        flush=True,
    )
    return True


def row_to_state_window(
    row: pd.Series,
    window_len: int,
    max_charge_current_a: float,
) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    max_i = float(max(1e-6, max_charge_current_a))
    for t in range(int(window_len)):
        prefix = f"window_t{t:02d}_f"
        soc = float(row.get(f"{prefix}00", 0.0))
        voltage_v = float(row.get(f"{prefix}01", 0.0)) * 100.0
        temp_c = float(row.get(f"{prefix}02", 0.0)) * 60.0
        imbalance_v = float(row.get(f"{prefix}03", 0.0)) * 0.2
        current_norm = float(np.clip(row.get(f"{prefix}04", 0.0), -2.0, 2.0))
        pack_current_a = -current_norm * max_i
        states.append(
            {
                "pack_soc": float(np.clip(soc, 0.0, 1.0)),
                "pack_voltage": float(voltage_v),
                "pack_temperature": float(temp_c),
                "voltage_imbalance": float(imbalance_v),
                "pack_current": float(pack_current_a),
                "safety_events": {},
            }
        )
    return states


def calibrate_antistall_from_validation(
    predictor: AdaptiveEnsemblePredictor,
    val_df: pd.DataFrame,
    config: TrainPolicyConfig,
) -> Dict[str, float]:
    if val_df.empty:
        return {"calibrated": 0.0, "n_samples": 0.0}

    max_rows = int(max(64, config.antistall_calibration_max_rows))
    sample_df = val_df.sample(
        n=min(len(val_df), max_rows),
        random_state=config.random_seed,
    ).copy()
    context_cols = parse_context_cols(sample_df.columns.tolist())
    sample_df["soc_gap"] = float(config.target_soc) - sample_df["pack_soc"].astype(float)
    gap_threshold = float(
        max(
            0.03,
            float(config.saerl_anti_stall_soc_gap) * float(config.antistall_calibration_soc_gap_mult),
        )
    )
    sample_df = sample_df[
        (sample_df["soc_gap"] >= gap_threshold)
        & (sample_df["target_delta_action"].astype(float) > 0.01)
    ].copy()
    if sample_df.empty:
        return {
            "calibrated": 0.0,
            "n_samples": 0.0,
            "gap_threshold": gap_threshold,
        }

    risk_scale = float(max(1e-6, predictor.config.anti_stall_risk_scale))
    normalized_risks: List[float] = []
    charge_fractions: List[float] = []
    for _, row in sample_df.iterrows():
        max_i = float(row.get("max_charge_current_a", config.max_charge_current_a))
        cv_v = float(row.get("cv_voltage_v", 84.0))
        states = row_to_state_window(
            row=row,
            window_len=predictor.config.window_len,
            max_charge_current_a=max_i,
        )
        ctx_vec = (
            row[context_cols].to_numpy(dtype=np.float32)
            if context_cols
            else np.zeros((0,), dtype=np.float32)
        )
        action = float(np.clip(row.get("target_action", row.get("action_mpc", 0.0)), -1.0, 1.0))
        fused = predictor.predict_fused(
            state_window=states,
            action=action,
            max_charge_current_a=max_i,
            cv_voltage_v=cv_v,
            context=ctx_vec,
        )
        raw_risk = float(fused.get("risk_score", 0.0))
        normalized_risk = float(max(0.0, raw_risk) / (max(0.0, raw_risk) + risk_scale))
        normalized_risks.append(normalized_risk)
        charge_fractions.append(float(np.clip(action, 0.0, 1.0)))

    calibrator = SafeAdaptiveEnsembleController(
        predictor=predictor,
        actor=None,
        config=predictor.config,
    )
    info = calibrator.calibrate_antistall_from_quantiles(
        normalized_risks=normalized_risks,
        charge_fractions=charge_fractions,
        risk_quantile=float(config.antistall_calibration_risk_quantile),
        floor_quantile=float(config.antistall_calibration_floor_quantile),
        min_samples=int(config.antistall_calibration_min_samples),
    )
    info["gap_threshold"] = gap_threshold
    return info


def main() -> None:
    config = parse_args()
    if maybe_run_objective_specific_heads(config=config):
        return
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(config.random_seed)

    df = pd.read_csv(
        config.dataset_csv,
        dtype={
            "episode_id": "string",
            "objective": "string",
            "dataset_family": "string",
            "dataset_case": "string",
        },
        low_memory=False,
    )
    if df.empty:
        raise SystemExit("Dataset is empty.")
    window_cols = parse_window_cols(df.columns.tolist())
    if not window_cols:
        raise SystemExit("No window_t* columns found.")
    context_cols = parse_context_cols(df.columns.tolist())
    if context_cols and config.context_feature_set == "none":
        config.context_feature_set = "source_v1"

    with Path(config.split_manifest_json).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    folds = manifest.get("folds", [])
    if not folds:
        raise SystemExit("No folds in split manifest.")
    if config.fold != "all":
        wanted = int(config.fold)
        folds = [f for f in folds if int(f.get("fold_id", -1)) == wanted]
        if not folds:
            raise SystemExit(f"Fold {wanted} not found.")

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
    scenario_map: Dict[Tuple[str, str], Dict[str, Any]] = {
        (str(s["family"]), str(s["case_id"])): s for s in scenarios
    }

    scope_family: str | None = None
    if config.chemistry_mode != "global":
        explicit_families = split_csv_arg(config.chemistry_families)
        if len(explicit_families) > 1:
            raise SystemExit(
                "For non-global policy training, pass exactly one family in --chemistry-families. "
                "Use orchestration to train multiple families."
            )
        if explicit_families:
            scope_family = explicit_families[0]
        elif config.chemistry_mode == "family_specific":
            raise SystemExit("family_specific mode requires --chemistry-families <family>.")

    profile_payload = load_optional_json(config.saerl_family_profile_json)
    profile_overrides = resolve_profile_overrides(
        profile_payload=profile_payload,
        family=scope_family,
        objective=config.objective if config.objective != "all" else None,
    )
    reward_soc_gain_weight = float(profile_overrides.get("reward_soc_gain_weight", config.reward_soc_gain_weight))
    reward_step_penalty = float(profile_overrides.get("reward_step_penalty", config.reward_step_penalty))
    reward_temp_penalty_weight = float(
        profile_overrides.get("reward_temp_penalty_weight", config.reward_temp_penalty_weight)
    )
    reward_imbalance_penalty_weight = float(
        profile_overrides.get("reward_imbalance_penalty_weight", config.reward_imbalance_penalty_weight)
    )
    reward_q_loss_penalty_weight = float(
        profile_overrides.get("reward_q_loss_penalty_weight", config.reward_q_loss_penalty_weight)
    )
    reward_safety_penalty_weight = float(
        profile_overrides.get("reward_safety_penalty_weight", config.reward_safety_penalty_weight)
    )
    eval_score_soc_weight = float(profile_overrides.get("eval_score_soc_weight", config.eval_score_soc_weight))
    eval_score_time_weight = float(profile_overrides.get("eval_score_time_weight", config.eval_score_time_weight))
    eval_score_safety_weight = float(
        profile_overrides.get("eval_score_safety_weight", config.eval_score_safety_weight)
    )
    reward_acceptance_progress_weight = float(
        profile_overrides.get("reward_acceptance_progress_weight", config.reward_acceptance_progress_weight)
    )
    reward_acceptance_terminal_weight = float(
        profile_overrides.get("reward_acceptance_terminal_weight", config.reward_acceptance_terminal_weight)
    )
    reward_acceptance_success_bonus = float(
        profile_overrides.get("reward_acceptance_success_bonus", config.reward_acceptance_success_bonus)
    )
    reward_acceptance_time_slack = float(
        max(1.0, profile_overrides.get("reward_acceptance_time_slack", config.reward_acceptance_time_slack))
    )
    reward_acceptance_soc_margin = float(
        max(0.0, profile_overrides.get("reward_acceptance_soc_margin", config.reward_acceptance_soc_margin))
    )
    reward_acceptance_start_fraction = float(
        np.clip(
            profile_overrides.get("reward_acceptance_start_fraction", config.reward_acceptance_start_fraction),
            0.0,
            1.0,
        )
    )
    reward_acceptance_final_soc_progress_mult = float(
        max(
            0.0,
            profile_overrides.get(
                "reward_acceptance_final_soc_progress_mult",
                config.reward_acceptance_final_soc_progress_mult,
            ),
        )
    )
    reward_acceptance_final_soc_terminal_mult = float(
        max(
            0.0,
            profile_overrides.get(
                "reward_acceptance_final_soc_terminal_mult",
                config.reward_acceptance_final_soc_terminal_mult,
            ),
        )
    )
    reward_acceptance_final_soc_success_bonus_mult = float(
        max(
            0.0,
            profile_overrides.get(
                "reward_acceptance_final_soc_success_bonus_mult",
                config.reward_acceptance_final_soc_success_bonus_mult,
            ),
        )
    )
    baseline_refs = load_baseline_acceptance_refs(
        baseline_root=config.baseline_results_root,
        target_soc=config.target_soc,
    )

    ensemble_root = Path(config.ensemble_root)
    output_root = Path(config.output_root)
    reports_root = Path(config.reports_root)
    if config.chemistry_mode == "family_specific":
        ensemble_root = scope_roots(mode="family_specific", root=ensemble_root, family=scope_family)
        output_root = scope_roots(mode="family_specific", root=output_root, family=scope_family)
        reports_root = scope_roots(mode="family_specific", root=reports_root, family=scope_family)
    elif config.chemistry_mode == "shared_plus_heads":
        ensemble_root = scope_roots(mode="shared_plus_heads", root=ensemble_root, family=scope_family)
        output_root = scope_roots(mode="shared_plus_heads", root=output_root, family=scope_family)
        reports_root = scope_roots(mode="shared_plus_heads", root=reports_root, family=scope_family)

    output_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    all_reports: List[Dict[str, Any]] = []

    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_df = select_rows_by_split(df=df, split_ids=fold["splits"], split_name="train")
        val_df = select_rows_by_split(df=df, split_ids=fold["splits"], split_name="val")
        internal_test_df = select_rows_by_split(df=df, split_ids=fold["splits"], split_name="internal_test")
        if scope_family is not None:
            train_df = filter_family(train_df, family=scope_family)
            val_df = filter_family(val_df, family=scope_family)
            internal_test_df = filter_family(internal_test_df, family=scope_family)

        if config.objective != "all":
            train_df = train_df[train_df["objective"].isin(objective_keys)].copy()
            val_df = val_df[val_df["objective"].isin(objective_keys)].copy()
            internal_test_df = internal_test_df[internal_test_df["objective"].isin(objective_keys)].copy()

        if train_df.empty:
            fallback_parts = [frame for frame in (val_df, internal_test_df) if not frame.empty]
            if fallback_parts:
                train_df = pd.concat(fallback_parts, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
                print(
                    f"Fold {fold_id}: promoted non-case rows into train split "
                    f"(family={scope_family or 'all'}, objective={config.objective})."
                )
            else:
                if config.objective != "all":
                    raise SystemExit(f"Fold {fold_id}: train split empty after objective filter.")
                raise SystemExit(f"Fold {fold_id}: empty train split.")
        if val_df.empty:
            val_df = train_df.sample(n=min(len(train_df), 1000), random_state=config.random_seed)

        x_train, y_train = build_actor_dataset(
            train_df,
            window_cols=window_cols,
            context_cols=context_cols,
            target_soc=config.target_soc,
        )
        x_val, y_val = build_actor_dataset(
            val_df,
            window_cols=window_cols,
            context_cols=context_cols,
            target_soc=config.target_soc,
        )

        actor = ResidualActorPolicy(
            input_dim=x_train.shape[1],
            delta_action_limit=SAERLConfig().delta_action_limit,
            context_dim=len(context_cols),
            context_feature_set=config.context_feature_set,
            context_columns=context_cols,
            device=str(device),
        )
        init_actor_root = Path(config.init_actor_root) if str(config.init_actor_root).strip() else None
        if config.chemistry_mode != "global" and not bool(config.allow_global_actor_init):
            init_actor_root = None
        elif (
            init_actor_root is None
            and config.chemistry_mode == "shared_plus_heads"
            and scope_family is not None
        ):
            init_actor_root = scope_roots(
                mode="shared_plus_heads",
                root=Path(config.output_root),
                family=None,
            )
        init_actor_path = None
        if init_actor_root is not None:
            candidate = init_actor_root / f"fold_{fold_id:02d}" / "residual_actor.pt"
            if candidate.exists():
                init_actor_path = candidate
                try:
                    loaded_actor = ResidualActorPolicy.load(path=candidate, device=str(device))
                    if int(loaded_actor.input_dim) == int(actor.input_dim):
                        actor = loaded_actor
                except Exception:
                    pass
        predictor = AdaptiveEnsemblePredictor.load(
            directory=ensemble_root / f"fold_{fold_id:02d}",
            device=str(device),
        )
        predictor.config.enable_shield = True
        predictor.config.score_soc_gain_weight = float(
            profile_overrides.get("saerl_score_soc_gain_weight", config.saerl_score_soc_gain_weight)
        )
        predictor.config.anti_stall_soc_gap = float(
            profile_overrides.get("saerl_anti_stall_soc_gap", config.saerl_anti_stall_soc_gap)
        )
        predictor.config.anti_stall_low_risk_threshold = float(
            profile_overrides.get(
                "saerl_anti_stall_low_risk_threshold",
                config.saerl_anti_stall_low_risk_threshold,
            )
        )
        predictor.config.anti_stall_duration_s = float(
            profile_overrides.get("saerl_anti_stall_duration_s", config.saerl_anti_stall_duration_s)
        )
        predictor.config.anti_stall_risk_scale = float(
            max(
                1e-6,
                profile_overrides.get(
                    "saerl_anti_stall_risk_scale",
                    config.saerl_anti_stall_risk_scale,
                ),
            )
        )
        predictor.config.min_safe_charge_fraction = float(
            profile_overrides.get("saerl_min_safe_charge_fraction", config.saerl_min_safe_charge_fraction)
        )
        predictor.config.enable_antistall = bool(config.saerl_enable_antistall)
        predictor.config.score_time_weight = float(
            profile_overrides.get("saerl_score_time_weight", config.saerl_score_time_weight)
        )
        predictor.config.score_temp_weight = float(
            profile_overrides.get("saerl_score_temp_weight", config.saerl_score_temp_weight)
        )
        predictor.config.score_degradation_weight = float(
            profile_overrides.get("saerl_score_degradation_weight", config.saerl_score_degradation_weight)
        )
        predictor.config.score_imbalance_weight = float(
            profile_overrides.get("saerl_score_imbalance_weight", config.saerl_score_imbalance_weight)
        )
        predictor.config.score_safety_weight = float(
            profile_overrides.get("saerl_score_safety_weight", config.saerl_score_safety_weight)
        )
        predictor.config.score_risk_weight = float(
            profile_overrides.get("saerl_score_risk_weight", config.saerl_score_risk_weight)
        )
        predictor.config.imbalance_margin_v = float(
            profile_overrides.get("saerl_imbalance_margin_v", config.saerl_imbalance_margin_v)
        )
        antistall_calibration = calibrate_antistall_from_validation(
            predictor=predictor,
            val_df=val_df,
            config=config,
        )

        # Offline behavior cloning warm-start.
        train_ds = TensorDataset(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train).float(),
        )
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
        x_val_t = torch.from_numpy(x_val).float().to(device)
        y_val_t = torch.from_numpy(y_val).float().to(device)

        optim_actor = torch.optim.Adam(
            actor.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        best_state = None
        best_val_mse = float("inf")
        for _ in range(config.offline_epochs):
            actor.model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mean, _ = actor.model(xb)
                pred = torch.tanh(mean) * actor.delta_action_limit
                loss = torch.mean((pred - yb) ** 2)
                optim_actor.zero_grad()
                loss.backward()
                optim_actor.step()
            actor.model.eval()
            with torch.no_grad():
                mean_val, _ = actor.model(x_val_t)
                pred_val = torch.tanh(mean_val) * actor.delta_action_limit
                mse = float(torch.mean((pred_val - y_val_t) ** 2).cpu().item())
            if mse < best_val_mse:
                best_val_mse = mse
                best_state = {k: v.detach().cpu().clone() for k, v in actor.model.state_dict().items()}
        if best_state is not None:
            actor.model.load_state_dict(best_state)

        # Build scenario tuples for online training from train split.
        train_tuples = sorted(
            {
                (str(r["objective"]), str(r["dataset_family"]), str(r["dataset_case"]))
                for _, r in train_df[["objective", "dataset_family", "dataset_case"]].drop_duplicates().iterrows()
                if (str(r["dataset_family"]), str(r["dataset_case"])) in scenario_map and str(r["objective"]) in objectives
            }
        )
        val_tuples = sorted(
            {
                (str(r["objective"]), str(r["dataset_family"]), str(r["dataset_case"]))
                for _, r in val_df[["objective", "dataset_family", "dataset_case"]].drop_duplicates().iterrows()
                if (str(r["dataset_family"]), str(r["dataset_case"])) in scenario_map and str(r["objective"]) in objectives
            }
        )
        setting_map: Dict[Tuple[str, str, str], Any] = {}
        online_max_steps_map: Dict[Tuple[str, str, str], int] = {}
        for tup in set(train_tuples + val_tuples):
            objective_key, family, case = tup
            if objective_key not in objectives:
                continue
            scenario = scenario_map[(family, case)]
            setting = build_setting_for_objective(
                run_config=run_config,
                objective_key=objective_key,
                objective=objectives[objective_key],
                scenario=scenario,
            )
            setting_map[tup] = setting
            if config.adaptive_online_horizon:
                scenario_steps, _ = recommend_episode_max_steps(
                    setting=setting,
                    base_max_steps=config.online_max_steps,
                    target_soc=config.target_soc,
                    min_episode_minutes=config.min_episode_minutes,
                    feasible_time_slack=config.feasible_time_slack,
                    max_steps_cap=config.max_steps_cap,
                )
            else:
                scenario_steps = int(max(1, config.online_max_steps))
            online_max_steps_map[tup] = int(max(1, scenario_steps))

        best_online_score = float("-inf")
        best_online_state = {k: v.detach().cpu().clone() for k, v in actor.model.state_dict().items()}
        no_improve = 0
        online_logs: List[Dict[str, Any]] = []

        for epoch in range(config.online_epochs):
            do_actor_update = epoch < 5
            all_x: List[np.ndarray] = []
            all_raw_delta: List[float] = []
            all_old_logp: List[float] = []
            all_adv: List[float] = []
            gate_x: List[np.ndarray] = []
            gate_t: List[np.ndarray] = []
            reward_means: List[float] = []

            for _ in range(config.online_rollout_episodes):
                if not train_tuples:
                    break
                tup = train_tuples[int(rng.integers(0, len(train_tuples)))]
                objective_key_ep, family_ep, case_ep = tup
                setting = setting_map[tup]
                episode_max_steps = int(max(1, online_max_steps_map.get(tup, config.online_max_steps)))
                env = make_env(setting=setting, max_steps=episode_max_steps, target_soc=config.target_soc)
                env.reset(initial_soc=setting.initial_soc, temperature=setting.initial_temp_c)
                trim_pack_histories(env.pack)
                apply_domain_randomization(env=env, rng=rng)
                acceptance_ref = baseline_refs.get(
                    (str(objective_key_ep).lower(), str(family_ep).lower(), str(case_ep))
                )
                ctx_vec = scenario_context_array(
                    setting.scenario,
                    context_feature_set=predictor.config.context_feature_set,
                    context_columns=predictor.config.context_columns,
                )

                mpc = RolloutMPCController(
                    config=chemistry_aware_mpc_config(
                        family=family_ep,
                        mode=config.saerl_mpc_anchor_mode,
                        objective_key=objective_key_ep,
                    ),
                    cv_voltage_v=setting.cv_voltage_v,
                    max_charge_current_a=setting.max_charge_current_a,
                    target_soc=config.target_soc,
                )
                mpc.reset()
                shield_helper = SafeAdaptiveEnsembleController(
                    predictor=predictor,
                    actor=None,
                    config=predictor.config,
                )
                shield_helper.reset()
                state = initial_state_from_env(env)
                state_window: Deque[Dict[str, Any]] = deque(maxlen=predictor.config.window_len)
                for _ in range(predictor.config.window_len):
                    state_window.append(dict(state))

                traj_x: List[np.ndarray] = []
                traj_raw: List[float] = []
                traj_logp: List[float] = []
                traj_rewards: List[float] = []
                prev_q_loss_total = float(sum(float(cell.Q_loss) for cell in env.pack.cells))
                t80_reached_min: float | None = None
                initial_soc_ep = float(setting.initial_soc)

                done = False
                step_idx = 0
                while not done and step_idx < episode_max_steps:
                    mpc_action, _ = mpc.act(state, env)
                    seq = window_to_sequence(
                        state_window=list(state_window),
                        window_len=predictor.config.window_len,
                        max_charge_current_a=setting.max_charge_current_a,
                    )
                    actor_input = ResidualActorPolicy.build_input(
                        sequence=seq,
                        mpc_action=float(mpc_action),
                        target_soc=config.target_soc,
                        context=ctx_vec,
                        context_dim=actor.context_dim,
                        context_columns=actor.context_columns,
                    )
                    x_t = torch.from_numpy(actor_input).float().unsqueeze(0).to(device)
                    actor.model.eval()
                    with torch.no_grad():
                        mean, std = actor.model(x_t)
                    dist = Normal(mean, std)
                    raw = dist.sample()
                    delta = float(torch.tanh(raw).cpu().item() * actor.delta_action_limit)
                    action = float(np.clip(float(mpc_action) + delta, -1.0, 1.0))

                    # Hard shield.
                    if not isfinite_action(action):
                        action = float(mpc_action)
                    fused = predictor.predict_fused(
                        state_window=list(state_window),
                        action=action,
                        max_charge_current_a=setting.max_charge_current_a,
                        cv_voltage_v=setting.cv_voltage_v,
                        context=ctx_vec,
                    )
                    action, _ = shield_helper.validate_and_shield_action(
                        state=state,
                        env=env,
                        proposed_action=action,
                        mpc_action=float(mpc_action),
                        cv_voltage_v=setting.cv_voltage_v,
                        risk_score=float(fused["risk_score"]),
                        apply_antistall=True,
                        update_counters=False,
                    )

                    _, _, done, next_state = env.step(action)
                    trim_pack_histories(env.pack)

                    # Reward shaping.
                    soc_gain = float(next_state["pack_soc"] - state["pack_soc"])
                    over_t = max(0.0, float(next_state["pack_temperature"]) - 42.0)
                    imbalance_mv = float(next_state["voltage_imbalance"]) * 1000.0
                    safety = int(count_safety_events(next_state.get("safety_events", {})))
                    q_loss_total = float(sum(float(cell.Q_loss) for cell in env.pack.cells))
                    q_loss_step = max(0.0, q_loss_total - prev_q_loss_total)
                    prev_q_loss_total = q_loss_total
                    reward = (
                        reward_soc_gain_weight * soc_gain
                        - reward_step_penalty
                        - reward_temp_penalty_weight * over_t * over_t
                        - reward_imbalance_penalty_weight * imbalance_mv
                        - reward_q_loss_penalty_weight * q_loss_step
                        - reward_safety_penalty_weight * safety
                    )
                    next_soc = float(next_state["pack_soc"])
                    elapsed_min = float(next_state.get("time", env.pack.cells[0].time if env.pack.cells else 0.0)) / 60.0
                    if t80_reached_min is None and next_soc >= config.target_soc:
                        t80_reached_min = elapsed_min
                    if (
                        acceptance_ref is not None
                        and reward_acceptance_progress_weight > 0.0
                    ):
                        progress_weight = reward_acceptance_progress_weight
                        if not acceptance_ref.best_hits_target:
                            progress_weight *= reward_acceptance_final_soc_progress_mult
                        progress_penalty = 0.0
                        if acceptance_ref.best_hits_target and np.isfinite(acceptance_ref.fastest_target_time_min):
                            allowed_min = max(1e-6, acceptance_ref.fastest_target_time_min * reward_acceptance_time_slack)
                            if elapsed_min >= reward_acceptance_start_fraction * allowed_min:
                                expected_soc = initial_soc_ep + (config.target_soc - initial_soc_ep) * min(
                                    1.0, elapsed_min / allowed_min
                                )
                                progress_penalty = max(0.0, expected_soc - next_soc)
                        else:
                            target_final_soc = acceptance_ref.best_final_soc + reward_acceptance_soc_margin
                            horizon_min = max(
                                1e-6,
                                acceptance_ref.horizon_time_min,
                                float(episode_max_steps * setting.dt_s / 60.0),
                            )
                            if elapsed_min >= reward_acceptance_start_fraction * horizon_min:
                                expected_soc = initial_soc_ep + (target_final_soc - initial_soc_ep) * min(
                                    1.0, elapsed_min / horizon_min
                                )
                                progress_penalty = max(0.0, expected_soc - next_soc)
                        reward -= progress_weight * progress_penalty

                    # Gate supervision signal from expert errors.
                    expert_outputs = predictor.predict_experts(
                        state_window=list(state_window),
                        action=action,
                        max_charge_current_a=setting.max_charge_current_a,
                        context=ctx_vec,
                    )
                    y_true = np.array(
                        [
                            float(next_state["pack_soc"]),
                            float(next_state["pack_voltage"]),
                            float(next_state["pack_temperature"]),
                            float(next_state["voltage_imbalance"]),
                        ],
                        dtype=np.float32,
                    )
                    err = []
                    unc = []
                    for k in ["gru", "mlp", "rf"]:
                        err.append(float(np.mean(np.abs(expert_outputs[k]["mean"] - y_true))))
                        unc.append(float(np.mean(expert_outputs[k]["uncertainty"])))
                    inv = 1.0 / np.clip(np.array(err) + 0.3 * np.array(unc), 1e-6, None)
                    w_target = inv / np.sum(inv)
                    seq_last = seq[-1]
                    gx = np.concatenate(
                        [
                            seq_last,
                            np.array([action], dtype=np.float32),
                            ctx_vec.astype(np.float32),
                            np.array(unc, dtype=np.float32),
                            np.array(err, dtype=np.float32),
                        ],
                        axis=0,
                    ).astype(np.float32)
                    gate_x.append(gx)
                    gate_t.append(w_target.astype(np.float32))

                    traj_x.append(actor_input.astype(np.float32))
                    raw_val = float(raw.cpu().item())
                    traj_raw.append(raw_val)
                    traj_logp.append(float(dist.log_prob(raw).sum().cpu().item()))
                    traj_rewards.append(float(reward))

                    state = {
                        "pack_soc": float(next_state["pack_soc"]),
                        "pack_voltage": float(next_state["pack_voltage"]),
                        "pack_temperature": float(next_state["pack_temperature"]),
                        "voltage_imbalance": float(next_state["voltage_imbalance"]),
                        "pack_current": float(next_state["pack_current"]),
                        "safety_events": next_state.get("safety_events", {}),
                    }
                    state_window.append(dict(state))
                    step_idx += 1

                if traj_rewards:
                    if acceptance_ref is not None and reward_acceptance_terminal_weight > 0.0:
                        terminal_penalty = 0.0
                        terminal_bonus = 0.0
                        terminal_weight = reward_acceptance_terminal_weight
                        success_bonus = reward_acceptance_success_bonus
                        if not acceptance_ref.best_hits_target:
                            terminal_weight *= reward_acceptance_final_soc_terminal_mult
                            success_bonus *= reward_acceptance_final_soc_success_bonus_mult
                        final_soc_ep = float(state.get("pack_soc", initial_soc_ep))
                        if acceptance_ref.best_hits_target and np.isfinite(acceptance_ref.fastest_target_time_min):
                            allowed_min = max(1e-6, acceptance_ref.fastest_target_time_min * reward_acceptance_time_slack)
                            success = bool(
                                t80_reached_min is not None
                                and float(t80_reached_min) <= allowed_min
                                and final_soc_ep >= config.target_soc
                            )
                            if success:
                                terminal_bonus = success_bonus
                            else:
                                if t80_reached_min is None:
                                    over = max(0.0, config.target_soc - final_soc_ep)
                                    terminal_penalty = 1.0 + 2.0 * over
                                else:
                                    terminal_penalty = max(0.0, (float(t80_reached_min) - allowed_min) / allowed_min)
                        else:
                            target_final_soc = acceptance_ref.best_final_soc + reward_acceptance_soc_margin
                            shortfall = max(0.0, target_final_soc - final_soc_ep)
                            if shortfall <= 1e-6:
                                terminal_bonus = success_bonus
                            else:
                                terminal_penalty = shortfall
                        traj_rewards[-1] += terminal_bonus
                        traj_rewards[-1] -= terminal_weight * terminal_penalty

                    rewards_np = np.asarray(traj_rewards, dtype=np.float32)
                    returns = compute_discounted_returns(rewards_np, gamma=config.gamma)
                    adv = (returns - np.mean(returns)) / max(np.std(returns), 1e-6)
                    all_x.extend(traj_x)
                    all_raw_delta.extend(traj_raw)
                    all_old_logp.extend(traj_logp)
                    all_adv.extend(adv.tolist())
                    reward_means.append(float(np.mean(rewards_np)))

            # Actor PPO update (epochs 1-5 only).
            if do_actor_update and all_x:
                x_np = np.asarray(all_x, dtype=np.float32)
                raw_np = np.asarray(all_raw_delta, dtype=np.float32).reshape(-1, 1)
                old_logp_np = np.asarray(all_old_logp, dtype=np.float32).reshape(-1, 1)
                adv_np = np.asarray(all_adv, dtype=np.float32).reshape(-1, 1)
                ds = TensorDataset(
                    torch.from_numpy(x_np).float(),
                    torch.from_numpy(raw_np).float(),
                    torch.from_numpy(old_logp_np).float(),
                    torch.from_numpy(adv_np).float(),
                )
                loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
                actor.model.train()
                for _ in range(config.ppo_epochs):
                    for xb, raw_b, old_lp_b, adv_b in loader:
                        xb = xb.to(device)
                        raw_b = raw_b.to(device)
                        old_lp_b = old_lp_b.to(device)
                        adv_b = adv_b.to(device)
                        mean, std = actor.model(xb)
                        dist = Normal(mean, std)
                        logp = dist.log_prob(raw_b).sum(dim=1, keepdim=True)
                        entropy = dist.entropy().sum(dim=1, keepdim=True)
                        ratio = torch.exp(logp - old_lp_b)
                        s1 = ratio * adv_b
                        s2 = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv_b
                        loss = -torch.mean(torch.minimum(s1, s2)) - config.entropy_coef * torch.mean(entropy)
                        optim_actor.zero_grad()
                        loss.backward()
                        optim_actor.step()

            # Gate update (epochs 1-5 together with actor, 6+ gate only).
            if gate_x:
                gx = torch.from_numpy(np.asarray(gate_x, dtype=np.float32)).float().to(device)
                gt = torch.from_numpy(np.asarray(gate_t, dtype=np.float32)).float().to(device)
                predictor.gate_model.train()
                optim_gate = torch.optim.Adam(
                    predictor.gate_model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                pred = predictor.gate_model(gx)
                loss_gate = torch.mean((pred - gt) ** 2)
                optim_gate.zero_grad()
                loss_gate.backward()
                optim_gate.step()

            val_score = evaluate_policy_short(
                actor=actor,
                predictor=predictor,
                scenario_tuples=val_tuples if val_tuples else train_tuples,
                setting_map=setting_map,
                max_steps=config.online_max_steps,
                max_steps_map=online_max_steps_map,
                saerl_mpc_anchor_mode=config.saerl_mpc_anchor_mode,
                target_soc=config.target_soc,
                eval_score_soc_weight=eval_score_soc_weight,
                eval_score_time_weight=eval_score_time_weight,
                eval_score_safety_weight=eval_score_safety_weight,
            )
            epoch_log = {
                "epoch": epoch,
                "do_actor_update": bool(do_actor_update),
                "mean_rollout_reward": float(np.mean(reward_means)) if reward_means else float("nan"),
                "val_score": float(val_score),
            }
            online_logs.append(epoch_log)

            if val_score > best_online_score:
                best_online_score = val_score
                best_online_state = {k: v.detach().cpu().clone() for k, v in actor.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= config.early_stop_patience:
                    break

        actor.model.load_state_dict(best_online_state)

        fold_model_dir = output_root / f"fold_{fold_id:02d}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        actor_path = fold_model_dir / "residual_actor.pt"
        actor.save(actor_path)

        # Save updated gate into ensemble directory for this fold.
        predictor.save(ensemble_root / f"fold_{fold_id:02d}")

        report_dir = reports_root / f"fold_{fold_id:02d}"
        report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "fold_id": fold_id,
            "config": asdict(config),
            "chemistry_mode": config.chemistry_mode,
            "chemistry_family": scope_family if scope_family is not None else "shared",
            "profile_overrides": profile_overrides,
            "antistall_calibration": antistall_calibration,
            "acceptance_reward": {
                "baseline_ref_count": int(len(baseline_refs)),
                "progress_weight": reward_acceptance_progress_weight,
                "terminal_weight": reward_acceptance_terminal_weight,
                "success_bonus": reward_acceptance_success_bonus,
                "time_slack": reward_acceptance_time_slack,
                "soc_margin": reward_acceptance_soc_margin,
                "start_fraction": reward_acceptance_start_fraction,
                "final_soc_progress_mult": reward_acceptance_final_soc_progress_mult,
                "final_soc_terminal_mult": reward_acceptance_final_soc_terminal_mult,
                "final_soc_success_bonus_mult": reward_acceptance_final_soc_success_bonus_mult,
            },
            "n_train_rows": int(len(train_df)),
            "n_val_rows": int(len(val_df)),
            "offline_best_val_mse": float(best_val_mse),
            "best_online_val_score": float(best_online_score),
            "actor_path": str(actor_path),
            "init_actor_path": str(init_actor_path) if init_actor_path is not None else "",
            "updated_ensemble_dir": str((ensemble_root / f"fold_{fold_id:02d}").resolve()),
            "online_logs": online_logs,
        }
        with (report_dir / "training_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=True)
        all_reports.append(report)
        print(
            f"Trained policy for fold {fold_id}: mode={config.chemistry_mode}, "
            f"family={scope_family or 'shared'} -> {actor_path}"
        )

    summary_path = reports_root / "summary_reports.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(config),
                "chemistry_mode": config.chemistry_mode,
                "chemistry_family": scope_family if scope_family is not None else "shared",
                "profile_overrides": profile_overrides,
                "acceptance_reward": {
                    "baseline_ref_count": int(len(baseline_refs)),
                    "progress_weight": reward_acceptance_progress_weight,
                    "terminal_weight": reward_acceptance_terminal_weight,
                    "success_bonus": reward_acceptance_success_bonus,
                    "time_slack": reward_acceptance_time_slack,
                    "soc_margin": reward_acceptance_soc_margin,
                    "start_fraction": reward_acceptance_start_fraction,
                    "final_soc_progress_mult": reward_acceptance_final_soc_progress_mult,
                    "final_soc_terminal_mult": reward_acceptance_final_soc_terminal_mult,
                    "final_soc_success_bonus_mult": reward_acceptance_final_soc_success_bonus_mult,
                },
                "reports": all_reports,
            },
            handle,
            indent=2,
            ensure_ascii=True,
        )
    print("Completed SAERL policy training.")
    print(f"Summary: {summary_path.resolve()}")


def isfinite_action(action: float) -> bool:
    return bool(np.isfinite(float(action)))


if __name__ == "__main__":
    main()
