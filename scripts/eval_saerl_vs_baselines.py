"""Evaluate SAERL against CCCV and MPC on held-out fold scenarios."""

from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from controllers.adaptive_ensemble_rl import (
    AdaptiveEnsemblePredictor,
    ResidualActorPolicy,
    SAERLConfig,
    SafeAdaptiveEnsembleController,
)
from pack_experiments import build_default_objectives
from scripts.run_baseline_benchmarks import (
    CCCVConfig,
    CCCVController,
    MPCConfig,
    RolloutMPCController,
    apply_publication_style,
    count_safety_events,
    plot_baseline_timeseries,
    plot_comparison_overlay,
    plot_metrics_bars,
    plot_phase_portraits,
    plot_tradeoff,
    save_json,
    trim_pack_histories,
)
from scripts.saerl_common import (
    build_setting_for_objective,
    compute_extended_metrics,
    initial_state_from_env,
    load_data_calibrated_scenarios,
    make_env,
)


@dataclass
class EvalConfig:
    dataset_csv: str = "data/training/saerl_phase2_dataset.csv"
    split_manifest_json: str = "data/training/saerl_phase2_splits.json"
    ensemble_root: str = "models/saerl_phase2/ensemble"
    ensemble_root_template: str = ""
    policy_root: str = "models/saerl_phase2/policy"
    policy_root_template: str = ""
    output_root: str = "results/saerl_phase2/evaluation"
    objective: str = "all"
    fold: str = "all"
    run_ablations: bool = True
    max_steps: int = 1200
    target_soc: float = 0.8
    n_series: int = 20
    n_parallel: int = 1
    max_charge_current_a: float = 10.0
    balancing_type: str = "passive"
    initial_soc: float = 0.2
    ambient_temp_c: float = 25.0
    standardized_root: str = "data/standardized"
    params_root: str = "data/standardized_params"
    dataset_families: str = "nasa,calce,matr"
    max_files_per_dataset: int = 3
    random_seed: int = 123
    chemistry_mode: str = "global"
    primary_saerl_mode: str = "shared_plus_heads"
    chemistry_families: str = ""
    chemistry_aware_baselines: bool = False
    saerl_eval_calibration_json: str = ""
    skip_detailed_figures: bool = False
    adaptive_horizon: bool = True
    min_episode_minutes: float = 120.0
    feasible_time_slack: float = 1.35
    max_steps_cap: int = 5000
    strict_chemistry_assets: bool = True


def parse_args() -> EvalConfig:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SAERL vs CCCV/MPC.")
    parser.add_argument("--dataset-csv", type=str, default="data/training/saerl_phase2_dataset.csv")
    parser.add_argument("--split-manifest-json", type=str, default="data/training/saerl_phase2_splits.json")
    parser.add_argument("--ensemble-root", type=str, default="models/saerl_phase2/ensemble")
    parser.add_argument(
        "--ensemble-root-template",
        type=str,
        default="",
        help="Optional template for objective-specific ensemble roots (e.g. models/saerl_phase2b/ensemble/{objective}).",
    )
    parser.add_argument("--policy-root", type=str, default="models/saerl_phase2/policy")
    parser.add_argument(
        "--policy-root-template",
        type=str,
        default="",
        help="Optional template for objective-specific policy roots (e.g. models/saerl_phase2b/policy/{objective}).",
    )
    parser.add_argument("--output-root", type=str, default="results/saerl_phase2/evaluation")
    parser.add_argument("--objective", type=str, default="all")
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--target-soc", type=float, default=0.8)
    parser.add_argument("--n-series", type=int, default=20)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--max-charge-current-a", type=float, default=10.0)
    parser.add_argument("--balancing-type", type=str, default="passive")
    parser.add_argument("--initial-soc", type=float, default=0.2)
    parser.add_argument("--ambient-temp-c", type=float, default=25.0)
    parser.add_argument("--standardized-root", type=str, default="data/standardized")
    parser.add_argument("--params-root", type=str, default="data/standardized_params")
    parser.add_argument("--dataset-families", type=str, default="nasa,calce,matr")
    parser.add_argument("--max-files-per-dataset", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument(
        "--chemistry-mode",
        type=str,
        default="global",
        choices=["global", "family_specific", "shared_plus_heads", "all"],
        help="SAERL checkpoint routing mode. Use 'all' to compare all chemistry-aware variants directly.",
    )
    parser.add_argument(
        "--primary-saerl-mode",
        type=str,
        default="shared_plus_heads",
        choices=["global", "family_specific", "shared_plus_heads"],
        help="Primary SAERL variant used for acceptance checks and legacy 'saerl' alias.",
    )
    parser.add_argument(
        "--chemistry-families",
        type=str,
        default="",
        help="Optional comma list of families used for chemistry-aware routing (default derives from dataset-families).",
    )
    parser.add_argument(
        "--chemistry-aware-baselines",
        action="store_true",
        help="Also evaluate chemistry-aware baseline variants (CCCV/MPC family_specific and shared_plus_heads).",
    )
    parser.add_argument(
        "--saerl-eval-calibration-json",
        type=str,
        default="",
        help=(
            "Optional JSON with SAERL eval-time family/objective calibration overrides. "
            "Supported sections: defaults, objectives, families, family_objectives."
        ),
    )
    parser.add_argument(
        "--skip-detailed-figures",
        action="store_true",
        help=(
            "Skip per-controller/per-scenario figure generation during evaluation "
            "(recommended for faster full sweeps)."
        ),
    )
    parser.add_argument(
        "--disable-adaptive-horizon",
        action="store_true",
        help=(
            "Disable horizon scaling based on dt/current feasibility. "
            "By default adaptive horizon is enabled for fair cross-chemistry timing."
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
    parser.add_argument(
        "--allow-chemistry-fallback",
        action="store_true",
        help=(
            "Allow relaxed checkpoint fallback across chemistry modes when exact mode checkpoints "
            "are missing. Disabled by default for strict mode-vs-mode comparisons."
        ),
    )
    args = parser.parse_args()

    return EvalConfig(
        dataset_csv=args.dataset_csv,
        split_manifest_json=args.split_manifest_json,
        ensemble_root=args.ensemble_root,
        ensemble_root_template=args.ensemble_root_template,
        policy_root=args.policy_root,
        policy_root_template=args.policy_root_template,
        output_root=args.output_root,
        objective=args.objective,
        fold=args.fold,
        run_ablations=bool(args.run_ablations),
        max_steps=args.max_steps,
        target_soc=args.target_soc,
        n_series=args.n_series,
        n_parallel=args.n_parallel,
        max_charge_current_a=args.max_charge_current_a,
        balancing_type=args.balancing_type,
        initial_soc=args.initial_soc,
        ambient_temp_c=args.ambient_temp_c,
        standardized_root=args.standardized_root,
        params_root=args.params_root,
        dataset_families=args.dataset_families,
        max_files_per_dataset=args.max_files_per_dataset,
        random_seed=args.random_seed,
        chemistry_mode=args.chemistry_mode,
        primary_saerl_mode=args.primary_saerl_mode,
        chemistry_families=args.chemistry_families,
        chemistry_aware_baselines=bool(args.chemistry_aware_baselines),
        saerl_eval_calibration_json=args.saerl_eval_calibration_json,
        skip_detailed_figures=bool(args.skip_detailed_figures),
        adaptive_horizon=not bool(args.disable_adaptive_horizon),
        min_episode_minutes=float(args.min_episode_minutes),
        feasible_time_slack=float(args.feasible_time_slack),
        max_steps_cap=int(args.max_steps_cap),
        strict_chemistry_assets=not bool(args.allow_chemistry_fallback),
    )


def save_figure(fig: plt.Figure, path_root: Path) -> None:
    path_root.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_root.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_root.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def resolve_objective_root(base_root: str, template: str, objective_key: str) -> Path:
    if template:
        if "{objective}" in template:
            return Path(template.format(objective=objective_key))
        return Path(template)
    return Path(base_root)


def split_csv_arg(value: str) -> List[str]:
    return [x.strip().lower() for x in str(value).split(",") if x.strip()]


def selected_saerl_modes(mode: str) -> List[str]:
    if mode == "all":
        return ["global", "family_specific", "shared_plus_heads"]
    return [str(mode).strip().lower()]


def saerl_mode_controller_name(mode: str) -> str:
    return f"saerl_{str(mode).strip().lower()}"


def resolve_chemistry_families(config: EvalConfig) -> List[str]:
    families = split_csv_arg(config.chemistry_families) if str(config.chemistry_families).strip() else []
    if not families:
        families = split_csv_arg(config.dataset_families)
    # Stable unique ordering.
    out: List[str] = []
    seen = set()
    for fam in families:
        if fam not in seen:
            out.append(fam)
            seen.add(fam)
    return out


EVAL_NUMERIC_OVERRIDES: Dict[str, str] = {
    "saerl_score_soc_gain_weight": "score_soc_gain_weight",
    "saerl_score_time_weight": "score_time_weight",
    "saerl_score_temp_weight": "score_temp_weight",
    "saerl_score_degradation_weight": "score_degradation_weight",
    "saerl_score_imbalance_weight": "score_imbalance_weight",
    "saerl_score_safety_weight": "score_safety_weight",
    "saerl_score_risk_weight": "score_risk_weight",
    "saerl_anti_stall_soc_gap": "anti_stall_soc_gap",
    "saerl_anti_stall_low_risk_threshold": "anti_stall_low_risk_threshold",
    "saerl_anti_stall_duration_s": "anti_stall_duration_s",
    "saerl_anti_stall_risk_scale": "anti_stall_risk_scale",
    "saerl_min_safe_charge_fraction": "min_safe_charge_fraction",
    "saerl_imbalance_margin_v": "imbalance_margin_v",
    "saerl_risk_uncertainty_weight": "risk_uncertainty_weight",
    "saerl_risk_temp_weight": "risk_temp_weight",
    "saerl_risk_voltage_weight": "risk_voltage_weight",
    "saerl_risk_imbalance_weight": "risk_imbalance_weight",
}
EVAL_BOOL_OVERRIDES: Dict[str, str] = {
    "saerl_enable_antistall": "enable_antistall",
    "saerl_enable_shield": "enable_shield",
    "saerl_use_adaptive_gate": "use_adaptive_gate",
}
EVAL_CAL_SCALE_KEYS: Dict[str, str] = {
    "saerl_calibration_scale_gru": "gru",
    "saerl_calibration_scale_mlp": "mlp",
    "saerl_calibration_scale_rf": "rf",
}


def load_optional_json(path: str) -> Dict[str, Any]:
    raw = str(path).strip()
    if not raw:
        return {}
    p = Path(raw)
    if not p.exists():
        raise FileNotFoundError(f"Eval calibration JSON not found: {p}")
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Eval calibration JSON must be a dict.")
    return payload


def _apply_eval_override_node(base: Dict[str, Any], node: Any) -> None:
    if not isinstance(node, dict):
        return
    for key in list(EVAL_NUMERIC_OVERRIDES.keys()) + list(EVAL_BOOL_OVERRIDES.keys()) + list(EVAL_CAL_SCALE_KEYS.keys()):
        if key not in node:
            continue
        base[key] = node[key]


def resolve_eval_calibration_overrides(
    payload: Dict[str, Any],
    family: str,
    objective: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    fam = str(family).strip().lower()
    obj = str(objective).strip().lower()

    _apply_eval_override_node(out, payload.get("defaults", {}))
    objectives = payload.get("objectives", {})
    if isinstance(objectives, dict):
        _apply_eval_override_node(out, objectives.get(obj, {}))
        _apply_eval_override_node(out, objectives.get(objective, {}))
    families = payload.get("families", {})
    if isinstance(families, dict):
        fam_node = families.get(fam, families.get(family, {}))
        _apply_eval_override_node(out, fam_node)
        if isinstance(fam_node, dict):
            fam_obj = fam_node.get("objectives", {})
            if isinstance(fam_obj, dict):
                _apply_eval_override_node(out, fam_obj.get(obj, {}))
                _apply_eval_override_node(out, fam_obj.get(objective, {}))
    fam_obj = payload.get("family_objectives", {})
    if isinstance(fam_obj, dict):
        fam_node = fam_obj.get(fam, fam_obj.get(family, {}))
        if isinstance(fam_node, dict):
            _apply_eval_override_node(out, fam_node.get(obj, {}))
            _apply_eval_override_node(out, fam_node.get(objective, {}))
    return out


def apply_saerl_eval_calibration(
    predictor: AdaptiveEnsemblePredictor,
    overrides: Dict[str, Any],
) -> None:
    if not overrides:
        return
    for key, attr in EVAL_NUMERIC_OVERRIDES.items():
        if key not in overrides:
            continue
        try:
            value = float(overrides[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            setattr(predictor.config, attr, value)
    for key, attr in EVAL_BOOL_OVERRIDES.items():
        if key not in overrides:
            continue
        value = overrides[key]
        if isinstance(value, str):
            v = value.strip().lower()
            bool_value = v in {"1", "true", "yes", "on"}
        else:
            bool_value = bool(value)
        setattr(predictor.config, attr, bool_value)
    for key, expert in EVAL_CAL_SCALE_KEYS.items():
        if key not in overrides:
            continue
        try:
            scale = float(overrides[key])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(scale) or scale <= 0.0:
            continue
        predictor.calibration[expert] = float(np.clip(predictor.calibration[expert] * scale, 0.25, 8.0))


def checkpoint_dir_candidates(
    mode: str,
    base_root: Path,
    family: str,
    fold_id: int,
    strict: bool = True,
) -> List[Path]:
    fold_dir = f"fold_{fold_id:02d}"
    family = str(family).lower()
    mode = str(mode).lower()
    if mode == "global":
        return [
            base_root / fold_dir,
            base_root / "global" / fold_dir,
            base_root / "shared" / fold_dir,
        ]
    if mode == "family_specific":
        cands = [
            base_root / "family_specific" / family / fold_dir,
            base_root / family / fold_dir,
        ]
        if not strict:
            cands.extend(
                [
                    base_root / "shared_plus_heads" / family / fold_dir,
                    base_root / "shared_plus_heads" / "shared" / fold_dir,
                    base_root / "global" / fold_dir,
                    base_root / fold_dir,
                ]
            )
        return cands
    if mode == "shared_plus_heads":
        cands = [
            base_root / "shared_plus_heads" / family / fold_dir,
            base_root / "shared_plus_heads" / "shared" / fold_dir,
        ]
        if not strict:
            cands.extend(
                [
                    base_root / family / fold_dir,
                    base_root / "family_specific" / family / fold_dir,
                    base_root / "global" / fold_dir,
                    base_root / fold_dir,
                ]
            )
        return cands
    return [base_root / fold_dir]


def try_load_saerl_assets(
    ensemble_root: Path,
    policy_root: Path,
    mode: str,
    family: str,
    fold_id: int,
    strict: bool = True,
) -> Optional[Tuple[AdaptiveEnsemblePredictor, ResidualActorPolicy, Path, Path, List[Path], List[Path]]]:
    ensemble_candidates = checkpoint_dir_candidates(
        mode=mode,
        base_root=ensemble_root,
        family=family,
        fold_id=fold_id,
        strict=strict,
    )
    policy_candidates = checkpoint_dir_candidates(
        mode=mode,
        base_root=policy_root,
        family=family,
        fold_id=fold_id,
        strict=strict,
    )

    predictor: Optional[AdaptiveEnsemblePredictor] = None
    actor: Optional[ResidualActorPolicy] = None
    used_ensemble = None
    used_policy = None

    for cand in ensemble_candidates:
        if (cand / "metadata.json").exists():
            predictor = AdaptiveEnsemblePredictor.load(directory=cand, device="cpu")
            used_ensemble = cand
            break
    for cand in policy_candidates:
        actor_path = cand / "residual_actor.pt"
        if actor_path.exists():
            actor = ResidualActorPolicy.load(path=actor_path, device="cpu")
            used_policy = cand
            break

    if predictor is None or actor is None:
        return None
    return predictor, actor, Path(used_ensemble), Path(used_policy), ensemble_candidates, policy_candidates


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
    alpha = 1.0 if mode == "family_specific" else 0.5
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
    alpha = 1.0 if mode == "family_specific" else 0.5
    w_soc_mult = _blend_multiplier(tune["mpc_w_soc_mult"], alpha) * float(obj_tune.get("mpc_w_soc_mult", 1.0))
    w_temp_mult = _blend_multiplier(tune["mpc_w_temp_mult"], alpha) * float(obj_tune.get("mpc_w_temp_mult", 1.0))
    w_imb_mult = _blend_multiplier(tune["mpc_w_imb_mult"], alpha) * float(obj_tune.get("mpc_w_imb_mult", 1.0))
    horizon_mult = _blend_multiplier(tune["mpc_horizon_mult"], alpha) * float(obj_tune.get("mpc_horizon_mult", 1.0))
    cfg.w_soc *= w_soc_mult
    cfg.w_temp *= w_temp_mult
    cfg.w_imbalance *= w_imb_mult
    cfg.horizon_steps = int(np.clip(round(cfg.horizon_steps * horizon_mult), 4, 16))
    return cfg


def plot_safety_envelope(results: pd.DataFrame, cv_voltage_v: float, output_path: Path) -> None:
    if results.empty:
        return
    t = results["time"].to_numpy(dtype=float) / 60.0
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    axes[0, 0].plot(t, results["pack_voltage"], color="#1D3557")
    axes[0, 0].axhline(cv_voltage_v, linestyle="--", color="#D62828")
    axes[0, 0].set_title("Voltage vs Safety Limit")
    axes[0, 0].set_xlabel("Time (min)")
    axes[0, 0].set_ylabel("Pack voltage (V)")

    axes[0, 1].plot(t, results["pack_temperature"], color="#E76F51")
    axes[0, 1].axhline(42.0, linestyle="--", color="black")
    axes[0, 1].set_title("Temperature Envelope")
    axes[0, 1].set_xlabel("Time (min)")
    axes[0, 1].set_ylabel("Pack temperature (C)")

    axes[1, 0].plot(t, -results["pack_current"], color="#2A9D8F")
    axes[1, 0].set_title("Charge Current Profile")
    axes[1, 0].set_xlabel("Time (min)")
    axes[1, 0].set_ylabel("Charge current (A)")

    safety_counts = [count_safety_events(x) for x in results["safety_events"]]
    axes[1, 1].plot(t, safety_counts, color="#6A4C93")
    axes[1, 1].set_title("Safety Event Count per Timestep")
    axes[1, 1].set_xlabel("Time (min)")
    axes[1, 1].set_ylabel("Event count")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_ensemble_diagnostics(results: pd.DataFrame, output_path: Path) -> None:
    if results.empty:
        return
    needed = {"risk_score", "expert_w_gru", "expert_w_mlp", "expert_w_rf"}
    if not needed.issubset(set(results.columns.tolist())):
        return
    t = results["time"].to_numpy(dtype=float) / 60.0
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True)
    axes[0].plot(t, results["expert_w_gru"], label="GRU")
    axes[0].plot(t, results["expert_w_mlp"], label="MLP")
    axes[0].plot(t, results["expert_w_rf"], label="RF")
    axes[0].set_ylabel("Gate weight")
    axes[0].set_title("Expert Weights Over Time")
    axes[0].legend(loc="best")
    axes[1].plot(t, results["risk_score"], color="#D62828")
    axes[1].set_title("Fused Risk Score")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylabel("Risk")
    fig.tight_layout()
    save_figure(fig, output_path)


def run_rollout(
    env,
    controller_name: str,
    controller_obj,
    initial_soc: float,
    initial_temp_c: float,
    target_soc: float,
    cv_voltage_v: float,
) -> pd.DataFrame:
    env.reset(initial_soc=initial_soc, temperature=initial_temp_c)
    trim_pack_histories(env.pack)
    if hasattr(controller_obj, "reset"):
        controller_obj.reset()

    rows: List[Dict[str, Any]] = []
    state = initial_state_from_env(env)

    # Base MPC is needed for SAERL residual action.
    mpc_for_saerl = RolloutMPCController(
        config=MPCConfig(),
        cv_voltage_v=cv_voltage_v,
        max_charge_current_a=env.max_charge_current_a,
        target_soc=target_soc,
    )
    mpc_for_saerl.reset()

    for _ in range(env.max_steps):
        if controller_name.lower().startswith("saerl"):
            mpc_action, mpc_info = mpc_for_saerl.act(state, env)
            t0 = time.perf_counter()
            action, info = controller_obj.act(
                state=state,
                env=env,
                mpc_action=float(mpc_action),
                cv_voltage_v=cv_voltage_v,
            )
            inference_ms = (time.perf_counter() - t0) * 1000.0
            info["mpc_action"] = float(mpc_action)
            if "mpc_best_cost" in mpc_info:
                info["mpc_best_cost"] = float(mpc_info["mpc_best_cost"])
        else:
            t0 = time.perf_counter()
            action, info = controller_obj.act(state, env)
            inference_ms = (time.perf_counter() - t0) * 1000.0

        _, reward, done, next_state = env.step(action)
        trim_pack_histories(env.pack)

        row = dict(next_state)
        row["controller"] = controller_name
        row["action"] = float(action)
        row["reward"] = float(reward)
        row["inference_latency_ms"] = float(inference_ms)
        row["q_loss_total"] = float(sum(float(cell.Q_loss) for cell in env.pack.cells))
        row["shield_used"] = bool(info.get("shield_used", False))
        row["antistall_used"] = bool(info.get("antistall_used", False))
        row["risk_score"] = float(info.get("risk_score", np.nan))
        row["risk_score_normalized"] = float(info.get("risk_score_normalized", np.nan))
        if isinstance(info.get("expert_weights"), dict):
            ew = info["expert_weights"]
            row["expert_w_gru"] = float(ew.get("gru", np.nan))
            row["expert_w_mlp"] = float(ew.get("mlp", np.nan))
            row["expert_w_rf"] = float(ew.get("rf", np.nan))
        if "proposal_delta_action" in info:
            row["proposal_delta_action"] = float(info["proposal_delta_action"])
        if "mpc_action" in info:
            row["mpc_action"] = float(info["mpc_action"])
        rows.append(row)

        if hasattr(controller_obj, "observe_transition"):
            controller_obj.observe_transition(next_state)

        state = {
            "pack_soc": float(next_state["pack_soc"]),
            "pack_voltage": float(next_state["pack_voltage"]),
            "pack_temperature": float(next_state["pack_temperature"]),
            "voltage_imbalance": float(next_state["voltage_imbalance"]),
            "pack_current": float(next_state["pack_current"]),
            "safety_events": next_state.get("safety_events", {}),
        }
        if done:
            break

    return pd.DataFrame(rows)


def acceptance_check(
    cccv_metrics: Dict[str, float],
    mpc_metrics: Dict[str, float],
    saerl_metrics: Dict[str, float],
) -> Dict[str, Any]:
    pass_safety_zero = int(saerl_metrics.get("safety_event_count", 1e9)) == 0
    temp_bound = min(
        float(cccv_metrics.get("peak_pack_temperature_c", np.inf)),
        float(mpc_metrics.get("peak_pack_temperature_c", np.inf)),
    ) + 0.2
    pass_temp = float(saerl_metrics.get("peak_pack_temperature_c", np.inf)) <= temp_bound

    q_mpc = float(mpc_metrics.get("q_loss_total", np.nan))
    q_saerl = float(saerl_metrics.get("q_loss_total", np.nan))
    if np.isfinite(q_mpc) and q_mpc > 0:
        pass_q_loss = bool(q_saerl <= 1.05 * q_mpc)
    else:
        pass_q_loss = True

    cccv_hit = float(cccv_metrics.get("final_soc", 0.0)) >= 0.8
    mpc_hit = float(mpc_metrics.get("final_soc", 0.0)) >= 0.8
    cccv_safe = int(cccv_metrics.get("safety_event_count", 1e9)) == 0
    mpc_safe = int(mpc_metrics.get("safety_event_count", 1e9)) == 0
    if cccv_hit or mpc_hit:
        candidate_times = []
        candidate_times_safe = []
        if cccv_hit:
            t = float(cccv_metrics.get("time_to_80_soc_min", np.nan))
            t_eff = t if np.isfinite(t) else float(cccv_metrics.get("charge_time_min", np.inf))
            candidate_times.append(t_eff)
            if cccv_safe:
                candidate_times_safe.append(t_eff)
        if mpc_hit:
            t = float(mpc_metrics.get("time_to_80_soc_min", np.nan))
            t_eff = t if np.isfinite(t) else float(mpc_metrics.get("charge_time_min", np.inf))
            candidate_times.append(t_eff)
            if mpc_safe:
                candidate_times_safe.append(t_eff)
        if candidate_times_safe:
            fastest = float(min(candidate_times_safe))
            perf_reference_source = "safe_baselines_only"
        else:
            fastest = float(min(candidate_times)) if candidate_times else np.inf
            perf_reference_source = "all_target_reaching_baselines"
        t_saerl = float(saerl_metrics.get("time_to_80_soc_min", np.nan))
        if not np.isfinite(t_saerl):
            t_saerl = float(saerl_metrics.get("charge_time_min", np.inf))
        pass_perf = bool(float(saerl_metrics.get("final_soc", 0.0)) >= 0.8 and t_saerl <= fastest * 1.10)
        perf_mode = "time_to_80"
    else:
        safe_socs: List[float] = []
        all_socs = [
            float(cccv_metrics.get("final_soc", 0.0)),
            float(mpc_metrics.get("final_soc", 0.0)),
        ]
        if cccv_safe:
            safe_socs.append(float(cccv_metrics.get("final_soc", 0.0)))
        if mpc_safe:
            safe_socs.append(float(mpc_metrics.get("final_soc", 0.0)))
        if safe_socs:
            best_baseline_soc = float(max(safe_socs))
            perf_reference_source = "safe_baselines_only_final_soc"
        else:
            best_baseline_soc = float(max(all_socs))
            perf_reference_source = "all_baselines_final_soc"
        pass_perf = bool(float(saerl_metrics.get("final_soc", 0.0)) >= best_baseline_soc + 0.02)
        perf_mode = "final_soc_gain"
        fastest = float("nan")

    scenario_pass = bool(pass_safety_zero and pass_temp and pass_q_loss and pass_perf)
    return {
        "pass_safety_zero": bool(pass_safety_zero),
        "pass_temp": bool(pass_temp),
        "pass_q_loss": bool(pass_q_loss),
        "pass_perf": bool(pass_perf),
        "perf_mode": perf_mode,
        "perf_reference_source": perf_reference_source,
        "perf_reference_time_min": float(fastest) if np.isfinite(fastest) else float("nan"),
        "scenario_pass": scenario_pass,
    }


def recommend_episode_max_steps(
    setting,
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


def main() -> None:
    config = parse_args()
    np.random.seed(config.random_seed)
    apply_publication_style()

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
    with Path(config.split_manifest_json).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    folds = manifest.get("folds", [])
    if config.fold != "all":
        fold_id = int(config.fold)
        folds = [f for f in folds if int(f.get("fold_id", -1)) == fold_id]
    if not folds:
        raise SystemExit("No folds selected.")

    objectives = build_default_objectives()
    selected_objectives = list(objectives.keys()) if config.objective == "all" else [config.objective]

    run_config, scenarios = load_data_calibrated_scenarios(
        standardized_root=config.standardized_root,
        params_root=config.params_root,
        dataset_families=config.dataset_families,
        max_files_per_dataset=config.max_files_per_dataset,
        n_series=config.n_series,
        n_parallel=config.n_parallel,
        max_charge_current_a=config.max_charge_current_a,
        balancing_type=config.balancing_type,
        initial_soc=config.initial_soc,
        target_soc=config.target_soc,
        ambient_temp_c=config.ambient_temp_c,
    )
    scenario_map = {(str(s["family"]), str(s["case_id"])): s for s in scenarios}

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    acceptance_rows: List[Dict[str, Any]] = []
    chemistry_families = resolve_chemistry_families(config=config)
    eval_calibration_payload = load_optional_json(config.saerl_eval_calibration_json)
    saerl_modes = selected_saerl_modes(config.chemistry_mode)
    primary_mode = str(config.primary_saerl_mode).strip().lower()
    if primary_mode not in saerl_modes:
        saerl_modes.append(primary_mode)

    for fold in folds:
        fold_id = int(fold["fold_id"])
        case_test_ids = set(fold["splits"].get("case_test", []))
        case_test_df = df[df["episode_id"].astype(str).isin(case_test_ids)].copy()
        case_tuples = sorted(
            {
                (str(r["objective"]), str(r["dataset_family"]), str(r["dataset_case"]))
                for _, r in case_test_df[["objective", "dataset_family", "dataset_case"]].drop_duplicates().iterrows()
                if str(r["objective"]) in selected_objectives
            }
        )
        if not case_tuples:
            continue

        saerl_asset_cache: Dict[
            Tuple[str, str, str],
            Tuple[AdaptiveEnsemblePredictor, ResidualActorPolicy, Path, Path, List[Path], List[Path]],
        ] = {}

        for objective_key, family, case in case_tuples:
            if (family, case) not in scenario_map:
                continue
            family_l = str(family).lower()
            eval_calibration = resolve_eval_calibration_overrides(
                payload=eval_calibration_payload,
                family=family_l,
                objective=objective_key,
            )
            ensemble_root = resolve_objective_root(
                base_root=config.ensemble_root,
                template=config.ensemble_root_template,
                objective_key=objective_key,
            )
            policy_root = resolve_objective_root(
                base_root=config.policy_root,
                template=config.policy_root_template,
                objective_key=objective_key,
            )

            setting = build_setting_for_objective(
                run_config=run_config,
                objective_key=objective_key,
                objective=objectives[objective_key],
                scenario=scenario_map[(family, case)],
            )
            effective_max_steps = int(config.max_steps)
            if config.adaptive_horizon:
                effective_max_steps, horizon_info = recommend_episode_max_steps(
                    setting=setting,
                    base_max_steps=config.max_steps,
                    target_soc=config.target_soc,
                    min_episode_minutes=config.min_episode_minutes,
                    feasible_time_slack=config.feasible_time_slack,
                    max_steps_cap=config.max_steps_cap,
                )
            else:
                dt_s = float(max(1e-6, setting.dt_s))
                horizon_info = {
                    "dt_s": dt_s,
                    "base_max_steps": float(config.max_steps),
                    "effective_max_steps": float(config.max_steps),
                    "base_horizon_s": float(config.max_steps * dt_s),
                    "effective_horizon_s": float(config.max_steps * dt_s),
                    "ideal_cc_time_to_target_s": float("nan"),
                    "required_current_for_base_horizon_a": float("nan"),
                    "configured_max_charge_current_a": float(setting.max_charge_current_a),
                    "current_feasibility_ratio_vs_base_horizon": float("nan"),
                }
            fold_case_root = output_root / f"fold_{fold_id:02d}" / objective_key / family / case
            fold_case_root.mkdir(parents=True, exist_ok=True)

            controllers: Dict[str, Any] = {
                "cccv": CCCVController(
                    config=CCCVConfig(),
                    cv_voltage_v=setting.cv_voltage_v,
                    max_charge_current_a=setting.max_charge_current_a,
                    target_soc=config.target_soc,
                ),
                "mpc": RolloutMPCController(
                    config=MPCConfig(),
                    cv_voltage_v=setting.cv_voltage_v,
                    max_charge_current_a=setting.max_charge_current_a,
                    target_soc=config.target_soc,
                ),
            }

            if config.chemistry_aware_baselines and family_l in chemistry_families:
                for baseline_mode in ["family_specific", "shared_plus_heads"]:
                    c_name = f"cccv_{baseline_mode}"
                    m_name = f"mpc_{baseline_mode}"
                    controllers[c_name] = CCCVController(
                        config=chemistry_aware_cccv_config(
                            family=family_l,
                            mode=baseline_mode,
                            objective_key=objective_key,
                        ),
                        cv_voltage_v=setting.cv_voltage_v,
                        max_charge_current_a=setting.max_charge_current_a,
                        target_soc=config.target_soc,
                    )
                    controllers[m_name] = RolloutMPCController(
                        config=chemistry_aware_mpc_config(
                            family=family_l,
                            mode=baseline_mode,
                            objective_key=objective_key,
                        ),
                        cv_voltage_v=setting.cv_voltage_v,
                        max_charge_current_a=setting.max_charge_current_a,
                        target_soc=config.target_soc,
                    )

            available_saerl_names: List[str] = []
            for mode in saerl_modes:
                cache_key = (objective_key, mode, family_l)
                if cache_key not in saerl_asset_cache:
                    loaded = try_load_saerl_assets(
                        ensemble_root=ensemble_root,
                        policy_root=policy_root,
                        mode=mode,
                        family=family_l,
                        fold_id=fold_id,
                        strict=config.strict_chemistry_assets,
                    )
                    if loaded is not None:
                        saerl_asset_cache[cache_key] = loaded
                    else:
                        cand_ens = checkpoint_dir_candidates(
                            mode=mode,
                            base_root=ensemble_root,
                            family=family_l,
                            fold_id=fold_id,
                            strict=config.strict_chemistry_assets,
                        )
                        cand_pol = checkpoint_dir_candidates(
                            mode=mode,
                            base_root=policy_root,
                            family=family_l,
                            fold_id=fold_id,
                            strict=config.strict_chemistry_assets,
                        )
                        print(
                            f"[WARN] Missing SAERL assets for mode={mode}, family={family_l}, fold={fold_id}. "
                            f"ensemble_candidates={','.join(str(x) for x in cand_ens)}; "
                            f"policy_candidates={','.join(str(x) for x in cand_pol)}"
                        )
                loaded_assets = saerl_asset_cache.get(cache_key)
                if loaded_assets is None:
                    continue
                predictor_obj, actor_obj, _, _, _, _ = loaded_assets
                predictor_for_eval = copy.deepcopy(predictor_obj)
                apply_saerl_eval_calibration(
                    predictor=predictor_for_eval,
                    overrides=eval_calibration,
                )
                c_name = saerl_mode_controller_name(mode)
                controllers[c_name] = SafeAdaptiveEnsembleController(
                    predictor=predictor_for_eval,
                    actor=copy.deepcopy(actor_obj),
                )
                available_saerl_names.append(c_name)

            primary_name = saerl_mode_controller_name(primary_mode)
            if primary_name not in available_saerl_names:
                if available_saerl_names and not config.strict_chemistry_assets:
                    print(
                        f"[WARN] Primary SAERL mode missing ({primary_name}); "
                        f"falling back to {available_saerl_names[0]} for fold={fold_id}, "
                        f"objective={objective_key}, case={family}/{case}."
                    )
                    primary_name = available_saerl_names[0]
                else:
                    print(
                        f"Skipping eval (missing SAERL checkpoints): fold={fold_id}, objective={objective_key}, case={family}/{case}"
                    )
                    continue
            controllers["saerl"] = copy.deepcopy(controllers[primary_name])

            if config.run_ablations:
                primary_mode_from_name = str(primary_name).replace("saerl_", "", 1)
                primary_assets = saerl_asset_cache.get((objective_key, primary_mode_from_name, family_l))
                if primary_assets is None:
                    primary_assets = saerl_asset_cache.get((objective_key, primary_mode, family_l))
                if primary_assets is None:
                    # Fallback to controller copy if cache key came from mode aliasing.
                    primary_ctrl = controllers["saerl"]
                    primary_predictor = copy.deepcopy(primary_ctrl.predictor)
                    primary_actor = copy.deepcopy(primary_ctrl.actor)
                else:
                    primary_predictor = copy.deepcopy(primary_assets[0])
                    primary_actor = copy.deepcopy(primary_assets[1])
                apply_saerl_eval_calibration(
                    predictor=primary_predictor,
                    overrides=eval_calibration,
                )

                p_no_gate = copy.deepcopy(primary_predictor)
                p_no_gate.config.use_adaptive_gate = False
                controllers["saerl_no_gate"] = SafeAdaptiveEnsembleController(
                    predictor=p_no_gate,
                    actor=copy.deepcopy(primary_actor),
                )

                p_no_rf = copy.deepcopy(primary_predictor)
                p_no_rf.set_disabled_experts(["rf"])
                controllers["saerl_no_rf"] = SafeAdaptiveEnsembleController(
                    predictor=p_no_rf,
                    actor=copy.deepcopy(primary_actor),
                )

                p_no_gru = copy.deepcopy(primary_predictor)
                p_no_gru.set_disabled_experts(["gru"])
                controllers["saerl_no_gru"] = SafeAdaptiveEnsembleController(
                    predictor=p_no_gru,
                    actor=copy.deepcopy(primary_actor),
                )

                p_no_antistall = copy.deepcopy(primary_predictor)
                p_no_antistall.config.enable_antistall = False
                controllers["saerl_no_antistall"] = SafeAdaptiveEnsembleController(
                    predictor=p_no_antistall,
                    actor=copy.deepcopy(primary_actor),
                )

                p_no_shield = copy.deepcopy(primary_predictor)
                p_no_shield.config.enable_shield = False
                controllers["saerl_no_shield"] = SafeAdaptiveEnsembleController(
                    predictor=p_no_shield,
                    actor=copy.deepcopy(primary_actor),
                )

            all_results: Dict[str, pd.DataFrame] = {}
            all_metrics: Dict[str, Dict[str, float]] = {}
            for name, ctrl in controllers.items():
                alias_source = None
                if name == "saerl" and primary_name != "saerl" and primary_name in all_results:
                    alias_source = primary_name

                if alias_source is not None:
                    # Avoid duplicate rollout: `saerl` is an alias of the selected primary SAERL mode.
                    results = all_results[alias_source].copy(deep=True)
                    metrics = dict(all_metrics[alias_source])
                    metrics["controller"] = name
                else:
                    env = make_env(
                        setting=setting,
                        max_steps=effective_max_steps,
                        target_soc=config.target_soc,
                    )
                    results = run_rollout(
                        env=env,
                        controller_name=name,
                        controller_obj=ctrl,
                        initial_soc=setting.initial_soc,
                        initial_temp_c=setting.initial_temp_c,
                        target_soc=config.target_soc,
                        cv_voltage_v=setting.cv_voltage_v,
                    )
                    metrics = compute_extended_metrics(
                        results=results,
                        target_soc=config.target_soc,
                        initial_soc=setting.initial_soc,
                        controller_name=name,
                    )
                all_results[name] = results
                all_metrics[name] = metrics

                algo_root = fold_case_root / name
                algo_root.mkdir(parents=True, exist_ok=True)
                results.to_csv(algo_root / "trajectory.csv", index=False)
                save_json(
                    algo_root / "metrics.json",
                    {
                        "fold_id": fold_id,
                        "objective": objective_key,
                        "dataset_family": family,
                        "dataset_case": case,
                        "algorithm": name,
                        "controller_chemistry_mode": (
                            str(name).replace("saerl_", "", 1)
                            if str(name).startswith("saerl_")
                            else (
                                "family_specific"
                                if str(name).endswith("_family_specific")
                                else ("shared_plus_heads" if str(name).endswith("_shared_plus_heads") else "global")
                            )
                        ),
                        "controller_chemistry_family": family_l,
                        "controller_eval_calibration": eval_calibration if str(name).startswith("saerl") else {},
                        "metrics": metrics,
                        "cv_voltage_v": setting.cv_voltage_v,
                        "max_charge_current_a": setting.max_charge_current_a,
                        "effective_max_steps": int(effective_max_steps),
                        "horizon_info": horizon_info,
                    },
                )
                if not config.skip_detailed_figures:
                    plot_baseline_timeseries(
                        results=results,
                        controller_name=name.upper(),
                        target_soc=config.target_soc,
                        cv_voltage_v=setting.cv_voltage_v,
                        output_path=algo_root / "figures" / "01_timeseries",
                    )
                    plot_phase_portraits(
                        results=results,
                        controller_name=name.upper(),
                        output_path=algo_root / "figures" / "02_phase_portraits",
                    )
                    plot_safety_envelope(
                        results=results,
                        cv_voltage_v=setting.cv_voltage_v,
                        output_path=algo_root / "figures" / "03_safety_envelope",
                    )
                    if name.startswith("saerl"):
                        plot_ensemble_diagnostics(
                            results=results,
                            output_path=algo_root / "figures" / "04_ensemble_diagnostics",
                        )

            metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
            metrics_df.index.name = "controller"
            comparison_root = fold_case_root / "comparison"
            comparison_root.mkdir(parents=True, exist_ok=True)
            metrics_df.to_csv(comparison_root / "metrics_summary.csv")
            save_json(
                comparison_root / "run_metadata.json",
                {
                    "fold_id": fold_id,
                    "objective": objective_key,
                    "dataset_family": family,
                    "dataset_case": case,
                    "eval_config": asdict(config),
                    "saerl_eval_calibration": eval_calibration,
                    "cv_voltage_v": setting.cv_voltage_v,
                    "max_charge_current_a": setting.max_charge_current_a,
                    "effective_max_steps": int(effective_max_steps),
                    "horizon_info": horizon_info,
                },
            )
            # Overlay figures use primary controllers.
            overlay_results = {
                "cccv": all_results["cccv"],
                "mpc": all_results["mpc"],
                "saerl": all_results["saerl"],
            }
            if not config.skip_detailed_figures:
                plot_comparison_overlay(
                    all_results=overlay_results,
                    target_soc=config.target_soc,
                    cv_voltage_v=setting.cv_voltage_v,
                    output_path=comparison_root / "figures" / "01_overlay",
                )
                primary_metrics_df = metrics_df.loc[[x for x in ["cccv", "mpc", "saerl"] if x in metrics_df.index]]
                plot_metrics_bars(
                    metrics_df=primary_metrics_df,
                    output_path=comparison_root / "figures" / "02_metrics_bar",
                )
                plot_tradeoff(
                    metrics_df=primary_metrics_df,
                    output_path=comparison_root / "figures" / "03_tradeoff",
                )

            # Acceptance criteria for main SAERL.
            check = acceptance_check(
                cccv_metrics=all_metrics["cccv"],
                mpc_metrics=all_metrics["mpc"],
                saerl_metrics=all_metrics["saerl"],
            )
            ideal_cc_time_s = float(horizon_info.get("ideal_cc_time_to_target_s", np.nan))
            base_horizon_s = float(horizon_info.get("base_horizon_s", np.nan))
            effective_horizon_s = float(horizon_info.get("effective_horizon_s", np.nan))
            feasibility_ratio = float(horizon_info.get("current_feasibility_ratio_vs_base_horizon", np.nan))
            feasible_to_target_base = bool(
                np.isfinite(base_horizon_s)
                and np.isfinite(ideal_cc_time_s)
                and base_horizon_s >= max(1e-9, config.feasible_time_slack * ideal_cc_time_s)
            )
            feasible_to_target_effective = bool(
                np.isfinite(effective_horizon_s)
                and np.isfinite(ideal_cc_time_s)
                and effective_horizon_s >= max(1e-9, config.feasible_time_slack * ideal_cc_time_s)
            )
            acceptance_row = {
                "fold_id": fold_id,
                "objective": objective_key,
                "dataset_family": family,
                "dataset_case": case,
                "saerl_primary_mode": str(primary_name).replace("saerl_", "", 1),
                **check,
                "horizon_feasibility_ratio_vs_base": feasibility_ratio,
                "horizon_ideal_cc_time_to_target_s": ideal_cc_time_s,
                "horizon_base_horizon_s": base_horizon_s,
                "horizon_effective_horizon_s": effective_horizon_s,
                "feasible_to_target_base_horizon": feasible_to_target_base,
                "feasible_to_target_effective_horizon": feasible_to_target_effective,
                "cccv_final_soc": float(all_metrics["cccv"].get("final_soc", np.nan)),
                "mpc_final_soc": float(all_metrics["mpc"].get("final_soc", np.nan)),
                "saerl_final_soc": float(all_metrics["saerl"].get("final_soc", np.nan)),
                "cccv_safety_event_count": float(all_metrics["cccv"].get("safety_event_count", np.nan)),
                "mpc_safety_event_count": float(all_metrics["mpc"].get("safety_event_count", np.nan)),
                "saerl_safety_event_count": float(all_metrics["saerl"].get("safety_event_count", np.nan)),
                "saerl_inference_latency_mean_ms": float(all_metrics["saerl"].get("inference_latency_mean_ms", np.nan)),
            }
            acceptance_rows.append(acceptance_row)
            print(
                f"Completed eval: fold={fold_id}, objective={objective_key}, "
                f"case={family}/{case}, pass={check['scenario_pass']}"
            )

    if not acceptance_rows:
        raise SystemExit("No evaluation rows produced.")

    acceptance_df = pd.DataFrame(acceptance_rows)
    acceptance_df.to_csv(output_root / "acceptance_summary.csv", index=False)

    global_stats = {
        "n_scenarios": int(len(acceptance_df)),
        "pass_rate": float(np.mean(acceptance_df["scenario_pass"].astype(float))),
        "pass_count": int(np.sum(acceptance_df["scenario_pass"].astype(bool))),
        "family_pass_counts": {
            str(k): int(np.sum(v["scenario_pass"].astype(bool)))
            for k, v in acceptance_df.groupby("dataset_family")
        },
        "family_total_counts": {
            str(k): int(len(v)) for k, v in acceptance_df.groupby("dataset_family")
        },
        "latency_mean_ms": float(np.nanmean(acceptance_df["saerl_inference_latency_mean_ms"].to_numpy(dtype=float))),
    }
    if "feasible_to_target_effective_horizon" in acceptance_df.columns:
        feasible_eff = acceptance_df[acceptance_df["feasible_to_target_effective_horizon"].astype(bool)].copy()
        global_stats["feasible_effective_total"] = int(len(feasible_eff))
        global_stats["feasible_effective_pass_count"] = int(np.sum(feasible_eff["scenario_pass"].astype(bool)))
        global_stats["feasible_effective_pass_rate"] = (
            float(np.mean(feasible_eff["scenario_pass"].astype(float))) if len(feasible_eff) else float("nan")
        )
    save_json(output_root / "global_acceptance_summary.json", global_stats)
    print("Completed SAERL evaluation.")
    print(f"Acceptance summary: {(output_root / 'acceptance_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
