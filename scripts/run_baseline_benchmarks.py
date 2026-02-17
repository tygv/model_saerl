"""Run CCCV and MPC charging baselines and generate publication-grade figures.

The two controllers are evaluated on the same HAMBRL pack environment.
Outputs are split into separate folders:
  - results/baselines/cccv
  - results/baselines/mpc
  - results/baselines/comparison

Use `--use-real-data` to run data-calibrated baselines from standardized CSVs
and fitted parameter JSON files (NASA/CALCE/MATR).
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from battery_pack_model import BatteryPack, PackConfiguration
from hambrl_pack_env import HAMBRLPackEnvironment
from pack_experiments import build_default_objectives


def apply_publication_style() -> None:
    """Configure a clean, paper-oriented matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "savefig.dpi": 350,
            "figure.dpi": 140,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def count_safety_events(safety_events: Dict) -> int:
    if not isinstance(safety_events, dict):
        return 0
    return int(
        sum(len(v) if isinstance(v, list) else (1 if v else 0) for v in safety_events.values())
    )


def time_to_soc_minutes(results: pd.DataFrame, threshold: float) -> float:
    hits = results[results["pack_soc"] >= threshold]
    if hits.empty:
        return float("nan")
    return float(hits.iloc[0]["time"] / 60.0)


def trapz_integral(y: np.ndarray, x: np.ndarray) -> float:
    if y.size == 0 or x.size == 0:
        return 0.0
    if y.size == 1 or x.size == 1:
        return 0.0
    return float(np.trapz(y, x))


@dataclass
class CCCVConfig:
    cv_hysteresis_v: float = 0.20
    kp_a_per_v: float = 8.0
    ki_a_per_vs: float = 0.03
    min_charge_current_a: float = 0.0
    soc_taper_window: float = 0.08


@dataclass
class MPCConfig:
    horizon_steps: int = 8
    discount_gamma: float = 0.98
    action_grid_points: int = 7
    temp_soft_limit_c: float = 42.0
    w_soc: float = 140.0
    w_voltage: float = 45.0
    w_temp: float = 30.0
    w_imbalance: float = 15.0
    w_current: float = 2.0
    w_smooth: float = 4.0
    terminal_reward: float = 220.0


@dataclass
class BenchmarkConfig:
    objective: str = "safe"
    output_root: str = "results/baselines"
    max_steps: int = 1800
    initial_soc: float = 0.2
    target_soc: float = 0.8
    ambient_temp_c: float = 25.0
    initial_temp_c: float = 25.0
    n_series: int = 20
    n_parallel: int = 1
    max_charge_current_a: float = 10.0
    balancing_type: str = "passive"
    use_real_data: bool = False
    standardized_root: str = "data/standardized"
    params_root: str = "data/standardized_params"
    dataset_families: str = "nasa,calce,matr"
    max_files_per_dataset: int = 1
    data_is_cell_level: bool = True
    include_cell_figures: bool = False
    adaptive_horizon: bool = True
    min_episode_minutes: float = 120.0
    feasible_time_slack: float = 1.35
    max_steps_cap: int = 5000


class CCCVController:
    """Constant-current/constant-voltage charging controller."""

    def __init__(
        self,
        config: CCCVConfig,
        cv_voltage_v: float,
        max_charge_current_a: float,
        target_soc: float,
    ) -> None:
        self.config = config
        self.cv_voltage_v = cv_voltage_v
        self.max_charge_current_a = max_charge_current_a
        self.target_soc = target_soc
        self.mode = "CC"
        self.integral_error = 0.0

    def reset(self) -> None:
        self.mode = "CC"
        self.integral_error = 0.0

    def act(self, state: Dict, env: HAMBRLPackEnvironment) -> Tuple[float, Dict]:
        if state["pack_voltage"] >= self.cv_voltage_v - self.config.cv_hysteresis_v:
            self.mode = "CV"

        if self.mode == "CC":
            charge_current_a = self.max_charge_current_a
        else:
            error_v = self.cv_voltage_v - state["pack_voltage"]
            self.integral_error = float(
                np.clip(self.integral_error + error_v * env.pack.dt, -50.0, 50.0)
            )
            if error_v <= 0.0:
                charge_current_a = 0.0
            else:
                charge_current_a = (
                    self.config.kp_a_per_v * error_v
                    + self.config.ki_a_per_vs * self.integral_error
                )
            charge_current_a = float(
                np.clip(
                    charge_current_a,
                    self.config.min_charge_current_a,
                    self.max_charge_current_a,
                )
            )

        soc_gap = max(0.0, self.target_soc - state["pack_soc"])
        taper_gain = float(np.clip(soc_gap / max(self.config.soc_taper_window, 1e-6), 0.15, 1.0))
        charge_current_a *= taper_gain
        charge_current_a = float(
            np.clip(
                charge_current_a,
                0.0,
                self.max_charge_current_a,
            )
        )

        desired_pack_current = -charge_current_a
        action = env.pack_current_to_action(desired_pack_current)
        return action, {
            "controller_mode": self.mode,
            "desired_pack_current": desired_pack_current,
        }


class RolloutMPCController:
    """Rollout MPC with receding horizon and action-grid search."""

    def __init__(
        self,
        config: MPCConfig,
        cv_voltage_v: float,
        max_charge_current_a: float,
        target_soc: float,
    ) -> None:
        self.config = config
        self.cv_voltage_v = cv_voltage_v
        self.max_charge_current_a = max_charge_current_a
        self.target_soc = target_soc
        self.action_grid = np.linspace(-1.0, 1.0, self.config.action_grid_points)
        self.prev_action = -1.0

    def reset(self) -> None:
        self.prev_action = -1.0

    def act(self, state: Dict, env: HAMBRLPackEnvironment) -> Tuple[float, Dict]:
        best_action = -1.0
        best_cost = float("inf")

        for candidate_action in self.action_grid:
            cost = self._rollout_cost(candidate_action, env)
            if cost < best_cost:
                best_cost = cost
                best_action = float(candidate_action)

        self.prev_action = best_action
        desired_pack_current = env.action_to_pack_current(best_action)
        return best_action, {
            "controller_mode": "MPC",
            "desired_pack_current": desired_pack_current,
            "mpc_best_cost": best_cost,
        }

    def _rollout_cost(self, first_action: float, env: HAMBRLPackEnvironment) -> float:
        simulated_pack = copy.deepcopy(env.pack)
        total_cost = 0.0
        discount = 1.0
        last_action = self.prev_action
        simulated_state: Dict = {
            "pack_soc": simulated_pack.pack_soc,
            "pack_voltage": simulated_pack.pack_voltage,
            "pack_temperature": simulated_pack.pack_temperature,
            "voltage_imbalance": simulated_pack.voltage_imbalance,
            "pack_current": simulated_pack.pack_current,
            "safety_events": simulated_pack.safety_events,
        }

        for step_idx in range(self.config.horizon_steps):
            if step_idx == 0:
                action = first_action
            else:
                action = self._terminal_policy(simulated_state, env)

            pack_current = env.action_to_pack_current(action)
            simulated_state = simulated_pack.step(pack_current, ambient_temp=env.ambient_temp)
            stage_cost = self._stage_cost(
                simulated_state=simulated_state,
                action=action,
                previous_action=last_action,
                env=env,
            )
            total_cost += discount * stage_cost

            if simulated_state["pack_soc"] >= self.target_soc:
                total_cost -= discount * self.config.terminal_reward
                break

            discount *= self.config.discount_gamma
            last_action = action

        return float(total_cost)

    def _terminal_policy(self, state: Dict, env: HAMBRLPackEnvironment) -> float:
        if state["pack_temperature"] >= self.config.temp_soft_limit_c:
            charge_current_a = 0.30 * self.max_charge_current_a
        elif state["pack_voltage"] >= self.cv_voltage_v - 0.12:
            charge_current_a = 0.40 * self.max_charge_current_a
        else:
            charge_current_a = self.max_charge_current_a

        soc_gap = max(0.0, self.target_soc - state["pack_soc"])
        charge_current_a *= float(np.clip(soc_gap / 0.10, 0.15, 1.0))
        desired_pack_current = -float(np.clip(charge_current_a, 0.0, self.max_charge_current_a))
        return env.pack_current_to_action(desired_pack_current)

    def _stage_cost(
        self,
        simulated_state: Dict,
        action: float,
        previous_action: float,
        env: HAMBRLPackEnvironment,
    ) -> float:
        soc_error = max(0.0, self.target_soc - simulated_state["pack_soc"])
        over_voltage = max(0.0, simulated_state["pack_voltage"] - self.cv_voltage_v)
        over_temp = max(0.0, simulated_state["pack_temperature"] - self.config.temp_soft_limit_c)
        imbalance_v = simulated_state["voltage_imbalance"]
        current_norm = abs(simulated_state["pack_current"]) / max(self.max_charge_current_a, 1e-6)
        smoothness = (action - previous_action) ** 2
        safety_count = count_safety_events(simulated_state.get("safety_events", {}))

        base_cost = (
            self.config.w_soc * (soc_error**2)
            + self.config.w_voltage * (over_voltage**2)
            + self.config.w_temp * (over_temp**2)
            + self.config.w_imbalance * (imbalance_v**2)
            + self.config.w_current * (current_norm**2)
            + self.config.w_smooth * smoothness
            + 500.0 * safety_count
        )

        hard_over_voltage = max(0.0, simulated_state["pack_voltage"] - env.pack.config.V_pack_max)
        hard_over_temp = max(0.0, simulated_state["pack_temperature"] - env.pack.config.T_pack_max)
        hard_penalty = 5000.0 * (hard_over_voltage**2 + hard_over_temp**2)
        return float(base_cost + hard_penalty)


def run_controller_episode(
    env: HAMBRLPackEnvironment,
    controller,
    name: str,
    initial_soc: float,
    initial_temp_c: float,
) -> pd.DataFrame:
    env.reset(initial_soc=initial_soc, temperature=initial_temp_c)
    trim_pack_histories(env.pack)
    controller.reset()

    records: List[Dict] = []
    current_state: Dict = {
        "pack_soc": env.pack.pack_soc,
        "pack_voltage": env.pack.pack_voltage,
        "pack_temperature": env.pack.pack_temperature,
        "voltage_imbalance": env.pack.voltage_imbalance,
        "pack_current": env.pack.pack_current,
        "safety_events": env.pack.safety_events,
    }

    for _ in range(env.max_steps):
        action, control_info = controller.act(current_state, env)
        _, reward, done, next_state = env.step(action)
        trim_pack_histories(env.pack)
        row = dict(next_state)
        row["action"] = float(action)
        row["reward"] = float(reward)
        row["controller"] = name
        row["controller_mode"] = str(control_info.get("controller_mode", ""))
        row["desired_pack_current"] = float(
            control_info.get("desired_pack_current", env.action_to_pack_current(action))
        )
        if "mpc_best_cost" in control_info:
            row["mpc_best_cost"] = float(control_info["mpc_best_cost"])
        records.append(row)

        current_state = next_state
        if done:
            break

    return pd.DataFrame(records)


def trim_pack_histories(pack, keep_last: int = 1) -> None:
    if keep_last <= 0:
        pack.history = []
        for cell in pack.cells:
            cell.history = []
        return

    if len(pack.history) > keep_last:
        pack.history = pack.history[-keep_last:]

    for cell in pack.cells:
        if len(cell.history) > keep_last:
            cell.history = cell.history[-keep_last:]


def compute_metrics(results: pd.DataFrame, target_soc: float) -> Dict[str, float]:
    if results.empty:
        return {}

    time_s = results["time"].to_numpy(dtype=float)
    voltage_v = results["pack_voltage"].to_numpy(dtype=float)
    current_a = results["pack_current"].to_numpy(dtype=float)
    charge_current_a = -current_a
    power_w = voltage_v * current_a

    event_counts = [count_safety_events(ev) for ev in results["safety_events"]]
    safety_total = int(sum(event_counts))
    safety_event_timesteps = int(sum(count > 0 for count in event_counts))
    metrics = {
        "steps": int(len(results)),
        "charge_time_min": float(time_s[-1] / 60.0),
        "final_soc": float(results["pack_soc"].iloc[-1]),
        "target_soc": float(target_soc),
        "time_to_80_soc_min": time_to_soc_minutes(results, 0.8),
        "time_to_90_soc_min": time_to_soc_minutes(results, 0.9),
        "peak_pack_voltage_v": float(results["pack_voltage"].max()),
        "peak_pack_temperature_c": float(results["pack_temperature"].max()),
        "peak_voltage_imbalance_mv": float(results["voltage_imbalance"].max() * 1000.0),
        "mean_charge_current_a": float(np.mean(charge_current_a)),
        "peak_charge_current_a": float(np.max(charge_current_a)),
        "rms_charge_current_a": float(np.sqrt(np.mean(charge_current_a**2))),
        "charge_throughput_ah": trapz_integral(charge_current_a, time_s) / 3600.0,
        "energy_in_kwh": trapz_integral(-power_w, time_s) / 3600.0 / 1000.0,
        "safety_event_count": safety_total,
        "safety_event_timesteps": safety_event_timesteps,
    }
    return metrics


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_without_suffix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_baseline_timeseries(
    results: pd.DataFrame,
    controller_name: str,
    target_soc: float,
    cv_voltage_v: float,
    output_path: Path,
) -> None:
    t_min = results["time"] / 60.0
    charge_current_a = -results["pack_current"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(t_min, results["pack_soc"] * 100.0, color="#155D27", label="SOC")
    axes[0, 0].axhline(target_soc * 100.0, linestyle="--", color="#0B090A", label="Target SOC")
    axes[0, 0].set_xlabel("Time (min)")
    axes[0, 0].set_ylabel("SOC (%)")
    axes[0, 0].set_title(f"{controller_name}: SOC Trajectory")
    axes[0, 0].legend()

    axes[0, 1].plot(t_min, results["pack_voltage"], color="#1D3557", label="Pack voltage")
    axes[0, 1].axhline(cv_voltage_v, linestyle="--", color="#E63946", label="CV reference")
    axes[0, 1].set_xlabel("Time (min)")
    axes[0, 1].set_ylabel("Voltage (V)")
    axes[0, 1].set_title(f"{controller_name}: Voltage Regulation")
    axes[0, 1].legend()

    axes[1, 0].plot(t_min, charge_current_a, color="#F4A261", label="Charge current")
    axes[1, 0].set_xlabel("Time (min)")
    axes[1, 0].set_ylabel("Current (A)")
    axes[1, 0].set_title(f"{controller_name}: Current Profile")
    axes[1, 0].legend()

    axes[1, 1].plot(t_min, results["pack_temperature"], color="#B5179E", label="Pack temp")
    axes[1, 1].axhline(40.0, linestyle="--", color="#6A040F", label="40 C guard")
    ax_twin = axes[1, 1].twinx()
    ax_twin.plot(t_min, results["voltage_imbalance"] * 1000.0, color="#457B9D", label="Delta V")
    axes[1, 1].set_xlabel("Time (min)")
    axes[1, 1].set_ylabel("Temperature (C)")
    ax_twin.set_ylabel("Voltage imbalance (mV)")
    axes[1, 1].set_title(f"{controller_name}: Thermal and Imbalance Behavior")
    lines_l, labels_l = axes[1, 1].get_legend_handles_labels()
    lines_r, labels_r = ax_twin.get_legend_handles_labels()
    axes[1, 1].legend(lines_l + lines_r, labels_l + labels_r, loc="upper left")

    fig.tight_layout()
    save_figure(fig, output_path)


def plot_cell_statistics(results: pd.DataFrame, controller_name: str, output_path: Path) -> None:
    t_min = results["time"] / 60.0
    cell_voltage_cols = [c for c in results.columns if c.startswith("cell_") and c.endswith("_voltage")]
    cell_soc_cols = [c for c in results.columns if c.startswith("cell_") and c.endswith("_soc")]
    cell_temp_cols = [c for c in results.columns if c.startswith("cell_") and c.endswith("_temperature")]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    if cell_voltage_cols:
        cell_v = results[cell_voltage_cols]
        axes[0, 0].plot(t_min, cell_v.min(axis=1), color="#1D3557", label="Min")
        axes[0, 0].plot(t_min, cell_v.mean(axis=1), color="#2A9D8F", label="Mean")
        axes[0, 0].plot(t_min, cell_v.max(axis=1), color="#E63946", label="Max")
        axes[0, 0].set_ylabel("Cell voltage (V)")
        axes[0, 0].set_xlabel("Time (min)")
        axes[0, 0].set_title("Cell Voltage Envelope")
        axes[0, 0].legend()

    if cell_soc_cols:
        cell_soc = results[cell_soc_cols]
        axes[0, 1].plot(t_min, cell_soc.min(axis=1) * 100.0, color="#1D3557", label="Min")
        axes[0, 1].plot(t_min, cell_soc.mean(axis=1) * 100.0, color="#2A9D8F", label="Mean")
        axes[0, 1].plot(t_min, cell_soc.max(axis=1) * 100.0, color="#E63946", label="Max")
        axes[0, 1].set_ylabel("Cell SOC (%)")
        axes[0, 1].set_xlabel("Time (min)")
        axes[0, 1].set_title("Cell SOC Envelope")
        axes[0, 1].legend()

    if cell_temp_cols:
        cell_temp = results[cell_temp_cols]
        axes[1, 0].plot(t_min, cell_temp.min(axis=1), color="#1D3557", label="Min")
        axes[1, 0].plot(t_min, cell_temp.mean(axis=1), color="#2A9D8F", label="Mean")
        axes[1, 0].plot(t_min, cell_temp.max(axis=1), color="#E63946", label="Max")
        axes[1, 0].set_ylabel("Cell temperature (C)")
        axes[1, 0].set_xlabel("Time (min)")
        axes[1, 0].set_title("Cell Temperature Envelope")
        axes[1, 0].legend()

    final_soc_values = (
        results[cell_soc_cols].iloc[-1].to_numpy(dtype=float) * 100.0 if cell_soc_cols else np.array([])
    )
    if final_soc_values.size:
        axes[1, 1].hist(final_soc_values, bins=16, color="#2A9D8F", edgecolor="black", alpha=0.8)
        axes[1, 1].set_xlabel("Final cell SOC (%)")
        axes[1, 1].set_ylabel("Cell count")
        axes[1, 1].set_title("Final SOC Distribution Across Cells")

    fig.suptitle(f"{controller_name}: Cell-Level Distribution Metrics", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_phase_portraits(results: pd.DataFrame, controller_name: str, output_path: Path) -> None:
    soc = results["pack_soc"].to_numpy(dtype=float) * 100.0
    voltage = results["pack_voltage"].to_numpy(dtype=float)
    temperature = results["pack_temperature"].to_numpy(dtype=float)
    charge_current = -results["pack_current"].to_numpy(dtype=float)
    time_min = results["time"].to_numpy(dtype=float) / 60.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    sc0 = axes[0].scatter(soc, voltage, c=time_min, cmap="viridis", s=12, alpha=0.9)
    axes[0].set_xlabel("SOC (%)")
    axes[0].set_ylabel("Pack voltage (V)")
    axes[0].set_title("Voltage vs SOC")
    cbar0 = fig.colorbar(sc0, ax=axes[0])
    cbar0.set_label("Time (min)")

    sc1 = axes[1].scatter(soc, temperature, c=time_min, cmap="plasma", s=12, alpha=0.9)
    axes[1].set_xlabel("SOC (%)")
    axes[1].set_ylabel("Pack temperature (C)")
    axes[1].set_title("Temperature vs SOC")
    cbar1 = fig.colorbar(sc1, ax=axes[1])
    cbar1.set_label("Time (min)")

    sc2 = axes[2].scatter(voltage, charge_current, c=soc, cmap="cividis", s=12, alpha=0.9)
    axes[2].set_xlabel("Pack voltage (V)")
    axes[2].set_ylabel("Charge current (A)")
    axes[2].set_title("Current-Voltage Operating Region")
    cbar2 = fig.colorbar(sc2, ax=axes[2])
    cbar2.set_label("SOC (%)")

    fig.suptitle(f"{controller_name}: Phase Portraits", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_comparison_overlay(
    all_results: Dict[str, pd.DataFrame],
    target_soc: float,
    cv_voltage_v: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = {"cccv": "#E76F51", "mpc": "#2A9D8F"}

    for name, results in all_results.items():
        label = name.upper()
        color = colors.get(name, None)
        t = results["time"] / 60.0
        axes[0, 0].plot(t, results["pack_soc"] * 100.0, label=label, color=color)
        axes[0, 1].plot(t, results["pack_voltage"], label=label, color=color)
        axes[1, 0].plot(t, -results["pack_current"], label=label, color=color)
        axes[1, 1].plot(t, results["pack_temperature"], label=label, color=color)

    axes[0, 0].axhline(target_soc * 100.0, linestyle="--", color="black", linewidth=1.2)
    axes[0, 0].set_title("SOC Comparison")
    axes[0, 0].set_xlabel("Time (min)")
    axes[0, 0].set_ylabel("SOC (%)")
    axes[0, 0].legend()

    axes[0, 1].axhline(cv_voltage_v, linestyle="--", color="black", linewidth=1.2)
    axes[0, 1].set_title("Voltage Comparison")
    axes[0, 1].set_xlabel("Time (min)")
    axes[0, 1].set_ylabel("Pack voltage (V)")
    axes[0, 1].legend()

    axes[1, 0].set_title("Charge Current Comparison")
    axes[1, 0].set_xlabel("Time (min)")
    axes[1, 0].set_ylabel("Charge current (A)")
    axes[1, 0].legend()

    axes[1, 1].set_title("Temperature Comparison")
    axes[1, 1].set_xlabel("Time (min)")
    axes[1, 1].set_ylabel("Pack temperature (C)")
    axes[1, 1].legend()

    fig.tight_layout()
    save_figure(fig, output_path)


def plot_metrics_bars(metrics_df: pd.DataFrame, output_path: Path) -> None:
    plot_columns = [
        "charge_time_min",
        "peak_pack_temperature_c",
        "peak_voltage_imbalance_mv",
        "energy_in_kwh",
        "safety_event_count",
    ]
    labels = [
        "Charge time (min)",
        "Peak temp (C)",
        "Peak imbalance (mV)",
        "Energy in (kWh)",
        "Safety events",
    ]

    fig, axes = plt.subplots(1, len(plot_columns), figsize=(16, 4.2))
    controllers = metrics_df.index.tolist()
    bar_colors = ["#E76F51", "#2A9D8F", "#264653", "#8AB17D", "#3A86FF"]

    for idx, (col, label) in enumerate(zip(plot_columns, labels)):
        values = metrics_df[col].to_numpy(dtype=float)
        axes[idx].bar(controllers, values, color=bar_colors[idx % len(bar_colors)], alpha=0.9)
        axes[idx].set_title(label)
        axes[idx].tick_params(axis="x", rotation=20)
        for x, y in enumerate(values):
            axes[idx].text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("CCCV vs MPC: Benchmark Metrics", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_tradeoff(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    x = metrics_df["charge_time_min"].to_numpy(dtype=float)
    y = metrics_df["peak_pack_temperature_c"].to_numpy(dtype=float)
    bubble = np.clip(metrics_df["peak_voltage_imbalance_mv"].to_numpy(dtype=float), 1.0, None) * 2.0

    for idx, name in enumerate(metrics_df.index.tolist()):
        ax.scatter(x[idx], y[idx], s=bubble[idx], alpha=0.75, label=name.upper())
        ax.text(x[idx], y[idx], f" {name.upper()}", va="center")

    ax.set_xlabel("Charge time to completion (min)")
    ax.set_ylabel("Peak pack temperature (C)")
    ax.set_title("Controller Tradeoff: Time vs Thermal Stress")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path)


def ensure_folders(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _split_csv_arg(value: str) -> List[str]:
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def derive_data_profile(
    csv_path: Path,
    default_initial_soc: float,
    default_ambient_temp_c: float,
) -> Dict[str, float]:
    header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    required = ["time", "pack_voltage", "pack_current"]
    if not set(required).issubset(set(header_cols)):
        missing = [col for col in required if col not in header_cols]
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    usecols = [col for col in ["time", "pack_voltage", "pack_current", "pack_temperature"] if col in header_cols]
    data = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    if len(data) > 120000:
        stride = max(1, len(data) // 120000)
        data = data.iloc[::stride].copy()

    data["time"] = _coerce_numeric(data["time"])
    data["pack_voltage"] = _coerce_numeric(data["pack_voltage"])
    data["pack_current"] = _coerce_numeric(data["pack_current"])
    if "pack_temperature" in data.columns:
        data["pack_temperature"] = _coerce_numeric(data["pack_temperature"])
    data = data.dropna(subset=["time", "pack_voltage", "pack_current"])
    if data.empty:
        raise ValueError(f"{csv_path} has no valid numeric rows")
    data = data.sort_values("time")

    time_s = data["time"].to_numpy(dtype=float)
    voltage_v = data["pack_voltage"].to_numpy(dtype=float)
    current_a = data["pack_current"].to_numpy(dtype=float)
    current_abs = np.abs(current_a)

    dt_candidates = np.diff(time_s)
    dt_candidates = dt_candidates[np.isfinite(dt_candidates) & (dt_candidates > 0)]
    dt_s = float(np.median(dt_candidates)) if dt_candidates.size else 1.0
    dt_s = float(np.clip(dt_s, 0.1, 30.0))

    v_q01 = float(np.nanquantile(voltage_v, 0.01))
    v_q50 = float(np.nanquantile(voltage_v, 0.50))
    v_q95 = float(np.nanquantile(voltage_v, 0.95))
    v_q99 = float(np.nanquantile(voltage_v, 0.99))
    i_abs_q95 = float(np.nanquantile(current_abs, 0.95))
    i_abs_q99 = float(np.nanquantile(current_abs, 0.99))
    initial_voltage = float(voltage_v[0])

    n_series_est = int(max(1, round(v_q95 / 4.05)))
    n_series_est = int(np.clip(n_series_est, 1, 200))

    voltage_span = max(v_q99 - v_q01, 1e-6)
    initial_soc = float(np.clip((initial_voltage - v_q01) / voltage_span, 0.05, 0.95))
    if not np.isfinite(initial_soc):
        initial_soc = float(np.clip(default_initial_soc, 0.05, 0.95))

    if "pack_temperature" in data.columns:
        temp_valid = data["pack_temperature"].dropna()
        if len(temp_valid):
            ambient_temp_c = float(temp_valid.median())
            initial_temp_c = float(temp_valid.iloc[0])
        else:
            ambient_temp_c = float(default_ambient_temp_c)
            initial_temp_c = float(default_ambient_temp_c)
    else:
        ambient_temp_c = float(default_ambient_temp_c)
        initial_temp_c = float(default_ambient_temp_c)

    return {
        "dt_s": dt_s,
        "n_rows_sampled": int(len(data)),
        "time_end_s": float(time_s[-1]),
        "initial_soc": initial_soc,
        "initial_temp_c": initial_temp_c,
        "ambient_temp_c": ambient_temp_c,
        "n_series_est": n_series_est,
        "voltage_q01_v": v_q01,
        "voltage_q50_v": v_q50,
        "voltage_q95_v": v_q95,
        "voltage_q99_v": v_q99,
        "initial_voltage_v": initial_voltage,
        "current_abs_q95_a": i_abs_q95,
        "current_abs_q99_a": i_abs_q99,
    }


def apply_fitted_params_to_pack(
    pack: BatteryPack,
    fitted_payload: Optional[Dict[str, Any]],
    source_is_cell_level: bool = False,
) -> None:
    if not fitted_payload:
        return
    results = fitted_payload.get("results", {})
    ecm = results.get("pack_ecm", {})
    thermal = results.get("pack_thermal", {})
    if not pack.cells:
        return

    n_series = max(1, pack.config.n_series)
    n_parallel = max(1, pack.config.n_parallel)
    if source_is_cell_level:
        r_pack_to_cell = 1.0
        c_pack_to_cell = 1.0
    else:
        r_pack_to_cell = n_parallel / n_series
        c_pack_to_cell = n_series / n_parallel
    n_cells = max(1, n_series * n_parallel)

    def _scale_cell_param(param_name: str, target_value: float, clamp: Tuple[float, float]) -> None:
        if not np.isfinite(target_value) or target_value <= 0:
            return
        current_vals = np.array([getattr(cell.params, param_name) for cell in pack.cells], dtype=float)
        mean_val = float(np.mean(current_vals))
        if not np.isfinite(mean_val) or mean_val <= 0:
            return
        scale = float(np.clip(target_value / mean_val, clamp[0], clamp[1]))
        for cell in pack.cells:
            setattr(cell.params, param_name, float(getattr(cell.params, param_name) * scale))

    r0_pack = float(ecm.get("R0_pack", np.nan))
    r1_pack = float(ecm.get("R1_pack", np.nan))
    r2_pack = float(ecm.get("R2_pack", np.nan))
    c1_pack = float(ecm.get("C1_pack", np.nan))
    c2_pack = float(ecm.get("C2_pack", np.nan))
    c_th_pack = float(thermal.get("C_th_pack", np.nan))
    hA_pack = float(thermal.get("hA_pack", np.nan))

    _scale_cell_param("R0", r0_pack * r_pack_to_cell, (0.20, 5.0))
    _scale_cell_param("R1", r1_pack * r_pack_to_cell, (0.20, 5.0))
    _scale_cell_param("R2", r2_pack * r_pack_to_cell, (0.20, 5.0))
    _scale_cell_param("C1", c1_pack * c_pack_to_cell, (0.20, 5.0))
    _scale_cell_param("C2", c2_pack * c_pack_to_cell, (0.20, 5.0))
    if source_is_cell_level:
        _scale_cell_param("C_th", c_th_pack, (0.20, 5.0))
        _scale_cell_param("hA", hA_pack, (0.20, 5.0))
    else:
        _scale_cell_param("C_th", c_th_pack / n_cells, (0.20, 5.0))
        _scale_cell_param("hA", hA_pack / n_cells, (0.20, 5.0))


def apply_data_profile_to_pack(
    pack: BatteryPack,
    profile: Optional[Dict[str, float]],
    cv_cell_v: Optional[float] = None,
) -> None:
    """Map observed cell-level voltage/temperature behavior into cell parameters."""
    if not profile or not pack.cells:
        return

    v_q01 = float(profile.get("voltage_q01_v", np.nan))
    v_q99 = float(profile.get("voltage_q99_v", np.nan))
    ambient_temp = float(profile.get("ambient_temp_c", 25.0))
    if not (np.isfinite(v_q01) and np.isfinite(v_q99) and (v_q99 > v_q01)):
        return

    v_cell_min = max(1.8, v_q01 - 0.03)
    v_cell_max = v_q99 + 0.03
    if cv_cell_v is not None and np.isfinite(cv_cell_v):
        v_cell_max = min(v_cell_max, float(cv_cell_v) + 0.05)
        v_cell_max = max(v_cell_max, float(cv_cell_v))
    if v_cell_max <= v_cell_min:
        v_cell_max = v_cell_min + 0.20

    for cell in pack.cells:
        base_ocv = np.asarray(cell.params.ocv_points, dtype=float)
        if base_ocv.size < 2:
            continue
        ocv_norm = (base_ocv - np.min(base_ocv)) / max(np.ptp(base_ocv), 1e-6)
        new_ocv = v_cell_min + ocv_norm * (v_cell_max - v_cell_min)
        new_soc = np.linspace(0.0, 1.0, new_ocv.size)

        cell.params.soc_points = new_soc
        cell.params.ocv_points = new_ocv
        cell.params.v_min = float(v_cell_min)
        cell.params.v_max = float(v_cell_max)
        cell.params.t_min = float(min(cell.params.t_min, ambient_temp - 20.0))
        cell.params.t_max = float(max(cell.params.t_max, ambient_temp + 35.0))

        soc_points_ref = new_soc.copy()
        ocv_points_ref = new_ocv.copy()
        cell.ocv_interp = lambda soc, sp=soc_points_ref, op=ocv_points_ref: np.interp(soc, sp, op)
        cell.voltage = float(cell.ocv_interp(cell.soc))


def collect_data_calibrated_scenarios(run_config: BenchmarkConfig) -> List[Dict[str, Any]]:
    standardized_root = Path(run_config.standardized_root)
    params_root = Path(run_config.params_root)
    families = _split_csv_arg(run_config.dataset_families)
    scenarios: List[Dict[str, Any]] = []

    for family in families:
        family_params_dir = params_root / family
        family_entries: List[Dict[str, Any]] = []
        if family_params_dir.exists():
            for param_path in sorted(family_params_dir.glob("*.json")):
                try:
                    payload = _load_json(param_path)
                except Exception:
                    continue
                source_rel = payload.get("source_relpath")
                if not source_rel:
                    continue
                csv_path = standardized_root / Path(str(source_rel).replace("\\", "/"))
                if not csv_path.exists():
                    continue
                n_rows = int(payload.get("n_rows", 0))
                family_entries.append(
                    {
                        "family": family,
                        "csv_path": csv_path,
                        "params_path": param_path,
                        "fitted_payload": payload,
                        "n_rows": n_rows,
                    }
                )

        # Fallback if no fitted params were found.
        if not family_entries:
            family_data_dir = standardized_root / family
            if family_data_dir.exists():
                for csv_path in sorted(family_data_dir.glob("*.csv")):
                    family_entries.append(
                        {
                            "family": family,
                            "csv_path": csv_path,
                            "params_path": None,
                            "fitted_payload": None,
                            "n_rows": 0,
                        }
                    )

        family_entries.sort(key=lambda item: item.get("n_rows", 0), reverse=True)
        selected = family_entries[: max(1, run_config.max_files_per_dataset)]
        for entry in selected:
            try:
                profile = derive_data_profile(
                    csv_path=entry["csv_path"],
                    default_initial_soc=run_config.initial_soc,
                    default_ambient_temp_c=run_config.ambient_temp_c,
                )
            except Exception:
                continue

            case_id = entry["csv_path"].stem
            scenarios.append(
                {
                    "family": family,
                    "case_id": case_id,
                    "csv_path": entry["csv_path"],
                    "params_path": entry["params_path"],
                    "fitted_payload": entry["fitted_payload"],
                    "profile": profile,
                }
            )

    return scenarios


def recommend_episode_max_steps(
    dt_s: float,
    base_max_steps: int,
    initial_soc: float,
    target_soc: float,
    capacity_ah: float,
    max_charge_current_a: float,
    min_episode_minutes: float,
    feasible_time_slack: float,
    max_steps_cap: int,
) -> Tuple[int, Dict[str, float]]:
    base_steps = int(max(1, base_max_steps))
    dt_s = float(max(1e-6, dt_s))
    base_horizon_s = float(base_steps * dt_s)
    capacity_ah = float(max(1e-6, capacity_ah))
    soc_gap = float(max(0.0, target_soc - initial_soc))
    max_current_a = float(max(1e-6, max_charge_current_a))

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


def execute_benchmark_setting(
    objective_key: str,
    objective,
    run_config: BenchmarkConfig,
    output_root: Path,
    pack_config: PackConfiguration,
    cv_voltage_v: float,
    max_charge_current_a: float,
    initial_soc: float,
    initial_temp_c: float,
    ambient_temp_c: float,
    dt_s: float,
    fitted_payload: Optional[Dict[str, Any]] = None,
    fitted_source_is_cell_level: bool = False,
    data_profile: Optional[Dict[str, float]] = None,
    cv_cell_v: Optional[float] = None,
    scenario_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    cccv_config = CCCVConfig()
    mpc_config = MPCConfig()
    cccv_root = output_root / "cccv"
    mpc_root = output_root / "mpc"
    comparison_root = output_root / "comparison"
    ensure_folders(
        [
            cccv_root / "figures",
            mpc_root / "figures",
            comparison_root / "figures",
        ]
    )

    controllers = {
        "cccv": CCCVController(
            config=cccv_config,
            cv_voltage_v=cv_voltage_v,
            max_charge_current_a=max_charge_current_a,
            target_soc=run_config.target_soc,
        ),
        "mpc": RolloutMPCController(
            config=mpc_config,
            cv_voltage_v=cv_voltage_v,
            max_charge_current_a=max_charge_current_a,
            target_soc=run_config.target_soc,
        ),
    }

    all_results: Dict[str, pd.DataFrame] = {}
    all_metrics: Dict[str, Dict] = {}
    capacity_ah = float(pack_config.get_total_capacity())
    if run_config.adaptive_horizon:
        effective_max_steps, horizon_info = recommend_episode_max_steps(
            dt_s=dt_s,
            base_max_steps=run_config.max_steps,
            initial_soc=initial_soc,
            target_soc=run_config.target_soc,
            capacity_ah=capacity_ah,
            max_charge_current_a=max_charge_current_a,
            min_episode_minutes=run_config.min_episode_minutes,
            feasible_time_slack=run_config.feasible_time_slack,
            max_steps_cap=run_config.max_steps_cap,
        )
    else:
        effective_max_steps = int(run_config.max_steps)
        horizon_info = {
            "dt_s": float(max(1e-6, dt_s)),
            "base_max_steps": float(run_config.max_steps),
            "effective_max_steps": float(run_config.max_steps),
            "base_horizon_s": float(run_config.max_steps * dt_s),
            "effective_horizon_s": float(run_config.max_steps * dt_s),
            "ideal_cc_time_to_target_s": float("nan"),
            "required_current_for_base_horizon_a": float("nan"),
            "configured_max_charge_current_a": float(max_charge_current_a),
            "current_feasibility_ratio_vs_base_horizon": float("nan"),
        }

    for name, controller in controllers.items():
        env = HAMBRLPackEnvironment(
            pack_config=pack_config,
            max_steps=effective_max_steps,
            target_soc=run_config.target_soc,
            ambient_temp=ambient_temp_c,
            max_charge_current_a=max_charge_current_a,
            dt=dt_s,
        )
        apply_fitted_params_to_pack(
            env.pack,
            fitted_payload,
            source_is_cell_level=fitted_source_is_cell_level,
        )
        apply_data_profile_to_pack(
            env.pack,
            profile=data_profile,
            cv_cell_v=cv_cell_v,
        )
        results = run_controller_episode(
            env=env,
            controller=controller,
            name=name,
            initial_soc=initial_soc,
            initial_temp_c=initial_temp_c,
        )
        all_results[name] = results
        all_metrics[name] = compute_metrics(results, target_soc=run_config.target_soc)

        algo_root = cccv_root if name == "cccv" else mpc_root
        results.to_csv(algo_root / "trajectory.csv", index=False)
        save_json(
            algo_root / "metrics.json",
            {
                "algorithm": name,
                "objective": objective_key,
                "objective_description": objective.description,
                "cv_voltage_v": cv_voltage_v,
                "max_charge_current_a": max_charge_current_a,
                "config": asdict(run_config),
                "scenario": scenario_metadata or {},
                "metrics": all_metrics[name],
                "effective_max_steps": int(effective_max_steps),
                "horizon_info": horizon_info,
            },
        )
        plot_baseline_timeseries(
            results=results,
            controller_name=name.upper(),
            target_soc=run_config.target_soc,
            cv_voltage_v=cv_voltage_v,
            output_path=algo_root / "figures" / "01_timeseries",
        )
        if run_config.include_cell_figures:
            plot_cell_statistics(
                results=results,
                controller_name=name.upper(),
                output_path=algo_root / "figures" / "02_cell_statistics",
            )
        plot_phase_portraits(
            results=results,
            controller_name=name.upper(),
            output_path=algo_root / "figures" / "03_phase_portraits",
        )

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    metrics_df.index.name = "controller"
    metrics_df.to_csv(comparison_root / "metrics_summary.csv")

    save_json(
        comparison_root / "run_metadata.json",
        {
            "objective": objective_key,
            "objective_description": objective.description,
            "objective_parameters": asdict(objective),
            "benchmark_config": asdict(run_config),
            "scenario": scenario_metadata or {},
            "cccv_config": asdict(cccv_config),
            "mpc_config": asdict(mpc_config),
            "cv_voltage_v": cv_voltage_v,
            "max_charge_current_a": max_charge_current_a,
            "dt_s": dt_s,
            "effective_max_steps": int(effective_max_steps),
            "horizon_info": horizon_info,
        },
    )

    plot_comparison_overlay(
        all_results=all_results,
        target_soc=run_config.target_soc,
        cv_voltage_v=cv_voltage_v,
        output_path=comparison_root / "figures" / "01_overlay",
    )
    plot_metrics_bars(
        metrics_df=metrics_df,
        output_path=comparison_root / "figures" / "02_metrics_bar",
    )
    plot_tradeoff(
        metrics_df=metrics_df,
        output_path=comparison_root / "figures" / "03_tradeoff",
    )


def parse_args() -> BenchmarkConfig:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run CCCV and MPC baselines on the pack environment."
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="safe",
        choices=tuple(list(build_default_objectives().keys()) + ["all"]),
        help="Charging objective from pack_experiments.py",
    )
    parser.add_argument("--output-root", type=str, default="results/baselines")
    parser.add_argument("--max-steps", type=int, default=1800)
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
    parser.add_argument("--initial-soc", type=float, default=0.2)
    parser.add_argument("--target-soc", type=float, default=0.8)
    parser.add_argument("--ambient-temp-c", type=float, default=25.0)
    parser.add_argument("--initial-temp-c", type=float, default=25.0)
    parser.add_argument("--n-series", type=int, default=20)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--max-charge-current-a", type=float, default=10.0)
    parser.add_argument("--balancing-type", type=str, default="passive")
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Calibrate benchmark settings from standardized dataset CSVs and fitted params.",
    )
    parser.add_argument(
        "--standardized-root",
        type=str,
        default="data/standardized",
        help="Root directory of standardized CSV datasets.",
    )
    parser.add_argument(
        "--params-root",
        type=str,
        default="data/standardized_params",
        help="Root directory of fitted parameter JSON files.",
    )
    parser.add_argument(
        "--dataset-families",
        type=str,
        default="nasa,calce,matr",
        help="Comma-separated dataset families to include (e.g. nasa,calce,matr).",
    )
    parser.add_argument(
        "--max-files-per-dataset",
        type=int,
        default=1,
        help="Number of representative files per dataset family when --use-real-data is enabled.",
    )
    parser.add_argument(
        "--data-is-pack-level",
        action="store_true",
        help=(
            "Treat standardized datasets as already pack-level. "
            "Default behavior treats them as cell-level and scales behavior to pack topology."
        ),
    )
    parser.add_argument(
        "--include-cell-figures",
        action="store_true",
        help="Also generate cell-level diagnostic figures (default off for pack-focused outputs).",
    )
    args = parser.parse_args()
    return BenchmarkConfig(
        objective=args.objective,
        output_root=args.output_root,
        max_steps=args.max_steps,
        initial_soc=args.initial_soc,
        target_soc=args.target_soc,
        ambient_temp_c=args.ambient_temp_c,
        initial_temp_c=args.initial_temp_c,
        n_series=args.n_series,
        n_parallel=args.n_parallel,
        max_charge_current_a=args.max_charge_current_a,
        balancing_type=args.balancing_type,
        use_real_data=args.use_real_data,
        standardized_root=args.standardized_root,
        params_root=args.params_root,
        dataset_families=args.dataset_families,
        max_files_per_dataset=args.max_files_per_dataset,
        data_is_cell_level=not args.data_is_pack_level,
        include_cell_figures=args.include_cell_figures,
        adaptive_horizon=not bool(args.disable_adaptive_horizon),
        min_episode_minutes=float(args.min_episode_minutes),
        feasible_time_slack=float(args.feasible_time_slack),
        max_steps_cap=int(args.max_steps_cap),
    )


def main() -> None:
    apply_publication_style()
    run_config = parse_args()
    objectives = build_default_objectives()
    selected_objective_keys = (
        list(objectives.keys()) if run_config.objective == "all" else [run_config.objective]
    )
    base_output_root = Path(run_config.output_root)

    if run_config.use_real_data:
        scenarios = collect_data_calibrated_scenarios(run_config)
        if not scenarios:
            raise SystemExit(
                "No data-calibrated scenarios were found. "
                "Check --standardized-root/--params-root/--dataset-families."
            )

        for objective_key in selected_objective_keys:
            objective = objectives[objective_key]
            for scenario in scenarios:
                profile = scenario["profile"]
                # Standardized public datasets are typically cell/channel recordings.
                # Build a pack-focused benchmark by scaling to configured topology.
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
                if run_config.data_is_cell_level:
                    data_current_limit_a = float(
                        max(0.20, profile["current_abs_q95_a"] * 1.10 * n_parallel)
                    )
                else:
                    data_current_limit_a = float(max(0.20, profile["current_abs_q95_a"] * 1.10))
                max_charge_current_a = float(
                    min(run_config.max_charge_current_a, objective_current_limit_a, data_current_limit_a)
                )
                max_charge_current_a = float(max(0.20, max_charge_current_a))
                pack_config.I_pack_max = float(max(2.0, 1.2 * max_charge_current_a))

                output_root = (
                    base_output_root
                    / "data_calibrated"
                    / objective_key
                    / scenario["family"]
                    / scenario["case_id"]
                )
                scenario_metadata = {
                    "mode": "data_calibrated",
                    "dataset_family": scenario["family"],
                    "dataset_case": scenario["case_id"],
                    "source_csv": str(scenario["csv_path"]),
                    "params_json": str(scenario["params_path"]) if scenario["params_path"] else None,
                    "data_is_cell_level": run_config.data_is_cell_level,
                    "profile": profile,
                }
                execute_benchmark_setting(
                    objective_key=objective_key,
                    objective=objective,
                    run_config=run_config,
                    output_root=output_root,
                    pack_config=pack_config,
                    cv_voltage_v=cv_voltage_v,
                    max_charge_current_a=max_charge_current_a,
                    initial_soc=float(run_config.initial_soc),
                    initial_temp_c=float(profile["initial_temp_c"]),
                    ambient_temp_c=float(profile["ambient_temp_c"]),
                    dt_s=float(profile["dt_s"]),
                    fitted_payload=scenario["fitted_payload"],
                    fitted_source_is_cell_level=run_config.data_is_cell_level,
                    data_profile=profile,
                    cv_cell_v=cv_cell_v,
                    scenario_metadata=scenario_metadata,
                )
                print("Completed baseline benchmark (data-calibrated).")
                print(
                    f"Objective: {objective_key}, Dataset: {scenario['family']}/{scenario['case_id']}"
                )
                print(f"CV voltage: {cv_voltage_v:.3f} V")
                print(f"Max charge current: {max_charge_current_a:.3f} A")
                print(f"Results written to: {output_root.resolve()}")
    else:
        for objective_key in selected_objective_keys:
            objective = objectives[objective_key]
            pack_config = PackConfiguration(
                n_series=run_config.n_series,
                n_parallel=run_config.n_parallel,
                balancing_type=run_config.balancing_type,
            )
            capacity_ah = pack_config.get_total_capacity()
            objective_current_limit_a = objective.i_max_c_rate * capacity_ah
            max_charge_current_a = float(
                min(run_config.max_charge_current_a, objective_current_limit_a)
            )
            cv_voltage_v = float(objective.v_max * run_config.n_series)

            output_root = (
                base_output_root / objective_key
                if run_config.objective == "all"
                else base_output_root
            )
            execute_benchmark_setting(
                objective_key=objective_key,
                objective=objective,
                run_config=run_config,
                output_root=output_root,
                pack_config=pack_config,
                cv_voltage_v=cv_voltage_v,
                max_charge_current_a=max_charge_current_a,
                initial_soc=run_config.initial_soc,
                initial_temp_c=run_config.initial_temp_c,
                ambient_temp_c=run_config.ambient_temp_c,
                dt_s=1.0,
                fitted_payload=None,
                scenario_metadata={"mode": "synthetic"},
            )
            print("Completed baseline benchmark.")
            print(f"Objective: {objective_key}")
            print(f"CV voltage: {cv_voltage_v:.3f} V")
            print(f"Max charge current: {max_charge_current_a:.3f} A")
            print(f"Results written to: {output_root.resolve()}")


if __name__ == "__main__":
    main()
