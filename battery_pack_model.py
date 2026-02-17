"""Battery pack model with cell-to-cell variation and balancing."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np

from physics_model import CellModel, CellParameters


class CellConnection(Enum):
    SERIES = "series"
    PARALLEL = "parallel"
    SERIES_PARALLEL = "series_parallel"


@dataclass
class CellVariance:
    """Statistical variation parameters for cell manufacturing."""
    Q_nominal_mean: float = 2.5
    Q_nominal_std: float = 0.05
    R0_mean: float = 0.02
    R0_std: float = 0.20
    C_th_mean: float = 75.0
    C_th_std: float = 0.10
    Q_R_correlation: float = -0.3
    aging_rate_std: float = 0.15


@dataclass
class PackConfiguration:
    """Battery pack topology and configuration."""
    n_series: int = 20
    n_parallel: int = 1
    connection: CellConnection = CellConnection.SERIES
    wiring_resistance: float = 0.001
    busbar_resistance: float = 0.0005
    contact_resistance: float = 0.0002
    cell_spacing: float = 0.002
    cooling_type: str = "forced_air"
    cooling_power: float = 50.0
    balancing_type: str = "passive"
    balancing_current: float = 0.1
    balancing_threshold: float = 0.01
    V_pack_max: float = 84.0
    V_pack_min: float = 50.0
    I_pack_max: float = 100.0
    T_pack_max: float = 60.0
    delta_V_max: float = 0.1

    def get_pack_voltage_limits(self) -> Tuple[float, float]:
        return (self.n_series * 2.5, self.n_series * 4.2)

    def get_total_capacity(self) -> float:
        return self.n_parallel * 2.5


class BatteryPack:
    """Complete battery pack model with multiple cells in series/parallel.

    The pack model aggregates individual cell models and adds:
    1. Series/Parallel Logic:
       V_pack = sum(V_series_strings)
       I_cell = I_pack / n_parallel

    2. Thermal Interconnection:
       T_new = ThermalMatrix @ T_old
       where ThermalMatrix models heat transfer between adjacent cells (conduction/convection).

    3. Balancing Logic:
       Passive balancing drains cells with SoC > mean(SoC) + threshold.
       Active balancing transfers charge from high-SoC to low-SoC cells (lossless assumption).

    4. Safety Logic:
       Checks limits for V_cell_max, V_cell_min, T_cell_max, and I_pack_max.
    """

    def __init__(
        self,
        config: PackConfiguration,
        cell_variance: CellVariance | None = None,
        dt: float = 1.0,
    ) -> None:
        self.config = config
        self.dt = dt
        if cell_variance is None:
            cell_variance = CellVariance()
        self.cell_variance = cell_variance

        self.cells = self._generate_cells()
        self.pack_voltage = 0.0
        self.pack_current = 0.0
        self.pack_soc = 0.0
        self.pack_temperature = 25.0
        self.ambient_temperature = 25.0
        self.balancing_active = False
        self.balancing_currents = np.zeros(config.n_series)
        self.thermal_matrix = self._create_thermal_interaction_matrix()
        self.history: List[Dict] = []
        self.safety_events: Dict = {}
        self._update_pack_state()

    def _generate_cells(self) -> List[CellModel]:
        cells: List[CellModel] = []
        n_cells = self.config.n_series * self.config.n_parallel

        base_params = CellParameters(
            Q_nominal=self.cell_variance.Q_nominal_mean,
            R0=self.cell_variance.R0_mean,
            R1=0.01,
            C1=2000.0,
            R2=0.02,
            C2=10000.0,
            eta=0.995,
            C_th=self.cell_variance.C_th_mean,
            hA=0.5,
            k1=1e-6,
            k2=0.05,
            k3=0.1,
        )

        np.random.seed(42)
        Q_variations = np.random.normal(1.0, self.cell_variance.Q_nominal_std, n_cells)
        R_variations = np.random.normal(1.0, self.cell_variance.R0_std, n_cells)

        for i in range(n_cells):
            R_adjustment = self.cell_variance.Q_R_correlation * (Q_variations[i] - 1.0)
            R_variations[i] = R_variations[i] - R_adjustment

        for i in range(n_cells):
            cell_params = CellParameters(**base_params.__dict__)
            cell_params.Q_nominal *= Q_variations[i]
            cell_params.R0 *= R_variations[i]
            cell_params.C_th *= np.random.normal(1.0, self.cell_variance.C_th_std)

            cell = CellModel(cell_id=i, params=cell_params, dt=self.dt)
            initial_soc = 0.5 + np.random.uniform(-0.05, 0.05)
            cell.reset(soc=initial_soc)
            cells.append(cell)

        return cells

    def _create_thermal_interaction_matrix(self) -> np.ndarray:
        n_cells = len(self.cells)
        matrix = np.eye(n_cells)

        if self.config.connection == CellConnection.SERIES:
            for i in range(n_cells):
                if i > 0:
                    distance = self.config.cell_spacing
                    matrix[i, i - 1] = 0.3 / (1 + distance * 100)
                if i < n_cells - 1:
                    distance = self.config.cell_spacing
                    matrix[i, i + 1] = 0.3 / (1 + distance * 100)

        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        return matrix

    def _update_pack_state(self) -> None:
        series_voltages = []
        for string_idx in range(self.config.n_parallel):
            start_idx = string_idx * self.config.n_series
            end_idx = start_idx + self.config.n_series
            string_cells = self.cells[start_idx:end_idx]
            string_voltage = sum(cell.voltage for cell in string_cells)
            series_voltages.append(string_voltage)

        self.pack_voltage = float(np.mean(series_voltages))
        total_capacity = sum(cell.Q_effective for cell in self.cells)
        weighted_soc = sum(cell.soc * cell.Q_effective for cell in self.cells)
        self.pack_soc = weighted_soc / total_capacity if total_capacity > 0 else 0.0
        self.pack_temperature = max(cell.temperature for cell in self.cells)
        self.cell_voltages = [cell.voltage for cell in self.cells]
        self.cell_socs = [cell.soc for cell in self.cells]
        self.cell_temperatures = [cell.temperature for cell in self.cells]
        self.voltage_imbalance = max(self.cell_voltages) - min(self.cell_voltages)
        self.soc_imbalance = max(self.cell_socs) - min(self.cell_socs)
        self.temperature_imbalance = max(self.cell_temperatures) - min(
            self.cell_temperatures
        )

    def _calculate_balancing_currents(self) -> np.ndarray:
        balancing_currents = np.zeros(len(self.cells))
        if self.config.balancing_type == "none":
            return balancing_currents

        socs = np.array([cell.soc for cell in self.cells])

        if self.config.balancing_type == "passive":
            soc_mean = np.mean(socs)
            soc_std = np.std(socs)
            if self.soc_imbalance > self.config.balancing_threshold:
                for i in range(len(self.cells)):
                    if socs[i] > soc_mean + soc_std / 2:
                        balancing_currents[i] = self.config.balancing_current

        elif self.config.balancing_type == "active":
            soc_mean = np.mean(socs)
            if self.soc_imbalance > self.config.balancing_threshold:
                above_mean = socs > soc_mean
                below_mean = socs < soc_mean
                excess = sum(
                    (socs[above_mean] - soc_mean)
                    * [self.cells[i].Q_effective for i in np.where(above_mean)[0]]
                )
                if excess > 0:
                    for i in range(len(self.cells)):
                        if above_mean[i]:
                            balancing_currents[i] = self.config.balancing_current * (
                                (socs[i] - soc_mean) / self.soc_imbalance
                            )
                        elif below_mean[i]:
                            balancing_currents[i] = -self.config.balancing_current * (
                                (soc_mean - socs[i]) / self.soc_imbalance
                            )

        return balancing_currents

    def _apply_thermal_interactions(self) -> None:
        if len(self.cells) <= 1:
            return
        temperatures = np.array([cell.temperature for cell in self.cells])
        new_temperatures = self.thermal_matrix @ temperatures
        for i, cell in enumerate(self.cells):
            cell.temperature = float(new_temperatures[i])

    def _check_safety(self) -> None:
        self.safety_events = {
            "over_voltage_cells": [],
            "under_voltage_cells": [],
            "over_temperature_cells": [],
            "voltage_imbalance": [],
            "current_limit": False,
            "pack_over_voltage": False,
            "pack_under_voltage": False,
        }

        for i, cell in enumerate(self.cells):
            safety = cell.get_safety_status()
            if safety["over_voltage"]:
                self.safety_events["over_voltage_cells"].append(i)
            if safety["under_voltage"]:
                self.safety_events["under_voltage_cells"].append(i)
            if safety["over_temperature"]:
                self.safety_events["over_temperature_cells"].append(i)

        if self.pack_voltage > self.config.V_pack_max:
            self.safety_events["pack_over_voltage"] = True
        if self.pack_voltage < self.config.V_pack_min:
            self.safety_events["pack_under_voltage"] = True
        if self.voltage_imbalance > self.config.delta_V_max:
            self.safety_events["voltage_imbalance"].append(self.voltage_imbalance)
        if abs(self.pack_current) > self.config.I_pack_max:
            self.safety_events["current_limit"] = True

    def step(self, pack_current: float, ambient_temp: float = 25.0) -> Dict:
        # Pack current convention: positive = discharge, negative = charge (A).
        self.ambient_temperature = ambient_temp
        self.pack_current = pack_current
        self.balancing_active = self.config.balancing_type != "none"
        self.balancing_currents = self._calculate_balancing_currents()

        for i, cell in enumerate(self.cells):
            cell_current = pack_current / self.config.n_parallel
            balancing_current = (
                self.balancing_currents[i] if i < len(self.balancing_currents) else 0.0
            )
            cell.step(cell_current, ambient_temp, balancing_current)

        self._apply_thermal_interactions()
        self._update_pack_state()
        self._check_safety()
        return self._record_state()

    def _record_state(self) -> Dict:
        pack_state = {
            "time": self.cells[0].time if self.cells else 0.0,
            "pack_voltage": self.pack_voltage,
            "pack_current": self.pack_current,
            "pack_soc": self.pack_soc,
            "pack_temperature": self.pack_temperature,
            "ambient_temperature": self.ambient_temperature,
            "voltage_imbalance": self.voltage_imbalance,
            "soc_imbalance": self.soc_imbalance,
            "temperature_imbalance": self.temperature_imbalance,
            "min_cell_voltage": min(self.cell_voltages),
            "max_cell_voltage": max(self.cell_voltages),
            "min_cell_soc": min(self.cell_socs),
            "max_cell_soc": max(self.cell_socs),
            "min_cell_temperature": min(self.cell_temperatures),
            "max_cell_temperature": max(self.cell_temperatures),
            "balancing_active": self.balancing_active,
            "safety_events": self.safety_events.copy(),
        }

        for i, cell in enumerate(self.cells):
            pack_state[f"cell_{i}_voltage"] = cell.voltage
            pack_state[f"cell_{i}_soc"] = cell.soc
            pack_state[f"cell_{i}_temperature"] = cell.temperature

        self.history.append(pack_state)
        return pack_state

    def simulate(
        self,
        current_profile: np.ndarray,
        ambient_profile: np.ndarray | None = None,
        verbose: bool = False,
    ) -> "np.ndarray":
        import pandas as pd

        if ambient_profile is None:
            ambient_profile = 25.0 * np.ones_like(current_profile)

        self.history = []
        n_steps = len(current_profile)
        for i in range(n_steps):
            if verbose and i % 100 == 0:
                print(f"Step {i}/{n_steps} - Pack SOC: {self.pack_soc:.2%}")
            self.step(current_profile[i], ambient_profile[i])

        return pd.DataFrame(self.history)

    def get_pack_state_of_health(self) -> Dict:
        cell_sohs = [cell.get_state_of_health() for cell in self.cells]
        return {
            "pack_soh": float(np.mean(cell_sohs)),
            "min_cell_soh": float(min(cell_sohs)),
            "max_cell_soh": float(max(cell_sohs)),
            "soh_imbalance": float(max(cell_sohs) - min(cell_sohs)),
            "weakest_cell": int(np.argmin(cell_sohs)),
            "cell_sohs": cell_sohs,
        }

    def reset(self, initial_soc: float = 0.5, temperature: float = 25.0) -> None:
        for cell in self.cells:
            cell.reset(initial_soc, temperature)

        self.pack_voltage = 0.0
        self.pack_current = 0.0
        self.pack_soc = initial_soc
        self.pack_temperature = temperature
        self.ambient_temperature = temperature
        self.balancing_active = False
        self.balancing_currents = np.zeros(len(self.cells))
        self.history = []
        self._update_pack_state()
