"""Physics-based single cell model (electro-thermal + degradation)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class CellParameters:
    """Parameter set for an equivalent circuit cell model."""
    Q_nominal: float = 2.5
    R0: float = 0.02
    R1: float = 0.01
    C1: float = 2000.0
    R2: float = 0.02
    C2: float = 10000.0
    eta: float = 0.995
    C_th: float = 75.0
    hA: float = 0.5
    k1: float = 1e-6
    k2: float = 0.05
    k3: float = 0.1
    T_ref: float = 25.0
    v_min: float = 2.5
    v_max: float = 4.2
    t_min: float = 0.0
    t_max: float = 60.0
    soc_points: np.ndarray = field(default_factory=lambda: np.linspace(0, 1, 11))
    ocv_points: np.ndarray = field(
        default_factory=lambda: np.array(
            [3.0, 3.3, 3.45, 3.55, 3.65, 3.72, 3.78, 3.85, 3.92, 4.05, 4.2]
        )
    )


class CellModel:
    """Physics-based model for a single battery cell.

    This class implements a 2nd-order Thevenin Equivalent Circuit Model (ECM) coupled with
    a lumped thermal model and a semi-empirical aging model.

    Governing Equations:
    --------------------
    1. State of Charge (SoC):
       d(SoC)/dt = -eta * I / (Q_nominal * 3600)

    2. Circuit Dynamics (2-RC pairs):
       d(v1)/dt = -v1 / (R1 * C1) + I / C1
       d(v2)/dt = -v2 / (R2 * C2) + I / C2
       V_terminal = V_ocv(SoC) - I * R0 - v1 - v2

    3. Thermal Dynamics (Lumped capacitance):
       d(T)/dt = (Q_gen - Q_diss) / C_th
       Q_gen = I^2 * (R0 + R0_growth)
       Q_diss = hA * (T - T_ambient)

    4. Degradation (Aging):
       Capacity Fade: d(Q_loss)/dt = k1 * S_I * S_T * S_SoC
       Resistance Growth: R0_growth = 2 * R0 * (Q_loss / Q_nominal)
       where S_I, S_T, S_SoC are stress factors for current, temperature, and SoC.

    Notation:
    - I: Current (Amps), positive = discharge
    - T: Cell temperature (C)
    - v1, v2: Overpotentials across RC pairs
    - R0, R1, R2: Ohmic and polarization resistances
    - C1, C2: Polarization capacitances
    """

    def __init__(self, cell_id: int, params: CellParameters | Dict, dt: float = 1.0):
        self.cell_id = cell_id
        self.dt = dt
        if isinstance(params, dict):
            params = CellParameters(**params)
        self.params = params

        self.Q_loss = 0.0
        self.cycles = 0
        self.R0_growth = 0.0
        self.Q_effective = self.params.Q_nominal

        self.ocv_interp = lambda soc: np.interp(
            soc, self.params.soc_points, self.params.ocv_points
        )
        self.reset()

    def reset(self, soc: float = 0.5, temperature: float = 25.0) -> None:
        self.soc = soc
        self.v1 = 0.0
        self.v2 = 0.0
        self.temperature = temperature
        self.voltage = self.ocv_interp(soc)
        self.current = 0.0
        self.time = 0.0
        self.history: List[Dict] = []
        self._record_state()

    def step(
        self, current: float, ambient_temp: float, balancing_current: float = 0.0
    ) -> Dict:
        """Advance one simulation step.

        Current convention (A): positive = discharge, negative = charge.
        """
        total_current = current + balancing_current

        # Coulomb counting with current in A, capacity in Ah, and time in s.
        soc_dot = -self.params.eta * total_current / (self.Q_effective * 3600)
        self.soc = float(np.clip(self.soc + soc_dot * self.dt, 0.0, 1.0))

        tau1 = self.params.R1 * self.params.C1
        if tau1 > 0:
            self.v1 += (-self.v1 / tau1 + total_current / self.params.C1) * self.dt

        tau2 = self.params.R2 * self.params.C2
        if tau2 > 0:
            self.v2 += (-self.v2 / tau2 + total_current / self.params.C2) * self.dt

        q_gen = total_current**2 * (self.params.R0 + self.R0_growth)
        q_diss = self.params.hA * (self.temperature - ambient_temp)
        self.temperature += ((q_gen - q_diss) / self.params.C_th) * self.dt

        self._update_degradation(total_current)

        ocv = self.ocv_interp(self.soc)
        effective_r0 = self.params.R0 + self.R0_growth
        # Thevenin terminal voltage (positive discharge current lowers voltage).
        self.voltage = ocv - total_current * effective_r0 - self.v1 - self.v2

        self.time += self.dt
        self.current = total_current
        return self._record_state(ambient_temp)

    def _update_degradation(self, current: float) -> None:
        current_stress = abs(current) / self.params.Q_nominal
        temp_stress = np.exp(self.params.k2 * (self.temperature - self.params.T_ref))
        soc_stress = np.exp(self.params.k3 * self.soc)

        q_loss_dot = self.params.k1 * current_stress * temp_stress * soc_stress
        self.Q_loss += q_loss_dot * self.dt

        capacity_fade = min(self.Q_loss / (0.2 * self.params.Q_nominal), 0.8)
        self.Q_effective = self.params.Q_nominal * (1 - capacity_fade)
        self.R0_growth = self.params.R0 * 2.0 * capacity_fade

        if self.soc >= 0.99 and self.history and self.history[-1]["soc"] < 0.99:
            self.cycles += 1

    def _record_state(self, ambient_temp: float = 25.0) -> Dict:
        measurement = {
            "time": self.time,
            "cell_id": self.cell_id,
            "soc": self.soc,
            "voltage": self.voltage,
            "current": self.current,
            "temperature": self.temperature,
            "ambient_temp": ambient_temp,
            "v1": self.v1,
            "v2": self.v2,
            "Q_effective": self.Q_effective,
            "Q_loss": self.Q_loss,
            "R0_effective": self.params.R0 + self.R0_growth,
            "cycles": self.cycles,
        }
        self.history.append(measurement)
        return measurement

    def get_state_of_health(self) -> float:
        capacity_soh = self.Q_effective / self.params.Q_nominal
        resistance_soh = max(0.0, 1.0 - self.R0_growth / (2 * self.params.R0))
        soh = 0.7 * capacity_soh + 0.3 * resistance_soh
        return max(0.0, min(1.0, soh))

    def get_safety_status(self) -> Dict[str, bool]:
        return {
            "over_voltage": self.voltage > self.params.v_max,
            "under_voltage": self.voltage < self.params.v_min,
            "over_temperature": self.temperature > self.params.t_max,
            "under_temperature": self.temperature < self.params.t_min,
            "over_current": abs(self.current) > 10,
        }
