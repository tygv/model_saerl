"""Residual controller utilities for phase-1 H-AMBRL development."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass
class ResidualPolicyConfig:
    """Configuration for a linear residual policy over an MPC action."""

    delta_action_scale: float = 0.30
    max_delta_action: float = 0.40
    temp_soft_limit_c: float = 42.0
    safety_voltage_margin_v: float = 0.03
    safety_temp_margin_c: float = 0.50
    ridge_lambda: float = 1e-3


class ResidualHAMBRLController:
    """Lightweight residual controller: a = clip(a_mpc + delta(s), -1, 1)."""

    FEATURE_DIM = 10

    def __init__(
        self,
        config: ResidualPolicyConfig | None = None,
        weights: np.ndarray | None = None,
        bias: float = 0.0,
    ) -> None:
        if config is None:
            config = ResidualPolicyConfig()
        self.config = config
        if weights is None:
            weights = np.zeros(self.FEATURE_DIM, dtype=float)
        self.weights = np.asarray(weights, dtype=float).reshape(-1)
        if self.weights.size != self.FEATURE_DIM:
            raise ValueError(
                f"weights must have size {self.FEATURE_DIM}, got {self.weights.size}"
            )
        self.bias = float(bias)

    @staticmethod
    def build_features(
        state: Dict,
        target_soc: float,
        cv_voltage_v: float,
        max_charge_current_a: float,
        temp_soft_limit_c: float,
    ) -> np.ndarray:
        """Construct normalized features from pack state for residual control."""
        soc = float(state.get("pack_soc", 0.0))
        pack_voltage = float(state.get("pack_voltage", 0.0))
        pack_temp = float(state.get("pack_temperature", 25.0))
        voltage_imbalance_v = float(state.get("voltage_imbalance", 0.0))
        pack_current = float(state.get("pack_current", 0.0))

        soc_gap = float(np.clip(target_soc - soc, -1.0, 1.0))
        voltage_margin = cv_voltage_v - pack_voltage
        temp_margin = temp_soft_limit_c - pack_temp
        charge_current_a = -pack_current
        current_norm = charge_current_a / max(max_charge_current_a, 1e-6)

        features = np.array(
            [
                soc,
                soc_gap,
                np.clip(soc_gap, 0.0, 1.0),
                pack_voltage / max(cv_voltage_v, 1e-6),
                voltage_margin / max(cv_voltage_v, 1e-6),
                pack_temp / 60.0,
                temp_margin / 20.0,
                voltage_imbalance_v * 1000.0 / 200.0,
                current_norm,
                current_norm * current_norm,
            ],
            dtype=float,
        )
        return features

    def predict_delta_from_features(self, features: np.ndarray) -> float:
        features = np.asarray(features, dtype=float).reshape(-1)
        if features.size != self.FEATURE_DIM:
            raise ValueError(
                f"features must have size {self.FEATURE_DIM}, got {features.size}"
            )
        raw = float(np.dot(self.weights, features) + self.bias)
        delta = self.config.delta_action_scale * np.tanh(raw)
        delta = float(np.clip(delta, -self.config.max_delta_action, self.config.max_delta_action))
        return delta

    def predict_delta(
        self,
        state: Dict,
        target_soc: float,
        cv_voltage_v: float,
        max_charge_current_a: float,
    ) -> Tuple[float, np.ndarray]:
        features = self.build_features(
            state=state,
            target_soc=target_soc,
            cv_voltage_v=cv_voltage_v,
            max_charge_current_a=max_charge_current_a,
            temp_soft_limit_c=self.config.temp_soft_limit_c,
        )
        delta = self.predict_delta_from_features(features)
        return delta, features

    def propose_action(
        self,
        state: Dict,
        mpc_action: float,
        target_soc: float,
        cv_voltage_v: float,
        max_charge_current_a: float,
    ) -> Tuple[float, Dict]:
        delta, features = self.predict_delta(
            state=state,
            target_soc=target_soc,
            cv_voltage_v=cv_voltage_v,
            max_charge_current_a=max_charge_current_a,
        )
        proposed_action = float(np.clip(float(mpc_action) + delta, -1.0, 1.0))
        return proposed_action, {
            "delta_action": delta,
            "features": features.tolist(),
            "raw_mpc_action": float(mpc_action),
        }

    def choose_safe_action(
        self,
        env,
        state: Dict,
        proposed_action: float,
        fallback_action: float,
        cv_voltage_v: float,
    ) -> Tuple[float, bool]:
        """Select a safe action using a discrete linear search (Safety Shield).

        Note: This implementation uses a discrete search over a linspace of candidates,
        checking the `env` (or a model copy) for safety violations. It is NOT
        differentiable in the PyTorch sense, but acts as a safety filter during rollout.
        """
        if self._is_safe_action(env, proposed_action, cv_voltage_v):
            return proposed_action, False

        if self._is_safe_action(env, fallback_action, cv_voltage_v):
            return fallback_action, True

        # Search from low-stress action (no charging) toward fallback.
        fallback = float(np.clip(fallback_action, -1.0, 1.0))
        candidates = np.linspace(-1.0, fallback, 11)
        for action in candidates:
            if self._is_safe_action(env, float(action), cv_voltage_v):
                return float(action), True

        return -1.0, True

    def act(
        self,
        state: Dict,
        env,
        mpc_action: float,
        cv_voltage_v: float,
    ) -> Tuple[float, Dict]:
        proposed_action, info = self.propose_action(
            state=state,
            mpc_action=mpc_action,
            target_soc=env.target_soc,
            cv_voltage_v=cv_voltage_v,
            max_charge_current_a=env.max_charge_current_a,
        )
        safe_action, shield_used = self.choose_safe_action(
            env=env,
            state=state,
            proposed_action=proposed_action,
            fallback_action=float(mpc_action),
            cv_voltage_v=cv_voltage_v,
        )
        info["proposed_action"] = proposed_action
        info["safe_action"] = safe_action
        info["shield_used"] = shield_used
        return safe_action, info

    def _is_safe_action(self, env, action: float, cv_voltage_v: float) -> bool:
        test_pack = copy.deepcopy(env.pack)
        pack_current = env.action_to_pack_current(action)
        next_state = test_pack.step(pack_current, ambient_temp=env.ambient_temp)

        voltage_limit = min(
            env.pack.config.V_pack_max,
            cv_voltage_v + self.config.safety_voltage_margin_v,
        )
        temp_limit = min(
            env.pack.config.T_pack_max,
            self.config.temp_soft_limit_c + self.config.safety_temp_margin_c,
        )
        if float(next_state["pack_voltage"]) > voltage_limit:
            return False
        if float(next_state["pack_temperature"]) > temp_limit:
            return False
        events = next_state.get("safety_events", {})
        if isinstance(events, dict):
            if events.get("pack_over_voltage", False):
                return False
            if events.get("pack_under_voltage", False):
                return False
            if events.get("current_limit", False):
                return False
        return True

    def fit_supervised(
        self,
        features: np.ndarray,
        target_delta: np.ndarray,
        ridge_lambda: float | None = None,
    ) -> Dict[str, float]:
        x = np.asarray(features, dtype=float)
        y = np.asarray(target_delta, dtype=float).reshape(-1)
        if x.ndim != 2 or x.shape[1] != self.FEATURE_DIM:
            raise ValueError(
                f"features shape must be (N, {self.FEATURE_DIM}), got {x.shape}"
            )
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                f"target length mismatch: {y.shape[0]} vs {x.shape[0]}"
            )

        y = np.clip(y, -self.config.max_delta_action, self.config.max_delta_action)
        lam = self.config.ridge_lambda if ridge_lambda is None else float(ridge_lambda)

        x_design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=float)], axis=1)
        reg = lam * np.eye(x_design.shape[1], dtype=float)
        reg[-1, -1] = 0.0  # do not regularize bias
        theta = np.linalg.solve(x_design.T @ x_design + reg, x_design.T @ y)

        self.weights = theta[:-1]
        self.bias = float(theta[-1])

        pred = self.predict_batch(x)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        mae = float(np.mean(np.abs(pred - y)))
        return {"rmse": rmse, "mae": mae, "n_samples": int(x.shape[0])}

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.FEATURE_DIM:
            raise ValueError(
                f"features shape must be (N, {self.FEATURE_DIM}), got {x.shape}"
            )
        raw = x @ self.weights + self.bias
        pred = self.config.delta_action_scale * np.tanh(raw)
        return np.clip(pred, -self.config.max_delta_action, self.config.max_delta_action)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "feature_dim": self.FEATURE_DIM,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)

    @classmethod
    def load(cls, path: str | Path) -> "ResidualHAMBRLController":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        config = ResidualPolicyConfig(**payload.get("config", {}))
        weights = np.asarray(payload.get("weights", []), dtype=float)
        bias = float(payload.get("bias", 0.0))
        return cls(config=config, weights=weights, bias=bias)
