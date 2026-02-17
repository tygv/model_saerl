"""Safe Adaptive Ensemble RL (SAERL) controller components."""

from __future__ import annotations

import copy
import json
import pickle
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor


TARGET_KEYS: Tuple[str, ...] = (
    "next_soc",
    "next_voltage",
    "next_temp",
    "next_imbalance",
)
EXPERT_KEYS: Tuple[str, ...] = ("gru", "mlp", "rf")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def state_to_feature_vector(state: Dict[str, Any], max_charge_current_a: float = 10.0) -> np.ndarray:
    soc = _safe_float(state.get("pack_soc", 0.0))
    pack_voltage = _safe_float(state.get("pack_voltage", 0.0))
    pack_temp = _safe_float(state.get("pack_temperature", 25.0))
    voltage_imbalance = _safe_float(state.get("voltage_imbalance", 0.0))
    pack_current = _safe_float(state.get("pack_current", 0.0))
    charge_current_a = -pack_current
    current_norm = charge_current_a / max(max_charge_current_a, 1e-6)
    return np.array(
        [
            soc,
            pack_voltage / 100.0,
            pack_temp / 60.0,
            np.clip(voltage_imbalance * 1000.0 / 200.0, -3.0, 3.0),
            np.clip(current_norm, -2.0, 2.0),
            np.clip(current_norm * current_norm, 0.0, 4.0),
        ],
        dtype=np.float32,
    )


def window_to_sequence(
    state_window: Sequence[Dict[str, Any]],
    window_len: int,
    max_charge_current_a: float,
) -> np.ndarray:
    if not state_window:
        return np.zeros((window_len, 6), dtype=np.float32)
    seq = [state_to_feature_vector(s, max_charge_current_a=max_charge_current_a) for s in state_window]
    if len(seq) >= window_len:
        seq = seq[-window_len:]
    else:
        seq = [seq[0].copy() for _ in range(window_len - len(seq))] + seq
    return np.stack(seq, axis=0).astype(np.float32)


def quantile_pinball_loss(
    y_true: torch.Tensor,
    y_pred_q10: torch.Tensor,
    y_pred_q50: torch.Tensor,
    y_pred_q90: torch.Tensor,
) -> torch.Tensor:
    def _pinball(q: float, y: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
        err = y - yp
        return torch.maximum(q * err, (q - 1.0) * err)

    return (
        _pinball(0.1, y_true, y_pred_q10).mean()
        + _pinball(0.5, y_true, y_pred_q50).mean()
        + _pinball(0.9, y_true, y_pred_q90).mean()
    )


class QuantileGRUModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        target_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.target_dim = int(target_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * self.target_dim),
        )

    def forward(self, x_seq: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x_seq)
        h_last = h[:, -1, :]
        return self.head(torch.cat([h_last, action], dim=1))


class QuantileMLPModel(nn.Module):
    def __init__(self, input_dim: int, target_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.target_dim = int(target_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * self.target_dim),
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        return self.net(x_flat)


class GateNetwork(nn.Module):
    def __init__(self, input_dim: int, n_experts: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class ResidualActorNetwork(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(x)
        std = torch.exp(self.log_std).clamp(min=1e-4, max=2.0)
        return mean, std


class ResidualActorPolicy:
    def __init__(
        self,
        input_dim: int,
        delta_action_limit: float = 0.40,
        device: str = "cpu",
    ) -> None:
        self.input_dim = int(input_dim)
        self.delta_action_limit = float(delta_action_limit)
        self.device = torch.device(device)
        self.model = ResidualActorNetwork(input_dim=self.input_dim).to(self.device)

    @staticmethod
    def build_input(sequence: np.ndarray, mpc_action: float, target_soc: float = 0.8) -> np.ndarray:
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 2:
            raise ValueError(f"sequence must be [T, F], got {seq.shape}")
        soc = float(seq[-1, 0])
        soc_gap = float(np.clip(target_soc - soc, -1.0, 1.0))
        return np.concatenate(
            [
                seq.reshape(-1),
                np.array([_safe_float(mpc_action), soc_gap], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

    def predict_delta(
        self,
        sequence: np.ndarray,
        mpc_action: float,
        target_soc: float = 0.8,
        stochastic: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        x = self.build_input(sequence=sequence, mpc_action=mpc_action, target_soc=target_soc)
        x_t = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            mean, std = self.model(x_t)
            if stochastic:
                eps = torch.randn_like(mean)
                a = mean + eps * std
            else:
                a = mean
            delta = float(torch.tanh(a).cpu().item() * self.delta_action_limit)
            mean_val = float(torch.tanh(mean).cpu().item() * self.delta_action_limit)
            std_val = float(std.cpu().item())
        return delta, {"mean_delta": mean_val, "std": std_val}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_dim": self.input_dim,
                "delta_action_limit": self.delta_action_limit,
                "model_state": self.model.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ResidualActorPolicy":
        payload = torch.load(Path(path), map_location=device)
        obj = cls(
            input_dim=int(payload["input_dim"]),
            delta_action_limit=float(payload["delta_action_limit"]),
            device=device,
        )
        obj.model.load_state_dict(payload["model_state"])
        obj.model.eval()
        return obj


@dataclass
class SAERLConfig:
    window_len: int = 20
    delta_action_limit: float = 0.40
    gate_update_interval: int = 50
    temp_soft_limit_c: float = 42.0
    voltage_margin_v: float = 0.03
    temp_margin_c: float = 0.50
    candidate_points: int = 9
    candidate_radius: float = 0.35
    include_cccv_anchor_candidate: bool = True
    include_full_charge_anchor: bool = True
    feature_dim: int = 6
    target_dim: int = 4
    risk_uncertainty_weight: float = 0.25
    risk_temp_weight: float = 8.0
    risk_voltage_weight: float = 8.0
    risk_imbalance_weight: float = 0.02
    score_soc_gain_weight: float = 120.0
    score_time_weight: float = 1.0
    score_temp_weight: float = 30.0
    score_degradation_weight: float = 15.0
    score_imbalance_weight: float = 0.005
    score_safety_weight: float = 300.0
    score_risk_weight: float = 1.0
    min_safe_charge_fraction: float = 0.20
    anti_stall_soc_gap: float = 0.25
    anti_stall_low_risk_threshold: float = 0.35
    anti_stall_duration_s: float = 120.0
    anti_stall_risk_scale: float = 3.0
    imbalance_margin_v: float = 0.095
    gate_error_momentum: float = 0.90
    use_adaptive_gate: bool = True
    enable_shield: bool = True
    enable_antistall: bool = True
    rf_uncertainty_tree_samples: int = 32


class AdaptiveEnsemblePredictor:
    """Adaptive ensemble predictor with GRU, MLP, and RF experts."""

    def __init__(self, config: SAERLConfig | None = None, device: str = "cpu") -> None:
        if config is None:
            config = SAERLConfig()
        self.config = config
        self.device = torch.device(device)
        self.gru_model = QuantileGRUModel(
            feature_dim=self.config.feature_dim,
            target_dim=self.config.target_dim,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
        ).to(self.device)
        flat_dim = self.config.window_len * self.config.feature_dim + 1
        self.mlp_model = QuantileMLPModel(input_dim=flat_dim, target_dim=self.config.target_dim).to(
            self.device
        )
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=5,
            random_state=123,
            n_jobs=1,
        )
        self.rf_fitted = False

        gate_input_dim = self.config.feature_dim + 1 + len(EXPERT_KEYS) + len(EXPERT_KEYS)
        self.gate_model = GateNetwork(input_dim=gate_input_dim, n_experts=len(EXPERT_KEYS)).to(
            self.device
        )
        self.gate_input_dim = gate_input_dim

        self.calibration = {"gru": 1.0, "mlp": 1.0, "rf": 1.0}
        self.cached_weights = np.ones(len(EXPERT_KEYS), dtype=np.float32) / float(len(EXPERT_KEYS))
        self.step_index = 0
        self.disabled_experts: set[str] = set()
        self.rolling_abs_error: Dict[str, np.ndarray] = {
            name: np.ones(self.config.target_dim, dtype=np.float32) * 0.05 for name in EXPERT_KEYS
        }

    def set_disabled_experts(self, disabled: Iterable[str]) -> None:
        disabled_set = {str(x).strip().lower() for x in disabled}
        self.disabled_experts = {x for x in disabled_set if x in EXPERT_KEYS}

    def _split_quantiles(self, output: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = output.detach().cpu().numpy().reshape(-1)
        td = self.config.target_dim
        q10 = arr[0:td]
        q50 = arr[td : 2 * td]
        q90 = arr[2 * td : 3 * td]
        q_low = np.minimum(np.minimum(q10, q50), q90)
        q_mid = q50
        q_high = np.maximum(np.maximum(q10, q50), q90)
        return q_low.astype(np.float32), q_mid.astype(np.float32), q_high.astype(np.float32)

    def _rf_tree_stats(self, x_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.rf_fitted:
            return np.zeros(self.config.target_dim, dtype=np.float32), np.ones(self.config.target_dim, dtype=np.float32) * 0.05
        x2 = x_flat.reshape(1, -1)
        mean = self.rf_model.predict(x2)[0].astype(np.float32)
        estimators = self.rf_model.estimators_
        n_trees = len(estimators)
        k = int(max(4, min(n_trees, self.config.rf_uncertainty_tree_samples)))
        if k < n_trees:
            idx = np.linspace(0, n_trees - 1, num=k, dtype=int)
            tree_preds = np.stack([estimators[i].predict(x2)[0] for i in idx], axis=0)
        else:
            tree_preds = np.stack([tree.predict(x2)[0] for tree in estimators], axis=0)
        std = np.std(tree_preds.astype(np.float32), axis=0)
        return mean, np.clip(std, 1e-6, None)

    def _rf_tree_stats_batch(self, x_flat_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.rf_fitted:
            n = x_flat_batch.shape[0]
            return (
                np.zeros((n, self.config.target_dim), dtype=np.float32),
                np.ones((n, self.config.target_dim), dtype=np.float32) * 0.05,
            )
        mean = self.rf_model.predict(x_flat_batch).astype(np.float32)
        estimators = self.rf_model.estimators_
        n_trees = len(estimators)
        k = int(max(4, min(n_trees, self.config.rf_uncertainty_tree_samples)))
        if k < n_trees:
            idx = np.linspace(0, n_trees - 1, num=k, dtype=int)
            tree_preds = np.stack([estimators[i].predict(x_flat_batch) for i in idx], axis=0)
        else:
            tree_preds = np.stack([tree.predict(x_flat_batch) for tree in estimators], axis=0)
        std = np.std(tree_preds.astype(np.float32), axis=0)
        return mean, np.clip(std, 1e-6, None)

    def _fallback_prediction(self, state_window: Sequence[Dict[str, Any]]) -> np.ndarray:
        state = state_window[-1] if state_window else {}
        return np.array(
            [
                _safe_float(state.get("pack_soc", 0.0)),
                _safe_float(state.get("pack_voltage", 0.0)),
                _safe_float(state.get("pack_temperature", 25.0)),
                _safe_float(state.get("voltage_imbalance", 0.0)),
            ],
            dtype=np.float32,
        )

    def _gate_input(
        self,
        state_window: Sequence[Dict[str, Any]],
        action: float,
        expert_outputs: Dict[str, Dict[str, np.ndarray]],
        max_charge_current_a: float,
    ) -> np.ndarray:
        seq = window_to_sequence(
            state_window=state_window,
            window_len=self.config.window_len,
            max_charge_current_a=max_charge_current_a,
        )
        last_feat = seq[-1]
        unc_values = []
        err_values = []
        for name in EXPERT_KEYS:
            out = expert_outputs[name]
            unc_values.append(float(np.mean(out["uncertainty"])))
            err_values.append(float(np.mean(self.rolling_abs_error[name])))
        gate_input = np.concatenate(
            [
                last_feat.astype(np.float32),
                np.array([_safe_float(action)], dtype=np.float32),
                np.array(unc_values, dtype=np.float32),
                np.array(err_values, dtype=np.float32),
            ],
            axis=0,
        )
        if gate_input.size != self.gate_input_dim:
            raise ValueError(
                f"gate input dim mismatch: expected {self.gate_input_dim}, got {gate_input.size}"
            )
        return gate_input.astype(np.float32)

    def _compute_weights(self, gate_input: np.ndarray) -> np.ndarray:
        enabled_mask = np.array([name not in self.disabled_experts for name in EXPERT_KEYS], dtype=np.float32)
        if enabled_mask.sum() <= 0:
            enabled_mask[:] = 1.0

        if not self.config.use_adaptive_gate:
            return (enabled_mask / enabled_mask.sum()).astype(np.float32)

        refresh = (self.step_index % max(1, self.config.gate_update_interval) == 0) or (
            self.step_index == 0
        )
        if refresh:
            self.gate_model.eval()
            with torch.no_grad():
                x = torch.from_numpy(gate_input).float().unsqueeze(0).to(self.device)
                w = self.gate_model(x).cpu().numpy().reshape(-1).astype(np.float32)
            w = w * enabled_mask
            if float(np.sum(w)) <= 1e-8:
                w = enabled_mask / enabled_mask.sum()
            else:
                w = w / np.sum(w)
            self.cached_weights = w.astype(np.float32)
        return self.cached_weights.copy()

    def predict_experts(
        self,
        state_window: Sequence[Dict[str, Any]],
        action: float,
        max_charge_current_a: float = 10.0,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Predict next-state targets and uncertainty for each expert."""
        seq = window_to_sequence(
            state_window=state_window,
            window_len=self.config.window_len,
            max_charge_current_a=max_charge_current_a,
        )
        x_seq = torch.from_numpy(seq).float().unsqueeze(0).to(self.device)
        action_t = torch.tensor([[float(action)]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            self.gru_model.eval()
            out_gru = self.gru_model(x_seq, action_t)
        q10_g, q50_g, q90_g = self._split_quantiles(out_gru)
        unc_gru = np.clip((q90_g - q10_g) * float(self.calibration["gru"]), 1e-6, None)

        x_flat = np.concatenate([seq.reshape(-1), np.array([float(action)], dtype=np.float32)], axis=0)
        x_flat_t = torch.from_numpy(x_flat).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.mlp_model.eval()
            out_mlp = self.mlp_model(x_flat_t)
        q10_m, q50_m, q90_m = self._split_quantiles(out_mlp)
        unc_mlp = np.clip((q90_m - q10_m) * float(self.calibration["mlp"]), 1e-6, None)

        rf_mean, rf_std = self._rf_tree_stats(x_flat)
        if not self.rf_fitted:
            rf_mean = self._fallback_prediction(state_window=state_window)
        unc_rf = np.clip(rf_std * float(self.calibration["rf"]), 1e-6, None)

        return {
            "gru": {
                "mean": q50_g.astype(np.float32),
                "uncertainty": unc_gru.astype(np.float32),
                "q10": q10_g.astype(np.float32),
                "q90": q90_g.astype(np.float32),
            },
            "mlp": {
                "mean": q50_m.astype(np.float32),
                "uncertainty": unc_mlp.astype(np.float32),
                "q10": q10_m.astype(np.float32),
                "q90": q90_m.astype(np.float32),
            },
            "rf": {
                "mean": rf_mean.astype(np.float32),
                "uncertainty": unc_rf.astype(np.float32),
                "q10": (rf_mean - 1.64 * unc_rf).astype(np.float32),
                "q90": (rf_mean + 1.64 * unc_rf).astype(np.float32),
            },
        }

    def predict_fused(
        self,
        state_window: Sequence[Dict[str, Any]],
        action: float,
        max_charge_current_a: float = 10.0,
        cv_voltage_v: Optional[float] = None,
    ) -> Dict[str, Any]:
        expert_outputs = self.predict_experts(
            state_window=state_window,
            action=action,
            max_charge_current_a=max_charge_current_a,
        )
        gate_input = self._gate_input(
            state_window=state_window,
            action=action,
            expert_outputs=expert_outputs,
            max_charge_current_a=max_charge_current_a,
        )
        weights = self._compute_weights(gate_input)
        self.step_index += 1

        means = np.stack([expert_outputs[k]["mean"] for k in EXPERT_KEYS], axis=0)
        uncs = np.stack([expert_outputs[k]["uncertainty"] for k in EXPERT_KEYS], axis=0)
        fused_mean = np.sum(weights[:, None] * means, axis=0)
        fused_unc = np.sum(weights[:, None] * uncs, axis=0)

        pred_soc = float(fused_mean[0])
        pred_v = float(fused_mean[1])
        pred_t = float(fused_mean[2])
        pred_imb = float(fused_mean[3])
        prev_soc = _safe_float(state_window[-1].get("pack_soc", 0.0)) if state_window else 0.0

        over_temp = max(0.0, pred_t - self.config.temp_soft_limit_c)
        over_v = 0.0 if cv_voltage_v is None else max(0.0, pred_v - float(cv_voltage_v + self.config.voltage_margin_v))
        risk_score = (
            self.config.risk_temp_weight * (over_temp**2)
            + self.config.risk_voltage_weight * (over_v**2)
            + self.config.risk_imbalance_weight * abs(pred_imb * 1000.0)
            + self.config.risk_uncertainty_weight * float(np.mean(fused_unc))
            + max(0.0, prev_soc - pred_soc) * 8.0
        )

        return {
            "prediction": {
                "next_soc": pred_soc,
                "next_voltage": pred_v,
                "next_temp": pred_t,
                "next_imbalance": pred_imb,
            },
            "fused_uncertainty": {
                "next_soc": float(fused_unc[0]),
                "next_voltage": float(fused_unc[1]),
                "next_temp": float(fused_unc[2]),
                "next_imbalance": float(fused_unc[3]),
            },
            "risk_score": float(risk_score),
            "expert_weights": {k: float(weights[i]) for i, k in enumerate(EXPERT_KEYS)},
            "expert_outputs": expert_outputs,
            "gate_input": gate_input,
        }

    def predict_fused_batch(
        self,
        state_window: Sequence[Dict[str, Any]],
        actions: Sequence[float],
        max_charge_current_a: float = 10.0,
        cv_voltage_v: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not actions:
            return []
        seq = window_to_sequence(
            state_window=state_window,
            window_len=self.config.window_len,
            max_charge_current_a=max_charge_current_a,
        )
        n = len(actions)
        seq_batch = np.repeat(seq[np.newaxis, :, :], repeats=n, axis=0).astype(np.float32)
        action_arr = np.asarray(actions, dtype=np.float32).reshape(-1, 1)

        x_seq_t = torch.from_numpy(seq_batch).float().to(self.device)
        action_t = torch.from_numpy(action_arr).float().to(self.device)
        with torch.no_grad():
            self.gru_model.eval()
            out_gru = self.gru_model(x_seq_t, action_t).cpu().numpy()

        seq_flat = seq.reshape(-1).astype(np.float32)
        x_flat_batch = np.concatenate(
            [
                np.repeat(seq_flat[np.newaxis, :], repeats=n, axis=0),
                action_arr,
            ],
            axis=1,
        ).astype(np.float32)
        with torch.no_grad():
            self.mlp_model.eval()
            out_mlp = self.mlp_model(torch.from_numpy(x_flat_batch).float().to(self.device)).cpu().numpy()

        rf_mean_batch, rf_std_batch = self._rf_tree_stats_batch(x_flat_batch)
        if not self.rf_fitted:
            fallback = self._fallback_prediction(state_window=state_window)
            rf_mean_batch = np.repeat(fallback[np.newaxis, :], repeats=n, axis=0).astype(np.float32)

        td = self.config.target_dim
        q10g = np.minimum(np.minimum(out_gru[:, 0:td], out_gru[:, td : 2 * td]), out_gru[:, 2 * td : 3 * td]).astype(np.float32)
        q50g = out_gru[:, td : 2 * td].astype(np.float32)
        q90g = np.maximum(np.maximum(out_gru[:, 0:td], out_gru[:, td : 2 * td]), out_gru[:, 2 * td : 3 * td]).astype(np.float32)
        q10m = np.minimum(np.minimum(out_mlp[:, 0:td], out_mlp[:, td : 2 * td]), out_mlp[:, 2 * td : 3 * td]).astype(np.float32)
        q50m = out_mlp[:, td : 2 * td].astype(np.float32)
        q90m = np.maximum(np.maximum(out_mlp[:, 0:td], out_mlp[:, td : 2 * td]), out_mlp[:, 2 * td : 3 * td]).astype(np.float32)
        unc_g = np.clip((q90g - q10g) * float(self.calibration["gru"]), 1e-6, None)
        unc_m = np.clip((q90m - q10m) * float(self.calibration["mlp"]), 1e-6, None)
        unc_rf = np.clip(rf_std_batch * float(self.calibration["rf"]), 1e-6, None)

        prev_soc = _safe_float(state_window[-1].get("pack_soc", 0.0)) if state_window else 0.0
        payloads: List[Dict[str, Any]] = []
        base_step = int(self.step_index)
        for i in range(n):
            self.step_index = base_step
            expert_outputs = {
                "gru": {"mean": q50g[i], "uncertainty": unc_g[i], "q10": q10g[i], "q90": q90g[i]},
                "mlp": {"mean": q50m[i], "uncertainty": unc_m[i], "q10": q10m[i], "q90": q90m[i]},
                "rf": {
                    "mean": rf_mean_batch[i].astype(np.float32),
                    "uncertainty": unc_rf[i].astype(np.float32),
                    "q10": (rf_mean_batch[i] - 1.64 * unc_rf[i]).astype(np.float32),
                    "q90": (rf_mean_batch[i] + 1.64 * unc_rf[i]).astype(np.float32),
                },
            }
            gate_input = self._gate_input(
                state_window=state_window,
                action=float(action_arr[i, 0]),
                expert_outputs=expert_outputs,
                max_charge_current_a=max_charge_current_a,
            )
            weights = self._compute_weights(gate_input)

            means = np.stack([expert_outputs[k]["mean"] for k in EXPERT_KEYS], axis=0)
            uncs = np.stack([expert_outputs[k]["uncertainty"] for k in EXPERT_KEYS], axis=0)
            fused_mean = np.sum(weights[:, None] * means, axis=0)
            fused_unc = np.sum(weights[:, None] * uncs, axis=0)
            pred_soc = float(fused_mean[0])
            pred_v = float(fused_mean[1])
            pred_t = float(fused_mean[2])
            pred_imb = float(fused_mean[3])
            over_temp = max(0.0, pred_t - self.config.temp_soft_limit_c)
            over_v = 0.0 if cv_voltage_v is None else max(0.0, pred_v - float(cv_voltage_v + self.config.voltage_margin_v))
            risk_score = (
                self.config.risk_temp_weight * (over_temp**2)
                + self.config.risk_voltage_weight * (over_v**2)
                + self.config.risk_imbalance_weight * abs(pred_imb * 1000.0)
                + self.config.risk_uncertainty_weight * float(np.mean(fused_unc))
                + max(0.0, prev_soc - pred_soc) * 8.0
            )
            payloads.append(
                {
                    "prediction": {
                        "next_soc": pred_soc,
                        "next_voltage": pred_v,
                        "next_temp": pred_t,
                        "next_imbalance": pred_imb,
                    },
                    "fused_uncertainty": {
                        "next_soc": float(fused_unc[0]),
                        "next_voltage": float(fused_unc[1]),
                        "next_temp": float(fused_unc[2]),
                        "next_imbalance": float(fused_unc[3]),
                    },
                    "risk_score": float(risk_score),
                    "expert_weights": {k: float(weights[j]) for j, k in enumerate(EXPERT_KEYS)},
                    "expert_outputs": expert_outputs,
                    "gate_input": gate_input,
                }
            )
        self.step_index = base_step + 1
        return payloads

    def update_error_statistics(
        self,
        expert_outputs: Dict[str, Dict[str, np.ndarray]],
        observed_next_state: Dict[str, Any],
    ) -> None:
        y_true = np.array(
            [
                _safe_float(observed_next_state.get("pack_soc", 0.0)),
                _safe_float(observed_next_state.get("pack_voltage", 0.0)),
                _safe_float(observed_next_state.get("pack_temperature", 25.0)),
                _safe_float(observed_next_state.get("voltage_imbalance", 0.0)),
            ],
            dtype=np.float32,
        )
        for name in EXPERT_KEYS:
            y_pred = expert_outputs[name]["mean"]
            abs_err = np.abs(y_pred - y_true).astype(np.float32)
            old = self.rolling_abs_error[name]
            mom = float(np.clip(self.config.gate_error_momentum, 0.0, 0.999))
            self.rolling_abs_error[name] = mom * old + (1.0 - mom) * abs_err

    def fit_rf(self, x_flat_action: np.ndarray, y_target: np.ndarray) -> None:
        self.rf_model.fit(x_flat_action, y_target)
        self.rf_fitted = True

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.gru_model.state_dict(), directory / "gru_quantile.pt")
        torch.save(self.mlp_model.state_dict(), directory / "mlp_quantile.pt")
        torch.save(self.gate_model.state_dict(), directory / "gate.pt")
        with (directory / "rf_model.pkl").open("wb") as handle:
            pickle.dump(self.rf_model, handle)
        metadata = {
            "config": asdict(self.config),
            "calibration": self.calibration,
            "rf_fitted": self.rf_fitted,
            "rolling_abs_error": {k: v.tolist() for k, v in self.rolling_abs_error.items()},
            "cached_weights": self.cached_weights.tolist(),
            "step_index": int(self.step_index),
        }
        with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=True)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        device: str = "cpu",
        config_override: SAERLConfig | None = None,
    ) -> "AdaptiveEnsemblePredictor":
        directory = Path(directory)
        with (directory / "metadata.json").open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        config = SAERLConfig(**meta.get("config", {}))
        if config_override is not None:
            config = config_override
        obj = cls(config=config, device=device)
        obj.gru_model.load_state_dict(torch.load(directory / "gru_quantile.pt", map_location=device))
        obj.mlp_model.load_state_dict(torch.load(directory / "mlp_quantile.pt", map_location=device))
        obj.gate_model.load_state_dict(torch.load(directory / "gate.pt", map_location=device))
        with (directory / "rf_model.pkl").open("rb") as handle:
            obj.rf_model = pickle.load(handle)
        if hasattr(obj.rf_model, "n_jobs"):
            obj.rf_model.n_jobs = 1
        obj.rf_fitted = bool(meta.get("rf_fitted", True))
        obj.calibration = {
            "gru": float(meta.get("calibration", {}).get("gru", 1.0)),
            "mlp": float(meta.get("calibration", {}).get("mlp", 1.0)),
            "rf": float(meta.get("calibration", {}).get("rf", 1.0)),
        }
        obj.rolling_abs_error = {
            k: np.asarray(meta.get("rolling_abs_error", {}).get(k, [0.05] * obj.config.target_dim), dtype=np.float32)
            for k in EXPERT_KEYS
        }
        obj.cached_weights = np.asarray(
            meta.get("cached_weights", [1.0 / len(EXPERT_KEYS)] * len(EXPERT_KEYS)),
            dtype=np.float32,
        )
        obj.step_index = int(meta.get("step_index", 0))
        obj.gru_model.eval()
        obj.mlp_model.eval()
        obj.gate_model.eval()
        return obj


class SafeAdaptiveEnsembleController:
    """Residual controller over MPC with adaptive ensemble scoring and hard shield."""

    def __init__(
        self,
        predictor: AdaptiveEnsemblePredictor,
        actor: Optional[ResidualActorPolicy] = None,
        config: SAERLConfig | None = None,
    ) -> None:
        self.predictor = predictor
        self.actor = actor
        self.config = predictor.config if config is None else config
        self.state_window: Deque[Dict[str, Any]] = deque(maxlen=self.config.window_len)
        self.low_risk_duration_s = 0.0
        self.step_count = 0
        self.shield_count = 0
        self.last_transition_cache: Dict[str, Any] = {}

    def reset(self) -> None:
        self.state_window.clear()
        self.low_risk_duration_s = 0.0
        self.step_count = 0
        self.shield_count = 0
        self.last_transition_cache = {}

    def _append_state(self, state: Dict[str, Any]) -> None:
        if len(self.state_window) == 0:
            for _ in range(self.config.window_len - 1):
                self.state_window.append(dict(state))
        self.state_window.append(dict(state))

    def _degradation_proxy(
        self,
        predicted_next: Dict[str, float],
        charge_current_a: float,
        max_charge_current_a: float,
    ) -> float:
        soc = float(np.clip(predicted_next["next_soc"], 0.0, 1.0))
        temp = float(predicted_next["next_temp"])
        c_rate = abs(charge_current_a) / max(max_charge_current_a, 1e-6)
        temp_factor = float(np.exp(max(0.0, temp - 25.0) / 18.0))
        soc_factor = float(np.exp(0.8 * soc))
        return float((c_rate**2) * temp_factor * soc_factor)

    def _candidate_score(
        self,
        fused: Dict[str, Any],
        state: Dict[str, Any],
        action: float,
        env,
        cv_voltage_v: float,
    ) -> float:
        pred = fused["prediction"]
        risk = float(fused["risk_score"])
        soc_gain = float(pred["next_soc"] - _safe_float(state.get("pack_soc", 0.0)))
        pred_v = float(pred["next_voltage"])
        pred_t = float(pred["next_temp"])
        pred_imb_mv = float(pred["next_imbalance"] * 1000.0)
        charge_current_a = -float(env.action_to_pack_current(action))
        deg_proxy = self._degradation_proxy(
            predicted_next=pred,
            charge_current_a=charge_current_a,
            max_charge_current_a=env.max_charge_current_a,
        )
        charge_rate = abs(charge_current_a) / max(float(env.max_charge_current_a), 1e-6)
        # Favor higher safe current (faster progress); penalize near-zero current candidates.
        time_proxy = 1.0 / max(charge_rate, 0.05)

        over_v = max(0.0, pred_v - float(cv_voltage_v + self.config.voltage_margin_v))
        over_t = max(0.0, pred_t - float(self.config.temp_soft_limit_c + self.config.temp_margin_c))
        safety_violation = 1.0 if (over_v > 0.0 or over_t > 0.0) else 0.0
        score = (
            -self.config.score_soc_gain_weight * soc_gain
            + self.config.score_time_weight * time_proxy
            + self.config.score_temp_weight * (over_t**2)
            + self.config.score_degradation_weight * deg_proxy
            + self.config.score_imbalance_weight * abs(pred_imb_mv)
            + self.config.score_safety_weight * safety_violation
            + self.config.score_risk_weight * risk
        )
        return float(score)

    def validate_and_shield_action(
        self,
        state: Dict[str, Any],
        env,
        proposed_action: float,
        mpc_action: float,
        cv_voltage_v: float,
        risk_score: Optional[float] = None,
        apply_antistall: bool = True,
        update_counters: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        selected_action = float(np.clip(proposed_action, -1.0, 1.0))
        mpc_action = float(np.clip(mpc_action, -1.0, 1.0))

        antistall_used = False
        if apply_antistall:
            antistall_action, antistall_used = self._apply_antistall(
                state=state,
                action=selected_action,
                risk_score=float(0.0 if risk_score is None else risk_score),
                env=env,
                cv_voltage_v=cv_voltage_v,
            )
            selected_action = float(antistall_action)

        shield_used = False
        shield_reason = "none"
        if self.config.enable_shield:
            if self._is_safe_action(env=env, action=selected_action, cv_voltage_v=cv_voltage_v):
                final_action = selected_action
            elif self._is_safe_action(env=env, action=mpc_action, cv_voltage_v=cv_voltage_v):
                final_action = mpc_action
                shield_used = True
                shield_reason = "fallback_mpc"
            else:
                final_action = self._least_stress_safe_action(
                    env=env,
                    mpc_action=mpc_action,
                    cv_voltage_v=cv_voltage_v,
                )
                shield_used = True
                shield_reason = "least_stress_safe"
        else:
            final_action = selected_action

        if update_counters:
            if shield_used:
                self.shield_count += 1
            self.step_count += 1

        return float(final_action), {
            "selected_action_pre_shield": float(selected_action),
            "safe_action": float(final_action),
            "shield_used": bool(shield_used),
            "shield_reason": str(shield_reason),
            "antistall_used": bool(antistall_used),
        }

    def _is_safe_action(self, env, action: float, cv_voltage_v: float) -> bool:
        pack_current = env.action_to_pack_current(action)
        next_state = self._simulate_next_state(env=env, pack_current=pack_current)

        voltage_limit = min(
            float(env.pack.config.V_pack_max),
            float(cv_voltage_v + self.config.voltage_margin_v),
        )
        temp_limit = min(
            float(env.pack.config.T_pack_max),
            float(self.config.temp_soft_limit_c + self.config.temp_margin_c),
        )
        if float(next_state["pack_voltage"]) > voltage_limit:
            return False
        if float(next_state["pack_temperature"]) > temp_limit:
            return False
        if abs(float(next_state["pack_current"])) > float(env.pack.config.I_pack_max):
            return False
        if abs(float(next_state.get("voltage_imbalance", 0.0))) > float(self.config.imbalance_margin_v):
            return False

        events = next_state.get("safety_events", {})
        if isinstance(events, dict):
            if len(events.get("over_voltage_cells", [])) > 0:
                return False
            if len(events.get("under_voltage_cells", [])) > 0:
                return False
            if events.get("pack_over_voltage", False):
                return False
            if events.get("pack_under_voltage", False):
                return False
            if events.get("current_limit", False):
                return False
            if len(events.get("over_temperature_cells", [])) > 0:
                return False
            if len(events.get("voltage_imbalance", [])) > 0:
                return False
        return True

    def _simulate_next_state(self, env, pack_current: float) -> Dict[str, Any]:
        """Fast one-step simulation with full state restore (avoids deepcopy overhead)."""
        pack = env.pack
        pack_hist_len = len(pack.history)
        saved_pack_fields = {
            "pack_current": float(pack.pack_current),
            "ambient_temperature": float(pack.ambient_temperature),
            "balancing_active": bool(pack.balancing_active),
            "balancing_currents": np.array(pack.balancing_currents, dtype=float).copy(),
            "safety_events": copy.deepcopy(pack.safety_events),
        }
        saved_cells = []
        for cell in pack.cells:
            saved_cells.append(
                {
                    "soc": float(cell.soc),
                    "v1": float(cell.v1),
                    "v2": float(cell.v2),
                    "temperature": float(cell.temperature),
                    "voltage": float(cell.voltage),
                    "current": float(cell.current),
                    "time": float(cell.time),
                    "Q_loss": float(cell.Q_loss),
                    "cycles": int(cell.cycles),
                    "R0_growth": float(cell.R0_growth),
                    "Q_effective": float(cell.Q_effective),
                    "history_len": len(cell.history),
                }
            )

        next_state = pack.step(float(pack_current), ambient_temp=env.ambient_temp)

        # Restore cell-level states.
        for cell, rec in zip(pack.cells, saved_cells):
            cell.soc = rec["soc"]
            cell.v1 = rec["v1"]
            cell.v2 = rec["v2"]
            cell.temperature = rec["temperature"]
            cell.voltage = rec["voltage"]
            cell.current = rec["current"]
            cell.time = rec["time"]
            cell.Q_loss = rec["Q_loss"]
            cell.cycles = rec["cycles"]
            cell.R0_growth = rec["R0_growth"]
            cell.Q_effective = rec["Q_effective"]
            if len(cell.history) > rec["history_len"]:
                cell.history = cell.history[: rec["history_len"]]

        # Restore pack-level states and history.
        if len(pack.history) > pack_hist_len:
            pack.history = pack.history[:pack_hist_len]
        pack._update_pack_state()
        pack.pack_current = saved_pack_fields["pack_current"]
        pack.ambient_temperature = saved_pack_fields["ambient_temperature"]
        pack.balancing_active = saved_pack_fields["balancing_active"]
        pack.balancing_currents = saved_pack_fields["balancing_currents"]
        pack.safety_events = saved_pack_fields["safety_events"]
        return next_state

    def _least_stress_safe_action(self, env, mpc_action: float, cv_voltage_v: float) -> float:
        top = float(np.clip(max(mpc_action, -1.0), -1.0, 1.0))
        for cand in np.linspace(-1.0, top, 21):
            if self._is_safe_action(env=env, action=float(cand), cv_voltage_v=cv_voltage_v):
                return float(cand)
        return -1.0

    def _apply_antistall(
        self,
        state: Dict[str, Any],
        action: float,
        risk_score: float,
        env,
        cv_voltage_v: float,
    ) -> Tuple[float, bool]:
        if not self.config.enable_antistall:
            return action, False

        # Normalize risk to [0, 1] so anti-stall threshold remains meaningful
        # across families whose raw risk magnitudes can vary by dataset.
        risk_scale = max(1e-6, float(self.config.anti_stall_risk_scale))
        normalized_risk = float(max(0.0, risk_score) / (max(0.0, risk_score) + risk_scale))
        soc_gap = float(env.target_soc - _safe_float(state.get("pack_soc", 0.0)))
        dt = float(getattr(env.pack, "dt", 1.0))
        if soc_gap > self.config.anti_stall_soc_gap and normalized_risk < self.config.anti_stall_low_risk_threshold:
            self.low_risk_duration_s += dt
        else:
            self.low_risk_duration_s = 0.0

        if self.low_risk_duration_s < self.config.anti_stall_duration_s:
            return action, False

        min_charge_current = max(
            0.2,
            self.config.min_safe_charge_fraction * float(env.max_charge_current_a),
        )
        floor_action = env.pack_current_to_action(-min_charge_current)
        candidate = float(max(action, floor_action))
        if candidate > action and self._is_safe_action(env=env, action=candidate, cv_voltage_v=cv_voltage_v):
            return candidate, True
        return action, False

    def calibrate_antistall_from_quantiles(
        self,
        normalized_risks: Sequence[float],
        charge_fractions: Sequence[float],
        risk_quantile: float = 0.75,
        floor_quantile: float = 0.65,
        min_samples: int = 48,
    ) -> Dict[str, float]:
        """Calibrate anti-stall thresholds from validation distribution statistics."""
        risks = np.asarray(list(normalized_risks), dtype=np.float64)
        floors = np.asarray(list(charge_fractions), dtype=np.float64)
        valid = np.isfinite(risks) & np.isfinite(floors)
        risks = risks[valid]
        floors = floors[valid]
        if risks.size < int(max(1, min_samples)):
            return {
                "calibrated": 0.0,
                "n_samples": float(risks.size),
                "anti_stall_low_risk_threshold": float(self.config.anti_stall_low_risk_threshold),
                "min_safe_charge_fraction": float(self.config.min_safe_charge_fraction),
            }

        rq = float(np.clip(risk_quantile, 0.05, 0.95))
        fq = float(np.clip(floor_quantile, 0.05, 0.95))
        risk_thr = float(np.clip(np.quantile(risks, rq), 0.05, 0.95))
        floor_fraction = float(np.clip(np.quantile(floors, fq), 0.05, 0.95))

        # Keep anti-stall activation permissive enough for stalled high-gap regimes.
        self.config.anti_stall_low_risk_threshold = float(
            max(self.config.anti_stall_low_risk_threshold, risk_thr)
        )
        self.config.min_safe_charge_fraction = float(
            max(self.config.min_safe_charge_fraction, floor_fraction)
        )
        return {
            "calibrated": 1.0,
            "n_samples": float(risks.size),
            "risk_quantile": rq,
            "floor_quantile": fq,
            "anti_stall_low_risk_threshold": float(self.config.anti_stall_low_risk_threshold),
            "min_safe_charge_fraction": float(self.config.min_safe_charge_fraction),
        }

    def _cccv_anchor_action(
        self,
        state: Dict[str, Any],
        env,
        cv_voltage_v: float,
    ) -> float:
        pack_voltage = _safe_float(state.get("pack_voltage", 0.0))
        max_charge_current = float(env.max_charge_current_a)
        if pack_voltage < float(cv_voltage_v - 0.05):
            target_charge_current = max_charge_current
        else:
            # Linear taper as pack voltage approaches/exceeds CV threshold.
            voltage_headroom = float(cv_voltage_v - pack_voltage)
            taper_scale = np.clip(voltage_headroom / 0.2, 0.05, 1.0)
            target_charge_current = max(0.2 * max_charge_current, taper_scale * max_charge_current)
        return float(env.pack_current_to_action(-target_charge_current))

    def act(
        self,
        state: Dict[str, Any],
        env,
        mpc_action: float,
        cv_voltage_v: float,
    ) -> Tuple[float, Dict[str, Any]]:
        self._append_state(state)
        state_window_list = list(self.state_window)
        mpc_action = float(np.clip(mpc_action, -1.0, 1.0))

        seq = window_to_sequence(
            state_window=state_window_list,
            window_len=self.config.window_len,
            max_charge_current_a=env.max_charge_current_a,
        )
        if self.actor is None:
            proposal_delta = 0.0
            actor_info = {"mean_delta": 0.0, "std": 0.0}
        else:
            proposal_delta, actor_info = self.actor.predict_delta(
                sequence=seq,
                mpc_action=mpc_action,
                target_soc=float(env.target_soc),
                stochastic=False,
            )
        proposal_delta = float(
            np.clip(proposal_delta, -self.config.delta_action_limit, self.config.delta_action_limit)
        )

        delta_candidates = np.linspace(
            proposal_delta - self.config.candidate_radius,
            proposal_delta + self.config.candidate_radius,
            int(max(3, self.config.candidate_points)),
        )
        delta_candidates = np.clip(
            delta_candidates,
            -self.config.delta_action_limit,
            self.config.delta_action_limit,
        )
        if 0.0 not in delta_candidates:
            delta_candidates = np.unique(np.concatenate([delta_candidates, np.array([0.0])], axis=0))

        best_score = float("inf")
        best_action = mpc_action
        best_payload: Optional[Dict[str, Any]] = None
        candidate_actions = [float(np.clip(mpc_action + float(delta), -1.0, 1.0)) for delta in delta_candidates]
        if self.config.include_cccv_anchor_candidate:
            candidate_actions.append(self._cccv_anchor_action(state=state, env=env, cv_voltage_v=cv_voltage_v))
        if self.config.include_full_charge_anchor:
            candidate_actions.append(1.0)
        candidate_actions.append(float(mpc_action))
        candidate_actions = sorted({float(np.clip(a, -1.0, 1.0)) for a in candidate_actions})
        payloads = self.predictor.predict_fused_batch(
            state_window=state_window_list,
            actions=candidate_actions,
            max_charge_current_a=env.max_charge_current_a,
            cv_voltage_v=cv_voltage_v,
        )
        for action, fused in zip(candidate_actions, payloads):
            score = self._candidate_score(
                fused=fused,
                state=state,
                action=action,
                env=env,
                cv_voltage_v=cv_voltage_v,
            )
            if score < best_score:
                best_score = score
                best_action = action
                best_payload = fused

        if best_payload is None:
            best_payload = self.predictor.predict_fused(
                state_window=state_window_list,
                action=best_action,
                max_charge_current_a=env.max_charge_current_a,
                cv_voltage_v=cv_voltage_v,
            )

        antistall_action, antistall_used = self._apply_antistall(
            state=state,
            action=best_action,
            risk_score=float(best_payload["risk_score"]),
            env=env,
            cv_voltage_v=cv_voltage_v,
        )
        selected_action = antistall_action

        final_action, shield_info = self.validate_and_shield_action(
            state=state,
            env=env,
            proposed_action=selected_action,
            mpc_action=mpc_action,
            cv_voltage_v=cv_voltage_v,
            risk_score=float(best_payload["risk_score"]),
            apply_antistall=False,
            update_counters=True,
        )
        shield_used = bool(shield_info["shield_used"])
        shield_reason = str(shield_info["shield_reason"])

        self.last_transition_cache = {
            "expert_outputs": best_payload["expert_outputs"],
            "state_window": state_window_list,
            "action": float(final_action),
        }

        info = {
            "controller_mode": "SAERL",
            "raw_mpc_action": float(mpc_action),
            "proposal_delta_action": float(proposal_delta),
            "proposed_action": float(best_action),
            "selected_action_pre_shield": float(selected_action),
            "safe_action": float(final_action),
            "shield_used": bool(shield_used),
            "shield_reason": str(shield_reason),
            "antistall_used": bool(antistall_used),
            "risk_score": float(best_payload["risk_score"]),
            "risk_score_normalized": float(
                max(0.0, float(best_payload["risk_score"]))
                / (
                    max(0.0, float(best_payload["risk_score"]))
                    + max(1e-6, float(self.config.anti_stall_risk_scale))
                )
            ),
            "expert_weights": {k: float(v) for k, v in best_payload["expert_weights"].items()},
            "fused_prediction": best_payload["prediction"],
            "fused_uncertainty": best_payload["fused_uncertainty"],
            "actor_mean_delta": float(actor_info.get("mean_delta", 0.0)),
            "actor_std": float(actor_info.get("std", 0.0)),
        }
        return float(final_action), info

    def observe_transition(self, next_state: Dict[str, Any]) -> None:
        if not self.last_transition_cache:
            return
        expert_outputs = self.last_transition_cache.get("expert_outputs")
        if isinstance(expert_outputs, dict):
            self.predictor.update_error_statistics(
                expert_outputs=expert_outputs,
                observed_next_state=next_state,
            )

    @property
    def shield_intervention_rate(self) -> float:
        if self.step_count <= 0:
            return 0.0
        return float(self.shield_count / self.step_count)
