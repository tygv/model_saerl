"""Unit/integration/runtime checks for SAERL components."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from battery_pack_model import PackConfiguration
from controllers.adaptive_ensemble_rl import (
    AdaptiveEnsemblePredictor,
    SAERLConfig,
    SafeAdaptiveEnsembleController,
    state_to_feature_vector,
    window_to_sequence,
)
from hambrl_pack_env import HAMBRLPackEnvironment
from scripts.run_baseline_benchmarks import CCCVConfig, CCCVController, MPCConfig, RolloutMPCController, count_safety_events, trim_pack_histories


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str


def initial_state(env: HAMBRLPackEnvironment) -> Dict:
    return {
        "pack_soc": env.pack.pack_soc,
        "pack_voltage": env.pack.pack_voltage,
        "pack_temperature": env.pack.pack_temperature,
        "voltage_imbalance": env.pack.voltage_imbalance,
        "pack_current": env.pack.pack_current,
        "safety_events": env.pack.safety_events,
    }


def unit_feature_shape() -> TestResult:
    s = {
        "pack_soc": 0.4,
        "pack_voltage": 72.0,
        "pack_temperature": 30.0,
        "voltage_imbalance": 0.02,
        "pack_current": -3.0,
    }
    feat = state_to_feature_vector(s, max_charge_current_a=10.0)
    ok = feat.shape == (6,)
    return TestResult("unit_feature_shape", ok, f"shape={feat.shape}")


def unit_window_order() -> TestResult:
    states = []
    for i in range(3):
        states.append(
            {
                "pack_soc": 0.2 + 0.1 * i,
                "pack_voltage": 70.0 + i,
                "pack_temperature": 25.0 + i,
                "voltage_imbalance": 0.01 * i,
                "pack_current": -1.0 * i,
            }
        )
    seq = window_to_sequence(states, window_len=5, max_charge_current_a=10.0)
    ok = bool(np.isclose(seq[-1, 0], 0.4))
    return TestResult("unit_window_order", ok, f"last_soc={seq[-1,0]:.4f}")


def unit_gate_simplex_and_uncertainty() -> TestResult:
    cfg = SAERLConfig(window_len=6)
    predictor = AdaptiveEnsemblePredictor(config=cfg, device="cpu")
    states = []
    for _ in range(6):
        states.append(
            {
                "pack_soc": 0.35,
                "pack_voltage": 71.0,
                "pack_temperature": 26.0,
                "voltage_imbalance": 0.015,
                "pack_current": -2.0,
            }
        )
    fused = predictor.predict_fused(states, action=0.2, max_charge_current_a=10.0, cv_voltage_v=83.0)
    w = fused["expert_weights"]
    wsum = float(w["gru"] + w["mlp"] + w["rf"])
    unc = fused["fused_uncertainty"]
    unc_vals = np.array(list(unc.values()), dtype=float)
    ok = bool(abs(wsum - 1.0) < 1e-5 and np.all(np.isfinite(unc_vals)) and np.all(unc_vals >= 0.0))
    return TestResult("unit_gate_simplex_and_uncertainty", ok, f"weight_sum={wsum:.6f}, unc_min={float(np.min(unc_vals)):.6g}")


def unit_shield_blocks_unsafe() -> TestResult:
    cfg = SAERLConfig(window_len=6, enable_shield=True)
    predictor = AdaptiveEnsemblePredictor(config=cfg, device="cpu")
    controller = SafeAdaptiveEnsembleController(predictor=predictor, actor=None, config=cfg)

    pack_cfg = PackConfiguration(n_series=20, n_parallel=1, V_pack_max=75.0, balancing_type="passive")
    env = HAMBRLPackEnvironment(
        pack_config=pack_cfg,
        max_steps=20,
        target_soc=0.8,
        ambient_temp=25.0,
        max_charge_current_a=10.0,
    )
    env.reset(initial_soc=0.7, temperature=25.0)
    st = initial_state(env)
    unsafe_action = 1.0
    is_safe = controller._is_safe_action(env=env, action=unsafe_action, cv_voltage_v=75.0)
    ok = not bool(is_safe)
    return TestResult("unit_shield_blocks_unsafe", ok, f"is_safe_forced_action={is_safe}")


def integration_short_rollout() -> TestResult:
    cfg = SAERLConfig(window_len=10, enable_shield=True)
    predictor = AdaptiveEnsemblePredictor(config=cfg, device="cpu")
    saerl = SafeAdaptiveEnsembleController(predictor=predictor, actor=None, config=cfg)

    env = HAMBRLPackEnvironment(
        pack_config=PackConfiguration(n_series=20, n_parallel=1, balancing_type="passive"),
        max_steps=120,
        target_soc=0.8,
        ambient_temp=25.0,
        max_charge_current_a=7.5,
    )

    cccv = CCCVController(CCCVConfig(), cv_voltage_v=83.0, max_charge_current_a=7.5, target_soc=0.8)
    mpc = RolloutMPCController(MPCConfig(), cv_voltage_v=83.0, max_charge_current_a=7.5, target_soc=0.8)
    controllers = {"cccv": cccv, "mpc": mpc, "saerl": saerl}

    summary = {}
    for name, ctrl in controllers.items():
        env.reset(initial_soc=0.2, temperature=25.0)
        trim_pack_histories(env.pack)
        if hasattr(ctrl, "reset"):
            ctrl.reset()
        state = initial_state(env)
        done = False
        steps = 0
        safety_total = 0
        mpc_for_saerl = RolloutMPCController(MPCConfig(), cv_voltage_v=83.0, max_charge_current_a=7.5, target_soc=0.8)
        mpc_for_saerl.reset()
        while not done and steps < 120:
            if name == "saerl":
                mpc_action, _ = mpc_for_saerl.act(state, env)
                action, _ = ctrl.act(state=state, env=env, mpc_action=float(mpc_action), cv_voltage_v=83.0)
            else:
                action, _ = ctrl.act(state, env)
            _, _, done, nxt = env.step(action)
            trim_pack_histories(env.pack)
            safety_total += int(count_safety_events(nxt.get("safety_events", {})))
            if hasattr(ctrl, "observe_transition"):
                ctrl.observe_transition(nxt)
            state = {
                "pack_soc": float(nxt["pack_soc"]),
                "pack_voltage": float(nxt["pack_voltage"]),
                "pack_temperature": float(nxt["pack_temperature"]),
                "voltage_imbalance": float(nxt["voltage_imbalance"]),
                "pack_current": float(nxt["pack_current"]),
                "safety_events": nxt.get("safety_events", {}),
            }
            steps += 1
        summary[name] = {"steps": steps, "final_soc": state["pack_soc"], "safety": safety_total}
    ok = bool(all(v["steps"] > 0 for v in summary.values()))
    return TestResult("integration_short_rollout", ok, str(summary))


def runtime_latency_check() -> TestResult:
    cfg = SAERLConfig(window_len=10)
    predictor = AdaptiveEnsemblePredictor(config=cfg, device="cpu")
    controller = SafeAdaptiveEnsembleController(predictor=predictor, actor=None, config=cfg)
    env = HAMBRLPackEnvironment(
        pack_config=PackConfiguration(n_series=20, n_parallel=1, balancing_type="passive"),
        max_steps=200,
        target_soc=0.8,
        ambient_temp=25.0,
        max_charge_current_a=7.5,
    )
    env.reset(initial_soc=0.2, temperature=25.0)
    state = initial_state(env)
    mpc = RolloutMPCController(MPCConfig(), cv_voltage_v=83.0, max_charge_current_a=7.5, target_soc=0.8)
    mpc.reset()

    latencies = []
    for _ in range(200):
        mpc_action, _ = mpc.act(state, env)
        t0 = time.perf_counter()
        action, _ = controller.act(state=state, env=env, mpc_action=float(mpc_action), cv_voltage_v=83.0)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        _, _, done, nxt = env.step(action)
        controller.observe_transition(nxt)
        state = {
            "pack_soc": float(nxt["pack_soc"]),
            "pack_voltage": float(nxt["pack_voltage"]),
            "pack_temperature": float(nxt["pack_temperature"]),
            "voltage_imbalance": float(nxt["voltage_imbalance"]),
            "pack_current": float(nxt["pack_current"]),
            "safety_events": nxt.get("safety_events", {}),
        }
        if done:
            break
    lat = np.asarray(latencies, dtype=float)
    mean_ms = float(np.mean(lat)) if len(lat) else float("inf")
    p95_ms = float(np.quantile(lat, 0.95)) if len(lat) else float("inf")
    ok = bool(mean_ms < 50.0)
    return TestResult("runtime_latency_check", ok, f"mean_ms={mean_ms:.3f}, p95_ms={p95_ms:.3f}")


def main() -> None:
    tests = [
        unit_feature_shape,
        unit_window_order,
        unit_gate_simplex_and_uncertainty,
        unit_shield_blocks_unsafe,
        integration_short_rollout,
        runtime_latency_check,
    ]
    results: List[TestResult] = []
    for fn in tests:
        try:
            results.append(fn())
        except Exception as exc:
            results.append(TestResult(fn.__name__, False, f"EXCEPTION: {exc}"))

    n_pass = sum(1 for r in results if r.passed)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}: {r.detail}")
    print(f"Passed {n_pass}/{len(results)} checks.")

    if n_pass != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

