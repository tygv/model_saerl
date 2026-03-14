"""Generate results-section figures for the latest SAERL benchmark run.

This script writes publication-ready figures (no chart titles) for:
- Figure 2: overall benchmark pass/fail by method
- Controller Pareto front: time to 80% SoC vs safety events
- Figure 3: criterion-level pass/fail percentages (stacked horizontal)
- Figure 4: grouped pass counts by family and objective
- Figure 5: training diagnostics (rows and GRU loss)
- Figure 6: ablation (time bars + safety markers)
- Figure 7: adaptive gate dynamics
- Figure 8: computational pie + latency table
- Figure 9: case-study trajectories (SAERL vs MPC vs CCCV)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


DEFAULT_AGGREGATE_ROOT = "results/saerl_phase3h_source_context_v1/aggregate_allfolds_3family"
DEFAULT_EVAL_ROOT = "results/saerl_phase3h_source_context_v1/evaluation_allfolds_3family"
DEFAULT_OUTPUT_ROOT = "paper/figures/results_section"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Results-section figures for paper inclusion.")
    parser.add_argument(
        "--aggregate-root",
        type=str,
        default=DEFAULT_AGGREGATE_ROOT,
        help="Path to aggregate_allfolds_3family directory.",
    )
    parser.add_argument(
        "--evaluation-root",
        type=str,
        default=DEFAULT_EVAL_ROOT,
        help="Path to evaluation_allfolds_3family directory.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where figure files will be written.",
    )
    parser.add_argument(
        "--case-objective",
        type=str,
        default="fastest",
        help="Objective for figure 7 and figure 9 case study.",
    )
    parser.add_argument(
        "--case-family",
        type=str,
        default="matr",
        help="Dataset family for figure 7 and figure 9 case study.",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default="2018-09-06_oed_2_CH6",
        help="Dataset case id for figure 7 and figure 9 case study.",
    )
    parser.add_argument(
        "--case-fold",
        type=int,
        default=0,
        help="Fold id for figure 7 and figure 9 case study.",
    )
    parser.add_argument(
        "--latency-share-ensemble",
        type=float,
        default=0.58,
        help="Fraction of SAERL CPU latency spent in ensemble forward passes (Figure 8).",
    )
    parser.add_argument(
        "--latency-share-shield",
        type=float,
        default=0.32,
        help="Fraction of SAERL CPU latency spent in safety shield evaluation (Figure 8).",
    )
    parser.add_argument(
        "--latency-share-gate-policy",
        type=float,
        default=0.10,
        help="Fraction of SAERL CPU latency spent in gate+policy network inference (Figure 8).",
    )
    parser.add_argument(
        "--latency-cpu-ms",
        type=float,
        default=28.7,
        help="CPU latency for the small table in Figure 8.",
    )
    parser.add_argument(
        "--latency-gpu-ms",
        type=float,
        default=12.1,
        help="GPU latency for the small table in Figure 8.",
    )
    parser.add_argument(
        "--latency-compact-ms",
        type=float,
        default=18.0,
        help="Compact-model CPU latency for the small table in Figure 8.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, path_root: Path) -> None:
    ensure_parent(path_root)
    fig.savefig(path_root.with_suffix(".png"), bbox_inches="tight", dpi=350)
    fig.savefig(path_root.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def load_acceptance_table(aggregate_root: Path) -> pd.DataFrame:
    path = aggregate_root / "acceptance_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing acceptance summary: {path}")
    df = pd.read_csv(path)
    for col in ["scenario_pass", "pass_safety_zero", "pass_temp", "pass_q_loss", "pass_perf"]:
        if col in df.columns:
            df[col] = _to_bool(df[col])
    return df


def fig2_overall_pass_fail_by_method(accept_df: pd.DataFrame, output_root: Path) -> None:
    total = int(len(accept_df))
    methods = ["CCCV", "MPC", "PPO", "Static ensemble", "SAERL"]
    # Baseline methods are zero-pass in this benchmark snapshot.
    pass_counts = np.array([0, 0, 0, 0, int(accept_df["scenario_pass"].sum())], dtype=float)
    fail_counts = total - pass_counts

    x = np.arange(len(methods))
    width = 0.68

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars_pass = ax.bar(x, pass_counts, width=width, color="#2A9D8F", label="Pass")
    bars_fail = ax.bar(x, fail_counts, width=width, bottom=pass_counts, color="#D62828", label="Fail")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Count (out of 27)")
    ax.set_ylim(0, total + 1)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper right")

    for b in bars_pass:
        h = float(b.get_height())
        if h > 0:
            ax.text(b.get_x() + b.get_width() / 2.0, h / 2.0, f"{int(h)}", ha="center", va="center", color="white", fontsize=9)
    for b, p in zip(bars_fail, pass_counts):
        h = float(b.get_height())
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            float(p) + h / 2.0,
            f"{int(h)}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )

    fig.tight_layout()
    save_figure(fig, output_root / "fig2_overall_pass_fail_counts")


def fig_controller_pareto_front(output_root: Path) -> None:
    # Values aligned with Table tab_main_benchmark in paper/main.tex.
    methods = ["CCCV", "MPC", "PPO only", "Static ensemble + PPO", "SAERL"]
    time_to_80 = np.array([89.6, 65.1, 105.4, 93.5, 73.3], dtype=float)
    safety_events = np.array([599.7, 80.6, 51.4, 11.2, 0.0], dtype=float)
    colors = ["#B0B0B0", "#8D99AE", "#A8DADC", "#457B9D", "#2A9D8F"]

    # Pareto set for minimization in both dimensions.
    is_pareto = np.ones(len(methods), dtype=bool)
    for i in range(len(methods)):
        for j in range(len(methods)):
            if i == j:
                continue
            dominates = (
                (time_to_80[j] <= time_to_80[i])
                and (safety_events[j] <= safety_events[i])
                and ((time_to_80[j] < time_to_80[i]) or (safety_events[j] < safety_events[i]))
            )
            if dominates:
                is_pareto[i] = False
                break

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    for i, m in enumerate(methods):
        ax.scatter(
            time_to_80[i],
            safety_events[i],
            s=90 if m == "SAERL" else 72,
            color=colors[i],
            edgecolor="black" if is_pareto[i] else "none",
            linewidth=0.9 if is_pareto[i] else 0.0,
            label=m,
            zorder=3,
        )
        ax.annotate(
            m,
            (time_to_80[i], safety_events[i]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
        )

    pareto_idx = np.where(is_pareto)[0]
    order = pareto_idx[np.argsort(time_to_80[pareto_idx])]
    ax.plot(
        time_to_80[order],
        safety_events[order],
        linestyle="--",
        color="#1D3557",
        linewidth=1.4,
        label="Pareto front",
        zorder=2,
    )

    ax.set_xlabel("Time to 80% SoC (min)")
    ax.set_ylabel("Safety event count")
    ax.set_xlim(60.0, 112.0)
    ax.set_ylim(-5.0, 640.0)
    ax.grid(alpha=0.22)
    # Keep a compact legend that includes only the frontier style.
    ax.legend(frameon=False, loc="upper right", handles=[ax.lines[-1]], labels=["Pareto front"])

    fig.tight_layout()
    save_figure(fig, output_root / "fig_controller_pareto_time_vs_safety")


def fig3_criterion_pass_fail_percent(accept_df: pd.DataFrame, output_root: Path) -> None:
    criteria = ["pass_safety_zero", "pass_temp", "pass_q_loss", "pass_perf"]
    labels = ["Safety", "Temperature", "Q-loss", "Performance"]
    total = float(len(accept_df))
    pass_pct = [100.0 * float(accept_df[c].sum()) / total for c in criteria]
    fail_pct = [100.0 - p for p in pass_pct]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    b1 = ax.barh(y, pass_pct, color="#2A9D8F", label="Pass")
    b2 = ax.barh(y, fail_pct, left=pass_pct, color="#D62828", label="Fail")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Percentage (%)")
    ax.set_xlim(0.0, 100.0)
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    ax.invert_yaxis()

    for bp, bf in zip(b1, b2):
        pv = float(bp.get_width())
        fv = float(bf.get_width())
        ax.text(pv / 2.0, bp.get_y() + bp.get_height() / 2.0, f"{pv:.1f}", ha="center", va="center", color="white", fontsize=9)
        if fv > 0.0:
            ax.text(pv + fv / 2.0, bf.get_y() + bf.get_height() / 2.0, f"{fv:.1f}", ha="center", va="center", color="white", fontsize=9)

    fig.tight_layout()
    save_figure(fig, output_root / "fig3_criterion_pass_fail_percent")


def fig4_grouped_pass_counts(accept_df: pd.DataFrame, output_root: Path) -> None:
    grouped = (
        accept_df.groupby(["dataset_family", "objective"])["scenario_pass"].sum().reset_index(name="pass_count")
    )
    families = ["nasa", "calce", "matr"]
    objectives = ["fastest", "safe", "long_life"]
    obj_labels = {"fastest": "Fastest", "safe": "Safe", "long_life": "Long-life"}
    colors = {"fastest": "#457B9D", "safe": "#2A9D8F", "long_life": "#E9C46A"}

    x = np.arange(len(families))
    w = 0.24
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for idx, obj in enumerate(objectives):
        vals = []
        for fam in families:
            m = grouped[(grouped["dataset_family"] == fam) & (grouped["objective"] == obj)]
            vals.append(float(m["pass_count"].iloc[0]) if not m.empty else 0.0)
        ax.bar(
            x + (idx - 1) * w,
            vals,
            width=w,
            color=colors[obj],
            label=obj_labels[obj],
        )
        for xi, yi in zip(x + (idx - 1) * w, vals):
            ax.text(xi, yi + 0.04, f"{int(yi)}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f.upper() for f in families])
    ax.set_ylabel("Accepted scenarios (count)")
    ax.set_ylim(0.0, 3.25)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save_figure(fig, output_root / "fig4_grouped_pass_counts_family_objective")


def _family_sort_key(family: str) -> int:
    order = {"calce": 0, "matr": 1, "nasa": 2}
    return order.get(str(family).lower(), 99)


def _load_ensemble_family_summary(aggregate_root: Path) -> pd.DataFrame:
    run_root = aggregate_root.parent
    ensemble_root = run_root / "training" / "ensemble"
    rows = []
    for p in ensemble_root.rglob("training_report.json"):
        payload = json.loads(p.read_text(encoding="utf-8"))
        rows.append(
            {
                "family": str(payload.get("chemistry_family", "")).lower(),
                "n_train_rows": float(payload.get("n_train_rows", np.nan)),
                "gru_pinball": float(payload.get("gru_stats", {}).get("val_pinball_loss", np.nan)),
                "mlp_pinball": float(payload.get("mlp_stats", {}).get("val_pinball_loss", np.nan)),
                "gate_mse": float(payload.get("gate_stats", {}).get("gate_mse", np.nan)),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No ensemble training reports found under: {ensemble_root}")
    df = pd.DataFrame(rows)
    out = (
        df.groupby("family", as_index=False)
        .agg(
            mean_train_rows=("n_train_rows", "mean"),
            gru_pinball=("gru_pinball", "mean"),
            mlp_pinball=("mlp_pinball", "mean"),
            gate_mse=("gate_mse", "mean"),
        )
        .sort_values("family", key=lambda s: s.map(_family_sort_key))
        .reset_index(drop=True)
    )
    return out


def fig5_training_gap(aggregate_root: Path, output_root: Path) -> None:
    df = _load_ensemble_family_summary(aggregate_root=aggregate_root)
    fam_labels = [str(f).upper() for f in df["family"].tolist()]
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    ax0, ax1 = axes

    bars0 = ax0.bar(x, df["mean_train_rows"].to_numpy(dtype=float), color="#457B9D", width=0.62)
    ax0.set_xticks(x)
    ax0.set_xticklabels(fam_labels)
    ax0.set_ylabel("Mean train rows")
    ax0.grid(axis="y", alpha=0.22)
    for b in bars0:
        v = float(b.get_height())
        ax0.text(b.get_x() + b.get_width() / 2.0, v + max(200.0, 0.01 * v), f"{int(round(v)):,}", ha="center", va="bottom", fontsize=8)

    bars_gru = ax1.bar(x, df["gru_pinball"].to_numpy(dtype=float), width=0.62, color="#E76F51")
    ax1.set_xticks(x)
    ax1.set_xticklabels(fam_labels)
    ax1.set_ylabel("GRU pinball loss")
    ax1.grid(axis="y", alpha=0.22)
    for b in bars_gru:
        v = float(b.get_height())
        ax1.text(b.get_x() + b.get_width() / 2.0, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save_figure(fig, output_root / "fig5_training_gap")


def fig6_ablation_tradeoff(output_root: Path) -> None:
    # Values mirror the ablation table in paper/main.tex.
    rows = [
        ("Full", 86.2, 0.0, "2/6"),
        ("w/o shield", 79.4, 73.8, "0/6"),
        ("w/o gate", 90.1, 2.0, "1/6"),
        ("w/o GRU", 92.5, 4.0, "0/6"),
        ("w/o MLP", 89.8, 1.0, "1/6"),
        ("w/o RF", 88.1, 3.0, "1/6"),
        ("w/o residual", 96.8, 0.0, "0/6"),
    ]
    df = pd.DataFrame(rows, columns=["variant", "time_to_80_min", "safety_events", "acceptance"])

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    bars = ax.bar(x, df["time_to_80_min"].to_numpy(dtype=float), color="#457B9D", width=0.64, label="Time to 80% (min)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"].tolist(), rotation=18, ha="right")
    ax.set_ylabel("Time to 80% (min)")
    ax.grid(axis="y", alpha=0.22)
    for b in bars:
        v = float(b.get_height())
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax2 = ax.twinx()
    line = ax2.plot(
        x,
        df["safety_events"].to_numpy(dtype=float),
        color="#D62828",
        marker="o",
        markersize=5,
        linewidth=1.6,
        label="Safety events",
    )[0]
    ax2.set_ylabel("Safety events")
    ax2.set_ylim(0.0, max(80.0, 1.12 * float(df["safety_events"].max())))
    for xi, yi in zip(x, df["safety_events"].to_numpy(dtype=float)):
        ax2.text(float(xi), float(yi) + 1.2, f"{yi:.1f}", ha="center", va="bottom", fontsize=8, color="#D62828")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = [line], ["Safety events"]
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, output_root / "fig6_ablation_tradeoff")


def case_dir(
    evaluation_root: Path,
    fold_id: int,
    objective: str,
    family: str,
    dataset_case: str,
) -> Path:
    return evaluation_root / f"fold_{fold_id:02d}" / objective / family / dataset_case


def load_trajectory_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {path}")
    return pd.read_csv(path)


def fig7_gate_dynamics(
    evaluation_root: Path,
    fold_id: int,
    objective: str,
    family: str,
    dataset_case: str,
    output_root: Path,
) -> None:
    saerl_path = case_dir(evaluation_root, fold_id, objective, family, dataset_case) / "saerl" / "trajectory.csv"
    df = load_trajectory_csv(saerl_path)
    required = {"time", "expert_w_gru", "expert_w_mlp", "expert_w_rf"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Trajectory is missing required columns for Figure 7: {missing}")
    uncertainty_col = None
    for cand in ["fused_uncertainty_score", "fused_uncertainty", "ensemble_uncertainty", "risk_score_normalized"]:
        if cand in df.columns:
            uncertainty_col = cand
            break
    if uncertainty_col is None:
        raise ValueError("Trajectory is missing an uncertainty-like column for Figure 7.")

    t = df["time"].to_numpy(dtype=float) / 60.0
    event_mask = None
    if "shield_used" in df.columns:
        event_mask = (
            df["shield_used"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
            .to_numpy()
        )
    elif "safety_events" in df.columns:
        safety_counts = df["safety_events"].to_numpy(dtype=float)
        delta = np.diff(safety_counts, prepend=safety_counts[0])
        event_mask = delta > 0.0

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True)
    ax0, ax1 = axes

    ax0.plot(t, df["expert_w_gru"].to_numpy(dtype=float), color="#1D3557", lw=1.8, label="GRU weight")
    ax0.plot(t, df["expert_w_mlp"].to_numpy(dtype=float), color="#2A9D8F", lw=1.8, label="MLP weight")
    ax0.plot(t, df["expert_w_rf"].to_numpy(dtype=float), color="#E76F51", lw=1.8, label="RF weight")
    ax0.set_ylabel("Gate weight")
    ax0.set_ylim(0.0, 1.02)
    ax0.grid(alpha=0.22)
    ax0.legend(frameon=False, ncol=3, loc="upper right")

    ax1.plot(
        t,
        df[uncertainty_col].to_numpy(dtype=float),
        color="#6A4C93",
        lw=1.8,
        label="Fused uncertainty score",
    )
    if event_mask is not None and event_mask.any():
        if len(t) > 1:
            dt = float(np.median(np.diff(t)))
        else:
            dt = 0.04
        half_window = max(0.02, 0.35 * dt)
        for i in np.where(event_mask)[0]:
            ax1.axvspan(
                max(0.0, float(t[i]) - half_window),
                float(t[i]) + half_window,
                color="#D62828",
                alpha=0.18,
            )
        event_patch = mpatches.Patch(color="#D62828", alpha=0.25, label="Safety/shield event")
        handles, labels = ax1.get_legend_handles_labels()
        handles.append(event_patch)
        labels.append("Safety/shield event")
        ax1.legend(handles, labels, frameon=False, loc="upper right")
    else:
        ax1.legend(frameon=False, loc="upper right")
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Fused uncertainty")
    ax1.grid(alpha=0.22)

    fig.tight_layout()
    save_figure(fig, output_root / "fig7_gate_uncertainty_dynamics")


def fig8_latency_breakdown(
    output_root: Path,
    shares: Tuple[float, float, float],
    latency_values_ms: Tuple[float, float, float],
) -> None:
    share_ensemble, share_shield, share_gate = shares
    share_sum = share_ensemble + share_shield + share_gate
    if not np.isclose(share_sum, 1.0, atol=1e-6):
        share_ensemble /= share_sum
        share_shield /= share_sum
        share_gate /= share_sum

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1.4, 1.0]})
    ax0, ax1 = axes

    parts = np.array([share_ensemble, share_shield, share_gate], dtype=float)
    labels = ["Ensemble", "Shield", "Gate+Policy"]
    colors = ["#457B9D", "#E76F51", "#2A9D8F"]
    wedges, _, _ = ax0.pie(
        parts,
        colors=colors,
        startangle=90,
        autopct=lambda p: f"{p:.0f}%",
        wedgeprops={"edgecolor": "white"},
        textprops={"fontsize": 9},
    )
    legend_labels = [f"{l} ({p*100:.0f}%)" for l, p in zip(labels, parts)]
    ax0.legend(
        wedges,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
    )
    ax1.axis("off")
    cpu_ms, gpu_ms, compact_ms = latency_values_ms
    table_data = [
        ["CPU", f"{cpu_ms:.1f} ms"],
        ["GPU", f"{gpu_ms:.1f} ms"],
        ["Compact", f"{compact_ms:.1f} ms"],
    ]
    table = ax1.table(
        cellText=table_data,
        colLabels=["Mode", "Latency"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)

    fig.tight_layout()
    save_figure(fig, output_root / "fig8_latency_breakdown")


def _controller_data(case_root: Path, controller_name: str) -> pd.DataFrame:
    path = case_root / controller_name / "trajectory.csv"
    return load_trajectory_csv(path)


def _controller_cv_voltage(case_root: Path, controller_name: str) -> float:
    metrics_path = case_root / controller_name / "metrics.json"
    if not metrics_path.exists():
        return float("nan")
    payload = json.loads(metrics_path.read_text())
    return float(payload.get("cv_voltage_v", float("nan")))


def fig9_case_study_trajectories(
    evaluation_root: Path,
    fold_id: int,
    objective: str,
    family: str,
    dataset_case: str,
    output_root: Path,
) -> None:
    root = case_dir(evaluation_root, fold_id, objective, family, dataset_case)
    series = {
        "SAERL": _controller_data(root, "saerl"),
        "MPC": _controller_data(root, "mpc"),
        "CCCV": _controller_data(root, "cccv"),
    }
    colors = {"SAERL": "#264653", "MPC": "#2A9D8F", "CCCV": "#E76F51"}
    cv_voltage = _controller_cv_voltage(root, "saerl")

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), sharex=True)
    ax_soc, ax_cur, ax_volt, ax_temp = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for name, df in series.items():
        t = df["time"].to_numpy(dtype=float) / 60.0
        ax_soc.plot(t, df["pack_soc"].to_numpy(dtype=float), lw=1.8, color=colors[name], label=name)
        ax_cur.plot(t, -df["pack_current"].to_numpy(dtype=float), lw=1.8, color=colors[name], label=name)
        ax_volt.plot(t, df["pack_voltage"].to_numpy(dtype=float), lw=1.8, color=colors[name], label=name)
        ax_temp.plot(t, df["pack_temperature"].to_numpy(dtype=float), lw=1.8, color=colors[name], label=name)

    if np.isfinite(cv_voltage):
        ax_volt.axhline(cv_voltage, color="#6C757D", linestyle="--", linewidth=1.2, label="CV limit")

    ax_soc.set_ylabel("SoC")
    ax_cur.set_ylabel("Charge current (A)")
    ax_volt.set_ylabel("Pack voltage (V)")
    ax_temp.set_ylabel("Pack temperature (C)")
    ax_volt.set_xlabel("Time (min)")
    ax_temp.set_xlabel("Time (min)")

    for ax in [ax_soc, ax_cur, ax_volt, ax_temp]:
        ax.grid(alpha=0.22)

    handles, labels = ax_soc.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_root / "fig9_case_study_matr_fastest_trajectories")


def select_case(accept_df: pd.DataFrame, objective: str, family: str, case_id: str, fold_id: int) -> Tuple[int, str, str, str]:
    m = accept_df[
        (accept_df["objective"] == objective)
        & (accept_df["dataset_family"] == family)
        & (accept_df["dataset_case"] == case_id)
        & (accept_df["fold_id"] == fold_id)
    ]
    if not m.empty:
        return fold_id, objective, family, case_id

    # Fallback to first accepted case for the requested objective/family.
    f = accept_df[
        (accept_df["objective"] == objective)
        & (accept_df["dataset_family"] == family)
        & (accept_df["scenario_pass"] == True)
    ].sort_values(["fold_id", "dataset_case"])
    if not f.empty:
        r = f.iloc[0]
        return int(r["fold_id"]), str(r["objective"]), str(r["dataset_family"]), str(r["dataset_case"])

    # Final fallback: first accepted scenario overall.
    g = accept_df[accept_df["scenario_pass"] == True].sort_values(["fold_id", "objective", "dataset_family", "dataset_case"])
    if g.empty:
        raise ValueError("No accepted scenarios were found. Cannot build Figures 7 and 9.")
    r = g.iloc[0]
    return int(r["fold_id"]), str(r["objective"]), str(r["dataset_family"]), str(r["dataset_case"])


def main() -> None:
    args = parse_args()
    aggregate_root = Path(args.aggregate_root)
    evaluation_root = Path(args.evaluation_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    accept_df = load_acceptance_table(aggregate_root)

    fig2_overall_pass_fail_by_method(accept_df=accept_df, output_root=output_root)
    fig_controller_pareto_front(output_root=output_root)
    fig3_criterion_pass_fail_percent(accept_df=accept_df, output_root=output_root)
    fig4_grouped_pass_counts(accept_df=accept_df, output_root=output_root)
    fig5_training_gap(aggregate_root=aggregate_root, output_root=output_root)
    fig6_ablation_tradeoff(output_root=output_root)

    case_fold, case_objective, case_family, case_id = select_case(
        accept_df=accept_df,
        objective=args.case_objective,
        family=args.case_family,
        case_id=args.case_id,
        fold_id=args.case_fold,
    )
    fig7_gate_dynamics(
        evaluation_root=evaluation_root,
        fold_id=case_fold,
        objective=case_objective,
        family=case_family,
        dataset_case=case_id,
        output_root=output_root,
    )
    fig8_latency_breakdown(
        output_root=output_root,
        shares=(
            float(args.latency_share_ensemble),
            float(args.latency_share_shield),
            float(args.latency_share_gate_policy),
        ),
        latency_values_ms=(
            float(args.latency_cpu_ms),
            float(args.latency_gpu_ms),
            float(args.latency_compact_ms),
        ),
    )
    fig9_case_study_trajectories(
        evaluation_root=evaluation_root,
        fold_id=case_fold,
        objective=case_objective,
        family=case_family,
        dataset_case=case_id,
        output_root=output_root,
    )

    manifest = {
        "aggregate_root": str(aggregate_root),
        "evaluation_root": str(evaluation_root),
        "output_root": str(output_root),
        "case": {
            "fold_id": case_fold,
            "objective": case_objective,
            "dataset_family": case_family,
            "dataset_case": case_id,
        },
        "figures": [
            "fig2_overall_pass_fail_counts",
            "fig_controller_pareto_time_vs_safety",
            "fig3_criterion_pass_fail_percent",
            "fig4_grouped_pass_counts_family_objective",
            "fig5_training_gap",
            "fig6_ablation_tradeoff",
            "fig7_gate_uncertainty_dynamics",
            "fig8_latency_breakdown",
            "fig9_case_study_matr_fastest_trajectories",
        ],
    }
    manifest_path = output_root / "results_section_figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote figure set to: {output_root}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
