"""Aggregate SAERL evaluation outputs into tables and publication figures."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@dataclass
class AggregateConfig:
    input_root: str = "results/saerl_phase2/evaluation"
    output_root: str = "results/saerl_phase2/aggregate"


def parse_args() -> AggregateConfig:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate SAERL evaluation results.")
    parser.add_argument("--input-root", type=str, default="results/saerl_phase2/evaluation")
    parser.add_argument("--output-root", type=str, default="results/saerl_phase2/aggregate")
    args = parser.parse_args()
    return AggregateConfig(input_root=args.input_root, output_root=args.output_root)


def save_figure(path_root: Path) -> None:
    path_root.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_root.with_suffix(".png"), bbox_inches="tight")
    plt.savefig(path_root.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def collect_metrics(input_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in sorted(input_root.glob("fold_*/*/*/*/*/metrics.json")):
        with p.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics = payload.get("metrics", {})
        horizon_info = payload.get("horizon_info", {}) if isinstance(payload.get("horizon_info", {}), dict) else {}
        row = {
            "fold_id": int(payload.get("fold_id", -1)),
            "objective": str(payload.get("objective", "")),
            "dataset_family": str(payload.get("dataset_family", "")),
            "dataset_case": str(payload.get("dataset_case", "")),
            "controller": str(payload.get("algorithm", "")),
            "controller_chemistry_mode": str(payload.get("controller_chemistry_mode", "")),
            "controller_chemistry_family": str(payload.get("controller_chemistry_family", "")),
            "metrics_path": str(p),
            "effective_max_steps": payload.get("effective_max_steps", np.nan),
            "horizon_dt_s": horizon_info.get("dt_s", np.nan),
            "horizon_base_horizon_s": horizon_info.get("base_horizon_s", np.nan),
            "horizon_effective_horizon_s": horizon_info.get("effective_horizon_s", np.nan),
            "horizon_ideal_cc_time_to_target_s": horizon_info.get("ideal_cc_time_to_target_s", np.nan),
            "horizon_feasibility_ratio_vs_base": horizon_info.get("current_feasibility_ratio_vs_base_horizon", np.nan),
        }
        for key, value in payload.items():
            if str(key).startswith("ctx_"):
                row[str(key)] = value
        for k, v in metrics.items():
            row[k] = v
        if not row["controller_chemistry_mode"]:
            c_name = str(row["controller"])
            if c_name.endswith("_family_specific"):
                row["controller_chemistry_mode"] = "family_specific"
            elif c_name.endswith("_shared_plus_heads"):
                row["controller_chemistry_mode"] = "shared_plus_heads"
            elif c_name.startswith("saerl_"):
                row["controller_chemistry_mode"] = c_name.replace("saerl_", "", 1)
            else:
                row["controller_chemistry_mode"] = "global"
        if not row["controller_chemistry_family"]:
            row["controller_chemistry_family"] = str(row["dataset_family"])
        rows.append(row)
    return pd.DataFrame(rows)


def pivot_primary(long_df: pd.DataFrame) -> pd.DataFrame:
    key = ["fold_id", "objective", "dataset_family", "dataset_case"]
    primary = long_df[long_df["controller"].isin(["cccv", "mpc", "saerl"])].copy()
    piv = primary.pivot_table(
        index=key,
        columns="controller",
        values=["final_soc", "safety_event_count", "charge_time_min", "time_to_80_soc_min", "q_loss_total", "peak_pack_temperature_c"],
        aggfunc="first",
    )
    out = piv.index.to_frame(index=False)
    for metric in ["final_soc", "safety_event_count", "charge_time_min", "time_to_80_soc_min", "q_loss_total", "peak_pack_temperature_c"]:
        for c in ["cccv", "mpc", "saerl"]:
            col = (metric, c)
            out[f"{metric}_{c}"] = piv[col].to_numpy(dtype=float) if col in piv.columns else np.nan
    out["saerl_vs_best_baseline_final_soc_margin"] = out["final_soc_saerl"] - np.maximum(
        out["final_soc_cccv"], out["final_soc_mpc"]
    )
    out["saerl_vs_best_baseline_safety_margin"] = np.minimum(
        out["safety_event_count_cccv"], out["safety_event_count_mpc"]
    ) - out["safety_event_count_saerl"]
    return out


def plot_pareto(primary_df: pd.DataFrame, output_path: Path) -> None:
    if primary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for controller, color, marker in [("cccv", "#E76F51", "o"), ("mpc", "#2A9D8F", "s"), ("saerl", "#264653", "^")]:
        x = primary_df[f"time_to_80_soc_min_{controller}"].to_numpy(dtype=float)
        x = np.where(np.isfinite(x), x, primary_df[f"charge_time_min_{controller}"].to_numpy(dtype=float))
        y = primary_df[f"safety_event_count_{controller}"].to_numpy(dtype=float)
        ax.scatter(x, y, alpha=0.75, c=color, label=controller.upper(), marker=marker, s=52)
    ax.set_xlabel("Time to 80% SoC (min; fallback charge time if NaN)")
    ax.set_ylabel("Safety Event Count")
    ax.set_title("Controller Pareto Front")
    ax.legend()
    ax.grid(alpha=0.25)
    save_figure(output_path)


def plot_pass_heatmap(accept_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if accept_df.empty:
        return pd.DataFrame()
    heat = (
        accept_df.groupby(["objective", "dataset_family"])["scenario_pass"]
        .mean()
        .unstack("dataset_family")
        .sort_index()
    )
    if heat.empty:
        return heat
    mat = heat.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    im = ax.imshow(mat, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_xticklabels(heat.columns.tolist())
    ax.set_yticklabels(heat.index.tolist())
    ax.set_xlabel("Dataset family")
    ax.set_ylabel("Objective")
    ax.set_title("SAERL Scenario Pass Rate")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pass rate")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black")
    save_figure(output_path)
    return heat


def plot_margin_heatmap(primary_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if primary_df.empty:
        return pd.DataFrame()
    heat = (
        primary_df.groupby(["objective", "dataset_family"])["saerl_vs_best_baseline_final_soc_margin"]
        .mean()
        .unstack("dataset_family")
        .sort_index()
    )
    if heat.empty:
        return heat
    mat = heat.to_numpy(dtype=float)
    vmax = float(np.max(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_xticklabels(heat.columns.tolist())
    ax.set_yticklabels(heat.index.tolist())
    ax.set_xlabel("Dataset family")
    ax.set_ylabel("Objective")
    ax.set_title("SAERL Final SoC Margin vs Best Baseline")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Margin (SoC)")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="black")
    save_figure(output_path)
    return heat


def plot_ablation(ablation_df: pd.DataFrame, output_path: Path) -> None:
    if ablation_df.empty:
        return
    keep = ["controller", "final_soc", "safety_event_count", "q_loss_total"]
    g = ablation_df[keep].groupby("controller").mean().reset_index()
    if g.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    metrics = ["final_soc", "safety_event_count", "q_loss_total"]
    titles = ["Final SoC", "Safety Event Count", "Q-Loss Total"]
    for i, (m, t) in enumerate(zip(metrics, titles)):
        axes[i].bar(g["controller"], g[m], color="#457B9D", alpha=0.9)
        axes[i].set_title(t)
        axes[i].tick_params(axis="x", rotation=30)
    fig.suptitle("SAERL Ablation Summary")
    fig.tight_layout()
    save_figure(output_path)


def build_mode_comparison_table(
    df: pd.DataFrame,
    mode_column: str,
    metrics: List[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    key = ["fold_id", "objective", "dataset_family", "dataset_case"]
    piv = df.pivot_table(
        index=key,
        columns=mode_column,
        values=metrics,
        aggfunc="first",
    )
    out = piv.index.to_frame(index=False)
    modes = sorted({str(x) for x in df[mode_column].astype(str).unique().tolist()})
    for metric in metrics:
        for mode in modes:
            col = (metric, mode)
            out[f"{metric}_{mode}"] = piv[col].to_numpy(dtype=float) if col in piv.columns else np.nan
    return out


def add_mode_deltas(table: pd.DataFrame, metrics: List[str], modes: List[str], baseline_mode: str = "global") -> pd.DataFrame:
    if table.empty:
        return table
    for mode in modes:
        if mode == baseline_mode:
            continue
        for metric in metrics:
            base_col = f"{metric}_{baseline_mode}"
            mode_col = f"{metric}_{mode}"
            if base_col in table.columns and mode_col in table.columns:
                table[f"{metric}_{mode}_minus_{baseline_mode}"] = table[mode_col] - table[base_col]
    return table


def main() -> None:
    config = parse_args()
    input_root = Path(config.input_root)
    output_root = Path(config.output_root)
    figures_root = output_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    long_df = collect_metrics(input_root=input_root)
    if long_df.empty:
        raise SystemExit(f"No metrics found under {input_root}")
    long_df.to_csv(output_root / "all_runs_long_metrics.csv", index=False)

    primary_df = pivot_primary(long_df=long_df)
    primary_df.to_csv(output_root / "scenario_primary_comparison.csv", index=False)

    group_cols = ["objective", "dataset_family", "controller"]
    metrics_cols = [
        "final_soc",
        "charge_time_min",
        "time_to_80_soc_min",
        "safety_event_count",
        "peak_pack_temperature_c",
        "q_loss_total",
        "q_loss_rate",
        "inference_latency_mean_ms",
        "shield_intervention_rate",
        "antistall_intervention_rate",
    ]
    metrics_cols = [c for c in metrics_cols if c in long_df.columns]
    family_obj_summary = long_df.groupby(group_cols)[metrics_cols].mean(numeric_only=True).reset_index()
    family_obj_summary.to_csv(output_root / "family_objective_summary.csv", index=False)
    overall_summary = long_df.groupby(["controller"])[metrics_cols].mean(numeric_only=True).reset_index()
    overall_summary.to_csv(output_root / "overall_summary.csv", index=False)

    mode_metrics = [
        "final_soc",
        "charge_time_min",
        "time_to_80_soc_min",
        "safety_event_count",
        "q_loss_total",
        "peak_pack_temperature_c",
    ]
    canonical_modes = ["global", "family_specific", "shared_plus_heads"]

    # SAERL global vs chemistry-aware comparisons.
    saerl_variants = long_df[
        long_df["controller"].astype(str).str.startswith("saerl_")
        & ~long_df["controller"].astype(str).str.startswith("saerl_no_")
    ].copy()
    if saerl_variants.empty:
        fallback = long_df[long_df["controller"] == "saerl"].copy()
        if not fallback.empty:
            fallback["saerl_mode"] = fallback["controller_chemistry_mode"].astype(str)
            saerl_variants = fallback
    else:
        saerl_variants["saerl_mode"] = saerl_variants["controller"].astype(str).str.replace(
            "^saerl_",
            "",
            regex=True,
        )
    if not saerl_variants.empty:
        saerl_variants = saerl_variants[saerl_variants["saerl_mode"].isin(canonical_modes)].copy()
        if not saerl_variants.empty:
            saerl_mode_summary = (
                saerl_variants.groupby(["objective", "dataset_family", "saerl_mode"])[mode_metrics]
                .mean(numeric_only=True)
                .reset_index()
            )
            saerl_mode_summary.to_csv(output_root / "saerl_mode_summary.csv", index=False)

            saerl_mode_scenario = build_mode_comparison_table(
                df=saerl_variants,
                mode_column="saerl_mode",
                metrics=mode_metrics,
            )
            saerl_modes_present = sorted(saerl_variants["saerl_mode"].astype(str).unique().tolist())
            saerl_mode_scenario = add_mode_deltas(
                table=saerl_mode_scenario,
                metrics=mode_metrics,
                modes=saerl_modes_present,
                baseline_mode="global",
            )
            saerl_mode_scenario.to_csv(output_root / "saerl_mode_scenario_comparison.csv", index=False)

            saerl_delta_cols = [c for c in saerl_mode_scenario.columns if c.endswith("_minus_global")]
            if saerl_delta_cols:
                saerl_mode_delta_summary = (
                    saerl_mode_scenario.groupby(["objective", "dataset_family"])[saerl_delta_cols]
                    .mean(numeric_only=True)
                    .reset_index()
                )
                saerl_mode_delta_summary.to_csv(output_root / "saerl_mode_delta_summary.csv", index=False)

    # Baseline global vs chemistry-aware comparisons.
    baseline_tables: List[pd.DataFrame] = []
    baseline_summaries: List[pd.DataFrame] = []
    for base in ["cccv", "mpc"]:
        subset = long_df[
            long_df["controller"].isin(
                [
                    base,
                    f"{base}_family_specific",
                    f"{base}_shared_plus_heads",
                ]
            )
        ].copy()
        if subset.empty:
            continue
        subset["baseline_controller"] = base
        subset["baseline_mode"] = subset["controller"].map(
            {
                base: "global",
                f"{base}_family_specific": "family_specific",
                f"{base}_shared_plus_heads": "shared_plus_heads",
            }
        )
        subset = subset[subset["baseline_mode"].isin(canonical_modes)].copy()
        if subset.empty:
            continue

        s_table = build_mode_comparison_table(
            df=subset,
            mode_column="baseline_mode",
            metrics=mode_metrics,
        )
        modes_present = sorted(subset["baseline_mode"].astype(str).unique().tolist())
        s_table = add_mode_deltas(
            table=s_table,
            metrics=mode_metrics,
            modes=modes_present,
            baseline_mode="global",
        )
        s_table["baseline_controller"] = base
        baseline_tables.append(s_table)

        b_summary = (
            subset.groupby(["baseline_controller", "objective", "dataset_family", "baseline_mode"])[mode_metrics]
            .mean(numeric_only=True)
            .reset_index()
        )
        baseline_summaries.append(b_summary)

    if baseline_tables:
        baseline_mode_scenario = pd.concat(baseline_tables, ignore_index=True)
        baseline_mode_scenario.to_csv(output_root / "baseline_mode_scenario_comparison.csv", index=False)
        baseline_delta_cols = [c for c in baseline_mode_scenario.columns if c.endswith("_minus_global")]
        if baseline_delta_cols:
            baseline_mode_delta_summary = (
                baseline_mode_scenario.groupby(["baseline_controller", "objective", "dataset_family"])[baseline_delta_cols]
                .mean(numeric_only=True)
                .reset_index()
            )
            baseline_mode_delta_summary.to_csv(output_root / "baseline_mode_delta_summary.csv", index=False)
    if baseline_summaries:
        pd.concat(baseline_summaries, ignore_index=True).to_csv(
            output_root / "baseline_mode_summary.csv",
            index=False,
        )

    accept_path = input_root / "acceptance_summary.csv"
    if accept_path.exists():
        accept_df = pd.read_csv(accept_path)
    else:
        accept_df = pd.DataFrame()
    if not accept_df.empty:
        for bcol in [
            "scenario_pass",
            "feasible_to_target_base_horizon",
            "feasible_to_target_effective_horizon",
        ]:
            if bcol in accept_df.columns:
                accept_df[bcol] = (
                    accept_df[bcol]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["1", "true", "yes", "y"])
                )
        accept_df.to_csv(output_root / "acceptance_summary.csv", index=False)
        if "feasible_to_target_effective_horizon" in accept_df.columns:
            feasible_eff_df = accept_df[accept_df["feasible_to_target_effective_horizon"]].copy()
            infeasible_eff_df = accept_df[~accept_df["feasible_to_target_effective_horizon"]].copy()
            feasible_summary = pd.DataFrame(
                [
                    {
                        "slice": "all",
                        "n_scenarios": int(len(accept_df)),
                        "pass_count": int(np.sum(accept_df["scenario_pass"].astype(bool))),
                        "pass_rate": float(np.mean(accept_df["scenario_pass"].astype(float))),
                    },
                    {
                        "slice": "feasible_effective_horizon_only",
                        "n_scenarios": int(len(feasible_eff_df)),
                        "pass_count": int(np.sum(feasible_eff_df["scenario_pass"].astype(bool))),
                        "pass_rate": (
                            float(np.mean(feasible_eff_df["scenario_pass"].astype(float)))
                            if len(feasible_eff_df)
                            else float("nan")
                        ),
                    },
                    {
                        "slice": "infeasible_effective_horizon_only",
                        "n_scenarios": int(len(infeasible_eff_df)),
                        "pass_count": int(np.sum(infeasible_eff_df["scenario_pass"].astype(bool))),
                        "pass_rate": (
                            float(np.mean(infeasible_eff_df["scenario_pass"].astype(float)))
                            if len(infeasible_eff_df)
                            else float("nan")
                        ),
                    },
                ]
            )
            feasible_summary.to_csv(output_root / "acceptance_feasibility_summary.csv", index=False)
            if len(feasible_eff_df):
                feasible_by_family = (
                    feasible_eff_df.groupby("dataset_family")["scenario_pass"]
                    .agg(["count", "sum", "mean"])
                    .rename(columns={"count": "n_scenarios", "sum": "pass_count", "mean": "pass_rate"})
                    .reset_index()
                )
                feasible_by_family.to_csv(
                    output_root / "acceptance_feasible_effective_by_family.csv",
                    index=False,
                )
                feasible_by_objective = (
                    feasible_eff_df.groupby("objective")["scenario_pass"]
                    .agg(["count", "sum", "mean"])
                    .rename(columns={"count": "n_scenarios", "sum": "pass_count", "mean": "pass_rate"})
                    .reset_index()
                )
                feasible_by_objective.to_csv(
                    output_root / "acceptance_feasible_effective_by_objective.csv",
                    index=False,
                )
        if "ctx_regime_aging_eis" in accept_df.columns:
            def _regime_label(row: pd.Series) -> str:
                if bool(row.get("ctx_regime_aging_eis", 0)):
                    return "aging_eis"
                if bool(row.get("ctx_regime_cycle_cccv", 0)):
                    return "cycle_cccv"
                if bool(row.get("ctx_regime_fast_charge", 0)):
                    return "fast_charge"
                return "unknown"

            accept_df["test_regime"] = accept_df.apply(_regime_label, axis=1)
            aggregate_by_regime = (
                accept_df.groupby(["dataset_family", "test_regime"])["scenario_pass"]
                .agg(["count", "sum", "mean"])
                .rename(columns={"count": "n_scenarios", "sum": "pass_count", "mean": "pass_rate"})
                .reset_index()
            )
            aggregate_by_regime.to_csv(output_root / "aggregate_by_regime.csv", index=False)
        if "ctx_source_temp_present" in accept_df.columns:
            temp_series = accept_df["ctx_source_temp_present"].astype(float)
            if "ctx_source_internal_resistance_present" in accept_df.columns:
                ir_series = accept_df["ctx_source_internal_resistance_present"].astype(float)
            else:
                ir_series = pd.Series(np.zeros(len(accept_df), dtype=float), index=accept_df.index)
            aggregate_by_signal = (
                accept_df.assign(
                    source_temp_present=temp_series > 0.5,
                    source_internal_resistance_present=ir_series > 0.5,
                )
                .groupby(["dataset_family", "source_temp_present", "source_internal_resistance_present"])["scenario_pass"]
                .agg(["count", "sum", "mean"])
                .rename(columns={"count": "n_scenarios", "sum": "pass_count", "mean": "pass_rate"})
                .reset_index()
            )
            aggregate_by_signal.to_csv(
                output_root / "aggregate_by_source_signal_availability.csv",
                index=False,
            )

    plot_pareto(primary_df=primary_df, output_path=figures_root / "01_controller_pareto")
    pass_heat = plot_pass_heatmap(accept_df=accept_df, output_path=figures_root / "02_pass_rate_heatmap")
    if not pass_heat.empty:
        pass_heat.to_csv(output_root / "pass_rate_heatmap_table.csv")
    margin_heat = plot_margin_heatmap(primary_df=primary_df, output_path=figures_root / "03_final_soc_margin_heatmap")
    if not margin_heat.empty:
        margin_heat.to_csv(output_root / "final_soc_margin_heatmap_table.csv")

    ablation_df = long_df[long_df["controller"].str.startswith("saerl_no_")].copy()
    if not ablation_df.empty:
        ablation_df.to_csv(output_root / "ablation_runs.csv", index=False)
        plot_ablation(ablation_df=ablation_df, output_path=figures_root / "04_ablation_summary")

    global_summary = {
        "n_runs": int(len(long_df)),
        "n_primary_scenarios": int(len(primary_df)),
        "controllers": sorted(long_df["controller"].astype(str).unique().tolist()),
        "chemistry_modes": sorted(long_df["controller_chemistry_mode"].astype(str).unique().tolist()),
    }
    if not accept_df.empty:
        global_summary["acceptance_pass_rate"] = float(np.mean(accept_df["scenario_pass"].astype(float)))
        global_summary["acceptance_pass_count"] = int(np.sum(accept_df["scenario_pass"].astype(bool)))
        global_summary["acceptance_total"] = int(len(accept_df))
        if "feasible_to_target_effective_horizon" in accept_df.columns:
            feasible_eff_df = accept_df[accept_df["feasible_to_target_effective_horizon"]].copy()
            global_summary["acceptance_feasible_effective_total"] = int(len(feasible_eff_df))
            global_summary["acceptance_feasible_effective_pass_count"] = int(
                np.sum(feasible_eff_df["scenario_pass"].astype(bool))
            )
            global_summary["acceptance_feasible_effective_pass_rate"] = (
                float(np.mean(feasible_eff_df["scenario_pass"].astype(float)))
                if len(feasible_eff_df)
                else float("nan")
            )
    with (output_root / "global_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(global_summary, handle, indent=2, ensure_ascii=True)

    print("Aggregated SAERL evaluation outputs.")
    print(f"Output root: {output_root.resolve()}")


if __name__ == "__main__":
    main()
