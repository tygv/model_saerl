"""Aggregate data-calibrated baseline results across objectives and datasets.

This script scans:
  results/baselines/data_calibrated/<objective>/<dataset_family>/<case>/<controller>/metrics.json

It builds:
  - Long-form metrics table
  - Pairwise scenario table (MPC vs CCCV deltas)
  - Grouped summary tables
  - Publication-style aggregated figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS: Tuple[str, ...] = (
    "final_soc",
    "charge_time_min",
    "peak_pack_temperature_c",
    "safety_event_count",
    "peak_voltage_imbalance_mv",
    "energy_in_kwh",
)

HIGHER_IS_BETTER: Set[str] = {
    "final_soc",
}

METRIC_LABELS: Dict[str, str] = {
    "final_soc": "Final SoC",
    "charge_time_min": "Charge Time (min)",
    "peak_pack_temperature_c": "Peak Temp (C)",
    "safety_event_count": "Safety Events",
    "peak_voltage_imbalance_mv": "Peak Imbalance (mV)",
    "energy_in_kwh": "Energy In (kWh)",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "savefig.dpi": 350,
            "figure.dpi": 140,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _split_csv_arg(csv_arg: str) -> List[str]:
    return [token.strip() for token in csv_arg.split(",") if token.strip()]


def _ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _coerce_float(value) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def collect_metrics_rows(input_root: Path, controllers: Set[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    pattern = "*/*/*/*/metrics.json"
    for metrics_path in sorted(input_root.glob(pattern)):
        controller = metrics_path.parent.name.lower()
        if controller not in controllers:
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        metrics = payload.get("metrics", {})
        scenario = payload.get("scenario", {})
        objective = str(payload.get("objective", metrics_path.parents[3].name))
        dataset_family = str(scenario.get("dataset_family", metrics_path.parents[2].name))
        dataset_case = str(scenario.get("dataset_case", metrics_path.parents[1].name))

        row: Dict[str, object] = {
            "objective": objective,
            "dataset_family": dataset_family,
            "dataset_case": dataset_case,
            "algorithm": str(payload.get("algorithm", controller)).lower(),
            "cv_voltage_v": _coerce_float(payload.get("cv_voltage_v")),
            "max_charge_current_a": _coerce_float(payload.get("max_charge_current_a")),
            "source_csv": scenario.get("source_csv"),
            "metrics_path": str(metrics_path),
        }

        for metric_name, metric_value in metrics.items():
            row[str(metric_name)] = _coerce_float(metric_value)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No metrics.json files found under: {input_root}")

    df = pd.DataFrame(rows)
    df["scenario_id"] = (
        df["objective"].astype(str)
        + "|"
        + df["dataset_family"].astype(str)
        + "|"
        + df["dataset_case"].astype(str)
    )
    return df


def compute_pairwise_table(
    long_df: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    key_cols = ["objective", "dataset_family", "dataset_case"]
    pivot = long_df.pivot_table(
        index=key_cols,
        columns="algorithm",
        values=list(metrics),
        aggfunc="first",
    )

    if ("final_soc", "cccv") not in pivot.columns or ("final_soc", "mpc") not in pivot.columns:
        raise RuntimeError("Pairwise table requires both CCCV and MPC rows for each scenario.")

    pair_df = pivot.index.to_frame(index=False)
    eps = 1e-9

    for metric in metrics:
        cccv_col = (metric, "cccv")
        mpc_col = (metric, "mpc")
        if cccv_col not in pivot.columns or mpc_col not in pivot.columns:
            continue

        cccv_vals = pivot[cccv_col].astype(float)
        mpc_vals = pivot[mpc_col].astype(float)
        pair_df[f"{metric}_cccv"] = cccv_vals.values
        pair_df[f"{metric}_mpc"] = mpc_vals.values

        # Positive delta always means MPC advantage.
        if metric in HIGHER_IS_BETTER:
            delta = mpc_vals - cccv_vals
        else:
            delta = cccv_vals - mpc_vals
        pair_df[f"{metric}_delta_mpc_adv"] = delta.values

        denom = cccv_vals.abs().clip(lower=1.0)
        pair_df[f"{metric}_delta_pct_mpc_adv"] = (100.0 * delta / denom).values

        winner = np.where(
            np.isfinite(delta.values),
            np.where(
                delta.values > eps,
                "mpc",
                np.where(delta.values < -eps, "cccv", "tie"),
            ),
            "nan",
        )
        pair_df[f"{metric}_winner"] = winner

    pair_df["scenario_id"] = (
        pair_df["objective"].astype(str)
        + "|"
        + pair_df["dataset_family"].astype(str)
        + "|"
        + pair_df["dataset_case"].astype(str)
    )
    return pair_df


def add_composite_score(pair_df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    z_cols = []
    for metric in metrics:
        delta_col = f"{metric}_delta_mpc_adv"
        if delta_col not in pair_df.columns:
            continue
        values = pair_df[delta_col].astype(float)
        sigma = float(values.std(skipna=True))
        mu = float(values.mean(skipna=True))
        z_col = f"{metric}_delta_z_mpc_adv"
        pair_df[z_col] = (values - mu) / (sigma + 1e-9)
        z_cols.append(z_col)

    if not z_cols:
        pair_df["composite_score_mpc_adv"] = float("nan")
        return pair_df

    z_matrix = pair_df[z_cols].to_numpy(dtype=float)
    pair_df["composite_score_mpc_adv"] = np.nanmean(z_matrix, axis=1)
    return pair_df


def compute_win_rates(pair_df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    for metric in metrics:
        delta_col = f"{metric}_delta_mpc_adv"
        if delta_col not in pair_df.columns:
            continue
        values = pair_df[delta_col].replace([np.inf, -np.inf], np.nan).dropna()
        total = int(values.shape[0])
        if total == 0:
            continue
        mpc_wins = int((values > 1e-9).sum())
        cccv_wins = int((values < -1e-9).sum())
        ties = int(total - mpc_wins - cccv_wins)
        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "n_scenarios": total,
                "mpc_wins": mpc_wins,
                "cccv_wins": cccv_wins,
                "ties": ties,
                "mpc_win_pct": 100.0 * mpc_wins / total,
                "cccv_win_pct": 100.0 * cccv_wins / total,
                "tie_pct": 100.0 * ties / total,
            }
        )

    return pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)


def save_figure(path_prefix: Path) -> None:
    plt.savefig(path_prefix.with_suffix(".png"), bbox_inches="tight")
    plt.savefig(path_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_win_rate_stacked(win_df: pd.DataFrame, output_path: Path) -> None:
    if win_df.empty:
        return

    labels = win_df["metric_label"].tolist()
    x = np.arange(len(labels))
    mpc = win_df["mpc_win_pct"].to_numpy(dtype=float)
    cccv = win_df["cccv_win_pct"].to_numpy(dtype=float)
    tie = win_df["tie_pct"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(labels)), 4.8))
    ax.bar(x, mpc, label="MPC wins", color="#1f77b4")
    ax.bar(x, cccv, bottom=mpc, label="CCCV wins", color="#ff7f0e")
    ax.bar(x, tie, bottom=mpc + cccv, label="Tie/NA", color="#9aa0a6")
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Scenario Share (%)")
    ax.set_title("Win Share by Metric (MPC vs CCCV)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(loc="upper right")
    save_figure(output_path)


def plot_pareto_scatter(pair_df: pd.DataFrame, output_path: Path) -> None:
    x_col = "charge_time_min_delta_mpc_adv"
    y_col = "safety_event_count_delta_mpc_adv"
    if x_col not in pair_df.columns or y_col not in pair_df.columns:
        return

    data = pair_df[[x_col, y_col, "objective", "dataset_family"]].dropna()
    if data.empty:
        return

    objective_colors = {
        objective: color
        for objective, color in zip(
            sorted(data["objective"].unique()),
            ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        )
    }
    family_markers = {"nasa": "o", "calce": "s", "matr": "^"}

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    for (objective, family), group in data.groupby(["objective", "dataset_family"]):
        ax.scatter(
            group[x_col],
            group[y_col],
            s=60,
            alpha=0.85,
            label=f"{objective}/{family}",
            c=objective_colors.get(objective, "#1f77b4"),
            marker=family_markers.get(family, "o"),
            edgecolors="black",
            linewidths=0.4,
        )

    ax.axvline(0.0, color="black", linestyle="--", alpha=0.6)
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.6)
    ax.set_xlabel("Charge Time Advantage (min, + means MPC faster)")
    ax.set_ylabel("Safety Event Advantage (count, + means MPC safer)")
    ax.set_title("Pareto Improvement Across Objectives and Datasets")
    ax.legend(loc="best", ncol=2, fontsize=7)
    save_figure(output_path)


def plot_composite_heatmap(pair_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if "composite_score_mpc_adv" not in pair_df.columns:
        return pd.DataFrame()

    heat_df = (
        pair_df.groupby(["objective", "dataset_family"])["composite_score_mpc_adv"]
        .mean()
        .unstack("dataset_family")
        .sort_index()
    )
    if heat_df.empty:
        return heat_df

    data = heat_df.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 1.0
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(heat_df.shape[1]))
    ax.set_yticks(np.arange(heat_df.shape[0]))
    ax.set_xticklabels(heat_df.columns.tolist())
    ax.set_yticklabels(heat_df.index.tolist())
    ax.set_xlabel("Dataset Family")
    ax.set_ylabel("Objective")
    ax.set_title("Composite MPC Advantage (z-score)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Higher is better for MPC")

    for row_idx in range(heat_df.shape[0]):
        for col_idx in range(heat_df.shape[1]):
            value = data[row_idx, col_idx]
            text = "nan" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    save_figure(output_path)
    return heat_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate baseline benchmark results across objectives and datasets."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="results/baselines/data_calibrated",
        help="Root containing objective/dataset/case/controller metrics.json files.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/baselines/aggregate",
        help="Directory to store aggregated tables and figures.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric names to aggregate.",
    )
    parser.add_argument(
        "--controllers",
        type=str,
        default="cccv,mpc",
        help="Comma-separated controller names to include.",
    )
    return parser.parse_args()


def main() -> None:
    apply_publication_style()
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    figures_root = output_root / "figures"
    _ensure_dirs([output_root, figures_root])

    requested_metrics = _split_csv_arg(args.metrics)
    controllers = set(_split_csv_arg(args.controllers))
    if not requested_metrics:
        raise SystemExit("No metrics selected. Provide --metrics with at least one metric.")
    if not controllers:
        raise SystemExit("No controllers selected. Provide --controllers.")

    long_df = collect_metrics_rows(input_root=input_root, controllers=controllers)
    available_metrics = [m for m in requested_metrics if m in long_df.columns]
    if not available_metrics:
        raise SystemExit(
            "None of the requested metrics were found. "
            f"Requested: {requested_metrics}"
        )

    pair_df = compute_pairwise_table(long_df=long_df, metrics=available_metrics)
    pair_df = add_composite_score(pair_df=pair_df, metrics=available_metrics)
    win_df = compute_win_rates(pair_df=pair_df, metrics=available_metrics)

    group_mean_df = (
        long_df.groupby(["objective", "dataset_family", "algorithm"])[available_metrics]
        .mean()
        .reset_index()
        .sort_values(["objective", "dataset_family", "algorithm"])
    )
    group_std_df = (
        long_df.groupby(["objective", "dataset_family", "algorithm"])[available_metrics]
        .std()
        .reset_index()
        .sort_values(["objective", "dataset_family", "algorithm"])
    )
    pair_mean_df = (
        pair_df.groupby(["objective", "dataset_family"])[
            [f"{m}_delta_mpc_adv" for m in available_metrics if f"{m}_delta_mpc_adv" in pair_df.columns]
        ]
        .mean()
        .reset_index()
        .sort_values(["objective", "dataset_family"])
    )

    long_df.sort_values(["objective", "dataset_family", "dataset_case", "algorithm"]).to_csv(
        output_root / "all_runs_long_metrics.csv", index=False
    )
    pair_df.sort_values(["objective", "dataset_family", "dataset_case"]).to_csv(
        output_root / "scenario_pairwise_deltas.csv", index=False
    )
    group_mean_df.to_csv(output_root / "group_means.csv", index=False)
    group_std_df.to_csv(output_root / "group_stds.csv", index=False)
    pair_mean_df.to_csv(output_root / "pairwise_delta_means.csv", index=False)
    win_df.to_csv(output_root / "win_rates.csv", index=False)

    plot_win_rate_stacked(win_df=win_df, output_path=figures_root / "01_win_share")
    plot_pareto_scatter(pair_df=pair_df, output_path=figures_root / "02_pareto_time_vs_safety")
    heat_df = plot_composite_heatmap(
        pair_df=pair_df, output_path=figures_root / "03_composite_heatmap"
    )
    if not heat_df.empty:
        heat_df.to_csv(output_root / "composite_heatmap_table.csv")

    print("Aggregated baseline analysis completed.")
    print(f"Input root: {input_root.resolve()}")
    print(f"Output root: {output_root.resolve()}")
    print(f"Scenarios aggregated: {pair_df.shape[0]}")
    print(f"Controllers aggregated: {sorted(set(long_df['algorithm'].tolist()))}")
    print(f"Metrics used: {available_metrics}")


if __name__ == "__main__":
    main()
