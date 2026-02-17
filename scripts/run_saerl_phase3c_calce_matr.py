"""Phase-3c sweep focused on CALCE/MATR with objective-specific family heads."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def split_csv_arg(value: str) -> List[str]:
    return [x.strip().lower() for x in str(value).split(",") if x.strip()]


def run_cmd(cmd: List[str]) -> None:
    print(">>", " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Phase-3c chemistry-aware SAERL sweep.")
    p.add_argument("--dataset-csv", type=str, default="data/training/saerl_phase3c_dataset.csv")
    p.add_argument("--dataset-meta-json", type=str, default="data/training/saerl_phase3c_dataset_meta.json")
    p.add_argument("--split-manifest-json", type=str, default="data/training/saerl_phase3c_splits.json")
    p.add_argument(
        "--cem-profile-json",
        type=str,
        default="configs/saerl_phase3c_cem_profile.json",
    )
    p.add_argument(
        "--family-policy-profile-json",
        type=str,
        default="configs/saerl_phase3c_family_policy_profile.json",
    )
    p.add_argument("--ensemble-root", type=str, default="models/saerl_phase3c/ensemble")
    p.add_argument("--policy-root", type=str, default="models/saerl_phase3c/policy")
    p.add_argument("--train-reports-root", type=str, default="results/saerl_phase3c/training")
    p.add_argument("--eval-output-root", type=str, default="results/saerl_phase3c/evaluation")
    p.add_argument("--aggregate-output-root", type=str, default="results/saerl_phase3c/aggregate")
    p.add_argument("--objectives", type=str, default="fastest,safe,long_life")
    p.add_argument("--fold", type=str, default="all")
    p.add_argument("--dataset-families", type=str, default="nasa,calce,matr")
    p.add_argument("--chemistry-families", type=str, default="nasa,calce,matr")
    p.add_argument("--max-files-per-dataset", type=int, default=3)
    p.add_argument("--episodes-per-setting", type=int, default=3)
    p.add_argument("--dataset-max-steps", type=int, default=500)
    p.add_argument("--eval-max-steps", type=int, default=1200)
    p.add_argument("--standardized-root", type=str, default="data/standardized")
    p.add_argument("--params-root", type=str, default="data/standardized_params")
    p.add_argument("--n-series", type=int, default=20)
    p.add_argument("--n-parallel", type=int, default=1)
    p.add_argument("--max-charge-current-a", type=float, default=10.0)
    p.add_argument("--initial-soc", type=float, default=0.2)
    p.add_argument("--target-soc", type=float, default=0.8)
    p.add_argument("--ambient-temp-c", type=float, default=25.0)
    p.add_argument("--random-seed", type=int, default=123)
    p.add_argument("--gru-epochs", type=int, default=28)
    p.add_argument("--mlp-epochs", type=int, default=28)
    p.add_argument("--gate-epochs", type=int, default=22)
    p.add_argument("--offline-epochs", type=int, default=24)
    p.add_argument("--online-epochs", type=int, default=14)
    p.add_argument("--primary-saerl-mode", type=str, default="family_specific")
    p.add_argument("--run-ablations", action="store_true")
    p.add_argument("--skip-dataset-generation", action="store_true")
    return p


def run_dataset_generation(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/generate_saerl_dataset.py",
        "--output-csv",
        args.dataset_csv,
        "--output-meta-json",
        args.dataset_meta_json,
        "--split-manifest-json",
        args.split_manifest_json,
        "--objective",
        "all",
        "--standardized-root",
        args.standardized_root,
        "--params-root",
        args.params_root,
        "--dataset-families",
        args.dataset_families,
        "--max-files-per-dataset",
        str(args.max_files_per_dataset),
        "--episodes-per-setting",
        str(args.episodes_per_setting),
        "--max-steps",
        str(args.dataset_max_steps),
        "--n-series",
        str(args.n_series),
        "--n-parallel",
        str(args.n_parallel),
        "--max-charge-current-a",
        str(args.max_charge_current_a),
        "--initial-soc",
        str(args.initial_soc),
        "--target-soc",
        str(args.target_soc),
        "--ambient-temp-c",
        str(args.ambient_temp_c),
        "--random-seed",
        str(args.random_seed),
        "--cem-profile-json",
        args.cem_profile_json,
    ]
    run_cmd(cmd)


def run_objective_training(args: argparse.Namespace, objective: str, chemistry_families: List[str]) -> None:
    objective_ensemble_root = Path(args.ensemble_root) / objective
    objective_policy_root = Path(args.policy_root) / objective
    objective_ensemble_reports = Path(args.train_reports_root) / objective / "ensemble"
    objective_policy_reports = Path(args.train_reports_root) / objective / "policy"

    # Global expert stack (reference controller).
    run_cmd(
        [
            sys.executable,
            "scripts/train_saerl_ensemble.py",
            "--dataset-csv",
            args.dataset_csv,
            "--split-manifest-json",
            args.split_manifest_json,
            "--output-root",
            str(objective_ensemble_root),
            "--reports-root",
            str(objective_ensemble_reports),
            "--fold",
            args.fold,
            "--objective",
            objective,
            "--random-seed",
            str(args.random_seed),
            "--gru-epochs",
            str(args.gru_epochs),
            "--mlp-epochs",
            str(args.mlp_epochs),
            "--gate-epochs",
            str(args.gate_epochs),
            "--chemistry-mode",
            "global",
        ]
    )

    # Objective-specific family heads (CALCE/MATR-focused with full-family coverage).
    run_cmd(
        [
            sys.executable,
            "scripts/train_saerl_ensemble.py",
            "--dataset-csv",
            args.dataset_csv,
            "--split-manifest-json",
            args.split_manifest_json,
            "--output-root",
            str(objective_ensemble_root),
            "--reports-root",
            str(objective_ensemble_reports),
            "--fold",
            args.fold,
            "--objective",
            objective,
            "--random-seed",
            str(args.random_seed),
            "--gru-epochs",
            str(args.gru_epochs),
            "--mlp-epochs",
            str(args.mlp_epochs),
            "--gate-epochs",
            str(args.gate_epochs),
            "--chemistry-mode",
            "family_specific",
            "--chemistry-families",
            ",".join(chemistry_families),
            "--init-from-root",
            str(objective_ensemble_root),
        ]
    )

    # Global policy per objective.
    run_cmd(
        [
            sys.executable,
            "scripts/train_saerl_policy.py",
            "--dataset-csv",
            args.dataset_csv,
            "--split-manifest-json",
            args.split_manifest_json,
            "--ensemble-root",
            str(objective_ensemble_root),
            "--output-root",
            str(objective_policy_root),
            "--reports-root",
            str(objective_policy_reports),
            "--fold",
            args.fold,
            "--objective",
            objective,
            "--dataset-families",
            args.dataset_families,
            "--max-files-per-dataset",
            str(args.max_files_per_dataset),
            "--standardized-root",
            args.standardized_root,
            "--params-root",
            args.params_root,
            "--max-charge-current-a",
            str(args.max_charge_current_a),
            "--initial-soc",
            str(args.initial_soc),
            "--target-soc",
            str(args.target_soc),
            "--ambient-temp-c",
            str(args.ambient_temp_c),
            "--random-seed",
            str(args.random_seed),
            "--offline-epochs",
            str(args.offline_epochs),
            "--online-epochs",
            str(args.online_epochs),
            "--chemistry-mode",
            "global",
        ]
    )

    # Objective-specific family policy heads with calibrated anti-stall.
    for family in chemistry_families:
        run_cmd(
            [
                sys.executable,
                "scripts/train_saerl_policy.py",
                "--dataset-csv",
                args.dataset_csv,
                "--split-manifest-json",
                args.split_manifest_json,
                "--ensemble-root",
                str(objective_ensemble_root),
                "--output-root",
                str(objective_policy_root),
                "--reports-root",
                str(objective_policy_reports),
                "--fold",
                args.fold,
                "--objective",
                objective,
                "--dataset-families",
                args.dataset_families,
                "--max-files-per-dataset",
                str(args.max_files_per_dataset),
                "--standardized-root",
                args.standardized_root,
                "--params-root",
                args.params_root,
                "--max-charge-current-a",
                str(args.max_charge_current_a),
                "--initial-soc",
                str(args.initial_soc),
                "--target-soc",
                str(args.target_soc),
                "--ambient-temp-c",
                str(args.ambient_temp_c),
                "--random-seed",
                str(args.random_seed),
                "--offline-epochs",
                str(args.offline_epochs),
                "--online-epochs",
                str(args.online_epochs),
                "--chemistry-mode",
                "family_specific",
                "--chemistry-families",
                family,
                "--init-actor-root",
                str(objective_policy_root),
                "--saerl-family-profile-json",
                args.family_policy_profile_json,
            ]
        )


def run_eval_and_aggregate(args: argparse.Namespace) -> None:
    ensemble_template = str(Path(args.ensemble_root) / "{objective}")
    policy_template = str(Path(args.policy_root) / "{objective}")
    eval_cmd = [
        sys.executable,
        "scripts/eval_saerl_vs_baselines.py",
        "--dataset-csv",
        args.dataset_csv,
        "--split-manifest-json",
        args.split_manifest_json,
        "--ensemble-root",
        args.ensemble_root,
        "--ensemble-root-template",
        ensemble_template,
        "--policy-root",
        args.policy_root,
        "--policy-root-template",
        policy_template,
        "--output-root",
        args.eval_output_root,
        "--objective",
        "all",
        "--fold",
        args.fold,
        "--standardized-root",
        args.standardized_root,
        "--params-root",
        args.params_root,
        "--dataset-families",
        args.dataset_families,
        "--max-files-per-dataset",
        str(args.max_files_per_dataset),
        "--max-steps",
        str(args.eval_max_steps),
        "--chemistry-mode",
        "all",
        "--primary-saerl-mode",
        args.primary_saerl_mode,
        "--chemistry-families",
        args.chemistry_families,
        "--chemistry-aware-baselines",
    ]
    if args.run_ablations:
        eval_cmd.append("--run-ablations")
    run_cmd(eval_cmd)

    run_cmd(
        [
            sys.executable,
            "scripts/aggregate_saerl_results.py",
            "--input-root",
            args.eval_output_root,
            "--output-root",
            args.aggregate_output_root,
        ]
    )


def main() -> None:
    args = parser().parse_args()
    objectives = split_csv_arg(args.objectives)
    if not objectives:
        raise SystemExit("No objectives were provided.")
    chemistry_families = split_csv_arg(args.chemistry_families)
    if not chemistry_families:
        chemistry_families = split_csv_arg(args.dataset_families)
    if not chemistry_families:
        raise SystemExit("No chemistry families were provided.")

    if not args.skip_dataset_generation:
        run_dataset_generation(args)

    for objective in objectives:
        print(f"\n=== Phase-3c training for objective={objective} ===", flush=True)
        run_objective_training(args=args, objective=objective, chemistry_families=chemistry_families)

    run_eval_and_aggregate(args=args)
    print("\nCompleted Phase-3c CALCE/MATR-focused sweep.")
    print(f"Evaluation: {Path(args.eval_output_root).resolve()}")
    print(f"Aggregate:  {Path(args.aggregate_output_root).resolve()}")


if __name__ == "__main__":
    main()
