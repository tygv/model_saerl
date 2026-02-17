"""Orchestrate Phase-3 chemistry-aware SAERL training/evaluation/aggregation."""

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
    p = argparse.ArgumentParser(description="Run Phase-3 chemistry-aware SAERL pipeline.")
    p.add_argument("--dataset-csv", type=str, default="data/training/saerl_phase2_dataset.csv")
    p.add_argument("--split-manifest-json", type=str, default="data/training/saerl_phase2_splits.json")
    p.add_argument("--ensemble-root", type=str, default="models/saerl_phase3/ensemble")
    p.add_argument("--policy-root", type=str, default="models/saerl_phase3/policy")
    p.add_argument("--train-reports-root", type=str, default="results/saerl_phase3/training")
    p.add_argument("--eval-output-root", type=str, default="results/saerl_phase3/evaluation")
    p.add_argument("--aggregate-output-root", type=str, default="results/saerl_phase3/aggregate")
    p.add_argument("--objective", type=str, default="all")
    p.add_argument("--fold", type=str, default="all")
    p.add_argument("--dataset-families", type=str, default="nasa,calce,matr")
    p.add_argument("--chemistry-families", type=str, default="")
    p.add_argument(
        "--chemistry-mode",
        type=str,
        default="all",
        choices=["global", "family_specific", "shared_plus_heads", "all"],
    )
    p.add_argument(
        "--primary-saerl-mode",
        type=str,
        default="shared_plus_heads",
        choices=["global", "family_specific", "shared_plus_heads"],
    )
    p.add_argument("--max-files-per-dataset", type=int, default=3)
    p.add_argument("--run-ablations", action="store_true")
    p.add_argument("--chemistry-aware-baselines", action="store_true")
    p.add_argument("--random-seed", type=int, default=123)
    p.add_argument("--offline-epochs", type=int, default=20)
    p.add_argument("--online-epochs", type=int, default=12)
    p.add_argument("--gru-epochs", type=int, default=25)
    p.add_argument("--mlp-epochs", type=int, default=25)
    p.add_argument("--gate-epochs", type=int, default=20)
    p.add_argument("--standardized-root", type=str, default="data/standardized")
    p.add_argument("--params-root", type=str, default="data/standardized_params")
    p.add_argument("--max-steps", type=int, default=1200)
    return p


def run_ensemble_train(args: argparse.Namespace, mode: str) -> None:
    cmd = [
        sys.executable,
        "scripts/train_saerl_ensemble.py",
        "--dataset-csv",
        args.dataset_csv,
        "--split-manifest-json",
        args.split_manifest_json,
        "--output-root",
        args.ensemble_root,
        "--reports-root",
        str(Path(args.train_reports_root) / "ensemble"),
        "--fold",
        args.fold,
        "--random-seed",
        str(args.random_seed),
        "--gru-epochs",
        str(args.gru_epochs),
        "--mlp-epochs",
        str(args.mlp_epochs),
        "--gate-epochs",
        str(args.gate_epochs),
        "--chemistry-mode",
        mode,
    ]
    fams = args.chemistry_families.strip()
    if fams:
        cmd.extend(["--chemistry-families", fams])
    run_cmd(cmd)


def run_policy_train(args: argparse.Namespace, mode: str, family: str | None, init_actor_root: str = "") -> None:
    cmd = [
        sys.executable,
        "scripts/train_saerl_policy.py",
        "--dataset-csv",
        args.dataset_csv,
        "--split-manifest-json",
        args.split_manifest_json,
        "--ensemble-root",
        args.ensemble_root,
        "--output-root",
        args.policy_root,
        "--reports-root",
        str(Path(args.train_reports_root) / "policy"),
        "--fold",
        args.fold,
        "--objective",
        args.objective,
        "--dataset-families",
        args.dataset_families,
        "--max-files-per-dataset",
        str(args.max_files_per_dataset),
        "--standardized-root",
        args.standardized_root,
        "--params-root",
        args.params_root,
        "--random-seed",
        str(args.random_seed),
        "--offline-epochs",
        str(args.offline_epochs),
        "--online-epochs",
        str(args.online_epochs),
        "--chemistry-mode",
        mode,
    ]
    if family:
        cmd.extend(["--chemistry-families", family])
    if init_actor_root:
        cmd.extend(["--init-actor-root", init_actor_root])
    run_cmd(cmd)


def run_eval_and_aggregate(args: argparse.Namespace) -> None:
    eval_cmd = [
        sys.executable,
        "scripts/eval_saerl_vs_baselines.py",
        "--dataset-csv",
        args.dataset_csv,
        "--split-manifest-json",
        args.split_manifest_json,
        "--ensemble-root",
        args.ensemble_root,
        "--policy-root",
        args.policy_root,
        "--output-root",
        args.eval_output_root,
        "--objective",
        args.objective,
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
        str(args.max_steps),
        "--chemistry-mode",
        args.chemistry_mode,
        "--primary-saerl-mode",
        args.primary_saerl_mode,
    ]
    if args.run_ablations:
        eval_cmd.append("--run-ablations")
    if args.chemistry_families.strip():
        eval_cmd.extend(["--chemistry-families", args.chemistry_families.strip()])
    if args.chemistry_aware_baselines:
        eval_cmd.append("--chemistry-aware-baselines")
    run_cmd(eval_cmd)

    agg_cmd = [
        sys.executable,
        "scripts/aggregate_saerl_results.py",
        "--input-root",
        args.eval_output_root,
        "--output-root",
        args.aggregate_output_root,
    ]
    run_cmd(agg_cmd)


def main() -> None:
    args = parser().parse_args()
    if not args.chemistry_families.strip():
        args.chemistry_families = args.dataset_families
    families = split_csv_arg(args.chemistry_families)
    modes = (
        ["global", "family_specific", "shared_plus_heads"]
        if args.chemistry_mode == "all"
        else [args.chemistry_mode]
    )

    if "global" in modes:
        run_ensemble_train(args=args, mode="global")
        run_policy_train(args=args, mode="global", family=None)

    if "family_specific" in modes:
        run_ensemble_train(args=args, mode="family_specific")
        for fam in families:
            run_policy_train(args=args, mode="family_specific", family=fam)

    if "shared_plus_heads" in modes:
        run_ensemble_train(args=args, mode="shared_plus_heads")
        run_policy_train(args=args, mode="shared_plus_heads", family=None)
        shared_actor_root = str(Path(args.policy_root) / "shared_plus_heads" / "shared")
        for fam in families:
            run_policy_train(
                args=args,
                mode="shared_plus_heads",
                family=fam,
                init_actor_root=shared_actor_root,
            )

    run_eval_and_aggregate(args=args)
    print("Completed Phase-3 chemistry-aware SAERL pipeline.")
    print(f"Evaluation: {Path(args.eval_output_root).resolve()}")
    print(f"Aggregate:  {Path(args.aggregate_output_root).resolve()}")


if __name__ == "__main__":
    main()
