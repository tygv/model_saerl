"""Reproducible runner for the locked Phase-3h progress stack."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class StageSpec:
    name: str
    dataset_families: str
    fold: str
    dataset_csv: str
    dataset_meta_json: str
    split_manifest_json: str
    model_root: str
    result_root: str
    aggregate_root: str


STAGES = {
    "fold0_calce_matr": StageSpec(
        name="fold0_calce_matr",
        dataset_families="calce,matr",
        fold="0",
        dataset_csv="data/training/saerl_phase3h_progress_dataset.csv",
        dataset_meta_json="data/training/saerl_phase3h_progress_dataset_meta.json",
        split_manifest_json="data/training/saerl_phase3h_progress_splits.json",
        model_root="models/saerl_phase3h_progress",
        result_root="results/saerl_phase3h_progress/evaluation_fold0_evalpush",
        aggregate_root="results/saerl_phase3h_progress/aggregate_fold0_evalpush",
    ),
    "allfolds_calce_matr": StageSpec(
        name="allfolds_calce_matr",
        dataset_families="calce,matr",
        fold="all",
        dataset_csv="data/training/saerl_phase3h_progress_dataset.csv",
        dataset_meta_json="data/training/saerl_phase3h_progress_dataset_meta.json",
        split_manifest_json="data/training/saerl_phase3h_progress_splits.json",
        model_root="models/saerl_phase3h_progress",
        result_root="results/saerl_phase3h_progress/evaluation_allfolds_calce_matr",
        aggregate_root="results/saerl_phase3h_progress/aggregate_allfolds_calce_matr",
    ),
    "allfolds_3family": StageSpec(
        name="allfolds_3family",
        dataset_families="nasa,calce,matr",
        fold="all",
        dataset_csv="data/training/saerl_phase3h_progress_full_dataset.csv",
        dataset_meta_json="data/training/saerl_phase3h_progress_full_dataset_meta.json",
        split_manifest_json="data/training/saerl_phase3h_progress_full_splits.json",
        model_root="models/saerl_phase3h_progress_full",
        result_root="results/saerl_phase3h_progress_full/evaluation_allfolds_3family",
        aggregate_root="results/saerl_phase3h_progress_full/aggregate_allfolds_3family",
    ),
}


def split_csv_arg(value: str) -> List[str]:
    return [x.strip().lower() for x in str(value).split(",") if x.strip()]


def run_cmd(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run locked Phase-3h progress SAERL stack.")
    p.add_argument(
        "--stage",
        type=str,
        default="fold0_calce_matr",
        choices=tuple(STAGES.keys()),
    )
    p.add_argument("--python-exec", type=str, default=sys.executable)
    p.add_argument("--skip-dataset", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-aggregate", action="store_true")
    p.add_argument("--random-seed", type=int, default=123)
    p.add_argument("--max-files-per-dataset", type=int, default=3)
    p.add_argument("--episodes-per-setting", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=160)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stage = STAGES[args.stage]
    families = split_csv_arg(stage.dataset_families)

    if not args.skip_dataset:
        run_cmd(
            [
                args.python_exec,
                "scripts/generate_saerl_dataset.py",
                "--output-csv",
                stage.dataset_csv,
                "--output-meta-json",
                stage.dataset_meta_json,
                "--split-manifest-json",
                stage.split_manifest_json,
                "--objective",
                "all",
                "--dataset-families",
                stage.dataset_families,
                "--max-files-per-dataset",
                str(args.max_files_per_dataset),
                "--episodes-per-setting",
                str(args.episodes_per_setting),
                "--max-steps",
                str(args.max_steps),
                "--cem-label-interval",
                "16",
                "--cem-horizon",
                "2",
                "--cem-iterations",
                "1",
                "--cem-population",
                "8",
                "--cem-elite-frac",
                "0.25",
                "--cem-profile-json",
                "configs/saerl_phase3h_cem_progress_profile.json",
                "--baseline-results-root",
                "results/baselines/data_calibrated",
                "--random-seed",
                str(args.random_seed),
            ]
        )

    if not args.skip_train:
        run_cmd(
            [
                args.python_exec,
                "scripts/train_saerl_ensemble.py",
                "--dataset-csv",
                stage.dataset_csv,
                "--split-manifest-json",
                stage.split_manifest_json,
                "--output-root",
                f"{stage.model_root}/ensemble",
                "--reports-root",
                f"{stage.model_root.replace('models', 'results')}/training/ensemble",
                "--fold",
                stage.fold,
                "--objective",
                "all",
                "--chemistry-mode",
                "family_specific",
                "--chemistry-families",
                stage.dataset_families,
                "--random-seed",
                str(args.random_seed),
            ]
        )
        for fam in families:
            run_cmd(
                [
                    args.python_exec,
                    "scripts/train_saerl_policy.py",
                    "--dataset-csv",
                    stage.dataset_csv,
                    "--split-manifest-json",
                    stage.split_manifest_json,
                    "--ensemble-root",
                    f"{stage.model_root}/ensemble",
                    "--output-root",
                    f"{stage.model_root}/policy",
                    "--reports-root",
                    f"{stage.model_root.replace('models', 'results')}/training/policy",
                    "--fold",
                    stage.fold,
                    "--objective",
                    "all",
                    "--dataset-families",
                    stage.dataset_families,
                    "--max-files-per-dataset",
                    str(args.max_files_per_dataset),
                    "--chemistry-mode",
                    "family_specific",
                    "--chemistry-families",
                    fam,
                    "--baseline-results-root",
                    "results/baselines/data_calibrated",
                    "--saerl-family-profile-json",
                    "configs/saerl_phase3h_policy_progress_profile.json",
                    "--random-seed",
                    str(args.random_seed),
                ]
            )

    if not args.skip_eval:
        run_cmd(
            [
                args.python_exec,
                "scripts/eval_saerl_vs_baselines.py",
                "--dataset-csv",
                stage.dataset_csv,
                "--split-manifest-json",
                stage.split_manifest_json,
                "--ensemble-root",
                f"{stage.model_root}/ensemble",
                "--policy-root-template",
                f"{stage.model_root}/policy/{{objective}}",
                "--output-root",
                stage.result_root,
                "--objective",
                "all",
                "--fold",
                stage.fold,
                "--dataset-families",
                stage.dataset_families,
                "--max-files-per-dataset",
                str(args.max_files_per_dataset),
                "--chemistry-mode",
                "family_specific",
                "--primary-saerl-mode",
                "family_specific",
                "--chemistry-families",
                stage.dataset_families,
                "--chemistry-aware-baselines",
                "--saerl-eval-calibration-json",
                "configs/saerl_phase3h_eval_push.json",
                "--skip-detailed-figures",
            ]
        )

    if not args.skip_aggregate:
        run_cmd(
            [
                args.python_exec,
                "scripts/aggregate_saerl_results.py",
                "--input-root",
                stage.result_root,
                "--output-root",
                stage.aggregate_root,
            ]
        )

    print(f"Completed stage: {stage.name}", flush=True)


if __name__ == "__main__":
    main()
