"""Run the full comparable Phase-3h 3-family benchmark with source_v1 context enabled."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TAG = "source_context_v1"


@dataclass
class ArtifactPaths:
    dataset_csv: str
    dataset_meta_json: str
    split_manifest_json: str
    model_root: str
    training_root: str
    eval_root: str
    aggregate_root: str


def run_cmd(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))


def build_paths(tag: str) -> ArtifactPaths:
    base = f"saerl_phase3h_{tag}"
    return ArtifactPaths(
        dataset_csv=f"data/training/{base}_dataset.csv",
        dataset_meta_json=f"data/training/{base}_dataset_meta.json",
        split_manifest_json=f"data/training/{base}_splits.json",
        model_root=f"models/{base}",
        training_root=f"results/{base}/training",
        eval_root=f"results/{base}/evaluation_allfolds_3family",
        aggregate_root=f"results/{base}/aggregate_allfolds_3family",
    )


def split_csv_arg(value: str) -> List[str]:
    return [x.strip().lower() for x in str(value).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full 3-family Phase-3h SAERL benchmark with source_v1 context."
    )
    p.add_argument("--python-exec", type=str, default=sys.executable)
    p.add_argument("--tag", type=str, default=DEFAULT_TAG)
    p.add_argument("--skip-dataset", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-ensemble", action="store_true")
    p.add_argument("--skip-policy", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-aggregate", action="store_true")
    p.add_argument("--random-seed", type=int, default=123)
    p.add_argument("--dataset-families", type=str, default="nasa,calce,matr")
    p.add_argument("--max-files-per-dataset", type=int, default=3)
    p.add_argument("--episodes-per-setting", type=int, default=2)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--min-episode-minutes", type=float, default=120.0)
    p.add_argument("--feasible-time-slack", type=float, default=1.35)
    p.add_argument("--max-steps-cap", type=int, default=5000)
    p.add_argument(
        "--saerl-mpc-anchor-mode",
        type=str,
        default="family_specific",
        choices=["global", "family_specific", "shared_plus_heads"],
    )
    p.add_argument(
        "--family-metadata-json",
        type=str,
        default="configs/source_family_metadata_v1.json",
    )
    p.add_argument(
        "--nasa-impedance-root",
        type=str,
        default="data/standardized/nasa_impedance",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_paths(args.tag)
    families = split_csv_arg(args.dataset_families)

    if not args.skip_dataset:
        run_cmd(
            [
                args.python_exec,
                "scripts/generate_saerl_dataset.py",
                "--output-csv",
                paths.dataset_csv,
                "--output-meta-json",
                paths.dataset_meta_json,
                "--split-manifest-json",
                paths.split_manifest_json,
                "--objective",
                "all",
                "--dataset-families",
                args.dataset_families,
                "--max-files-per-dataset",
                str(args.max_files_per_dataset),
                "--episodes-per-setting",
                str(args.episodes_per_setting),
                "--n-folds",
                str(args.n_folds),
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
                "--saerl-mpc-anchor-mode",
                args.saerl_mpc_anchor_mode,
                "--context-feature-set",
                "source_v1",
                "--family-metadata-json",
                args.family_metadata_json,
                "--nasa-impedance-root",
                args.nasa_impedance_root,
                "--min-episode-minutes",
                str(args.min_episode_minutes),
                "--feasible-time-slack",
                str(args.feasible_time_slack),
                "--max-steps-cap",
                str(args.max_steps_cap),
                "--random-seed",
                str(args.random_seed),
            ]
        )

    if not args.skip_train:
        if not args.skip_ensemble:
            run_cmd(
                [
                    args.python_exec,
                    "scripts/train_saerl_ensemble.py",
                    "--dataset-csv",
                    paths.dataset_csv,
                    "--split-manifest-json",
                    paths.split_manifest_json,
                    "--output-root",
                    f"{paths.model_root}/ensemble",
                    "--reports-root",
                    f"{paths.training_root}/ensemble",
                    "--fold",
                    "all",
                    "--objective",
                    "all",
                    "--chemistry-mode",
                    "family_specific",
                    "--chemistry-families",
                    args.dataset_families,
                    "--context-feature-set",
                    "source_v1",
                    "--family-metadata-json",
                    args.family_metadata_json,
                    "--random-seed",
                    str(args.random_seed),
                ]
            )
        if not args.skip_policy:
            for fam in families:
                run_cmd(
                    [
                        args.python_exec,
                        "scripts/train_saerl_policy.py",
                        "--dataset-csv",
                        paths.dataset_csv,
                        "--split-manifest-json",
                        paths.split_manifest_json,
                        "--ensemble-root",
                        f"{paths.model_root}/ensemble",
                        "--output-root",
                        f"{paths.model_root}/policy",
                        "--reports-root",
                        f"{paths.training_root}/policy",
                        "--fold",
                        "all",
                        "--objective",
                        "all",
                        "--dataset-families",
                        args.dataset_families,
                        "--max-files-per-dataset",
                        str(args.max_files_per_dataset),
                        "--chemistry-mode",
                        "family_specific",
                        "--chemistry-families",
                        fam,
                        "--baseline-results-root",
                        "results/baselines/data_calibrated",
                        "--saerl-mpc-anchor-mode",
                        args.saerl_mpc_anchor_mode,
                        "--context-feature-set",
                        "source_v1",
                        "--family-metadata-json",
                        args.family_metadata_json,
                        "--nasa-impedance-root",
                        args.nasa_impedance_root,
                        "--min-episode-minutes",
                        str(args.min_episode_minutes),
                        "--feasible-time-slack",
                        str(args.feasible_time_slack),
                        "--max-steps-cap",
                        str(args.max_steps_cap),
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
                paths.dataset_csv,
                "--split-manifest-json",
                paths.split_manifest_json,
                "--ensemble-root",
                f"{paths.model_root}/ensemble",
                "--policy-root-template",
                f"{paths.model_root}/policy/{{objective}}",
                "--output-root",
                paths.eval_root,
                "--objective",
                "all",
                "--fold",
                "all",
                "--dataset-families",
                args.dataset_families,
                "--max-files-per-dataset",
                str(args.max_files_per_dataset),
                "--chemistry-mode",
                "family_specific",
                "--primary-saerl-mode",
                "family_specific",
                "--chemistry-families",
                args.dataset_families,
                "--chemistry-aware-baselines",
                "--saerl-mpc-anchor-mode",
                args.saerl_mpc_anchor_mode,
                "--context-feature-set",
                "source_v1",
                "--family-metadata-json",
                args.family_metadata_json,
                "--nasa-impedance-root",
                args.nasa_impedance_root,
                "--saerl-eval-calibration-json",
                "configs/saerl_phase3h_eval_push.json",
                "--skip-detailed-figures",
                "--max-steps",
                str(args.max_steps),
                "--min-episode-minutes",
                str(args.min_episode_minutes),
                "--feasible-time-slack",
                str(args.feasible_time_slack),
                "--max-steps-cap",
                str(args.max_steps_cap),
            ]
        )

    if not args.skip_aggregate:
        run_cmd(
            [
                args.python_exec,
                "scripts/aggregate_saerl_results.py",
                "--input-root",
                paths.eval_root,
                "--output-root",
                paths.aggregate_root,
            ]
        )

    print(f"Completed source-context tag: {args.tag}", flush=True)


if __name__ == "__main__":
    main()
