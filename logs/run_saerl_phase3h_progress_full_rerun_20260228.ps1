$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$log = Join-Path $root "logs/saerl_phase3h_progress_full_rerun_20260228.log"
$python = "python"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "s"), $Message
    $line | Out-File -FilePath $log -Append -Encoding utf8
}

function Invoke-Step {
    param(
        [string]$Label,
        [string[]]$CommandArgs
    )
    Write-Log "START $Label"
    Write-Log ("CMD " + ($CommandArgs -join " "))
    & $python -u @CommandArgs *>> $log
    Write-Log "DONE $Label"
}

try {
    New-Item -ItemType Directory -Force (Join-Path $root "logs") | Out-Null
    Write-Log "Pipeline launch"

    Invoke-Step -Label "dataset" -CommandArgs @(
        "scripts/generate_saerl_dataset.py",
        "--output-csv", "data/training/saerl_phase3h_progress_full_rerun_20260228_dataset.csv",
        "--output-meta-json", "data/training/saerl_phase3h_progress_full_rerun_20260228_dataset_meta.json",
        "--split-manifest-json", "data/training/saerl_phase3h_progress_full_rerun_20260228_splits.json",
        "--objective", "all",
        "--dataset-families", "nasa,calce,matr",
        "--max-files-per-dataset", "3",
        "--episodes-per-setting", "2",
        "--max-steps", "160",
        "--cem-label-interval", "16",
        "--cem-horizon", "2",
        "--cem-iterations", "1",
        "--cem-population", "8",
        "--cem-elite-frac", "0.25",
        "--cem-profile-json", "configs/saerl_phase3h_cem_progress_profile.json",
        "--baseline-results-root", "results/baselines/data_calibrated",
        "--saerl-mpc-anchor-mode", "family_specific",
        "--min-episode-minutes", "120",
        "--feasible-time-slack", "1.35",
        "--max-steps-cap", "5000",
        "--random-seed", "123"
    )

    Invoke-Step -Label "ensemble" -CommandArgs @(
        "scripts/train_saerl_ensemble.py",
        "--dataset-csv", "data/training/saerl_phase3h_progress_full_rerun_20260228_dataset.csv",
        "--split-manifest-json", "data/training/saerl_phase3h_progress_full_rerun_20260228_splits.json",
        "--output-root", "models/saerl_phase3h_progress_full_rerun_20260228/ensemble",
        "--reports-root", "results/saerl_phase3h_progress_full_rerun_20260228/training/ensemble",
        "--fold", "all",
        "--objective", "all",
        "--chemistry-mode", "family_specific",
        "--chemistry-families", "nasa,calce,matr",
        "--random-seed", "123"
    )

    foreach ($family in @("nasa", "calce", "matr")) {
        Invoke-Step -Label "policy_$family" -CommandArgs @(
            "scripts/train_saerl_policy.py",
            "--dataset-csv", "data/training/saerl_phase3h_progress_full_rerun_20260228_dataset.csv",
            "--split-manifest-json", "data/training/saerl_phase3h_progress_full_rerun_20260228_splits.json",
            "--ensemble-root", "models/saerl_phase3h_progress_full_rerun_20260228/ensemble",
            "--output-root", "models/saerl_phase3h_progress_full_rerun_20260228/policy",
            "--reports-root", "results/saerl_phase3h_progress_full_rerun_20260228/training/policy",
            "--fold", "all",
            "--objective", "all",
            "--dataset-families", "nasa,calce,matr",
            "--max-files-per-dataset", "3",
            "--chemistry-mode", "family_specific",
            "--chemistry-families", $family,
            "--baseline-results-root", "results/baselines/data_calibrated",
            "--saerl-mpc-anchor-mode", "family_specific",
            "--min-episode-minutes", "120",
            "--feasible-time-slack", "1.35",
            "--max-steps-cap", "5000",
            "--saerl-family-profile-json", "configs/saerl_phase3h_policy_progress_profile.json",
            "--random-seed", "123"
        )
    }

    Invoke-Step -Label "eval" -CommandArgs @(
        "scripts/eval_saerl_vs_baselines.py",
        "--dataset-csv", "data/training/saerl_phase3h_progress_full_rerun_20260228_dataset.csv",
        "--split-manifest-json", "data/training/saerl_phase3h_progress_full_rerun_20260228_splits.json",
        "--ensemble-root", "models/saerl_phase3h_progress_full_rerun_20260228/ensemble",
        "--policy-root-template", "models/saerl_phase3h_progress_full_rerun_20260228/policy/{objective}",
        "--output-root", "results/saerl_phase3h_progress_full_rerun_20260228/evaluation_allfolds_3family",
        "--objective", "all",
        "--fold", "all",
        "--dataset-families", "nasa,calce,matr",
        "--max-files-per-dataset", "3",
        "--chemistry-mode", "family_specific",
        "--primary-saerl-mode", "family_specific",
        "--chemistry-families", "nasa,calce,matr",
        "--chemistry-aware-baselines",
        "--saerl-mpc-anchor-mode", "family_specific",
        "--saerl-eval-calibration-json", "configs/saerl_phase3h_eval_push.json",
        "--skip-detailed-figures"
    )

    Invoke-Step -Label "aggregate" -CommandArgs @(
        "scripts/aggregate_saerl_results.py",
        "--input-root", "results/saerl_phase3h_progress_full_rerun_20260228/evaluation_allfolds_3family",
        "--output-root", "results/saerl_phase3h_progress_full_rerun_20260228/aggregate_allfolds_3family"
    )

    Write-Log "Pipeline completed"
}
catch {
    Write-Log ("FAILED " + $_.Exception.Message)
    throw
}
