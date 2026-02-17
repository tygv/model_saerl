"""Train SAERL ensemble experts and adaptive gate per fold."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from controllers.adaptive_ensemble_rl import (
    AdaptiveEnsemblePredictor,
    GateNetwork,
    QuantileGRUModel,
    QuantileMLPModel,
    SAERLConfig,
    quantile_pinball_loss,
)


@dataclass
class TrainEnsembleConfig:
    dataset_csv: str = "data/training/saerl_phase2_dataset.csv"
    split_manifest_json: str = "data/training/saerl_phase2_splits.json"
    output_root: str = "models/saerl_phase2/ensemble"
    reports_root: str = "results/saerl_phase2/training/ensemble"
    fold: str = "all"
    objective: str = "all"
    random_seed: int = 123
    batch_size: int = 256
    gru_epochs: int = 25
    mlp_epochs: int = 25
    gate_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 5
    window_len: int = 20
    feature_dim: int = 6
    chemistry_mode: str = "global"
    chemistry_families: str = ""
    init_from_root: str = ""


def parse_args() -> TrainEnsembleConfig:
    import argparse

    parser = argparse.ArgumentParser(description="Train SAERL ensemble per fold.")
    parser.add_argument("--dataset-csv", type=str, default="data/training/saerl_phase2_dataset.csv")
    parser.add_argument(
        "--split-manifest-json",
        type=str,
        default="data/training/saerl_phase2_splits.json",
    )
    parser.add_argument("--output-root", type=str, default="models/saerl_phase2/ensemble")
    parser.add_argument(
        "--reports-root",
        type=str,
        default="results/saerl_phase2/training/ensemble",
    )
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument("--objective", type=str, default="all")
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gru-epochs", type=int, default=25)
    parser.add_argument("--mlp-epochs", type=int, default=25)
    parser.add_argument("--gate-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--window-len", type=int, default=20)
    parser.add_argument("--feature-dim", type=int, default=6)
    parser.add_argument(
        "--chemistry-mode",
        type=str,
        default="global",
        choices=["global", "family_specific", "shared_plus_heads"],
        help="Checkpoint layout mode for chemistry-aware training.",
    )
    parser.add_argument(
        "--chemistry-families",
        type=str,
        default="",
        help="Optional comma list of chemistry families (defaults to dataset families in CSV).",
    )
    parser.add_argument(
        "--init-from-root",
        type=str,
        default="",
        help="Optional checkpoint root to warm-start from (used for family heads).",
    )
    args = parser.parse_args()

    return TrainEnsembleConfig(
        dataset_csv=args.dataset_csv,
        split_manifest_json=args.split_manifest_json,
        output_root=args.output_root,
        reports_root=args.reports_root,
        fold=args.fold,
        objective=args.objective,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        gru_epochs=args.gru_epochs,
        mlp_epochs=args.mlp_epochs,
        gate_epochs=args.gate_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        window_len=args.window_len,
        feature_dim=args.feature_dim,
        chemistry_mode=args.chemistry_mode,
        chemistry_families=args.chemistry_families,
        init_from_root=args.init_from_root,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_window_cols(columns: Sequence[str]) -> List[str]:
    pattern = re.compile(r"^window_t(\d+)_f(\d+)$")
    parsed = []
    for c in columns:
        m = pattern.match(c)
        if not m:
            continue
        t = int(m.group(1))
        f = int(m.group(2))
        parsed.append((t, f, c))
    parsed.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in parsed]


def split_csv_arg(value: str) -> List[str]:
    return [x.strip().lower() for x in str(value).split(",") if x.strip()]


def resolve_chemistry_families(config: TrainEnsembleConfig, df: pd.DataFrame) -> List[str]:
    explicit = split_csv_arg(config.chemistry_families)
    if explicit:
        return explicit
    families = sorted(df["dataset_family"].astype(str).str.lower().unique().tolist())
    return [str(x) for x in families if str(x)]


def filter_family(df: pd.DataFrame, family: str | None) -> pd.DataFrame:
    if family is None:
        return df
    fam = str(family).lower()
    return df[df["dataset_family"].astype(str).str.lower() == fam].copy()


def scope_roots(
    mode: str,
    output_root: Path,
    reports_root: Path,
    family: str | None,
) -> Tuple[Path, Path]:
    if mode == "global":
        return output_root, reports_root
    if mode == "family_specific":
        if family is None:
            raise ValueError("family is required for family_specific mode")
        return (
            output_root / "family_specific" / family,
            reports_root / "family_specific" / family,
        )
    if mode == "shared_plus_heads":
        key = "shared" if family is None else family
        return (
            output_root / "shared_plus_heads" / key,
            reports_root / "shared_plus_heads" / key,
        )
    raise ValueError(f"Unsupported chemistry mode: {mode}")


def build_arrays(df: pd.DataFrame, window_cols: List[str], window_len: int, feature_dim: int) -> Dict[str, np.ndarray]:
    x_window_flat = df[window_cols].to_numpy(dtype=np.float32)
    x_seq = x_window_flat.reshape(-1, window_len, feature_dim).astype(np.float32)
    actions = df["action_behavior"].to_numpy(dtype=np.float32).reshape(-1, 1)
    x_flat_action = np.concatenate([x_window_flat, actions], axis=1).astype(np.float32)
    y = df[["next_soc", "next_voltage", "next_temp", "next_imbalance"]].to_numpy(dtype=np.float32)
    return {"x_seq": x_seq, "actions": actions, "x_flat_action": x_flat_action, "y": y}


def split_by_episode(df: pd.DataFrame, split_ids: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for split_name, ids in split_ids.items():
        ids_set = set(ids)
        out[split_name] = df[df["episode_id"].astype(str).isin(ids_set)].copy()
    return out


def train_quantile_gru(
    model: QuantileGRUModel,
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    config: TrainEnsembleConfig,
    device: torch.device,
) -> Dict[str, float]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_ds = TensorDataset(
        torch.from_numpy(train_data["x_seq"]).float(),
        torch.from_numpy(train_data["actions"]).float(),
        torch.from_numpy(train_data["y"]).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)

    x_val_seq = torch.from_numpy(val_data["x_seq"]).float().to(device)
    x_val_act = torch.from_numpy(val_data["actions"]).float().to(device)
    y_val = torch.from_numpy(val_data["y"]).float().to(device)

    best_state = None
    best_val = float("inf")
    patience = 0

    for _ in range(config.gru_epochs):
        model.train()
        for x_seq_b, a_b, y_b in train_loader:
            x_seq_b = x_seq_b.to(device)
            a_b = a_b.to(device)
            y_b = y_b.to(device)
            out = model(x_seq_b, a_b)
            td = y_b.shape[1]
            q10 = out[:, 0:td]
            q50 = out[:, td : 2 * td]
            q90 = out[:, 2 * td : 3 * td]
            loss = quantile_pinball_loss(y_true=y_b, y_pred_q10=q10, y_pred_q50=q50, y_pred_q90=q90)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(x_val_seq, x_val_act)
            td = y_val.shape[1]
            q10v = out_val[:, 0:td]
            q50v = out_val[:, td : 2 * td]
            q90v = out_val[:, 2 * td : 3 * td]
            val_loss = float(
                quantile_pinball_loss(y_true=y_val, y_pred_q10=q10v, y_pred_q50=q50v, y_pred_q90=q90v).cpu().item()
            )

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"val_pinball_loss": float(best_val)}


def train_quantile_mlp(
    model: QuantileMLPModel,
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    config: TrainEnsembleConfig,
    device: torch.device,
) -> Dict[str, float]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_ds = TensorDataset(
        torch.from_numpy(train_data["x_flat_action"]).float(),
        torch.from_numpy(train_data["y"]).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)

    x_val = torch.from_numpy(val_data["x_flat_action"]).float().to(device)
    y_val = torch.from_numpy(val_data["y"]).float().to(device)

    best_state = None
    best_val = float("inf")
    patience = 0

    for _ in range(config.mlp_epochs):
        model.train()
        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            out = model(x_b)
            td = y_b.shape[1]
            q10 = out[:, 0:td]
            q50 = out[:, td : 2 * td]
            q90 = out[:, 2 * td : 3 * td]
            loss = quantile_pinball_loss(y_true=y_b, y_pred_q10=q10, y_pred_q50=q50, y_pred_q90=q90)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(x_val)
            td = y_val.shape[1]
            q10v = out_val[:, 0:td]
            q50v = out_val[:, td : 2 * td]
            q90v = out_val[:, 2 * td : 3 * td]
            val_loss = float(
                quantile_pinball_loss(y_true=y_val, y_pred_q10=q10v, y_pred_q50=q50v, y_pred_q90=q90v).cpu().item()
            )

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"val_pinball_loss": float(best_val)}


def predict_quantiles_gru(model: QuantileGRUModel, x_seq: np.ndarray, actions: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        out = model(
            torch.from_numpy(x_seq).float().to(device),
            torch.from_numpy(actions).float().to(device),
        ).cpu().numpy()
    td = out.shape[1] // 3
    q10 = out[:, 0:td]
    q50 = out[:, td : 2 * td]
    q90 = out[:, 2 * td : 3 * td]
    q_low = np.minimum(np.minimum(q10, q50), q90)
    q_high = np.maximum(np.maximum(q10, q50), q90)
    return q_low.astype(np.float32), q50.astype(np.float32), q_high.astype(np.float32)


def predict_quantiles_mlp(model: QuantileMLPModel, x_flat_action: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x_flat_action).float().to(device)).cpu().numpy()
    td = out.shape[1] // 3
    q10 = out[:, 0:td]
    q50 = out[:, td : 2 * td]
    q90 = out[:, 2 * td : 3 * td]
    q_low = np.minimum(np.minimum(q10, q50), q90)
    q_high = np.maximum(np.maximum(q10, q50), q90)
    return q_low.astype(np.float32), q50.astype(np.float32), q_high.astype(np.float32)


def rf_predict_with_std(rf: RandomForestRegressor, x_flat_action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = rf.predict(x_flat_action).astype(np.float32)
    tree_preds = np.stack([tree.predict(x_flat_action) for tree in rf.estimators_], axis=0).astype(np.float32)
    std = np.std(tree_preds, axis=0)
    std = np.clip(std, 1e-6, None)
    return mean, std


def fit_gate(
    gate_model: GateNetwork,
    gate_x: np.ndarray,
    gate_target: np.ndarray,
    config: TrainEnsembleConfig,
    device: torch.device,
) -> Dict[str, float]:
    gate_model.train()
    ds = TensorDataset(
        torch.from_numpy(gate_x).float(),
        torch.from_numpy(gate_target).float(),
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    optim = torch.optim.Adam(
        gate_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    for _ in range(config.gate_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = gate_model(xb)
            loss = torch.mean((pred - yb) ** 2)
            optim.zero_grad()
            loss.backward()
            optim.step()
    gate_model.eval()
    with torch.no_grad():
        pred_val = gate_model(torch.from_numpy(gate_x).float().to(device)).cpu().numpy()
    mse = float(np.mean((pred_val - gate_target) ** 2))
    return {"gate_mse": mse}


def make_reliability_plot(
    q10: np.ndarray,
    q90: np.ndarray,
    y_true: np.ndarray,
    title: str,
    output_root: Path,
) -> float:
    coverage = float(np.mean((y_true >= q10) & (y_true <= q90)))
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.bar(["Observed"], [coverage], color="#2A9D8F", alpha=0.9)
    ax.axhline(0.80, linestyle="--", color="#D62828", label="Target 80%")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Interval Coverage")
    ax.set_title(title)
    ax.legend()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_root.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_root.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return coverage


def main() -> None:
    config = parse_args()
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(
        config.dataset_csv,
        dtype={
            "episode_id": "string",
            "objective": "string",
            "dataset_family": "string",
            "dataset_case": "string",
        },
        low_memory=False,
    )
    if df.empty:
        raise SystemExit("Dataset is empty.")
    if config.objective != "all":
        objective_key = str(config.objective).strip().lower()
        df = df[df["objective"].astype(str).str.lower() == objective_key].copy()
        if df.empty:
            raise SystemExit(f"No rows matched objective={config.objective!r}.")
    window_cols = parse_window_cols(df.columns.tolist())
    if not window_cols:
        raise SystemExit("No window_tXX_fYY columns found.")

    with Path(config.split_manifest_json).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    folds = manifest.get("folds", [])
    if not folds:
        raise SystemExit("No folds found in split manifest.")
    if config.fold != "all":
        wanted = int(config.fold)
        folds = [f for f in folds if int(f.get("fold_id", -1)) == wanted]
        if not folds:
            raise SystemExit(f"Fold {wanted} not found.")

    output_root = Path(config.output_root)
    reports_root = Path(config.reports_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    families = resolve_chemistry_families(config=config, df=df)
    init_root = Path(config.init_from_root) if str(config.init_from_root).strip() else None

    scopes: List[Dict[str, Any]] = []
    if config.chemistry_mode == "global":
        model_root, report_root = scope_roots(
            mode="global",
            output_root=output_root,
            reports_root=reports_root,
            family=None,
        )
        scopes.append(
            {
                "mode": "global",
                "family": None,
                "model_root": model_root,
                "report_root": report_root,
                "init_root": init_root,
            }
        )
    elif config.chemistry_mode == "family_specific":
        for family in families:
            model_root, report_root = scope_roots(
                mode="family_specific",
                output_root=output_root,
                reports_root=reports_root,
                family=family,
            )
            scopes.append(
                {
                    "mode": "family_specific",
                    "family": family,
                    "model_root": model_root,
                    "report_root": report_root,
                    "init_root": init_root,
                }
            )
    elif config.chemistry_mode == "shared_plus_heads":
        shared_model_root, shared_report_root = scope_roots(
            mode="shared_plus_heads",
            output_root=output_root,
            reports_root=reports_root,
            family=None,
        )
        scopes.append(
            {
                "mode": "shared_plus_heads",
                "family": None,
                "model_root": shared_model_root,
                "report_root": shared_report_root,
                "init_root": init_root,
            }
        )
        for family in families:
            head_model_root, head_report_root = scope_roots(
                mode="shared_plus_heads",
                output_root=output_root,
                reports_root=reports_root,
                family=family,
            )
            scopes.append(
                {
                    "mode": "shared_plus_heads",
                    "family": family,
                    "model_root": head_model_root,
                    "report_root": head_report_root,
                    "init_root": shared_model_root,
                }
            )
    else:
        raise SystemExit(f"Unsupported chemistry mode: {config.chemistry_mode}")

    all_scope_reports: List[Dict[str, Any]] = []
    for scope in scopes:
        mode = str(scope["mode"])
        family = scope["family"]
        model_root = Path(scope["model_root"])
        report_root = Path(scope["report_root"])
        scope_init_root = scope.get("init_root")
        scope_init_root = Path(scope_init_root) if scope_init_root else None
        model_root.mkdir(parents=True, exist_ok=True)
        report_root.mkdir(parents=True, exist_ok=True)
        fold_reports: List[Dict[str, Any]] = []

        for fold in folds:
            fold_id = int(fold["fold_id"])
            split_data = split_by_episode(df, split_ids=fold["splits"])
            train_df = filter_family(split_data.get("train", pd.DataFrame()), family=family)
            val_df = filter_family(split_data.get("val", pd.DataFrame()), family=family)
            if train_df.empty:
                print(
                    f"Skipping mode={mode}, family={family or 'shared'}, fold={fold_id}: empty train split."
                )
                continue
            if val_df.empty:
                n_val = max(1, int(0.2 * len(train_df)))
                val_df = train_df.sample(n=n_val, random_state=config.random_seed + fold_id).copy()

            train_arr = build_arrays(train_df, window_cols, config.window_len, config.feature_dim)
            val_arr = build_arrays(val_df, window_cols, config.window_len, config.feature_dim)

            saerl_cfg = SAERLConfig(window_len=config.window_len, feature_dim=config.feature_dim)
            predictor: AdaptiveEnsemblePredictor
            init_dir = None
            if scope_init_root is not None:
                init_dir = scope_init_root / f"fold_{fold_id:02d}"
            if init_dir is not None and (init_dir / "metadata.json").exists():
                predictor = AdaptiveEnsemblePredictor.load(directory=init_dir, device=str(device), config_override=saerl_cfg)
            else:
                predictor = AdaptiveEnsemblePredictor(config=saerl_cfg, device=str(device))

            gru_stats = train_quantile_gru(
                model=predictor.gru_model,
                train_data=train_arr,
                val_data=val_arr,
                config=config,
                device=device,
            )
            mlp_stats = train_quantile_mlp(
                model=predictor.mlp_model,
                train_data=train_arr,
                val_data=val_arr,
                config=config,
                device=device,
            )

            rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=5,
                random_state=config.random_seed + fold_id,
                n_jobs=-1,
            )
            rf.fit(train_arr["x_flat_action"], train_arr["y"])
            predictor.rf_model = rf
            predictor.rf_fitted = True

            q10_g, q50_g, q90_g = predict_quantiles_gru(
                model=predictor.gru_model,
                x_seq=val_arr["x_seq"],
                actions=val_arr["actions"],
                device=device,
            )
            q10_m, q50_m, q90_m = predict_quantiles_mlp(
                model=predictor.mlp_model,
                x_flat_action=val_arr["x_flat_action"],
                device=device,
            )
            rf_mean, rf_std = rf_predict_with_std(rf, val_arr["x_flat_action"])
            rf_q10 = rf_mean - 1.64 * rf_std
            rf_q90 = rf_mean + 1.64 * rf_std

            cov_gru = float(np.mean((val_arr["y"] >= q10_g) & (val_arr["y"] <= q90_g)))
            cov_mlp = float(np.mean((val_arr["y"] >= q10_m) & (val_arr["y"] <= q90_m)))
            cov_rf = float(np.mean((val_arr["y"] >= rf_q10) & (val_arr["y"] <= rf_q90)))
            predictor.calibration = {
                "gru": float(np.clip(0.8 / max(cov_gru, 1e-3), 0.4, 3.0)),
                "mlp": float(np.clip(0.8 / max(cov_mlp, 1e-3), 0.4, 3.0)),
                "rf": float(np.clip(0.8 / max(cov_rf, 1e-3), 0.4, 3.0)),
            }

            # Gate labels based on inverse error + uncertainty.
            err_gru = np.mean(np.abs(q50_g - val_arr["y"]), axis=1, keepdims=True)
            err_mlp = np.mean(np.abs(q50_m - val_arr["y"]), axis=1, keepdims=True)
            err_rf = np.mean(np.abs(rf_mean - val_arr["y"]), axis=1, keepdims=True)
            unc_gru = np.mean(np.clip(q90_g - q10_g, 1e-6, None), axis=1, keepdims=True)
            unc_mlp = np.mean(np.clip(q90_m - q10_m, 1e-6, None), axis=1, keepdims=True)
            unc_rf = np.mean(np.clip(rf_std, 1e-6, None), axis=1, keepdims=True)
            inv = 1.0 / np.clip(
                np.concatenate(
                    [
                        err_gru + 0.3 * unc_gru,
                        err_mlp + 0.3 * unc_mlp,
                        err_rf + 0.3 * unc_rf,
                    ],
                    axis=1,
                ),
                1e-6,
                None,
            )
            gate_target = inv / np.sum(inv, axis=1, keepdims=True)
            last_feat = val_arr["x_seq"][:, -1, :]
            gate_x = np.concatenate(
                [
                    last_feat,
                    val_arr["actions"],
                    np.concatenate([unc_gru, unc_mlp, unc_rf], axis=1),
                    np.concatenate([err_gru, err_mlp, err_rf], axis=1),
                ],
                axis=1,
            ).astype(np.float32)
            gate_stats = fit_gate(
                gate_model=predictor.gate_model,
                gate_x=gate_x,
                gate_target=gate_target.astype(np.float32),
                config=config,
                device=device,
            )

            fold_model_dir = model_root / f"fold_{fold_id:02d}"
            predictor.save(fold_model_dir)

            fold_report_dir = report_root / f"fold_{fold_id:02d}"
            fold_report_dir.mkdir(parents=True, exist_ok=True)
            cov_gru_plot = make_reliability_plot(
                q10=q10_g,
                q90=q90_g,
                y_true=val_arr["y"],
                title=f"Fold {fold_id}: GRU 80% Interval Coverage",
                output_root=fold_report_dir / "01_reliability_gru",
            )
            cov_mlp_plot = make_reliability_plot(
                q10=q10_m,
                q90=q90_m,
                y_true=val_arr["y"],
                title=f"Fold {fold_id}: MLP 80% Interval Coverage",
                output_root=fold_report_dir / "02_reliability_mlp",
            )
            cov_rf_plot = make_reliability_plot(
                q10=rf_q10,
                q90=rf_q90,
                y_true=val_arr["y"],
                title=f"Fold {fold_id}: RF 80% Interval Coverage",
                output_root=fold_report_dir / "03_reliability_rf",
            )

            report = {
                "fold_id": fold_id,
                "chemistry_mode": mode,
                "chemistry_family": family if family is not None else "shared",
                "n_train_rows": int(len(train_df)),
                "n_val_rows": int(len(val_df)),
                "model_dir": str(fold_model_dir),
                "init_dir": str(init_dir) if init_dir is not None else "",
                "gru_stats": gru_stats,
                "mlp_stats": mlp_stats,
                "gate_stats": gate_stats,
                "coverage": {
                    "gru": cov_gru_plot,
                    "mlp": cov_mlp_plot,
                    "rf": cov_rf_plot,
                },
                "calibration": predictor.calibration,
                "gate_input_dim": int(predictor.gate_input_dim),
            }
            fold_reports.append(report)
            with (fold_report_dir / "training_report.json").open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, ensure_ascii=True)
            print(
                f"Trained fold {fold_id}: mode={mode}, family={family or 'shared'} -> {fold_model_dir}"
            )

        scope_summary_path = report_root / "summary_reports.json"
        with scope_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": asdict(config),
                    "chemistry_mode": mode,
                    "chemistry_family": family if family is not None else "shared",
                    "fold_reports": fold_reports,
                },
                handle,
                indent=2,
                ensure_ascii=True,
            )
        all_scope_reports.append(
            {
                "chemistry_mode": mode,
                "chemistry_family": family if family is not None else "shared",
                "model_root": str(model_root),
                "report_root": str(report_root),
                "n_fold_reports": int(len(fold_reports)),
            }
        )

    summary_path = reports_root / "summary_reports.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(config),
                "scopes": all_scope_reports,
            },
            handle,
            indent=2,
            ensure_ascii=True,
        )
    print("Completed SAERL ensemble training.")
    print(f"Summary report: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
