from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rages_q import load_q_checkpoint, predict_q_tensors, tensor_outputs_to_q  # noqa: E402
from rages_sampling import enumerate_priority_profiles  # noqa: E402
from rages_scoring import (  # noqa: E402
    LexicographicPreference,
    VERIFIER_METRIC_KEYS,
    rank_candidates,
    rank_q_candidates,
    select_best_q,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Stage 2 Q fixture.")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/q_data/stage2_q_v0.pth")
    parser.add_argument("--q-ckpt", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=ROOT / "data/q_data/q_v0_report.json")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--p-conv-threshold", type=float, default=0.5)
    parser.add_argument("--ece-bins", type=int, default=10)
    return parser.parse_args()


def split_indices(data: Mapping[str, torch.Tensor], split: str) -> np.ndarray:
    if split == "all" or "split_id" not in data:
        return np.arange(int(data["x0"].shape[0]), dtype=int)
    split_id = {"train": 0, "val": 1, "test": 2}[split]
    idx = torch.where(data["split_id"].to(torch.long) == split_id)[0].cpu().numpy()
    if len(idx) == 0 and split == "test":
        idx = torch.where(data["split_id"].to(torch.long) == 1)[0].cpu().numpy()
    if len(idx) == 0:
        idx = np.arange(int(data["x0"].shape[0]), dtype=int)
    return idx


def ece_score(p: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    if len(p) == 0:
        return float("nan")
    total = float(len(p))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        ece += float(np.sum(mask)) / total * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return float(ece)


def convergence_metrics(p: np.ndarray, y: np.ndarray, *, threshold: float, n_bins: int) -> Dict[str, float]:
    p = np.clip(p.astype(float), 1e-6, 1.0 - 1e-6)
    y = y.astype(float)
    pred_pos = p >= float(threshold)
    fp = pred_pos & (y < 0.5)
    return {
        "ece": ece_score(p, y, n_bins),
        "brier": float(np.mean((p - y) ** 2)) if len(p) else float("nan"),
        "nll": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))) if len(p) else float("nan"),
        "false_positive_rate": float(np.sum(fp) / max(1, np.sum(pred_pos))),
        "predicted_feasible_count": int(np.sum(pred_pos)),
        "false_positive_count": int(np.sum(fp)),
    }


def metric_errors(
    pred_metrics: np.ndarray,
    true_metrics: np.ndarray,
    converged: np.ndarray,
    metric_keys: Sequence[str],
) -> Dict[str, Any]:
    mask = converged.astype(bool) & np.all(np.isfinite(true_metrics), axis=1)
    per_metric: Dict[str, Dict[str, float]] = {}
    if not np.any(mask):
        for key in metric_keys:
            per_metric[str(key)] = {"mae": float("nan"), "rmse": float("nan")}
        return {"num_converged_eval": 0, "per_metric": per_metric, "mean_mae": float("nan"), "mean_rmse": float("nan")}

    abs_errs = []
    rmses = []
    for j, key in enumerate(metric_keys):
        err = pred_metrics[mask, j] - true_metrics[mask, j]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        per_metric[str(key)] = {"mae": mae, "rmse": rmse}
        abs_errs.append(mae)
        rmses.append(rmse)
    return {
        "num_converged_eval": int(np.sum(mask)),
        "per_metric": per_metric,
        "mean_mae": float(np.mean(abs_errs)),
        "mean_rmse": float(np.mean(rmses)),
    }


def metric_row(vals: np.ndarray, metric_keys: Sequence[str]) -> Dict[str, float]:
    return {str(k): float(vals[j]) for j, k in enumerate(metric_keys)}


def ranking_metrics(
    data: Mapping[str, torch.Tensor],
    indices: Sequence[int],
    p_conv: np.ndarray,
    pred_metrics: np.ndarray,
    *,
    metric_keys: Sequence[str],
    threshold: float,
) -> Dict[str, Any]:
    scenario_ids = data["scenario_id"].cpu().numpy()
    converged = data["converged"].cpu().numpy() > 0.5
    true_metrics = data["metrics"].cpu().numpy()
    selected = list(int(i) for i in indices)
    by_scenario: Dict[int, List[int]] = {}
    for pos, idx in enumerate(selected):
        by_scenario.setdefault(int(scenario_ids[idx]), []).append(pos)

    taus: List[float] = []
    regrets: List[float] = []
    groups_used = 0
    priorities = enumerate_priority_profiles()
    for positions in by_scenario.values():
        if len(positions) < 2:
            continue
        true_rows = [metric_row(true_metrics[selected[pos]], metric_keys) for pos in positions]
        true_feasible = [bool(converged[selected[pos]]) for pos in positions]
        if not any(true_feasible):
            continue
        q_outputs = tensor_outputs_to_q(
            torch.as_tensor(p_conv[positions], dtype=torch.float32),
            torch.as_tensor(pred_metrics[positions], dtype=torch.float32),
            metric_keys,
        )
        groups_used += 1
        for priority in priorities:
            preference = LexicographicPreference(
                priority=tuple(priority),
                p_conv_threshold=float(threshold),
            )
            true_ranks = rank_candidates(true_rows, preference, feasible=true_feasible)
            pred_ranks = rank_q_candidates(q_outputs, preference)
            if len(set(true_ranks)) > 1 and len(set(pred_ranks)) > 1:
                tau = kendalltau(true_ranks, pred_ranks).statistic
                if np.isfinite(tau):
                    taus.append(float(tau))

            pred_best = select_best_q(q_outputs, preference)
            if pred_best is None:
                regrets.append(float(max(true_ranks) + 1))
            else:
                regrets.append(float(true_ranks[int(pred_best)]))

    return {
        "groups_used": int(groups_used),
        "priority_profiles": int(len(priorities)),
        "kendall_tau_mean": float(np.mean(taus)) if taus else float("nan"),
        "kendall_tau_count": int(len(taus)),
        "top1_dense_rank_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "top1_regret_count": int(len(regrets)),
    }


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main() -> None:
    args = parse_args()
    dataset = torch.load(args.data_path, map_location="cpu")
    data = dataset["data"]
    meta = dataset.get("meta", {})
    idx = split_indices(data, args.split)
    bundle = load_q_checkpoint(args.q_ckpt, device=torch.device("cpu"))
    pred = predict_q_tensors(bundle, data, idx, batch_size=args.batch_size)

    p_conv = pred["p_conv"].cpu().numpy()
    pred_metrics = pred["metric_means"].cpu().numpy()
    true_conv = data["converged"][idx].cpu().numpy()
    true_metrics = data["metrics"][idx].cpu().numpy()
    metric_keys = list(bundle.get("metric_keys") or meta.get("metric_keys") or VERIFIER_METRIC_KEYS)

    report = {
        "data_path": str(args.data_path),
        "q_ckpt": str(args.q_ckpt),
        "split": args.split,
        "num_rows": int(len(idx)),
        "convergence": convergence_metrics(
            p_conv,
            true_conv,
            threshold=float(args.p_conv_threshold),
            n_bins=int(args.ece_bins),
        ),
        "metrics": metric_errors(pred_metrics, true_metrics, true_conv > 0.5, metric_keys),
        "ranking": ranking_metrics(
            data,
            idx,
            p_conv,
            pred_metrics,
            metric_keys=metric_keys,
            threshold=float(args.p_conv_threshold),
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(jsonable(report), f, indent=2, allow_nan=True)
    print(json.dumps(jsonable(report), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
