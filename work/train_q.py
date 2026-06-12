from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rages_q import (  # noqa: E402
    QConfig,
    QDataset,
    QNetwork,
    compute_input_stats,
    compute_metric_stats,
    fit_platt_calibration,
    q_loss,
    save_q_checkpoint,
)
from rages_scoring import VERIFIER_METRIC_KEYS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2 Q_psi V0.")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/q_data/stage2_q_v0.pth")
    parser.add_argument("--out-path", type=Path, default=ROOT / "model/q_model/q_v0.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-hidden-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--metric-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def indices_for_split(data: Dict[str, torch.Tensor], split_id: int) -> np.ndarray:
    return torch.where(data["split_id"].to(torch.long) == int(split_id))[0].cpu().numpy()


def nonempty_eval_indices(data: Dict[str, torch.Tensor]) -> np.ndarray:
    for split_id in (1, 2, 0):
        idx = indices_for_split(data, split_id)
        if len(idx) > 0:
            return idx
    raise ValueError("Dataset has no rows.")


def make_loader(
    data: Dict[str, torch.Tensor],
    indices: Sequence[int],
    cfg: QConfig,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    metric_mean: torch.Tensor,
    metric_std: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = QDataset(data, indices, cfg, input_mean, input_std, metric_mean, metric_std)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def eval_epoch(
    model: QNetwork,
    loader: DataLoader,
    device: torch.device,
    *,
    metric_weight: float,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {"loss": 0.0, "conv_loss": 0.0, "metric_loss": 0.0}
    n_batches = 0
    with torch.no_grad():
        for x, y_conv, metrics, metric_mask in loader:
            x = x.to(device)
            y_conv = y_conv.to(device)
            metrics = metrics.to(device)
            metric_mask = metric_mask.to(device)
            loss, parts = q_loss(
                model(x),
                y_conv,
                metrics,
                metric_mask,
                metric_weight=metric_weight,
            )
            _ = loss
            for key in totals:
                totals[key] += parts[key]
            n_batches += 1
    model.train()
    if n_batches == 0:
        return {key: float("nan") for key in totals}
    return {key: val / n_batches for key, val in totals.items()}


def collect_logits(
    model: QNetwork,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for x, y_conv, _, _ in loader:
            out = model(x.to(device))
            logits.append(out["conv_logit"].detach().cpu())
            labels.append(y_conv.detach().cpu())
    model.train()
    if not logits:
        return torch.empty(0), torch.empty(0)
    return torch.cat(logits, dim=0), torch.cat(labels, dim=0)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torch.load(args.data_path, map_location="cpu")
    data = dataset["data"]
    meta = dataset.get("meta", {})
    train_idx = indices_for_split(data, 0)
    if len(train_idx) == 0:
        train_idx = np.arange(int(data["x0"].shape[0]), dtype=int)
    eval_idx = nonempty_eval_indices(data)

    max_phase = int(meta.get("max_phase", data["b_seq"].shape[1]))
    b_seq_num_classes = int(meta.get("b_seq_num_classes", 11))
    cfg = QConfig(
        input_dim=1,
        n_metrics=int(data["metrics"].shape[1]),
        max_phase=max_phase,
        hidden_dim=int(args.hidden_dim),
        n_hidden_layers=int(args.n_hidden_layers),
        dropout=float(args.dropout),
        b_seq_num_classes=b_seq_num_classes,
    )
    input_mean, input_std = compute_input_stats(data, train_idx, cfg)
    cfg.input_dim = int(input_mean.numel())
    metric_mean, metric_std = compute_metric_stats(data, train_idx)

    train_loader = make_loader(
        data,
        train_idx,
        cfg,
        input_mean,
        input_std,
        metric_mean,
        metric_std,
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = make_loader(
        data,
        eval_idx,
        cfg,
        input_mean,
        input_std,
        metric_mean,
        metric_std,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = QNetwork(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    for epoch in range(int(args.epochs)):
        model.train()
        batch_losses: List[float] = []
        for x, y_conv, metrics, metric_mask in train_loader:
            x = x.to(device)
            y_conv = y_conv.to(device)
            metrics = metrics.to(device)
            metric_mask = metric_mask.to(device)
            loss, _ = q_loss(
                model(x),
                y_conv,
                metrics,
                metric_mask,
                metric_weight=float(args.metric_weight),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        if epoch == 0 or epoch == int(args.epochs) - 1 or (epoch + 1) % 10 == 0:
            eval_parts = eval_epoch(
                model,
                eval_loader,
                device,
                metric_weight=float(args.metric_weight),
            )
            print(
                f"epoch={epoch + 1} train_loss={np.mean(batch_losses):.4f} "
                f"eval_loss={eval_parts['loss']:.4f} "
                f"eval_conv={eval_parts['conv_loss']:.4f} "
                f"eval_metric={eval_parts['metric_loss']:.4f}"
            )

    logits, labels = collect_logits(model, eval_loader, device)
    calibration = fit_platt_calibration(logits, labels)
    save_q_checkpoint(
        args.out_path,
        model,
        cfg,
        input_mean=input_mean,
        input_std=input_std,
        metric_mean=metric_mean,
        metric_std=metric_std,
        metric_keys=meta.get("metric_keys") or VERIFIER_METRIC_KEYS,
        calibration=calibration,
        meta={
            "source_data_path": str(args.data_path),
            "source_data_meta": meta,
            "epochs": int(args.epochs),
            "metric_weight": float(args.metric_weight),
            "seed": int(args.seed),
        },
    )
    print(f"saved {args.out_path}")
    print(f"calibration={calibration}")


if __name__ == "__main__":
    main()
