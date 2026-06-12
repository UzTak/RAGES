from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from parameters import DEFAULT_B_SEQ_ENCODING, DEFAULT_B_SEQ_NUM_CLASSES
from rages_scoring import VERIFIER_METRIC_KEYS
from wyp_predictor import build_input_from_data, build_input_slices


DEFAULT_Q_INPUTS: Tuple[str, ...] = (
    "x0",
    "tof",
    "oec0_modified",
    "artms_scale_range_1e3",
    "koz_dim",
    "b_seq",
)


@dataclass
class QConfig:
    input_dim: int
    n_metrics: int
    max_phase: int = 3
    hidden_dim: int = 256
    n_hidden_layers: int = 3
    dropout: float = 0.0
    b_seq_encoding: str = DEFAULT_B_SEQ_ENCODING
    b_seq_num_classes: int = DEFAULT_B_SEQ_NUM_CLASSES


@dataclass(frozen=True)
class QOutput:
    p_conv: float
    metric_means: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "p_conv": float(self.p_conv),
            "metric_means": dict(self.metric_means),
        }


class QNetwork(nn.Module):
    def __init__(self, cfg: QConfig):
        super().__init__()
        layers: List[nn.Module] = []
        d = int(cfg.input_dim)
        for _ in range(int(cfg.n_hidden_layers)):
            layers.append(nn.Linear(d, int(cfg.hidden_dim)))
            layers.append(nn.GELU())
            if float(cfg.dropout) > 0.0:
                layers.append(nn.Dropout(float(cfg.dropout)))
            d = int(cfg.hidden_dim)
        self.backbone = nn.Sequential(*layers)
        self.conv_head = nn.Linear(d, 1)
        self.metric_head = nn.Linear(d, int(cfg.n_metrics))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "conv_logit": self.conv_head(h).squeeze(-1),
            "metric_scaled": self.metric_head(h),
        }


def _as_index_tensor(indices: Sequence[int]) -> torch.Tensor:
    return torch.as_tensor(list(indices), dtype=torch.long)


def split_indices(data: Mapping[str, torch.Tensor], split_id: int) -> np.ndarray:
    split = data.get("split_id")
    if split is None:
        return np.arange(int(data["x0"].shape[0]), dtype=int)
    return torch.where(split.to(torch.long) == int(split_id))[0].cpu().numpy()


def input_slices_for_data(data: Mapping[str, torch.Tensor], cfg: QConfig) -> Dict[str, slice]:
    return build_input_slices(
        dict(data),
        list(DEFAULT_Q_INPUTS),
        b_seq_encoding=cfg.b_seq_encoding,
        b_seq_num_classes=cfg.b_seq_num_classes,
    )


def build_q_input(
    data: Mapping[str, torch.Tensor],
    idx: int,
    cfg: QConfig,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
) -> torch.Tensor:
    x = build_input_from_data(
        dict(data),
        int(idx),
        list(DEFAULT_Q_INPUTS),
        b_seq_encoding=cfg.b_seq_encoding,
        b_seq_num_classes=cfg.b_seq_num_classes,
    )
    return (x - input_mean) / input_std.clamp_min(1e-6)


def compute_input_stats(
    data: Mapping[str, torch.Tensor],
    indices: Sequence[int],
    cfg: QConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = list(int(i) for i in indices)
    if not idx:
        raise ValueError("Cannot compute Q input stats on an empty index set.")
    xs = torch.stack(
        [
            build_input_from_data(
                dict(data),
                i,
                list(DEFAULT_Q_INPUTS),
                b_seq_encoding=cfg.b_seq_encoding,
                b_seq_num_classes=cfg.b_seq_num_classes,
            )
            for i in idx
        ],
        dim=0,
    )
    mean = xs.mean(dim=0)
    std = xs.std(dim=0, unbiased=False).clamp_min(1e-6)
    if cfg.b_seq_encoding == "one_hot":
        slices = input_slices_for_data(data, cfg)
        b_slice = slices["b_seq"]
        mean[b_slice] = 0.0
        std[b_slice] = 1.0
    return mean, std


def compute_metric_stats(
    data: Mapping[str, torch.Tensor],
    indices: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = _as_index_tensor(indices)
    metrics = data["metrics"][idx].to(torch.float32)
    converged = data["converged"][idx].to(torch.bool)
    usable = metrics[converged]
    if usable.numel() == 0:
        n_metrics = int(metrics.shape[-1])
        return torch.zeros(n_metrics), torch.ones(n_metrics)
    mean = torch.nanmean(usable, dim=0)
    filled = torch.where(torch.isfinite(usable), usable, mean)
    std = filled.std(dim=0, unbiased=False).clamp_min(1e-6)
    mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
    std = torch.where(torch.isfinite(std), std, torch.ones_like(std))
    return mean, std


class QDataset(Dataset):
    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        indices: Sequence[int],
        cfg: QConfig,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        metric_mean: torch.Tensor,
        metric_std: torch.Tensor,
    ) -> None:
        self.data = data
        self.indices = list(int(i) for i in indices)
        self.cfg = cfg
        self.input_mean = input_mean.to(torch.float32)
        self.input_std = input_std.to(torch.float32)
        self.metric_mean = metric_mean.to(torch.float32)
        self.metric_std = metric_std.to(torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, row: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.indices[int(row)]
        x = build_q_input(self.data, idx, self.cfg, self.input_mean, self.input_std)
        y_conv = self.data["converged"][idx].to(torch.float32)
        metrics = self.data["metrics"][idx].to(torch.float32)
        metric_mask = (y_conv > 0.5) & torch.isfinite(metrics)
        metric_scaled = (metrics - self.metric_mean) / self.metric_std.clamp_min(1e-6)
        metric_scaled = torch.where(torch.isfinite(metric_scaled), metric_scaled, torch.zeros_like(metric_scaled))
        return x, y_conv, metric_scaled, metric_mask.to(torch.float32)


def q_loss(
    out: Mapping[str, torch.Tensor],
    y_conv: torch.Tensor,
    metric_scaled: torch.Tensor,
    metric_mask: torch.Tensor,
    *,
    metric_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    conv_loss = nn.functional.binary_cross_entropy_with_logits(out["conv_logit"], y_conv)
    denom = metric_mask.sum().clamp_min(1.0)
    metric_loss = (((out["metric_scaled"] - metric_scaled) ** 2) * metric_mask).sum() / denom
    loss = conv_loss + float(metric_weight) * metric_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "conv_loss": float(conv_loss.detach().cpu()),
        "metric_loss": float(metric_loss.detach().cpu()),
    }


def fit_platt_calibration(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    logits = logits.detach().to(torch.float32).reshape(-1)
    labels = labels.detach().to(torch.float32).reshape(-1)
    if logits.numel() == 0 or torch.unique(labels).numel() < 2:
        return {"scale": 1.0, "bias": 0.0}

    scale = torch.tensor(1.0, requires_grad=True)
    bias = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([scale, bias], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(scale * logits + bias, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "scale": float(scale.detach().clamp(-20.0, 20.0).cpu()),
        "bias": float(bias.detach().clamp(-20.0, 20.0).cpu()),
    }


def save_q_checkpoint(
    path: Path,
    model: QNetwork,
    cfg: QConfig,
    *,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    metric_mean: torch.Tensor,
    metric_std: torch.Tensor,
    metric_keys: Sequence[str] = VERIFIER_METRIC_KEYS,
    calibration: Optional[Mapping[str, float]] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "cfg": asdict(cfg),
        "inputs_arg": list(DEFAULT_Q_INPUTS),
        "input_mean": input_mean.detach().cpu(),
        "input_std": input_std.detach().cpu(),
        "metric_mean": metric_mean.detach().cpu(),
        "metric_std": metric_std.detach().cpu(),
        "metric_keys": list(metric_keys),
        "calibration": dict(calibration or {"scale": 1.0, "bias": 0.0}),
        "meta": dict(meta or {}),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_q_checkpoint(path: Path, *, device: Optional[torch.device] = None) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location="cpu")
    cfg = QConfig(**dict(ckpt["cfg"]))
    model = QNetwork(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return {
        "cfg": cfg,
        "model": model,
        "device": device,
        "input_mean": ckpt["input_mean"].to(device),
        "input_std": ckpt["input_std"].to(device),
        "metric_mean": ckpt["metric_mean"].to(device),
        "metric_std": ckpt["metric_std"].to(device),
        "metric_keys": list(ckpt.get("metric_keys", VERIFIER_METRIC_KEYS)),
        "calibration": dict(ckpt.get("calibration", {"scale": 1.0, "bias": 0.0})),
        "meta": dict(ckpt.get("meta", {})),
    }


@torch.no_grad()
def predict_q_tensors(
    bundle: Mapping[str, Any],
    data: Mapping[str, torch.Tensor],
    indices: Optional[Sequence[int]] = None,
    *,
    batch_size: int = 1024,
) -> Dict[str, torch.Tensor]:
    cfg: QConfig = bundle["cfg"]
    model: QNetwork = bundle["model"]
    device: torch.device = bundle["device"]
    input_mean = bundle["input_mean"].detach().cpu()
    input_std = bundle["input_std"].detach().cpu()
    metric_mean = bundle["metric_mean"].to(device)
    metric_std = bundle["metric_std"].to(device)
    calibration = bundle.get("calibration", {"scale": 1.0, "bias": 0.0})
    scale = float(calibration.get("scale", 1.0))
    bias = float(calibration.get("bias", 0.0))

    idx = list(range(int(data["x0"].shape[0]))) if indices is None else list(int(i) for i in indices)
    p_chunks: List[torch.Tensor] = []
    metric_chunks: List[torch.Tensor] = []
    logit_chunks: List[torch.Tensor] = []

    for start in range(0, len(idx), int(batch_size)):
        rows = idx[start : start + int(batch_size)]
        x = torch.stack(
            [build_q_input(data, i, cfg, input_mean, input_std) for i in rows],
            dim=0,
        ).to(device)
        out = model(x)
        logits = out["conv_logit"]
        p_chunks.append(torch.sigmoid(scale * logits + bias).detach().cpu())
        logit_chunks.append(logits.detach().cpu())
        metrics = out["metric_scaled"] * metric_std + metric_mean
        metric_chunks.append(metrics.detach().cpu())

    return {
        "p_conv": torch.cat(p_chunks, dim=0) if p_chunks else torch.empty(0),
        "conv_logit": torch.cat(logit_chunks, dim=0) if logit_chunks else torch.empty(0),
        "metric_means": torch.cat(metric_chunks, dim=0) if metric_chunks else torch.empty(0),
    }


def tensor_outputs_to_q(
    p_conv: torch.Tensor,
    metric_means: torch.Tensor,
    metric_keys: Sequence[str] = VERIFIER_METRIC_KEYS,
) -> List[QOutput]:
    outs: List[QOutput] = []
    for i in range(int(p_conv.shape[0])):
        outs.append(
            QOutput(
                p_conv=float(p_conv[i].item()),
                metric_means={
                    str(k): float(metric_means[i, j].item())
                    for j, k in enumerate(metric_keys)
                },
            )
        )
    return outs
