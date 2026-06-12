from __future__ import annotations

"""
Waypoint generation model (`p_phi`) library.

Code moved verbatim from `work/train_wyp_predictor.py` (config, featurization,
scaling/stats, model classes, decoding, dataset) and from
`work/datagen_reasoning.py` (checkpoint loading and waypoint inference), so the
training/datagen scripts only keep their script-specific logic.

Model classes share an informal protocol used by training and inference:

    forward(x_in, y, m) -> out dict
    compute_loss(y_true, m_missing, out, weights, phase_valid) -> loss
    sample_y(x_in, y_in, m_known, cfg, phase_valid, ...) -> y (scaled)

followed by `constrained_fill` to combine known values and enforce the dt
simplex. The future flow-matching model should implement the same protocol.
"""

from dataclasses import MISSING, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from parameters import DEFAULT_B_SEQ_ENCODING, DEFAULT_B_SEQ_NUM_CLASSES


@dataclass
class FillerConfig:
    max_phase: int                 # maximum number of behavior phases
    input_dim: int                 # concatenated input dimension
    hidden_dim: int = 256
    n_hidden_layers: int = 3
    dropout: float = 0.0
    K: int = 3                          # number of mixture components
    latent_dim: int = 16                # VAE latent dimension
    kl_beta: float = 1e-3               # VAE KL weight
    b_seq_encoding: str = "one_hot"      # "scalar" or "one_hot"
    b_seq_num_classes: int = 11         # used when b_seq_encoding == "one_hot"

def _encode_b_seq(
    b_seq: torch.Tensor,
    *,
    encoding: str,
    num_classes: int,
) -> torch.Tensor:
    b_seq = b_seq.reshape(-1).to(torch.long)
    if encoding == "scalar":
        return b_seq.to(torch.float32)
    if encoding == "one_hot":
        max_phase = int(b_seq.shape[0])
        out = torch.zeros((max_phase, int(num_classes)), dtype=torch.float32, device=b_seq.device)
        valid = (b_seq >= 1) & (b_seq <= int(num_classes))
        if torch.any(valid):
            rows = torch.where(valid)[0]
            cols = b_seq[valid] - 1
            out[rows, cols] = 1.0
        return out.reshape(-1)
    raise ValueError(f"Unsupported b_seq encoding: {encoding}")


def _expand_phase_valid_for_b_seq(phase_valid: torch.Tensor, b_seq_width: int, max_phase: int) -> torch.Tensor:
    if int(b_seq_width) == int(max_phase):
        return phase_valid
    if int(b_seq_width) % int(max_phase) != 0:
        raise ValueError(
            f"b_seq input width {b_seq_width} is not divisible by max_phase={max_phase}."
        )
    rep = int(b_seq_width) // int(max_phase)
    return phase_valid.repeat_interleave(rep, dim=-1)


def _input_field_size(
    data: Dict[str, torch.Tensor],
    name: str,
    *,
    b_seq_encoding: str = "scalar",
    b_seq_num_classes: int = 11,
) -> int:
    if name == "tof":
        return 1
    if name not in data:
        raise KeyError(f"Missing input field in data: {name}")
    t = data[name]
    if name == "b_seq":
        if t.ndim != 2:
            raise ValueError(f"Unsupported b_seq shape: {t.shape}")
        if b_seq_encoding == "scalar":
            return int(t.shape[1])
        if b_seq_encoding == "one_hot":
            return int(t.shape[1]) * int(b_seq_num_classes)
        raise ValueError(f"Unsupported b_seq encoding: {b_seq_encoding}")
    if t.ndim == 1:
        return 1
    if t.ndim == 2:
        return t.shape[1]
    if t.ndim == 3:
        return t.shape[1] * t.shape[2]
    raise ValueError(f"Unsupported tensor rank for input field {name}: {t.shape}")


def build_input_from_data(
    data: Dict[str, torch.Tensor],
    idx: int,
    inputs_arg: List[str],
    b_seq_encoding: str = "scalar",
    b_seq_num_classes: int = 11,
) -> torch.Tensor:
    parts = []
    for name in inputs_arg:
        key = name
        if key == "tof":
            tof = data["tof"][idx].reshape(1)
            parts.append(tof)
            continue
        if key == "b_seq":
            parts.append(
                _encode_b_seq(
                    data["b_seq"][idx],
                    encoding=b_seq_encoding,
                    num_classes=b_seq_num_classes,
                )
            )
            continue
        if key == "x_seq":
            parts.append(data["x_seq"][idx].reshape(-1).to(torch.float32))
            continue
        parts.append(data[key][idx].reshape(-1).to(torch.float32))
    return torch.cat(parts, dim=-1)


def build_input_slices(
    data: Dict[str, torch.Tensor],
    inputs_arg: List[str],
    *,
    b_seq_encoding: str = "scalar",
    b_seq_num_classes: int = 11,
) -> Dict[str, slice]:
    slices: Dict[str, slice] = {}
    offset = 0
    for name in inputs_arg:
        key = name
        if key in slices:
            raise ValueError(f"Duplicate input field: {key}")
        size = _input_field_size(
            data,
            key,
            b_seq_encoding=b_seq_encoding,
            b_seq_num_classes=b_seq_num_classes,
        )
        slices[key] = slice(offset, offset + size)
        offset += size
    return slices


def build_cond_input(x_in: torch.Tensor, y: Optional[torch.Tensor], m: Optional[torch.Tensor], cfg) -> torch.Tensor:
    B = x_in.shape[0]
    device = x_in.device
    if y is None:
        y = torch.zeros((B, cfg.max_phase * 6 + cfg.max_phase), device=device)
    if m is None:
        m = torch.zeros((B, cfg.max_phase * 6 + cfg.max_phase), device=device)
    return torch.cat([x_in, y, m], dim=-1)

def _masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    if mask is None:
        return torch.softmax(logits, dim=dim)
    mask = mask.to(dtype=torch.bool)
    neg = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask, neg)
    # If all entries are masked, return zeros to avoid NaNs
    all_masked = (~mask).all(dim=dim, keepdim=True)
    out = torch.softmax(masked_logits, dim=dim)
    return torch.where(all_masked, torch.zeros_like(out), out)


def scale_var(y, mean, std):
    return (y - mean) / std


def unscale_var(y, mean, std):
    return y * std + mean


def build_y_from_data(data: Dict[str, torch.Tensor], idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x_seq = data["x_seq"][idx]
    dt_seq = data["dt_seq"][idx]
    phase_valid = data["phase_valid"][idx]
    y = torch.cat([x_seq.reshape(-1), dt_seq], dim=-1)
    return y, phase_valid


def build_y_stats_from_stats(stats: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    x_mean = stats["x_seq"]["mean"].reshape(-1)
    x_std = stats["x_seq"]["std"].reshape(-1)
    max_phase = int(stats["x_seq"]["mean"].shape[0])
    dt_mean = torch.zeros(max_phase, dtype=x_mean.dtype)
    dt_std = torch.ones(max_phase, dtype=x_std.dtype)
    y_mean = torch.cat([x_mean, dt_mean], dim=-1)
    y_std = torch.cat([x_std, dt_std], dim=-1)
    return y_mean, y_std


def build_X_stats_from_stats(
    stats: Dict[str, Dict[str, torch.Tensor]],
    inputs_arg: List[str],
    *,
    b_seq_encoding: str = "scalar",
    b_seq_num_classes: int = 11,
) -> Tuple[torch.Tensor, torch.Tensor]:
    means = []
    stds = []
    for key in inputs_arg:
        if key == "b_seq" and b_seq_encoding == "one_hot":
            max_phase = int(stats["b_seq"]["mean"].numel())
            dtype = stats["b_seq"]["mean"].dtype
            mean = torch.zeros(max_phase * int(b_seq_num_classes), dtype=dtype)
            std = torch.ones(max_phase * int(b_seq_num_classes), dtype=dtype)
        else:
            mean = stats[key]["mean"].reshape(-1)
            std = stats[key]["std"].reshape(-1)
        means.append(mean)
        stds.append(std)
    X_mean = torch.cat(means, dim=-1)
    X_std = torch.cat(stds, dim=-1)
    return X_mean, X_std


class ConditionalGMM(nn.Module):
    """
    Masked conditional MLP for blank filling.

    Inputs:
      x_in: (B, input_dim) concatenated inputs defined by inputs_arg
      y:    (B, y_dim) provided optional values with zeros in missing entries (can be None at inference)
      m:    (B, y_dim) mask: 1 if provided/known, 0 if missing (can be None -> assume all missing)

    Output:
      y_mean: (B, y_dim)
      y_std:  (B, y_dim) if predict_std else None
    """
    def __init__(self, cfg: FillerConfig):
        super().__init__()
        # y = [x1..xN, dt1..dtN], where N = max_phase
        self.y_dim = cfg.max_phase * 6 + (cfg.max_phase * 1)  # e.g., 3 phases => 18 + 3 = 21
        self.cfg = cfg

        in_dim = cfg.input_dim
        in_dim += self.y_dim     # y values (with zeros for missing)
        in_dim += self.y_dim     # y mask

        layers = []
        d = in_dim
        for _ in range(cfg.n_hidden_layers):
            layers += [nn.Linear(d, cfg.hidden_dim), nn.GELU()]
            if cfg.dropout > 0:
                layers += [nn.Dropout(cfg.dropout)]
            d = cfg.hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.head_logits = nn.Linear(cfg.hidden_dim, cfg.K)           # mixture weights
        self.head_mean   = nn.Linear(cfg.hidden_dim, cfg.K * self.y_dim)   # component means
        self.head_logstd = nn.Linear(cfg.hidden_dim, cfg.K * self.y_dim)   # (diag) stds

        # small init helps stability
        nn.init.zeros_(self.head_mean.bias)
        if self.head_logstd is not None:
            nn.init.constant_(self.head_logstd.bias, -1.0)

    def forward(self, x_in, y=None, m=None):
        B = x_in.shape[0]
        z = build_cond_input(x_in, y, m, self.cfg)

        # backbone
        h = self.backbone(z)                  # (B, hidden_dim)
        pi = torch.softmax(self.head_logits(h), dim=-1)     # (B, K)
        mu = self.head_mean(h).view(B, self.cfg.K, self.y_dim)  # (B, K, y_dim)

        logstd = self.head_logstd(h).view(B, self.cfg.K, self.y_dim)
        # Prevent extremely small std from producing negative NLL (pdf > 1).
        logstd = torch.clamp(logstd, -0.9, 4.0)
        std = torch.exp(logstd)

        return {
            "pi": pi,
            "mu": mu,
            "std": std,
        }

    def compute_loss(self, y_true, m_missing, out, weights=None, phase_valid=None):
        pi = out["pi"]
        mu = out["mu"]
        std = out["std"]
        w_split = self.cfg.max_phase * 6
        # Waypoint Gaussian loss
        loss_w = masked_mdn_nll(
            pi,
            mu[:, :, :w_split],
            std[:, :, :w_split],
            y_true[:, :w_split],
            m_missing[:, :w_split],
            weights=weights,
        )

        # DT simplex loss (deterministic softmax over logits)
        dt_logits = (pi.unsqueeze(-1) * mu[:, :, w_split:]).sum(dim=1)
        dt_mask = m_missing[:, w_split:]
        dt_valid = phase_valid if phase_valid is not None else (dt_mask > 0)
        dt_pred = _masked_softmax(dt_logits, dt_valid, dim=-1)
        dt_true = y_true[:, w_split:]

        diff = (dt_pred - dt_true) * dt_mask
        denom = dt_mask.sum(dim=-1).clamp_min(1.0)
        per_sample = (diff ** 2).sum(dim=-1) / denom
        if weights is None:
            loss_dt = per_sample.mean()
        else:
            w = weights.view(-1)
            w = w / w.sum().clamp_min(1e-8)
            loss_dt = (per_sample * w).sum()

        return loss_w + loss_dt

    @torch.no_grad()
    def sample_y(
        self,
        x_in: torch.Tensor,
        y_in: torch.Tensor,
        m_known: torch.Tensor,
        cfg: FillerConfig,
        phase_valid: Optional[torch.Tensor] = None,
        dt_mode: str = "dirichlet_sample",
        use_mean_w: bool = False,
    ) -> torch.Tensor:
        out = self.forward(x_in, y=y_in, m=m_known)
        pi = out["pi"]
        mu = out["mu"]
        std = out["std"]
        B, K, D = mu.shape
        cat = torch.distributions.Categorical(probs=pi)
        k = cat.sample()
        mu_k = mu[torch.arange(B), k]
        std_k = std[torch.arange(B), k]
        if use_mean_w:
            return mu_k
        eps = torch.randn_like(std_k)
        return mu_k + std_k * eps


class ConditionalVAE(nn.Module):
    """
    Conditional VAE for masked waypoint filling.
    Waypoints use a Gaussian decoder; dt uses a Dirichlet simplex decoder.
    """
    def __init__(self, cfg: FillerConfig):
        super().__init__()
        self.cfg = cfg
        self.w_dim = cfg.max_phase * 6
        self.dt_dim = cfg.max_phase
        self.y_dim = self.w_dim + self.dt_dim

        in_dim = cfg.input_dim
        in_dim += self.y_dim
        in_dim += self.y_dim

        enc_layers = []
        d = in_dim
        for _ in range(cfg.n_hidden_layers):
            enc_layers += [nn.Linear(d, cfg.hidden_dim), nn.GELU()]
            if cfg.dropout > 0:
                enc_layers += [nn.Dropout(cfg.dropout)]
            d = cfg.hidden_dim
        self.encoder = nn.Sequential(*enc_layers)
        self.head_z_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.head_z_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        dec_layers = []
        d = in_dim + cfg.latent_dim
        for _ in range(cfg.n_hidden_layers):
            dec_layers += [nn.Linear(d, cfg.hidden_dim), nn.GELU()]
            if cfg.dropout > 0:
                dec_layers += [nn.Dropout(cfg.dropout)]
            d = cfg.hidden_dim
        self.decoder = nn.Sequential(*dec_layers)
        self.head_w_mean = nn.Linear(cfg.hidden_dim, self.w_dim)
        self.head_w_logstd = nn.Linear(cfg.hidden_dim, self.w_dim)
        self.head_dt_alpha = nn.Linear(cfg.hidden_dim, self.dt_dim)

        nn.init.zeros_(self.head_w_mean.bias)
        nn.init.constant_(self.head_w_logstd.bias, -1.0)
        nn.init.zeros_(self.head_dt_alpha.bias)

        self.kl_beta = float(cfg.kl_beta)
        self.last_kl = torch.tensor(0.0)

    def forward(self, x_in, y=None, m=None):
        cond = build_cond_input(x_in, y, m, self.cfg)

        h = self.encoder(cond)
        z_mu = self.head_z_mu(h)
        z_logvar = self.head_z_logvar(h)
        z_logvar = torch.clamp(z_logvar, -8.0, 8.0)
        z_std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_std)
        z_latent = z_mu + eps * z_std

        dec_in = torch.cat([cond, z_latent], dim=-1)
        h_dec = self.decoder(dec_in)
        w_mu = self.head_w_mean(h_dec)
        w_logstd = self.head_w_logstd(h_dec)
        w_logstd = torch.clamp(w_logstd, -0.9, 4.0)
        w_std = torch.exp(w_logstd)

        dt_alpha = torch.nn.functional.softplus(self.head_dt_alpha(h_dec)) + 1e-4

        # KL(q(z|x,y_obs) || p(z))
        kl = 0.5 * torch.sum(torch.exp(z_logvar) + z_mu ** 2 - 1.0 - z_logvar, dim=-1)
        self.last_kl = kl.mean()

        # Return as a single-component mixture for compatibility
        pi = torch.ones((w_mu.shape[0], 1), device=w_mu.device)
        dt_logits = torch.log(dt_alpha)
        mu = torch.cat([w_mu, dt_logits], dim=-1).unsqueeze(1)
        std = torch.cat([w_std, torch.ones_like(dt_logits)], dim=-1).unsqueeze(1)
        return {
            "pi": pi,
            "mu": mu,
            "std": std,
            "dt_alpha": dt_alpha,
        }

    def compute_loss(self, y_true, m_missing, out, weights=None, phase_valid=None):
        pi = out["pi"]
        mu = out["mu"]
        std = out["std"]
        dt_alpha = out["dt_alpha"]
        w_split = self.cfg.max_phase * 6

        recon = masked_mdn_nll(
            pi,
            mu[:, :, :w_split],
            std[:, :, :w_split],
            y_true[:, :w_split],
            m_missing[:, :w_split],
            weights=weights,
        )

        dt_true = y_true[:, w_split:]
        if phase_valid is None:
            dt_valid = (dt_true > 0.0)
        else:
            dt_valid = phase_valid

        # Dirichlet NLL over valid phases
        dt_true = torch.clamp(dt_true, min=1e-8)
        sum_alpha = (dt_alpha * dt_valid).sum(dim=-1)
        logB = torch.lgamma(sum_alpha) - torch.sum(torch.lgamma(dt_alpha) * dt_valid, dim=-1)
        log_term = torch.sum((dt_alpha - 1.0) * torch.log(dt_true) * dt_valid, dim=-1)
        nll = -(logB + log_term)

        if weights is None:
            loss_dt = nll.mean()
        else:
            w = weights.view(-1)
            w = w / w.sum().clamp_min(1e-8)
            loss_dt = (nll * w).sum()

        return recon + loss_dt + self.kl_beta * self.last_kl

    @torch.no_grad()
    def sample_y(
        self,
        x_in: torch.Tensor,
        y_in: torch.Tensor,
        m_known: torch.Tensor,
        cfg: FillerConfig,
        phase_valid: Optional[torch.Tensor] = None,
        dt_mode: str = "dirichlet_sample",
        use_mean_w: bool = False,
    ) -> torch.Tensor:
        out = self.forward(x_in, y=y_in, m=m_known)
        w_split = cfg.max_phase * 6
        mu = out["mu"][:, 0, :w_split]
        std = out["std"][:, 0, :w_split]
        if use_mean_w:
            w_sample = mu
        else:
            eps = torch.randn_like(std)
            w_sample = mu + std * eps

        alpha = out["dt_alpha"]
        if dt_mode == "dirichlet_mean":
            dt_frac = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            dt_frac = torch.distributions.Dirichlet(alpha).sample()

        if phase_valid is not None:
            dt_frac = dt_frac * phase_valid
            dt_frac = dt_frac / dt_frac.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        dt_logits = torch.log(dt_frac.clamp_min(1e-8))
        return torch.cat([w_sample, dt_logits], dim=-1)


def masked_mdn_nll(pi, mu, std, y_true, mask_missing, weights=None, eps: float = 1e-8):
    """
    Masked negative log-likelihood for a diagonal Gaussian mixture (MDN).

    pi:           (B, K)        mixture weights, sum to 1
    mu, std:      (B, K, D)
    y_true:       (B, D)
    mask_missing: (B, D)        1 for supervised dims (missing), 0 otherwise
    """
    B, K, D = mu.shape
    mask = mask_missing.unsqueeze(1)  # (B, 1, D)

    # Expand y to (B, K, D)
    y = y_true.unsqueeze(1).expand(-1, K, -1)
    var = (std ** 2).clamp_min(eps)

    # Per-dimension Gaussian NLL: 0.5*(log(2pi var) + (err^2)/var)
    nll_dim = 0.5 * (torch.log(2.0 * torch.pi * var) + ((y - mu) ** 2) / var)  # (B, K, D)

    # Apply mask over dimensions, then sum dims -> per-component nll
    nll_k = (nll_dim * mask).sum(dim=-1)  # (B, K)

    # Combine with mixture weights: -log sum_k pi_k * exp(-nll_k)
    # Use log-sum-exp for stability:
    log_pi = torch.log(pi.clamp_min(eps))          # (B, K)
    log_prob = torch.logsumexp(log_pi - nll_k, dim=-1)  # (B,)

    # Average over samples that actually have missing dims
    denom = mask_missing.sum(dim=-1).clamp_min(1.0)  # (B,)
    per_sample = -log_prob / denom

    if weights is None:
        loss = per_sample.mean()
    else:
        w = weights.view(-1)
        w = w / w.sum().clamp_min(eps)
        loss = (per_sample * w).sum()

    return loss

@torch.no_grad()
def constrained_fill(y_sample, y_in, m_known, tof, cfg, phase_valid=None):
    """
    y_sample: (B, D) - Raw output from the MDN (mu or sampled). DT portion is logits.
    y_in:     (B, D) - Input vector with known values and zeros
    m_known:  (B, D) - Mask (1=known, 0=missing)
    tof:      (B, 1) - Total campaign time (steps). Optional; not used for fractions.
    cfg:      FillerConfig
    phase_valid: (B, max_phase) - 1 if phase is real, 0 if padded (optional)
    """
    B = y_sample.shape[0]

    # 1. Split indices
    # Waypoints: first (max_phase * 6) elements
    # Delta-Ts: remaining (max_phase) elements
    w_split = cfg.max_phase * 6

    # --- Part A: Waypoints (Standard Fill) ---
    w_sample = y_sample[:, :w_split]
    w_in = y_in[:, :w_split]
    w_mask = m_known[:, :w_split]

    w_filled = torch.where(w_mask.bool(), w_in, w_sample)

    # --- Part B: Time Fractions (Simplex via logits) ---
    # dt_sample represents logits; enforce sum to 1 over valid phases
    dt_logits = y_sample[:, w_split:]
    dt_in = y_in[:, w_split:]
    dt_mask = m_known[:, w_split:]

    if phase_valid is None:
        dt_valid = torch.ones_like(dt_in)
    else:
        # dt corresponds to phases 1..N (x0->x1, ..., x_{N-1}->x_N)
        dt_valid = phase_valid  # (B, max_phase)

    # Sum known fractions (only over valid dt slots)
    dt_known = dt_in * dt_mask * dt_valid
    known_sum = dt_known.sum(dim=-1, keepdim=True)

    # If known_sum > 1, renormalize known part to sum to 1
    over = known_sum > 1.0
    dt_known = torch.where(over, dt_known / known_sum.clamp_min(1e-8), dt_known)
    known_sum = torch.where(over, torch.ones_like(known_sum), known_sum)

    remaining = torch.clamp(1.0 - known_sum, min=0.0)
    missing = (~dt_mask.bool()) & (dt_valid.bool())

    dt_soft = _masked_softmax(dt_logits, missing, dim=-1)
    dt_allocated = dt_soft * remaining

    # Combine known + allocated (zero out padded slots)
    dt_filled = (dt_known + dt_allocated) * dt_valid

    # 3. Re-concatenate
    return torch.cat([w_filled, dt_filled], dim=-1)


class WypDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        cfg: FillerConfig,
        inputs_arg: List[str],
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        X_mean: torch.Tensor,
        X_std: torch.Tensor,
        beta: float = 1.0,
    ):
        self.data = data
        self.cfg = cfg
        self.inputs_arg = inputs_arg
        self.input_slices = build_input_slices(
            data,
            inputs_arg,
            b_seq_encoding=self.cfg.b_seq_encoding,
            b_seq_num_classes=self.cfg.b_seq_num_classes,
        )
        self.y_mean = y_mean
        self.y_std = y_std
        self.X_mean = X_mean
        self.X_std = X_std
        self.beta = float(beta)

    def __len__(self):
        return int(self.data["x0"].shape[0])

    def __getitem__(self, idx):
        y, phase_valid = build_y_from_data(self.data, idx)

        # build valid mask for y dims
        wyp_valid = phase_valid.repeat_interleave(6)  # x1..xN
        dt_valid = phase_valid                      # dt1..dtN
        valid_mask = torch.cat([wyp_valid, dt_valid], dim=-1)

        # 70%: all missing; 30%: 20% known among valid dims
        if np.random.rand() < 0.7:
            m_known = torch.zeros_like(y)
        else:
            m_known = torch.zeros_like(y)
            valid_idx = torch.where(valid_mask > 0.0)[0]
            n_known = max(1, int(0.2 * len(valid_idx)))
            chosen = np.random.choice(valid_idx.numpy(), size=n_known, replace=False)
            m_known[chosen] = 1.0

        x_full = build_input_from_data(
            self.data,
            idx,
            self.inputs_arg,
            b_seq_encoding=self.cfg.b_seq_encoding,
            b_seq_num_classes=self.cfg.b_seq_num_classes,
        )
        x_full = scale_var(x_full, self.X_mean, self.X_std)
        if "b_seq" in self.input_slices:
            b_slice = self.input_slices["b_seq"]
            b_width = b_slice.stop - b_slice.start
            b_phase_valid = _expand_phase_valid_for_b_seq(
                phase_valid,
                b_width,
                self.cfg.max_phase,
            )
            x_full[b_slice] = x_full[b_slice] * b_phase_valid

        y_scaled = scale_var(y, self.y_mean, self.y_std)
        y_in = y_scaled * m_known
        m_missing = valid_mask - m_known
        m_missing = torch.clamp(m_missing, min=0.0)

        reward = self.data["reward"][idx].item() if "reward" in self.data else 1.0
        weight = torch.tensor([reward], dtype=torch.float32)

        return (
            x_full,
            y_scaled,
            y_in,
            m_known,
            m_missing,
            phase_valid,
            weight,
        )


def load_model(ckpt_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("device for waypoint inference:", device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_raw = ckpt["cfg"]
    cfg_dict = dict(cfg_raw) if isinstance(cfg_raw, dict) else dict(vars(cfg_raw))

    for name, field in FillerConfig.__dataclass_fields__.items():
        if name not in cfg_dict and field.default is not MISSING:
            cfg_dict[name] = field.default

    fields = set(FillerConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg_dict.items() if k in fields}
    cfg = FillerConfig(**filtered)

    model_type = ckpt.get("model_type", "gmm")
    model = ConditionalVAE(cfg) if model_type == "vae" else ConditionalGMM(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    def _to_device_if_tensor(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x

    return {
        "cfg": cfg,
        "model": model,
        "device": device,
        "y_mean": _to_device_if_tensor(ckpt["y_mean"]),
        "y_std": _to_device_if_tensor(ckpt["y_std"]),
        "X_mean": _to_device_if_tensor(ckpt["X_mean"]),
        "X_std": _to_device_if_tensor(ckpt["X_std"]),
        "inputs_arg": ckpt.get(
            "inputs_arg",
            ["x0", "tof", "oec0_modified", "artms_scale_range_1e3", "koz_dim", "b_seq"],
        ),
    }


def build_data_from_values(values: Dict[str, Any], max_phase: int) -> Dict[str, torch.Tensor]:
    data: Dict[str, torch.Tensor] = {}
    data["x0"] = torch.as_tensor(values["x0"], dtype=torch.float32).reshape(1, -1)
    data["tof"] = torch.as_tensor([[values["tof"]]], dtype=torch.float32)
    data["oec0_modified"] = torch.as_tensor(values["oec0_modified"], dtype=torch.float32).reshape(1, -1)
    data["artms_scale_range_1e3"] = torch.as_tensor(values["artms_scale_range_1e3"], dtype=torch.float32).reshape(1, -1)
    data["koz_dim"] = torch.as_tensor(values["koz_dim"], dtype=torch.float32).reshape(1, -1)

    b_pad = np.zeros((1, max_phase), dtype=np.float32)
    b_seq = np.asarray(values["b_seq"], dtype=np.float32)
    b_pad[0, : min(len(b_seq), max_phase)] = b_seq[:max_phase]
    data["b_seq"] = torch.as_tensor(b_pad, dtype=torch.float32)

    x_seq = np.zeros((1, max_phase, 6), dtype=np.float32)
    data["x_seq"] = torch.as_tensor(x_seq, dtype=torch.float32)
    return data


def predict_wyp_seq(
    model_bundle: Dict[str, Any],
    input_slices: Dict[str, slice],
    x0: np.ndarray,
    tof_steps: int,
    b_seq: Sequence[int],
    oec0_mod: np.ndarray,
    artms: np.ndarray,
    koz_dim: np.ndarray,
    use_mean_w: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg: FillerConfig = model_bundle["cfg"]
    model = model_bundle["model"]
    device: torch.device = model_bundle["device"]
    y_mean, y_std = model_bundle["y_mean"], model_bundle["y_std"]
    X_mean, X_std = model_bundle["X_mean"], model_bundle["X_std"]
    inputs_arg = model_bundle["inputs_arg"]

    max_phase = cfg.max_phase
    phase_valid = np.zeros(max_phase, dtype=bool)
    phase_valid[: len(b_seq)] = True
    phase_valid_t = torch.tensor(phase_valid, dtype=torch.float32, device=device).unsqueeze(0)

    y_dim = max_phase * 6 + max_phase
    y_in = torch.zeros(1, y_dim, device=device)
    m_known = torch.zeros(1, y_dim, device=device)

    values = {
        "x0": x0,
        "tof": tof_steps,
        "oec0_modified": oec0_mod,
        "artms_scale_range_1e3": artms,
        "koz_dim": koz_dim,
        "b_seq": b_seq,
    }
    data_like = build_data_from_values(values, max_phase)
    x_full = build_input_from_data(
        data_like,
        0,
        inputs_arg,
        # Dataset b_seq stays scalar IDs; build_input_from_data handles one-hot expansion.
        b_seq_encoding=getattr(cfg, "b_seq_encoding", DEFAULT_B_SEQ_ENCODING),
        b_seq_num_classes=int(getattr(cfg, "b_seq_num_classes", DEFAULT_B_SEQ_NUM_CLASSES)),
    ).unsqueeze(0).to(device)
    x_full = scale_var(x_full, X_mean, X_std)

    if "b_seq" in input_slices:
        b_slice = input_slices["b_seq"]
        b_width = int(b_slice.stop - b_slice.start)
        if b_width == int(max_phase):
            b_phase_valid = phase_valid_t
        elif b_width % int(max_phase) == 0:
            rep = b_width // int(max_phase)
            b_phase_valid = phase_valid_t.repeat_interleave(rep, dim=-1)
        else:
            raise ValueError(
                f"b_seq input width {b_width} is incompatible with max_phase={max_phase}."
            )
        x_full[:, b_slice.start : b_slice.stop] = (
            x_full[:, b_slice.start : b_slice.stop] * b_phase_valid
        )

    with torch.no_grad():
        y_scaled = model.sample_y(x_full, y_in, m_known, cfg, phase_valid=phase_valid_t, use_mean_w=use_mean_w)
        y_unscaled = unscale_var(y_scaled, y_mean, y_std)

    tof_raw_t = torch.tensor([[tof_steps]], dtype=torch.float32, device=device)
    y_filled = constrained_fill(y_unscaled, y_in, m_known, tof_raw_t, cfg, phase_valid=phase_valid_t)
    x_pred = y_filled[0, : max_phase * 6].reshape(max_phase, 6).detach().cpu().numpy()
    dt_pred = y_filled[0, max_phase * 6 :].detach().cpu().numpy()
    return x_pred[: len(b_seq)], dt_pred[: len(b_seq)]
