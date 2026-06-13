from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parameters import DEFAULT_B_SEQ_ENCODING, DEFAULT_B_SEQ_NUM_CLASSES
from utils import contiguous_train_eval_index_ranges
from wyp_predictor import (
    ConditionalFlowMatcher,
    ConditionalGMM,
    ConditionalVAE,
    FillerConfig,
    WypDataset,
    _input_field_size,
    build_input_slices,
    build_X_stats_from_stats,
    build_y_stats_from_stats,
)


DEFAULT_INPUTS = ["x0", "tof", "oec0_modified", "artms_scale_range_1e3", "koz_dim", "b_seq"]


def _repo_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a waypoint predictor p_phi.")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/wyp_data/data_v5.pth")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "model/wyp_model")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-type", choices=("gmm", "vae", "flow"), default="gmm")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument(
        "--b-seq-encoding",
        choices=("scalar", "one_hot"),
        default=DEFAULT_B_SEQ_ENCODING,
    )
    parser.add_argument("--b-seq-num-classes", type=int, default=DEFAULT_B_SEQ_NUM_CLASSES)
    parser.add_argument("--weighted-il", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--weight-power",
        type=float,
        default=1.0,
        help="Exponent applied to nonnegative reward weights; values <1 smooth support.",
    )
    parser.add_argument(
        "--weight-clip-max",
        type=float,
        default=None,
        help="Optional upper clip for reward weights before per-batch normalization.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=5000)
    parser.add_argument("--initial-lr", type=float, default=1e-4)
    parser.add_argument("--final-lr", type=float, default=1e-4)
    parser.add_argument("--decay-frac", type=float, default=0.02)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-hidden-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--kl-beta", type=float, default=1e-3)
    parser.add_argument("--flow-steps", type=int, default=32)
    parser.add_argument("--mask-beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-loader-workers", type=int, default=0)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional short-run cap for smoke tests.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Optional eval cap for smoke tests.",
    )
    args = parser.parse_args()

    if args.weight_power <= 0.0:
        raise ValueError("--weight-power must be positive.")
    if args.weight_clip_max is not None and args.weight_clip_max <= 0.0:
        raise ValueError("--weight-clip-max must be positive when provided.")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1).")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive.")

    args.data_path = _repo_path(args.data_path)
    args.out_dir = _repo_path(args.out_dir)
    if args.model_name is None:
        weight_tag = "weighted" if args.weighted_il else "unweighted"
        args.model_name = (
            f"model_{args.model_type}_{args.data_path.stem}_{weight_tag}_{args.b_seq_encoding}.pt"
        )
    return args


def build_model(cfg: FillerConfig, model_type: str, device: torch.device) -> torch.nn.Module:
    if model_type == "gmm":
        return ConditionalGMM(cfg=cfg).to(device)
    if model_type == "vae":
        return ConditionalVAE(cfg=cfg).to(device)
    if model_type == "flow":
        return ConditionalFlowMatcher(cfg=cfg).to(device)
    raise ValueError(f"Unknown model_type: {model_type}")


def move_batch_to_device(batch, device: torch.device, weighted_il: bool, args: argparse.Namespace):
    x_in, y_true, y_in, m_known, m_missing, phase_valid, weights = batch
    x_in = x_in.to(device)
    y_true = y_true.to(device)
    y_in = y_in.to(device)
    m_known = m_known.to(device)
    m_missing = m_missing.to(device)
    phase_valid = phase_valid.to(device)
    if weighted_il:
        weights = weights.to(device).clamp_min(0.0)
        if args.weight_power != 1.0:
            weights = weights.pow(args.weight_power)
        if args.weight_clip_max is not None:
            weights = weights.clamp_max(float(args.weight_clip_max))
    else:
        weights = None
    return x_in, y_true, y_in, m_known, m_missing, phase_valid, weights


def serializable_args(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data_path}. Generate data_v5 with "
            "`uv run python work/datagen_wyp.py` or pass --data-path."
        )

    dataset = torch.load(args.data_path, map_location="cpu")
    data = dataset["data"]
    stats = dataset["stats"]
    meta = dataset["meta"]
    data_sha256 = sha256_file(args.data_path)

    # derive dataset stats
    max_phase = int(meta.get("max_phase", data["b_seq"].shape[1]))
    b_seq_num_classes = int(args.b_seq_num_classes)
    max_behavior_id = int(torch.max(data["b_seq"]).item())
    if args.b_seq_encoding == "one_hot" and max_behavior_id > b_seq_num_classes:
        raise ValueError(
            f"Dataset contains behavior id {max_behavior_id}, but "
            f"--b-seq-num-classes={b_seq_num_classes}."
        )
    print(f"device: {device}")
    print(f"data_path: {args.data_path}")
    print(f"data_sha256: {data_sha256}")
    print(f"max_phase: {max_phase}")
    print(f"b_seq_num_classes: {b_seq_num_classes}")
    input_dim = sum(
        _input_field_size(
            data,
            name,
            b_seq_encoding=args.b_seq_encoding,
            b_seq_num_classes=b_seq_num_classes,
        )
        for name in args.inputs
    )
    cfg = FillerConfig(
        max_phase=max_phase,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        dropout=args.dropout,
        K=args.K,
        latent_dim=args.latent_dim,
        kl_beta=args.kl_beta,
        b_seq_encoding=args.b_seq_encoding,
        b_seq_num_classes=b_seq_num_classes,
        flow_steps=args.flow_steps,
    )

    model = build_model(cfg, args.model_type, device)

    # split train/test as contiguous 90/10 tail eval to match analysis scripts
    num_samples = int(data["x0"].shape[0])
    train_range, test_range = contiguous_train_eval_index_ranges(
        n_rows=num_samples,
        val_ratio=args.val_ratio,
    )
    if test_range is None:
        raise ValueError("val_ratio must be > 0 for train/test split.")
    train_idx = np.arange(train_range.start, train_range.stop, dtype=int)
    test_idx = np.arange(test_range.start, test_range.stop, dtype=int)

    # compute scaling stats on train split
    train_data = {k: v[train_idx] for k, v in data.items()}
    test_data = {k: v[test_idx] for k, v in data.items()}
    y_mean, y_std = build_y_stats_from_stats(stats)
    X_mean, X_std = build_X_stats_from_stats(
        stats,
        args.inputs,
        b_seq_encoding=cfg.b_seq_encoding,
        b_seq_num_classes=cfg.b_seq_num_classes,
    )

    print("y_mean:", y_mean)
    print("y_std:", y_std)
    print("X_mean:", X_mean)
    print("X_std:", X_std)

    y_mean_cpu = y_mean.clone()
    y_std_cpu = y_std.clone()
    X_mean_cpu = X_mean.clone()
    X_std_cpu = X_std.clone()

    train_ds = WypDataset(
        train_data,
        cfg,
        args.inputs,
        y_mean_cpu,
        y_std_cpu,
        X_mean_cpu,
        X_std_cpu,
        beta=args.mask_beta,
    )
    test_ds = WypDataset(
        test_data,
        cfg,
        args.inputs,
        y_mean_cpu,
        y_std_cpu,
        X_mean_cpu,
        X_std_cpu,
        beta=args.mask_beta,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_loader_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_loader_workers,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)
    if args.max_train_batches is None:
        train_batches_per_epoch = len(train_loader)
    else:
        train_batches_per_epoch = min(len(train_loader), int(args.max_train_batches))
    total_steps = max(1, args.num_epochs * max(1, train_batches_per_epoch))
    decay_steps = max(1, int(args.decay_frac * total_steps))

    def lr_lambda(step: int) -> float:
        progress = min(step, decay_steps) / decay_steps
        return (args.final_lr / args.initial_lr) ** progress

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_losses = []
    test_losses = []
    step = 0
    last_saved_step = -1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / args.model_name
    out_path = args.out_dir / f"{model_path.stem}_loss_curve.png"

    checkpoint_common = {
        "cfg": cfg.__dict__,
        "inputs_arg": list(args.inputs),
        "input_slices": {
            k: (v.start, v.stop)
            for k, v in build_input_slices(
                train_data,
                args.inputs,
                b_seq_encoding=cfg.b_seq_encoding,
                b_seq_num_classes=cfg.b_seq_num_classes,
            ).items()
        },
        "y_mean": y_mean_cpu,
        "y_std": y_std_cpu,
        "X_mean": X_mean_cpu,
        "X_std": X_std_cpu,
        "model_type": args.model_type,
        "train_args": serializable_args(args),
        "data_path": str(args.data_path),
        "data_sha256": data_sha256,
        "data_meta": meta,
        "train_range": (int(train_range.start), int(train_range.stop)),
        "test_range": (int(test_range.start), int(test_range.stop)),
        "decode_contract": {
            "deterministic_decode": "use_mean_w=True",
            "gmm": "argmax mixture component mean",
            "vae": "latent-mean decoder with Dirichlet mean dt fractions",
            "flow": "Euler ODE decode from zero base sample",
        },
    }

    def save_checkpoint(current_step: int, test_loss: float | None) -> None:
        nonlocal last_saved_step
        torch.save(
            {
                **checkpoint_common,
                "model_state_dict": model.state_dict(),
                "step": int(current_step),
                "latest_train_loss": float(train_losses[-1]) if train_losses else None,
                "latest_test_loss": None if test_loss is None else float(test_loss),
            },
            model_path,
        )
        last_saved_step = int(current_step)

    def eval_loss():
        model.eval()
        losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if args.max_eval_batches is not None and batch_idx >= args.max_eval_batches:
                    break
                x_in, y_true, y_in, m_known, m_missing, phase_valid, weights = move_batch_to_device(
                    batch, device, args.weighted_il, args
                )

                out = model(x_in, y=y_in, m=m_known)
                loss = model.compute_loss(
                    y_true, m_missing, out, weights=weights, phase_valid=phase_valid
                )
                losses.append(loss.item())
        model.train()
        return float(np.mean(losses)) if losses else 0.0

    print(f"model_type: {args.model_type}")
    print(f"model_path: {model_path}")
    print("Starting training...")

    latest_test_loss = None
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            x_in, y_true, y_in, m_known, m_missing, phase_valid, weights = move_batch_to_device(
                batch, device, args.weighted_il, args
            )

            out = model(x_in, y=y_in, m=m_known)
            loss = model.compute_loss(
                y_true, m_missing, out, weights=weights, phase_valid=phase_valid
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            step += 1

            if step < 5:
                debug_parts = [
                    f"[step {step}]",
                    f"x_in max={x_in.abs().max():.2e}",
                    f"y_scaled max={y_true.abs().max():.2e}",
                    f"mask_missing mean={m_missing.float().mean().item():.2f}",
                ]
                if "mu" in out and "std" in out:
                    mu = out["mu"]
                    std = out["std"]
                    debug_parts.extend(
                        [
                            f"mu max={mu.abs().max():.2e}",
                            f"std min/max={std.min().item():.2e}/{std.max().item():.2e}",
                        ]
                    )
                if "cond" in out:
                    debug_parts.append(f"cond max={out['cond'].abs().max():.2e}")
                print(" ".join(debug_parts))
                if args.model_type == "vae":
                    print(f"[step {step}] kl={model.last_kl.item():.4f}")

            if step % args.save_every == 0:
                latest_test_loss = eval_loss()
                test_losses.append((step, latest_test_loss))
                plt.figure(figsize=(6, 4))
                plt.plot(train_losses, label="train")
                if test_losses:
                    xs, ys = zip(*test_losses)
                    plt.plot(xs, ys, label="test")
                plt.yscale("log")
                plt.xlabel("iteration")
                plt.ylabel("loss")
                plt.ylim(top=5)
                plt.legend()
                # plt.ylim(bottom=1e-6)
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close()
                save_checkpoint(step, latest_test_loss)

                # print progress
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Step {step} (lr={lr:.6e}): train loss = {loss.item():.3f}, test loss = {latest_test_loss:.3f}"
                )

    if step > 0 and step != last_saved_step:
        latest_test_loss = eval_loss()
        save_checkpoint(step, latest_test_loss)
        print(f"Saved final checkpoint at step {step}: {model_path}")


if __name__ == "__main__":
    main()
