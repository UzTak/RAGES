from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import contiguous_train_eval_index_ranges
from wyp_predictor import (
    ConditionalGMM,
    ConditionalVAE,
    FillerConfig,
    WypDataset,
    _input_field_size,
    build_input_slices,
    build_X_stats_from_stats,
    build_y_stats_from_stats,
)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #### INPUT DATA ###########
    root = Path(__file__).resolve().parents[2]
    data_path = root / "rpod" / "rages" / "wyp_data" / "data_v4.pth"
    model_name = "model_gmm_v4_weighted_one_hot.pt"

    weighted_IL = True
    model_type = "gmm"  # "gmm" or "vae"
    inputs_arg = ["x0", "tof", "oec0_modified", "artms_scale_range_1e3", "koz_dim", "b_seq"]
    b_seq_encoding = "one_hot"

    val_ratio = 0.1

    ############################

    dataset = torch.load(data_path, map_location="cpu")
    data = dataset["data"]
    stats = dataset["stats"]
    meta = dataset["meta"]

    # derive dataset stats
    max_phase = int(meta.get("max_phase", data["b_seq"].shape[1]))
    b_seq_num_classes = int(torch.max(data["b_seq"]).item())
    print(f"max_phase: {max_phase}")
    input_dim = sum(
        _input_field_size(
            data,
            name,
            b_seq_encoding=b_seq_encoding,
            b_seq_num_classes=b_seq_num_classes,
        )
        for name in inputs_arg
    )
    cfg = FillerConfig(
        max_phase=max_phase,
        input_dim=input_dim,
        hidden_dim=256,
        n_hidden_layers=3,
        K=1,
        b_seq_encoding=b_seq_encoding,
        b_seq_num_classes=b_seq_num_classes,
    )

    if model_type == "gmm":
        model = ConditionalGMM(cfg=cfg).to(device)
    elif model_type == "vae":
        model = ConditionalVAE(cfg=cfg).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # split train/test as contiguous 90/10 tail eval to match analysis scripts
    num_samples = int(data["x0"].shape[0])
    train_range, test_range = contiguous_train_eval_index_ranges(
        n_rows=num_samples,
        val_ratio=val_ratio,
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
        inputs_arg,
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
        inputs_arg,
        y_mean_cpu,
        y_std_cpu,
        X_mean_cpu,
        X_std_cpu,
        beta=0.1,
    )
    test_ds = WypDataset(
        test_data,
        cfg,
        inputs_arg,
        y_mean_cpu,
        y_std_cpu,
        X_mean_cpu,
        X_std_cpu,
        beta=0.1,
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    num_epochs = 5000
    initial_lr = 1e-4
    final_lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    total_steps = max(1, num_epochs * len(train_loader))
    decay_frac = 0.02
    decay_steps = max(1, int(decay_frac * total_steps))

    def lr_lambda(step: int) -> float:
        progress = min(step, decay_steps) / decay_steps
        return (final_lr / initial_lr) ** progress

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_losses = []
    test_losses = []
    step = 0
    save_every = 100
    out_dir = root / "rpod" / "rages" / "wyp_model"
    out_path = out_dir / "loss_curve.png"
    model_path = out_dir / model_name

    def eval_loss():
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in test_loader:
                x_in, y_true, y_in, m_known, m_missing, phase_valid, weights = batch
                x_in = x_in.to(device)
                y_true = y_true.to(device)
                y_in = y_in.to(device)
                m_known = m_known.to(device)
                m_missing = m_missing.to(device)
                phase_valid = phase_valid.to(device)
                weights = weights.to(device) if weighted_IL else None

                out = model(x_in, y=y_in, m=m_known)
                loss = model.compute_loss(
                    y_true, m_missing, out, weights=weights, phase_valid=phase_valid
                )
                losses.append(loss.item())
        model.train()
        return float(np.mean(losses)) if losses else 0.0

    print("Starting training...")

    for epoch in range(num_epochs):
        for batch in train_loader:
            x_in, y_true, y_in, m_known, m_missing, phase_valid, weights = batch
            x_in = x_in.to(device)
            y_true = y_true.to(device)
            y_in = y_in.to(device)
            m_known = m_known.to(device)
            m_missing = m_missing.to(device)
            phase_valid = phase_valid.to(device)
            weights = weights.to(device) if weighted_IL else None

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
                mu = out["mu"]
                std = out["std"]
                print(
                    f"[step {step}] "
                    f"x_in max={x_in.abs().max():.2e} "
                    f"y_scaled max={y_true.abs().max():.2e} "
                    f"mu max={mu.abs().max():.2e} "
                    f"std min/max={std.min().item():.2e}/{std.max().item():.2e} "
                    f"mask_missing mean={m_missing.float().mean().item():.2f}"
                )
                if model_type == "vae":
                    print(f"[step {step}] kl={model.last_kl.item():.4f}")

            if step % save_every == 0:
                test_loss = eval_loss()
                test_losses.append((step, test_loss))
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
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "inputs_arg": inputs_arg,
                        "input_slices": {
                            k: (v.start, v.stop)
                            for k, v in build_input_slices(
                                train_data,
                                inputs_arg,
                                b_seq_encoding=cfg.b_seq_encoding,
                                b_seq_num_classes=cfg.b_seq_num_classes,
                            ).items()
                        },
                        "y_mean": y_mean_cpu,
                        "y_std": y_std_cpu,
                        "X_mean": X_mean_cpu,
                        "X_std": X_std_cpu,
                        "model_type": model_type,
                        "step": step,
                    },
                    model_path,
                )

                # print progress
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Step {step} (lr={lr:.6e}): train loss = {loss.item():.3f}, test loss = {test_loss:.3f}"
                )

        # end-of-epoch test
        test_loss = eval_loss()
        test_losses.append((step, test_loss))
