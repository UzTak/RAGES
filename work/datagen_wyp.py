# dataset generation for the waypoint-based mission planning task (RAGES+ item 6)
#
# All rows (converged and failed) are retained so Stage 2 Q training has
# access to the full candidate population.  Converged rows get a scalar reward
# derived from fuel_dv; failed rows get reward=0 and NaN metrics.
import itertools
import sys
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

root_folder = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_folder / "src"))

from parameters import OK_STATUS
from rages_sampling import (
    SplitConfig,
    WaypointPlan,
    sample_curated_rollout,
    verify_waypoint_plan,
)
from rages_scoring import VERIFIER_METRIC_KEYS


def _generate_one_sample(args: Tuple) -> Dict:
    """
    Deterministically sample one scenario+action (via sample_curated_rollout)
    and verify it through the full CVX+SCP stack.  Always returns a row; failed
    rows have converged=False and NaN metrics so they are excluded from wyp
    training weights but kept for Q training.
    """
    sample_id, seed, split = args
    rollout = sample_curated_rollout(sample_id, seed=seed, split=split)
    plan = WaypointPlan(
        waypoint_states=rollout.waypoint_states,
        dt_fractions=rollout.dt_fractions,
    )
    result = verify_waypoint_plan(
        rollout.scenario,
        rollout.curated_action.action,
        plan,
        candidate_id=0,
    )

    converged = result.status_scp in OK_STATUS
    metrics_vec = [float(result.metrics.get(k, float("nan"))) for k in VERIFIER_METRIC_KEYS]
    # Scalar reward for wyp reward-weighted MLE; NaN signals "do not train on this row"
    reward = -result.metrics["fuel_dv"] if converged else float("nan")

    return {
        # scenario
        "x0": np.asarray(rollout.scenario.x0, dtype=np.float32),
        "tof": int(rollout.curated_action.action.tof_steps),
        "oec0_modified": np.asarray(rollout.scenario.oec0_modified, dtype=np.float32),
        "koz_dim": np.asarray(rollout.scenario.koz_dim, dtype=np.float32).reshape(-1),
        "artms_scale_range_1e3": np.asarray(
            rollout.scenario.artms_scale_range_1e3, dtype=np.float32
        ),
        # action
        "b_seq": list(rollout.curated_action.action.b_seq),
        # waypoint plan
        "x_seq": [np.asarray(x, dtype=np.float32) for x in rollout.waypoint_states],
        "dt_seq": np.asarray(rollout.dt_fractions, dtype=np.float32),
        # verifier
        "converged": bool(converged),
        "status_cvx": result.status_cvx,
        "status_scp": result.status_scp,
        "metrics": metrics_vec,   # (4,) in VERIFIER_METRIC_KEYS order; NaN if not converged
        "reward": float(reward),  # -fuel_dv for converged; NaN for failed
        # metadata
        "campaign": rollout.curated_action.curation.policy,
        "split": rollout.scenario.split,
        "sample_id": int(sample_id),
    }


def _masked_mean_std(x: torch.Tensor, mask: Optional[torch.Tensor] = None):
    if mask is None:
        mask = torch.ones_like(x)
    sum_x = (x * mask).sum(dim=0)
    sumsq_x = (x ** 2 * mask).sum(dim=0)
    count = mask.sum(dim=0).clamp_min(1.0)
    mean = sum_x / count
    var = sumsq_x / count - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-6))
    return mean, std


def generate_dataset(
    num_samples: int = 10,
    num_workers: int = 1,
    max_phase: int = 3,
    seed: int = 42,
    split_config: Optional[SplitConfig] = None,
) -> Dict:
    """
    Generate a fixed-size dataset.

    Every row is stored (converged and failed).  Failed rows have
    ``converged=0``, ``reward=0``, and NaN metric entries so they receive zero
    weight in wyp reward-weighted MLE but are available for Q-head training.
    """
    split_config = split_config or SplitConfig()

    # ---- infer shapes from a single sample ----
    first_args = (0, seed, split_config.split_for_index(0, num_samples))
    first_row = _generate_one_sample(first_args)

    oe_dim = int(np.atleast_1d(first_row["oec0_modified"]).shape[-1])
    koz_d = int(np.atleast_1d(first_row["koz_dim"]).shape[-1])
    artms_d = int(np.atleast_1d(first_row["artms_scale_range_1e3"]).shape[-1])
    n_metrics = len(VERIFIER_METRIC_KEYS)

    # ---- pre-allocate tensors ----
    data: Dict[str, torch.Tensor] = {
        "x0":                   torch.zeros((num_samples, 6),            dtype=torch.float32),
        "tof":                  torch.zeros((num_samples, 1),            dtype=torch.float32),
        "b_seq":                torch.zeros((num_samples, max_phase),    dtype=torch.float32),
        "phase_valid":          torch.zeros((num_samples, max_phase),    dtype=torch.float32),
        "oec0_modified":        torch.zeros((num_samples, oe_dim),       dtype=torch.float32),
        "koz_dim":              torch.zeros((num_samples, koz_d),        dtype=torch.float32),
        "artms_scale_range_1e3":torch.zeros((num_samples, artms_d),      dtype=torch.float32),
        "x_seq":                torch.zeros((num_samples, max_phase, 6), dtype=torch.float32),
        "dt_seq":               torch.zeros((num_samples, max_phase),    dtype=torch.float32),
        "converged":            torch.zeros((num_samples,),              dtype=torch.float32),
        "reward":               torch.zeros((num_samples,),              dtype=torch.float32),
        "metrics":              torch.full((num_samples, n_metrics), float("nan"), dtype=torch.float32),
    }

    def _fill_from_row(row: Dict, idx: int) -> None:
        b_seq  = row["b_seq"]
        dt_seq = row["dt_seq"]
        x_seq  = row["x_seq"]
        if len(b_seq) != len(dt_seq) or len(b_seq) != len(x_seq):
            raise ValueError("b_seq, dt_seq, x_seq length mismatch.")
        n_phase = len(b_seq)
        if n_phase > max_phase:
            raise ValueError(f"Sequence length {n_phase} > max_phase={max_phase}.")

        b_pad  = torch.zeros(max_phase, dtype=torch.float32)
        dt_pad = torch.zeros(max_phase, dtype=torch.float32)
        x_pad  = torch.zeros((max_phase, 6), dtype=torch.float32)
        b_pad[:n_phase]    = torch.as_tensor(b_seq, dtype=torch.float32)
        dt_pad[:n_phase]   = torch.as_tensor(dt_seq, dtype=torch.float32)
        x_pad[:n_phase]    = torch.as_tensor(np.stack(x_seq), dtype=torch.float32)
        phase_valid        = torch.zeros(max_phase, dtype=torch.float32)
        phase_valid[:n_phase] = 1.0

        data["x0"][idx]                    = torch.as_tensor(row["x0"],                    dtype=torch.float32)
        data["tof"][idx]                   = torch.tensor([float(row["tof"])],              dtype=torch.float32)
        data["b_seq"][idx]                 = b_pad
        data["dt_seq"][idx]                = dt_pad
        data["x_seq"][idx]                 = x_pad
        data["phase_valid"][idx]           = phase_valid
        data["oec0_modified"][idx]         = torch.as_tensor(row["oec0_modified"],          dtype=torch.float32)
        data["koz_dim"][idx]               = torch.as_tensor(row["koz_dim"],                dtype=torch.float32)
        data["artms_scale_range_1e3"][idx] = torch.as_tensor(row["artms_scale_range_1e3"], dtype=torch.float32)
        data["converged"][idx]             = float(row["converged"])
        # reward: NaN → 0 here; shift applied later only for converged rows
        r = row["reward"]
        data["reward"][idx]                = float(r) if np.isfinite(r) else 0.0
        m = row["metrics"]
        data["metrics"][idx]               = torch.tensor(m, dtype=torch.float32)

    # ---- collect rows ----
    def _args_iter():
        for i in itertools.count():
            yield (i, seed, split_config.split_for_index(i % num_samples, num_samples))

    filled = 0
    _fill_from_row(first_row, filled)
    filled += 1

    pbar = tqdm(total=num_samples)
    pbar.update(1)

    if num_workers > 1:
        ctx = get_context("spawn")
        pool = ctx.Pool(processes=num_workers)
        try:
            it = pool.imap_unordered(_generate_one_sample, _args_iter())
            for row in it:
                _fill_from_row(row, filled)
                filled += 1
                pbar.update(1)
                if filled >= num_samples:
                    pool.terminate()
                    break
        finally:
            pool.join()
    else:
        gen = _args_iter()
        next(gen)           # skip sample_id=0 already filled
        while filled < num_samples:
            row = _generate_one_sample(next(gen))
            _fill_from_row(row, filled)
            filled += 1
            pbar.update(1)

    pbar.close()

    # ---- reward shift (converged rows only; failed rows keep weight=0) ----
    converged_mask = data["converged"] > 0.5
    reward_shift = 0.0
    if converged_mask.any():
        min_r = float(data["reward"][converged_mask].min().item())
        if min_r < 0.0:
            reward_shift = -min_r + 1e-3
            data["reward"][converged_mask] = data["reward"][converged_mask] + reward_shift

    n_conv = int(converged_mask.sum().item())
    print(f"Converged: {n_conv}/{num_samples} ({100*n_conv/num_samples:.1f}%)")

    # ---- stats (all rows for X; converged-only for metrics) ----
    phase_valid = data["phase_valid"]
    stats: Dict = {k: {} for k in [
        "x0", "tof", "oec0_modified", "koz_dim", "artms_scale_range_1e3", "b_seq", "x_seq",
    ]}
    stats["x0"]["mean"],             stats["x0"]["std"]             = _masked_mean_std(data["x0"])
    stats["tof"]["mean"],            stats["tof"]["std"]            = _masked_mean_std(data["tof"])
    stats["oec0_modified"]["mean"],  stats["oec0_modified"]["std"]  = _masked_mean_std(data["oec0_modified"])
    stats["koz_dim"]["mean"],        stats["koz_dim"]["std"]        = _masked_mean_std(data["koz_dim"])
    stats["artms_scale_range_1e3"]["mean"], stats["artms_scale_range_1e3"]["std"] = _masked_mean_std(
        data["artms_scale_range_1e3"]
    )
    stats["b_seq"]["mean"],  stats["b_seq"]["std"]  = _masked_mean_std(data["b_seq"], mask=phase_valid)
    stats["x_seq"]["mean"],  stats["x_seq"]["std"]  = _masked_mean_std(
        data["x_seq"], mask=phase_valid.unsqueeze(-1)
    )

    return {
        "data": data,
        "stats": stats,
        "meta": {
            "num_samples":    num_samples,
            "max_phase":      max_phase,
            "oe_dim":         oe_dim,
            "koz_dim":        koz_d,
            "artms_dim":      artms_d,
            "n_metrics":      n_metrics,
            "metric_keys":    list(VERIFIER_METRIC_KEYS),
            "reward_shift":   reward_shift,
            "seed":           seed,
        },
    }


if __name__ == "__main__":
    N_data    = 80_000
    N_proc    = 20
    SEED      = 42

    dataset = generate_dataset(N_data, num_workers=N_proc, max_phase=3, seed=SEED)

    save_path = root_folder / "data" / "wyp_data" / "data_v5.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, save_path)
    print(f"Dataset saved to {save_path}")
    meta = dataset["meta"]
    n_conv = int((dataset["data"]["converged"] > 0.5).sum().item())
    print(f"  samples: {meta['num_samples']}  converged: {n_conv}  metric_keys: {meta['metric_keys']}")
