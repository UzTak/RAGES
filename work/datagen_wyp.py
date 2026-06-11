# dataset generation for the waypoint-based mission planning task
import numpy as np
import torch
import itertools
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import os, sys
from multiprocessing import get_context
from contextlib import nullcontext
from tqdm import tqdm

# Add root to path
def find_root_path(path:str, word:str):
    parts = path.split(word, 1)
    return parts[0] + word if len(parts) > 1 else path 
root_folder = Path(__file__).resolve().parents[1]

from optimization import parameters as param
from dynamics.dynamics_trans import (
    propagate_oe,
    true_to_mean_anomaly,
    modify_koe,
)
from optimization.optimization import NonConvexOCP
from optimization.scvx import solve_scvx
from parameters import (
    ARTMS_SCALE_FACTORS,
    BEHAVIOR_IDS,
    DT_SEC,
    KOZ_DIMS,
    MissionPolicy,
    NODES,
    N_TIME_MAX,
    POLICY_REGISTRY,
    TRUE_ANOMALY_GRID_RAD,
)
from utils import time_grid_from_orbits, waypoint_times_from_dts

_time_grid_from_orbits = time_grid_from_orbits
_waypoint_times_from_dts = waypoint_times_from_dts

def enumerate_policy_paths(
    policy: MissionPolicy,
    start_node: str,
    max_steps: int,
) -> List[Tuple[List[int], List[Tuple[float, float]]]]:
    """
    Enumerate unique behavior sequences reachable within max_steps.
    Returns list of (behavior_sequence, dt_ranges).
    """
    paths = [(start_node, [], [])]  # (node, beh_seq, dt_ranges)
    for step_index in range(max_steps):
        new_paths = []
        for node, beh_seq, dt_ranges in paths:
            options = policy.get_next_options(node, step_index)
            if not options:
                new_paths.append((node, beh_seq, dt_ranges))
                continue
            for next_node, beh, dt_range in options:
                if beh == 0:
                    new_paths.append((next_node, beh_seq, dt_ranges))
                else:
                    if len(beh_seq) + 1 <= max_steps:
                        new_paths.append((next_node, beh_seq + [beh], dt_ranges + [dt_range]))
        paths = new_paths

    # unique by behavior sequence
    uniq = {}
    for _, beh_seq, dt_ranges in paths:
        key = tuple(beh_seq)
        if key and key not in uniq:
            uniq[key] = (beh_seq, dt_ranges)
    return list(uniq.values())

# --- 3. THE GENERATOR (The Engine) ---

def _sample_koz_dim() -> np.ndarray:
    """ return 2D array with shape (1,3) """
    dim = float(np.random.choice(KOZ_DIMS))
    return np.array([[dim, dim, dim]], dtype=float)

def _sample_artms_param() -> np.ndarray:
    nominal = np.asarray(param.artms_scale_range_1e3, dtype=float).copy()
    scale = float(np.random.choice(ARTMS_SCALE_FACTORS))
    return nominal * scale

def _sample_oec0() -> np.ndarray:
    oec0 = np.asarray(param.oec0, dtype=float).copy()
    nu = float(np.random.choice(TRUE_ANOMALY_GRID_RAD))
    oec0[5] = true_to_mean_anomaly(nu, oec0[1])
    return oec0

def _compute_reward(prob: NonConvexOCP, roe, actions) -> float:
    # Waypoint selection minimizes DV cost (= SCP objective).
    dv_total = float(np.linalg.norm(actions, axis=1).sum())
    return -dv_total

def _generate_one_sample(_: int) -> Optional[Dict[str, object]]:
    # 1. Pick Campaign
    camp_type = np.random.choice(list(POLICY_REGISTRY.keys()))
    policy = POLICY_REGISTRY[camp_type]

    # 2. Initialize
    start_node = np.random.choice(policy.get_valid_start_nodes())
    current_state = NODES[start_node].sample()
    current_node_id = start_node

    # Buffers
    states = [current_state]  # x0
    behaviors = []            # b1, b2...
    dts_orbit = []            # dt1, dt2...

    # 3. Walk the Policy
    step = 0
    while True:
        res = policy.get_next_step(current_node_id, step)
        if res is None:
            break

        next_node, beh, (t_min, t_max) = res

        # Handle "Dummy" ops (Logic for "If already there, do nothing")
        if beh == 0:
            step += 1
            continue

        # Generate Data
        # For station-keeping (behavior_id == 1), keep the state unchanged
        if beh == 1:
            next_state = current_state.copy()
        else:
            next_state = NODES[next_node].sample()
        dt_orbit = float(np.random.choice(np.linspace(t_min, t_max, 3)))

        states.append(next_state)
        behaviors.append(beh)
        dts_orbit.append(dt_orbit)
        
        current_node_id = next_node
        current_state = next_state
        step += 1

    # Problem-specific parameters
    koz_dim = _sample_koz_dim()
    artms_scale_range_1e3 = _sample_artms_param()
    oec0 = _sample_oec0()
    oec0_modified = modify_koe(oec0)
    n_time, tvec_sec = time_grid_from_orbits(
        float(np.sum(dts_orbit)),
        DT_SEC,
        N_TIME_MAX,
        oec0[0],
    )

    # Build OCP for cost/constraint evaluation
    if len(states) < 2:
        return None
    x0 = states[0]
    xf = states[-1]
    waypoints = states[1:-1]
    t_idx_wyp = waypoint_times_from_dts(dts_orbit, n_time)
    # convert orbit-based durations to integer timestep durations
    times = [0] + t_idx_wyp + [n_time - 1]
    dt_steps = np.diff(times).astype(int)
    tof_steps = int(n_time - 1)
    if tof_steps <= 0:
        return None
    dt_frac = dt_steps.astype(np.float32) / float(tof_steps)

    current_obs = {
        "state": x0,
        "goal": xf,
        "ttg": tvec_sec[-1],
        "dt": DT_SEC,
        "oe": oec0,
    }
    prob = NonConvexOCP(
        prob_definition={
            "t_i": 0,
            "t_f": n_time,
            "tvec_sec": tvec_sec,
            "chance": True,
            "ct": False,
            "current_obs": current_obs,
            "waypoint_times": t_idx_wyp,
            "waypoints": waypoints,
            "koz_dim": koz_dim,
            "artms_scale_range_1e3": artms_scale_range_1e3,
        }
    )
    sol_cvx = prob.ocp_cvx()
    if sol_cvx["status"] not in ["optimal", "optimal_inaccurate"]:
        return None

    roe_cvx = sol_cvx["z"]["state"]
    actions_cvx = sol_cvx["z"]["action"]
    oec = propagate_oe(oec0, tvec_sec)
    rtn_cvx = prob.f_2rtn(roe_cvx, oec)

    # solve SCP 
    prob.zref = {'state': roe_cvx, 'action': actions_cvx}
    prob.sol_0 = {"z": prob.zref}
    prob.generate_scaling(roe_cvx, actions_cvx)
    sol_scp, log_scp = solve_scvx(prob)
    feas_scp = sol_scp['status']
    # print(f"campaign type: {camp_type}, SCP Solution Status: {feas_scp}")

    if feas_scp not in ['optimal', 'optimal_inaccurate']:
        converged = False
        reward = -1.0
        rtn_scp = None
    else:
        converged = True
        roe_scp = sol_scp["z"]["state"]
        actions_scp = sol_scp["z"]["action"]
        rtn_scp = prob.f_2rtn(roe_scp, oec)
        reward = _compute_reward(prob, roe_scp, actions_scp)

    # 4. Pack Row
    row = {
        # input (X)
        "x0": x0,
        "tof": tof_steps,
        "oec0_modified": oec0_modified,
        "koz_dim": koz_dim[0],
        "artms_scale_range_1e3": artms_scale_range_1e3,
        "b_seq": behaviors,
        # output (y)
        "x_seq": states[1:],
        "dt_seq": dt_frac,
        # reward
        "reward": reward,
        "converged": converged,
        # other info
        "campaign": camp_type,
        "rtn_cvx": rtn_cvx,
        "rtn_scp": rtn_scp,
    }
    return row

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


def generate_dataset(num_samples=10, num_workers: Optional[int] = 1, max_phase: int = 3):
    """
    Generate a fixed-size dataset using pre-allocated tensors.
    Saves rows "as is" and computes mean/std for input/output variables.
    """
    # Grab one valid sample to infer shapes
    first_row = None
    while first_row is None:
        first_row = _generate_one_sample(0)

    oe_dim = int(np.atleast_1d(first_row["oec0_modified"]).shape[-1])
    koz_dim = int(np.atleast_1d(first_row["koz_dim"]).shape[-1])
    artms_dim = int(np.atleast_1d(first_row["artms_scale_range_1e3"]).shape[-1])

    # Pre-allocate tensors
    data = {
        # input (X)
        "x0": torch.zeros((num_samples, 6), dtype=torch.float32),
        "tof": torch.zeros((num_samples, 1), dtype=torch.float32),
        "b_seq": torch.zeros((num_samples, max_phase), dtype=torch.float32),
        "phase_valid": torch.zeros((num_samples, max_phase), dtype=torch.float32),
        "oec0_modified": torch.zeros((num_samples, oe_dim), dtype=torch.float32),
        "koz_dim": torch.zeros((num_samples, koz_dim), dtype=torch.float32),
        "artms_scale_range_1e3": torch.zeros((num_samples, artms_dim), dtype=torch.float32),
        # reward (for analysis)
        "reward": torch.zeros((num_samples,), dtype=torch.float32),
        "converged": torch.zeros((num_samples,), dtype=torch.float32),
        # output (y)
        "x_seq": torch.zeros((num_samples, max_phase, 6), dtype=torch.float32),
        "dt_seq": torch.zeros((num_samples, max_phase), dtype=torch.float32),
    }

    def _fill_from_row(row, idx):
        x0 = torch.as_tensor(row["x0"], dtype=torch.float32)
        b_seq = row["b_seq"]
        dt_seq = row["dt_seq"]
        tof = float(row["tof"])
        x_seq = row["x_seq"]
        reward = float(row["reward"])
        converged = float(bool(row["converged"]))
        if len(b_seq) != len(dt_seq) or len(b_seq) != len(x_seq):
            raise ValueError("b_seq, dt_seq, and x_seq must have the same length.")
        n_phase = len(b_seq)
        if n_phase > max_phase:
            raise ValueError(f"Sequence length {n_phase} exceeds max_phase={max_phase}.")

        b_pad = torch.zeros(max_phase, dtype=torch.float32)
        dt_pad = torch.zeros(max_phase, dtype=torch.float32)
        x_pad = torch.zeros((max_phase, 6), dtype=torch.float32)
        b_pad[:n_phase] = torch.as_tensor(b_seq, dtype=torch.float32)
        dt_pad[:n_phase] = torch.as_tensor(dt_seq, dtype=torch.float32)
        x_pad[:n_phase] = torch.as_tensor(np.asarray(x_seq, dtype=np.float32), dtype=torch.float32)

        phase_valid = torch.zeros(max_phase, dtype=torch.float32)
        phase_valid[:n_phase] = 1.0

        oec0_mod = row["oec0_modified"] if "oec0_modified" in row else row["oec0"]
        oec0_mod = torch.as_tensor(np.asarray(oec0_mod, dtype=np.float32), dtype=torch.float32)

        koz = torch.as_tensor(np.asarray(row["koz_dim"], dtype=np.float32), dtype=torch.float32)
        artms = torch.as_tensor(
            np.asarray(row["artms_scale_range_1e3"], dtype=np.float32), dtype=torch.float32
        )

        data["x0"][idx] = x0
        data["tof"][idx] = torch.tensor([tof], dtype=torch.float32)
        data["b_seq"][idx] = b_pad
        data["dt_seq"][idx] = dt_pad
        data["x_seq"][idx] = x_pad
        data["phase_valid"][idx] = phase_valid
        data["oec0_modified"][idx] = oec0_mod
        data["koz_dim"][idx] = koz
        data["artms_scale_range_1e3"][idx] = artms
        data["reward"][idx] = reward
        data["converged"][idx] = converged

    # Fill first sample
    filled = 0
    _fill_from_row(first_row, filled)
    filled += 1

    # Fill remaining samples
    pbar = tqdm(total=num_samples)
    pbar.update(1)
    if num_workers > 1:
        ctx = get_context("spawn")
        pool = ctx.Pool(processes=num_workers)
        try:
            iterator = pool.imap_unordered(_generate_one_sample, itertools.count())
            for row in iterator:
                if row is None:
                    continue
                _fill_from_row(row, filled)
                filled += 1
                pbar.update(1)
                if filled >= num_samples:
                    pool.terminate()
                    break
        finally:
            pool.join()
    else:
        while filled < num_samples:
            row = _generate_one_sample(filled)
            if row is None:
                continue
            _fill_from_row(row, filled)
            filled += 1
            pbar.update(1)
    pbar.close()

    # shift total reward so all values are positive
    min_reward = float(data["reward"].min().item())
    reward_shift = 0.0
    if min_reward < 0.0:
        reward_shift = -min_reward + 1e-3
        data["reward"] = data["reward"] + reward_shift

    # Compute stats (masked for variable-length sequences)
    phase_valid = data["phase_valid"]
    stats = {
        "x0": {}, "tof": {}, "oec0_modified": {}, "koz_dim": {}, "artms_scale_range_1e3": {}, "b_seq": {},
        "x_seq": {},
    }

    # input (X)
    stats["x0"]["mean"], stats["x0"]["std"] = _masked_mean_std(data["x0"])
    stats["tof"]["mean"], stats["tof"]["std"] = _masked_mean_std(data["tof"])
    stats["oec0_modified"]["mean"], stats["oec0_modified"]["std"] = _masked_mean_std(data["oec0_modified"])
    stats["koz_dim"]["mean"], stats["koz_dim"]["std"] = _masked_mean_std(data["koz_dim"])
    stats["artms_scale_range_1e3"]["mean"], stats["artms_scale_range_1e3"]["std"] = _masked_mean_std(
        data["artms_scale_range_1e3"]
    )
    stats["b_seq"]["mean"], stats["b_seq"]["std"] = _masked_mean_std(
        data["b_seq"], mask=phase_valid
    )
    # output (y)
    stats["x_seq"]["mean"], stats["x_seq"]["std"] = _masked_mean_std(
        data["x_seq"], mask=phase_valid.unsqueeze(-1)
    )
    # dt_seq stores fractions that sum to 1; no stats needed
    
    dataset = {
        "data": data,
        "stats": stats,
        "meta": {
            "num_samples": num_samples,
            "max_phase": max_phase,
            "oe_dim": oe_dim,
            "koz_dim": koz_dim,
            "artms_dim": artms_dim,
            "dt_sec": float(DT_SEC),
            "n_time_max": int(N_TIME_MAX),
            "reward_shift": reward_shift,
        },
    }
    return dataset

if __name__ == "__main__":
    
    N_data = 80_000
    N_proc = 20
    
    dataset = generate_dataset(N_data, num_workers=N_proc, max_phase=3)
    
    # save to file 
    save_path = root_folder / "rpod" / "rages" / "wyp_data" / "test.pth"
    torch.save(dataset, save_path)
    print(f"Dataset with {dataset['meta']['num_samples']} samples saved to {save_path}")
