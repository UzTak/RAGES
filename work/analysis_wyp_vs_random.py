from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


def find_root_path(path: str, word: str) -> str:
    parts = path.split(word, 1)
    return parts[0] + word if len(parts) > 1 else path


ROOT_FOLDER = Path(__file__).resolve().parents[1]

from datagen_reasoning import (
    generate_behavior_seq,
    load_dataset,
)
from optimization.optimization import generate_traj_with_wyp
from parameters import NODES, OK_STATUS
from wyp_predictor import (
    build_input_slices,
    load_model,
    predict_wyp_seq,
)
from datagen_wyp import _compute_reward  
from utils import (
    behavior_seq_to_text,
    classify_orbital_domain,
    contiguous_train_eval_index_ranges,
    recover_target_domains_with_policy,
    sample_state_from_node,
)
import optimization.parameters as param  


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate waypoint generator against random domain-constrained waypoint baseline."
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="rpod/rages/wyp_model/model_gmm_v3_unweighted_one_hot.pt",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="rpod/rages/wyp_data/data_v3_discrete.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rpod/rages/out",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="wyp_vs_random_unweighted_test",
    )
    parser.add_argument("--num-cases", type=int, default=500)
    parser.add_argument("--m-scenarios", type=int, default=1)
    parser.add_argument("--num-process", dest="num_process", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Train-Eval split ratio")
    parser.add_argument("--obj-type", type=str, default="min_fuel", choices=["min_fuel", "feasibility"])
    parser.add_argument(
        "--error-margin",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 10.0],
        help=(
            "Waypoint-domain tolerance values applied equally to d_lambda, d_ey, and d_iy. "
            "Use one or more values to compare shared margins, e.g. 0 5 10."
        ),
    )
    return parser.parse_args()


def _parse_error_margin(raw_margin: Sequence[float]) -> np.ndarray:
    margin_arr = np.asarray(raw_margin, dtype=float)
    if margin_arr.ndim == 2:
        if margin_arr.shape[1] != 2:
            raise ValueError(
                "--error-margin 2D input must have shape (n_margin, 2): "
                "shared [margin, margin] pairs."
            )
        if not np.allclose(margin_arr[:, 0], margin_arr[:, 1]):
            raise ValueError(
                "--error-margin no longer supports different d_lambda/d_ey/d_iy values. "
                "Use shared scalar margins only."
            )
        margin = margin_arr[:, 0].reshape(-1)
    else:
        margin = margin_arr.reshape(-1)

    if margin.size < 1:
        raise ValueError("--error-margin requires at least one value.")
    if np.any(margin < 0.0):
        raise ValueError("--error-margin values must be >= 0.")
    return np.repeat(margin.reshape(-1, 1), 2, axis=1).astype(float)


def _in_range_with_margin(x: float, r: Tuple[float, float], margin: Any) -> np.ndarray:
    lo, hi = float(r[0]), float(r[1])
    margin_arr = np.asarray(margin, dtype=float).reshape(-1)
    return ((lo - margin_arr) <= x) & (x <= (hi + margin_arr))


def _in_multirange_with_margin(x: float, r: Any, margin: Any) -> np.ndarray:
    first = r[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        checks = [_in_range_with_margin(x, rr, margin) for rr in r]
        return np.logical_or.reduce(checks)
    return _in_range_with_margin(x, r, margin)


def _is_state_in_target_domain(
    state: np.ndarray,
    target_domain: str,
    error_margin: np.ndarray,
) -> np.ndarray:
    vol = NODES.get(str(target_domain))
    margins = np.asarray(error_margin, dtype=float)
    if margins.ndim != 2 or margins.shape[1] != 2:
        raise ValueError("error_margin must have shape (n_margin, 2).")

    if vol is None:
        return np.zeros(margins.shape[0], dtype=bool)

    s = np.asarray(state, dtype=float).reshape(-1)
    if s.size < 6:
        return np.zeros(margins.shape[0], dtype=bool)

    dl = float(s[1])
    dey = float(s[3])
    diy = float(s[5])
    return (
        _in_range_with_margin(dl, vol.d_lambda_range, margins[:, 0])
        & _in_multirange_with_margin(dey, vol.d_ex_range, margins[:, 1])
        & _in_multirange_with_margin(diy, vol.d_iy_range, margins[:, 1])
    )


def _is_trajectory_domain_correct(
    x_pred: np.ndarray,
    target_domains: Optional[Sequence[str]],
    error_margin: np.ndarray,
) -> np.ndarray:
    margins = np.asarray(error_margin, dtype=float)
    if margins.ndim != 2 or margins.shape[1] != 2:
        raise ValueError("error_margin must have shape (n_margin, 2).")

    n_margin = int(margins.shape[0])
    if target_domains is None or len(target_domains) != len(x_pred):
        return np.zeros(n_margin, dtype=bool)

    in_domain = np.ones(n_margin, dtype=bool)
    for state, target_domain in zip(x_pred, target_domains):
        in_domain &= _is_state_in_target_domain(
            state=np.asarray(state),
            target_domain=str(target_domain),
            error_margin=margins,
        )
        if not np.any(in_domain):
            break
    return in_domain


def _reward_or_nan(solved: Dict[str, Any]) -> float:
    if solved.get("status_scp") not in OK_STATUS:
        return float("nan")
    rew = _compute_reward(
        prob=solved["prob"],
        roe=solved["roe_scp"],
        actions=solved["actions_scp"],
    )
    return float(rew["reward"])


_CASE_WORKER_DATA = None
_CASE_WORKER_META = None
_CASE_WORKER_MODEL_BUNDLE = None
_CASE_WORKER_INPUT_SLICES = None
_CASE_WORKER_DT_SEC = float(param.dt_sec)
_CASE_WORKER_MAX_PHASE = 0
_CASE_WORKER_M_SCENARIOS = 1
_CASE_WORKER_ERROR_MARGIN = np.zeros((1, 2), dtype=float)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return torch.device(device_arg)


def _move_bundle_to_device(model_bundle: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    model_bundle["model"] = model_bundle["model"].to(device)
    model_bundle["model"].eval()
    model_bundle["device"] = device
    for k in ("y_mean", "y_std", "X_mean", "X_std"):
        v = model_bundle.get(k, None)
        if torch.is_tensor(v):
            model_bundle[k] = v.to(device)
    return model_bundle


def _init_case_worker(
    ckpt_path: str,
    data_path: str,
    m_scenarios: int,
    device_arg: str,
    error_margin: Sequence[float],
) -> None:
    global _CASE_WORKER_DATA
    global _CASE_WORKER_META
    global _CASE_WORKER_MODEL_BUNDLE
    global _CASE_WORKER_INPUT_SLICES
    global _CASE_WORKER_DT_SEC
    global _CASE_WORKER_MAX_PHASE
    global _CASE_WORKER_M_SCENARIOS
    global _CASE_WORKER_ERROR_MARGIN

    _CASE_WORKER_MODEL_BUNDLE = load_model(Path(ckpt_path))
    _CASE_WORKER_MODEL_BUNDLE = _move_bundle_to_device(
        _CASE_WORKER_MODEL_BUNDLE,
        _resolve_device(device_arg),
    )
    _CASE_WORKER_DATA, _CASE_WORKER_META = load_dataset(Path(data_path))
    cfg = _CASE_WORKER_MODEL_BUNDLE["cfg"]
    _CASE_WORKER_INPUT_SLICES = build_input_slices(
        _CASE_WORKER_DATA,
        _CASE_WORKER_MODEL_BUNDLE["inputs_arg"],
        b_seq_encoding=getattr(cfg, "b_seq_encoding", "scalar"),
        b_seq_num_classes=int(getattr(cfg, "b_seq_num_classes", 11)),
    )
    _CASE_WORKER_DT_SEC = float(_CASE_WORKER_META.get("dt_sec", param.dt_sec))
    _CASE_WORKER_MAX_PHASE = int(_CASE_WORKER_MODEL_BUNDLE["cfg"].max_phase)
    _CASE_WORKER_M_SCENARIOS = int(m_scenarios)
    _CASE_WORKER_ERROR_MARGIN = _parse_error_margin(error_margin)


def run_case(
    case_id: int,
    dataset_idx: int,
    scenario_seed: int,
    random_seed: int,
    data: Dict[str, Any],
    model_bundle: Dict[str, Any],
    input_slices: Dict[str, slice],
    dt_sec: float,
    max_phase: int,
    m_scenarios: int,
    obj_type: str,
    error_margin: np.ndarray,
) -> Dict[str, Any]:
    margins = np.asarray(error_margin, dtype=float)
    if margins.ndim != 2 or margins.shape[1] != 2:
        raise ValueError("error_margin must have shape (n_margin, 2).")
    n_margin = int(margins.shape[0])

    x0 = data["x0"][dataset_idx].numpy()
    oec0_mod = data["oec0_modified"][dataset_idx].numpy()
    artms = data["artms_scale_range_1e3"][dataset_idx].numpy()
    koz_dim = data["koz_dim"][dataset_idx].numpy()

    out_rows: List[Dict[str, Any]] = []
    skipped_no_domain = 0
    skipped_no_scenario = 0

    start_domains = classify_orbital_domain(x0)
    if not start_domains:
        skipped_no_domain = 1
        return {
            "case_id": case_id,
            "rows": out_rows,
            "skipped_no_domain": skipped_no_domain,
            "skipped_no_scenario": skipped_no_scenario,
        }
    start_domain = str(start_domains[0])

    scenarios = generate_behavior_seq(
        x0=x0,
        M=m_scenarios,
        max_phase=max_phase,
        seed=scenario_seed,
    )
    if not scenarios:
        skipped_no_scenario = 1
        return {
            "case_id": case_id,
            "rows": out_rows,
            "skipped_no_domain": skipped_no_domain,
            "skipped_no_scenario": skipped_no_scenario,
        }

    rng = np.random.default_rng(int(random_seed))

    for scenario_id, scenario in enumerate(scenarios):
        b_seq = [int(b) for b in scenario["b_seq"]]
        tof_steps = int(scenario["tof_steps"])
        
        # waypoint generation model 
        x_pred_model, dt_pred_model = predict_wyp_seq(
            model_bundle=model_bundle,
            input_slices=input_slices,
            x0=x0,
            tof_steps=tof_steps,
            b_seq=b_seq,
            oec0_mod=oec0_mod,
            artms=artms,
            koz_dim=koz_dim,
        )
        solved_model = generate_traj_with_wyp(
            x0=x0,
            x_pred=x_pred_model,
            dt_pred=dt_pred_model,
            tof_steps=tof_steps,
            koz_dim=koz_dim,
            artms=artms,
            dt_sec=dt_sec,
            obj_type=obj_type
        )
        reward_model = _reward_or_nan(solved_model)

        target_domains = recover_target_domains_with_policy(
            policy_name=str(scenario["policy"]),
            start_node=start_domain,
            b_seq=b_seq,
        )
        traj_domain_correct_model = _is_trajectory_domain_correct(
            x_pred=x_pred_model,
            target_domains=target_domains,
            error_margin=margins,
        )

        # random baseline 
        if target_domains is None or len(target_domains) != len(b_seq):
            status_rand_cvx = "rand_invalid"
            status_rand_scp = "rand_invalid"
            reward_rand = float("nan")
            traj_domain_correct_rand = np.zeros(n_margin, dtype=bool)
        else:
            x_pred_rand = np.stack(
                [sample_state_from_node(node, rng) for node in target_domains],
                axis=0,
            )
            traj_domain_correct_rand = _is_trajectory_domain_correct(
                x_pred=x_pred_rand,
                target_domains=target_domains,
                error_margin=margins,
            )
            dt_pred_rand = np.asarray(scenario["dt_orbits"], dtype=float)
            solved_rand = generate_traj_with_wyp(
                x0=x0,
                x_pred=x_pred_rand,
                dt_pred=dt_pred_rand,
                tof_steps=tof_steps,
                koz_dim=koz_dim,
                artms=artms,
                dt_sec=dt_sec,
                obj_type=obj_type
            )
            status_rand_cvx = str(solved_rand["status_cvx"])
            status_rand_scp = str(solved_rand["status_scp"])
            reward_rand = _reward_or_nan(solved_rand)

        out_rows.append(
            {
                "case_id": case_id,
                "dataset_idx": dataset_idx,
                "scenario_id": scenario_id,
                "seed_scenario": scenario_seed,
                "policy": str(scenario["policy"]),
                "start_domain": start_domain,
                "target_domains": target_domains,
                "b_seq": b_seq,
                "b_seq_text": behavior_seq_to_text(b_seq),
                "tof_steps": tof_steps,
                "status_model_cvx": str(solved_model["status_cvx"]),
                "status_model_scp": str(solved_model["status_scp"]),
                "status_rand_cvx": status_rand_cvx,
                "status_rand_scp": status_rand_scp,
                "reward_model": reward_model,
                "reward_rand": reward_rand,
                "traj_domain_correct_model": bool(traj_domain_correct_model[0]),
                "traj_domain_correct_rand": bool(traj_domain_correct_rand[0]),
                "traj_domain_correct_model_by_margin": np.asarray(traj_domain_correct_model, dtype=bool),
                "traj_domain_correct_rand_by_margin": np.asarray(traj_domain_correct_rand, dtype=bool),
            }
        )

    return {
        "case_id": case_id,
        "rows": out_rows,
        "skipped_no_domain": skipped_no_domain,
        "skipped_no_scenario": skipped_no_scenario,
    }


def _run_case_from_spec(case_spec: Dict[str, int]) -> Dict[str, Any]:
    if (
        _CASE_WORKER_DATA is None
        or _CASE_WORKER_MODEL_BUNDLE is None
        or _CASE_WORKER_INPUT_SLICES is None
    ):
        raise RuntimeError("Case worker context is not initialized.")

    return run_case(
        case_id=int(case_spec["case_id"]),
        dataset_idx=int(case_spec["dataset_idx"]),
        scenario_seed=int(case_spec["scenario_seed"]),
        random_seed=int(case_spec["random_seed"]),
        data=_CASE_WORKER_DATA,
        model_bundle=_CASE_WORKER_MODEL_BUNDLE,
        input_slices=_CASE_WORKER_INPUT_SLICES,
        dt_sec=float(_CASE_WORKER_DT_SEC),
        max_phase=int(_CASE_WORKER_MAX_PHASE),
        m_scenarios=int(_CASE_WORKER_M_SCENARIOS),
        obj_type=str(case_spec.get("obj_type", "min_fuel")),
        error_margin=np.asarray(_CASE_WORKER_ERROR_MARGIN, dtype=float),
    )


def main() -> None:
    args = parse_args()
    error_margin = _parse_error_margin(args.error_margin)
    if args.num_cases < 1:
        raise ValueError("--num-cases must be >= 1")
    if args.m_scenarios < 1:
        raise ValueError("--m-scenarios must be >= 1")
    if args.num_process < 1:
        raise ValueError("--num-process must be >= 1")
    selected_device = _resolve_device(args.device)

    ckpt_path = ROOT_FOLDER / args.ckpt_path
    data_path = ROOT_FOLDER / args.data_path
    output_dir = ROOT_FOLDER / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.save_name}.npz"
    print(
        f"[config] device={selected_device} "
        f"error_margin={np.array2string(error_margin, precision=6)}"
    )

    data_for_specs, _ = load_dataset(data_path)
    n_dataset = int(data_for_specs["x0"].shape[0])
    _, eval_dataset_indices = contiguous_train_eval_index_ranges(
        n_rows=n_dataset,
        val_ratio=float(args.val_ratio),
    )
    if eval_dataset_indices is None:
        raise ValueError("--val-ratio must be > 0 to sample from a validation split.")
    eval_indices = [int(idx) for idx in eval_dataset_indices]
    print(
        f"[config] val_ratio={float(args.val_ratio):.6f} "
        f"eval_pool={len(eval_indices)}"
    )

    spec_rng = np.random.default_rng(int(args.seed))

    case_specs: List[Dict[str, int]] = []
    for i in range(int(args.num_cases)):
        case_specs.append(
            {
                "case_id": i,
                "dataset_idx": int(eval_indices[int(spec_rng.integers(0, len(eval_indices)))]),
                "scenario_seed": int(spec_rng.integers(0, 2**31 - 1)),
                "random_seed": int(spec_rng.integers(0, 2**31 - 1)),
                "obj_type": args.obj_type,  
            }
        )

    case_outs: List[Dict[str, Any]] = []
    if case_specs:
        if args.num_process == 1:
            _init_case_worker(
                str(ckpt_path),
                str(data_path),
                int(args.m_scenarios),
                str(selected_device),
                args.error_margin,
            )
            iterator = (_run_case_from_spec(spec) for spec in case_specs)
            for case in tqdm(iterator, total=len(case_specs), desc="cases", unit="case"):
                case_outs.append(case)
        else:
            n_workers = min(int(args.num_process), len(case_specs))
            with mp.get_context("spawn").Pool(
                processes=n_workers,
                initializer=_init_case_worker,
                initargs=(
                    str(ckpt_path),
                    str(data_path),
                    int(args.m_scenarios),
                    str(selected_device),
                    args.error_margin,
                ),
            ) as pool:
                iterator = pool.imap_unordered(_run_case_from_spec, case_specs)
                for case in tqdm(iterator, total=len(case_specs), desc="cases", unit="case"):
                    case_outs.append(case)

    case_outs.sort(key=lambda c: int(c["case_id"]))
    rows: List[Dict[str, Any]] = [r for c in case_outs for r in c["rows"]]
    n_skipped_no_domain = int(sum(int(c["skipped_no_domain"]) for c in case_outs))
    n_skipped_no_scenario = int(sum(int(c["skipped_no_scenario"]) for c in case_outs))

    if not rows:
        raise RuntimeError("No evaluation rows generated. Check inputs / scenario generation.")

    def col(name: str, dtype: Optional[Any] = None) -> np.ndarray:
        vals = [r[name] for r in rows]
        if dtype is None:
            return np.asarray(vals)
        return np.asarray(vals, dtype=dtype)

    status_model_cvx = col("status_model_cvx", dtype=object)
    status_model_scp = col("status_model_scp", dtype=object)
    status_rand_cvx = col("status_rand_cvx", dtype=object)
    status_rand_scp = col("status_rand_scp", dtype=object)
    traj_domain_correct_model = col("traj_domain_correct_model", dtype=bool)
    traj_domain_correct_rand = col("traj_domain_correct_rand", dtype=bool)
    traj_domain_correct_model_by_margin = np.stack(col("traj_domain_correct_model_by_margin"), axis=0).astype(bool)
    traj_domain_correct_rand_by_margin = np.stack(col("traj_domain_correct_rand_by_margin"), axis=0).astype(bool)

    i_infeas_model_cvx = np.where(~np.isin(status_model_cvx, list(OK_STATUS)))[0]
    i_infeas_model_scp = np.where(~np.isin(status_model_scp, list(OK_STATUS)))[0]
    i_infeas_rand_cvx = np.where(~np.isin(status_rand_cvx, list(OK_STATUS)))[0]
    i_infeas_rand_scp = np.where(~np.isin(status_rand_scp, list(OK_STATUS)))[0]
    i_rand_invalid = np.where(status_rand_scp == "rand_invalid")[0]

    np.savez_compressed(
        out_path,
        case_id=col("case_id", dtype=int),
        dataset_idx=col("dataset_idx", dtype=int),
        scenario_id=col("scenario_id", dtype=int),
        seed_scenario=col("seed_scenario", dtype=int),
        policy=col("policy", dtype=object),
        start_domain=col("start_domain", dtype=object),
        target_domains=col("target_domains", dtype=object),
        b_seq=col("b_seq", dtype=object),
        b_seq_text=col("b_seq_text", dtype=object),
        tof_steps=col("tof_steps", dtype=int),
        reward_model=col("reward_model", dtype=float),
        reward_rand=col("reward_rand", dtype=float),
        status_model_cvx=status_model_cvx,
        status_model_scp=status_model_scp,
        status_rand_cvx=status_rand_cvx,
        status_rand_scp=status_rand_scp,
        traj_domain_correct_model=traj_domain_correct_model,
        traj_domain_correct_rand=traj_domain_correct_rand,
        traj_domain_correct_model_by_margin=traj_domain_correct_model_by_margin,
        traj_domain_correct_rand_by_margin=traj_domain_correct_rand_by_margin,
        i_infeas_model_cvx=i_infeas_model_cvx.astype(int),
        i_infeas_model_scp=i_infeas_model_scp.astype(int),
        i_infeas_rand_cvx=i_infeas_rand_cvx.astype(int),
        i_infeas_rand_scp=i_infeas_rand_scp.astype(int),
        i_rand_invalid=i_rand_invalid.astype(int),
        num_cases_requested=np.array([int(args.num_cases)], dtype=int),
        m_scenarios_requested=np.array([int(args.m_scenarios)], dtype=int),
        num_rows=np.array([len(rows)], dtype=int),
        skipped_no_domain=np.array([n_skipped_no_domain], dtype=int),
        skipped_no_scenario=np.array([n_skipped_no_scenario], dtype=int),
        seed=np.array([int(args.seed)], dtype=int),
        val_ratio=np.array([float(args.val_ratio)], dtype=float),
        num_process=np.array([int(args.num_process)], dtype=int),
        device=np.array([str(selected_device)], dtype=object),
        error_margin=np.asarray(error_margin, dtype=float),
    )

    n_model_scp_ok = int(np.sum(np.isin(status_model_scp, list(OK_STATUS))))
    n_rand_scp_ok = int(np.sum(np.isin(status_rand_scp, list(OK_STATUS))))
    n_model_domain_ok = int(np.sum(traj_domain_correct_model))
    n_rand_domain_ok = int(np.sum(traj_domain_correct_rand))
    n_model_domain_ok_by_margin = np.sum(traj_domain_correct_model_by_margin, axis=0).astype(int)
    n_rand_domain_ok_by_margin = np.sum(traj_domain_correct_rand_by_margin, axis=0).astype(int)
    print(f"[done] saved: {out_path}")
    print(
        f"[done] rows={len(rows)} "
        f"model_scp_ok={n_model_scp_ok}/{len(rows)} "
        f"rand_scp_ok={n_rand_scp_ok}/{len(rows)}"
    )
    print(
        f"[done] traj_domain_ok_model={n_model_domain_ok}/{len(rows)} "
        f"traj_domain_ok_rand={n_rand_domain_ok}/{len(rows)}"
    )
    for i in range(int(error_margin.shape[0])):
        dl_margin = float(error_margin[i, 0])
        dey_margin = float(error_margin[i, 1])
        print(
            f"[done] traj_domain_ok@margin[{i}] "
            f"(d_lambda={dl_margin:.6g}, d_ey/d_iy={dey_margin:.6g}) "
            f"model={int(n_model_domain_ok_by_margin[i])}/{len(rows)} "
            f"rand={int(n_rand_domain_ok_by_margin[i])}/{len(rows)}"
        )


if __name__ == "__main__":
    main()
