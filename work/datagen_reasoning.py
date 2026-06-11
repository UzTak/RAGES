from __future__ import annotations

import argparse
import json
import os, sys
import re
import multiprocessing as mp
from dataclasses import MISSING
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

def find_root_path(path: str, word: str) -> str:
    parts = path.split(word, 1)
    return parts[0] + word if len(parts) > 1 else path


ROOT_FOLDER = Path(__file__).resolve().parents[1]

from optimization.optimization import NonConvexOCP
from optimization.scvx import solve_scvx
import optimization.parameters as param
from dynamics.dynamics_trans import propagate_oe, propagate_ct, restore_koe
from parameters import (
    DEFAULT_B_SEQ_ENCODING,
    DEFAULT_B_SEQ_NUM_CLASSES,
    DEFAULT_INTENT_PRIORITY,
    DT_SEC,
    INTENT_TO_METRIC,
    METRIC_PREF,
    N_TIME_MAX,
    OK_STATUS,
    POLICY_REGISTRY,
)
from train_wyp_predictor import (
    ConditionalGMM,
    ConditionalVAE,
    constrained_fill,
    scale_var,
    unscale_var,
    FillerConfig,
    build_input_from_data,
    build_input_slices,
)
from datagen_wyp import (
    enumerate_policy_paths,
)
from utils import (
    behavior_seq_to_text,
    classify_orbital_domain,
    time_grid_from_orbits,
    waypoint_times_from_dts,
)
BASE_SEED = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data generation of reasoning traces.")
    parser.add_argument(
        "--ckpt-path", type=str,
        default="rpod/rages/wyp_model/model_gmm_v4_weighted_one_hot.pt",
    )
    parser.add_argument(
        "--data-path", type=str,
        default="rpod/rages/wyp_data/data_v4.pth",
    )
    parser.add_argument(
        "--dataset-path", type=str,
        default="rpod/rages/reasoning_data/reasoning_dataset30k_v4.json",
    )
    
    parser.add_argument("--num-cases", type=int, default=30000)
    parser.add_argument("--m-candidates", type=int, default=4)
    parser.add_argument("--num-process", dest="num_process", type=int, default=25)
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")

    return parser.parse_args()


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


def load_dataset(data_path: Path):
    dataset = torch.load(data_path, map_location="cpu")
    return dataset["data"], dataset["meta"]


def generate_behavior_seq(
    x0: np.ndarray,
    M: int,
    max_phase: int,
    seed: int,
) -> List[Dict[str, Any]]:
    start_nodes = classify_orbital_domain(x0)
    if not start_nodes:
        return []
    start_node = start_nodes[0]

    all_sequences = []
    for name, pol in POLICY_REGISTRY.items():
        if start_node not in pol.get_valid_start_nodes():
            continue
        seqs = enumerate_policy_paths(pol, start_node, max_steps=max_phase)
        for beh_seq, dt_ranges in seqs:
            if 0 < len(beh_seq) <= max_phase:
                all_sequences.append((name, beh_seq, dt_ranges))

    if not all_sequences:
        return []

    rng = np.random.default_rng(seed)
    scenarios = []
    for _ in range(M):
        name, beh_seq, dt_ranges = all_sequences[int(rng.integers(len(all_sequences)))]
        dt_orbits = [float(rng.uniform(lo, hi)) for (lo, hi) in dt_ranges]
        total_orbits = float(np.sum(dt_orbits))
        n_time, _ = time_grid_from_orbits(total_orbits, DT_SEC, N_TIME_MAX, param.oec0[0])
        tof_steps = int(n_time - 1)
        scenarios.append(
            {
                "policy": name,
                "b_seq": [int(b) for b in beh_seq],
                "tof_steps": tof_steps,
                "dt_orbits": dt_orbits,
            }
        )
    return scenarios


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


def _restore_oec0_modified(oec0_mod: np.ndarray) -> np.ndarray:
    oec0 = np.asarray(restore_koe(np.asarray(oec0_mod, dtype=float)), dtype=float).reshape(-1)
    if oec0.shape != (6,):
        raise ValueError(f"Restored OE must have shape (6,), got {oec0.shape}")
    return oec0


def generate_traj_with_wyp(
    x0: np.ndarray,
    x_pred: np.ndarray,
    dt_pred: np.ndarray,
    tof_steps: int,
    koz_dim: np.ndarray,
    artms: np.ndarray,
    dt_sec: float,
    oec0_mod: np.ndarray | None = None,
    obj_type: str = "min_fuel",
) -> Dict[str, Any]:
    n_time = int(tof_steps) + 1
    if n_time < 2:
        return {"status_cvx": "invalid_tof", "status_scp": "invalid_tof"}

    tvec_sec = np.arange(n_time, dtype=float) * dt_sec
    oec0 = _restore_oec0_modified(oec0_mod) if oec0_mod is not None else np.asarray(param.oec0, dtype=float)
    t_idx_wyp = waypoint_times_from_dts(list(dt_pred), n_time)
    wyp = x_pred[:-1] if len(x_pred) > 1 else np.empty((0, 6))
    goal = x_pred[-1] if len(x_pred) > 0 else x0

    current_obs = {"state": x0, "goal": goal, "ttg": tvec_sec[-1], "dt": dt_sec, "oe": oec0}
    prob = NonConvexOCP(
        prob_definition={
            "t_i": 0,
            "t_f": n_time,
            "tvec_sec": tvec_sec,
            "chance": True,
            "ct": False,
            "current_obs": current_obs,
            "waypoint_times": t_idx_wyp,
            "waypoints": wyp,
            "waypoint_type": "roe",
            "koz_dim": koz_dim,
            "artms_scale_range_1e3": artms,
        }
    )

    sol_dict = {
        "wyp": x_pred,  # include the terminal state 
        "t_idx_wyp": t_idx_wyp,
    }

    # try cvx (waypoint hopping only) 
    sol_cvx = prob.ocp_cvx()
    status_cvx = sol_cvx["status"]
    if status_cvx not in OK_STATUS:
        sol_dict.update({"status_cvx": status_cvx, "status_scp": "cvx_failed"})
        return sol_dict
    roe_cvx = sol_cvx["z"]["state"]
    actions_cvx = sol_cvx["z"]["action"]

    # SCP 
    prob.zref = {"state": roe_cvx, "action": actions_cvx}
    prob.sol_0 = {"z": prob.zref}
    prob.generate_scaling(roe_cvx, actions_cvx)
    
    if obj_type == "feasibility":
        prob.type = "feasibility"
        prob._cvx_built_AL = False; prob.update_flag = True
    
    sol_scp, _ = solve_scvx(prob)
    status_scp = sol_scp["status"]
    if status_scp not in OK_STATUS:
        rtn_cvx = prob.f_2rtn(roe_cvx, propagate_oe(oec0, tvec_sec))
        rtn_cvx_ct = propagate_ct(roe_cvx, actions_cvx, propagate_oe(oec0, tvec_sec), tvec_sec, n=10)
        sol_dict.update({"status_cvx": status_cvx, 
                         "status_scp": status_scp,
                         "prob": prob,
                         "roe_cvx": roe_cvx,
                         "actions_cvx": actions_cvx,
                         "rtn_cvx": rtn_cvx,
                         "rtn_cvx_ct": rtn_cvx_ct,
                         })
        return sol_dict

    roe_scp = sol_scp["z"]["state"]
    actions_scp = sol_scp["z"]["action"]
    oec = propagate_oe(oec0, tvec_sec)
    rtn_scp = prob.f_2rtn(roe_scp, oec)
    _, _, rtn_scp_ct = propagate_ct(roe_scp, actions_scp, oec, tvec_sec, n=10)
    
    sol_dict.update({
        "status_cvx": status_cvx,
        "status_scp": status_scp,
        "prob": prob,
        "roe_scp": roe_scp,
        "actions_scp": actions_scp,
        "rtn_scp": rtn_scp,
        "rtn_scp_ct": rtn_scp_ct,
    })
    return sol_dict


def compute_obs_score(prob: NonConvexOCP, roe: np.ndarray) -> float:
    oec = propagate_oe(prob.oe_i, prob.tvec_sec)
    rtn = prob.f_2rtn(roe, oec)
    ranges = np.linalg.norm(rtn[:, :3], axis=1)
    koz_radius = float(np.atleast_1d(prob.koz_dim).ravel()[0])
    threshold = koz_radius + 50.0

    within = ranges <= threshold
    if not np.any(within):
        return 0.0
    first_idx = int(np.argmax(within))
    last_idx = int(len(within) - 1 - np.argmax(within[::-1]))
    region = ranges[first_idx : last_idx + 1]
    return float(-np.sum(region) / prob.n_time)


def compute_metrics(prob: NonConvexOCP, roe: np.ndarray, actions: np.ndarray, rtn_ct: np.ndarray) -> Dict[str, float]:
    fuel_dv = float(np.linalg.norm(actions, axis=1).sum())
    transfer_time_sec = float(prob.tvec_sec[-1])
    observation_score = compute_obs_score(prob, roe)

    ranges_ct = np.linalg.norm(rtn_ct[:, :3], axis=1)
    min_separation_m = float(np.min(ranges_ct))
    koz_radius = float(np.atleast_1d(prob.koz_dim).ravel()[0])
    safety_margin_m = float(min_separation_m - koz_radius)

    return {
        "fuel_dv": fuel_dv,
        "transfer_time_sec": transfer_time_sec,
        "observation_score": observation_score,
        "safety_margin_m": safety_margin_m,
    }


def rank_det(candidates: List[Dict[str, Any]], intent_priority: Sequence[str]) -> Optional[int]:
    feasible = [c for c in candidates if c["status_scp"] in OK_STATUS]
    if not feasible:
        return None

    priority_metrics = [INTENT_TO_METRIC[p] for p in intent_priority]

    def _sort_key(c: Dict[str, Any]):
        invalid = 0
        key = []
        for metric in priority_metrics:
            val = float(c.get(metric, np.nan))
            if not np.isfinite(val):
                invalid = 1
                val = np.inf
            if METRIC_PREF[metric] == "min":
                key.append(val)
            else:
                key.append(-val)
        key.append(int(c["candidate_id"]))
        return tuple([invalid] + key)

    best = min(feasible, key=_sort_key)
    return int(best["candidate_id"])


def format_table(candidates: List[Dict[str, Any]]) -> str:
    headers = ["id", "policy", "fuel_dv", "time_sec", "obs", "safety_margin"]
    lines = [",".join(headers)]
    for c in candidates:
        if c["status_scp"] not in OK_STATUS:
            continue
        lines.append(
            ",".join(
                [
                    str(c["candidate_id"]),
                    str(c["policy"]),
                    f"{float(c['fuel_dv']):.6f}",
                    f"{float(c['transfer_time_sec']):.3f}",
                    f"{float(c['observation_score']):.6f}",
                    f"{float(c['safety_margin_m']):.6f}",
                ]
            )
        )
    return "\n".join(lines)


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def rank_with_llm(
    candidates: List[Dict[str, Any]],
    intent_priority: Sequence[str],
    model: str,
    api_key: str,
) -> Dict[str, Any]:
    feasible = [c for c in candidates if c["status_scp"] in OK_STATUS]
    if not feasible:
        return {"best_candidate_id": None, "one_line_reason": "No feasible candidates."}

    priority_metrics = [INTENT_TO_METRIC[p] for p in intent_priority]
    table_csv = format_table(candidates)
    priority_text = " > ".join(intent_priority)
    metric_text = ", ".join(priority_metrics)

    system_msg = (
        "You're an expert spacecraft operator for rendezvous missions.\n"
        "You select one trajectory candidate from metric tables. "
        "Follow the priority order (lexicographic), not weighted sum. "
        "Output only valid JSON. "
        "For one_line_reason, use probabilistic wording to describe why the selected candidate seems favorable based on its metrics and the intent priority."
        "Avoid absolute superlatives or explicit comparisons, as the candidates may have tradeoffs and there are no guarantees."
        "Focus on the strengths of the chosen candidate in relation to the mission intent, without directly stating it is the best or comparing it to others. "
    )
    user_msg = (
        f"Priority order: {priority_text}\n"
        f"Metrics: {metric_text}\n"
        "Rules:\n"
        "- Lower is better for fuel_dv and time_sec.\n"
        "- Higher is better for obs and safety_margin.\n"
        "- All candidates are already safe; safety_margin is a metric of conservatism.\n\n"
        f"Candidates CSV:\n{table_csv}\n\n"
        'Return JSON with keys: {"best_candidate_id": <int>, "one_line_reason": "<short sentence>"}\n'
        "Reasoning style constraints:\n"
        "- one_line_reason must be exactly one short sentence.\n"
        "- Do not mention candidate IDs or names.\n"
        "- Avoid comparative/superlative words: lower, higher, lowest, highest, better, best, worse, worst, more, less.\n"
        "- Avoid ranking symbols or explicit comparisons: >, <, >=, <=, versus, than.\n"
        "- Prefer probabilistic phrasing like: it seems to lead to low delta-v, is expected to keep transfer time short, has a high chance of supporting observation.\n"
    )

    client = OpenAI(api_key=api_key)
    rsp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=1.0,
        max_tokens=300,
    )
    raw = (rsp.choices[0].message.content or "").strip()
    parsed = _extract_json_obj(raw)
    if parsed is None:
        return {"best_candidate_id": None, "one_line_reason": "LLM response parse failed.", "raw": raw}

    out_id = parsed.get("best_candidate_id", None)
    reason = str(parsed.get("one_line_reason", "")).strip()
    valid_ids = {int(c["candidate_id"]) for c in feasible}
    if not isinstance(out_id, int) or out_id not in valid_ids:
        return {"best_candidate_id": None, "one_line_reason": "LLM selected invalid candidate.", "raw": raw}
    return {"best_candidate_id": out_id, "one_line_reason": reason, "raw": raw}


def _build_candidate_row(
    case_id: int,
    dataset_idx: int,
    candidate_id: int,
    scenario: Dict[str, Any],
    model_bundle: Dict[str, Any],
    input_slices: Dict[str, slice],
    x0: np.ndarray,
    oec0_mod: np.ndarray,
    artms: np.ndarray,
    koz_dim: np.ndarray,
    dt_sec: float,
    use_mean_w: bool = True,
) -> Dict[str, Any]:
    b_seq = scenario["b_seq"]
    
    # waypoint prediction using the trained model
    x_pred, dt_pred = predict_wyp_seq(
        model_bundle=model_bundle,
        input_slices=input_slices,
        x0=x0,
        tof_steps=scenario["tof_steps"],
        b_seq=b_seq,
        oec0_mod=oec0_mod,
        artms=artms,
        koz_dim=koz_dim,
        use_mean_w=use_mean_w
    )
    
    # generate trajectory using SCP with the waypoint constr. 
    solved = generate_traj_with_wyp(
        x0=x0,
        x_pred=x_pred,
        dt_pred=dt_pred,
        tof_steps=scenario["tof_steps"],
        koz_dim=koz_dim,
        artms=artms,
        dt_sec=dt_sec,
    )

    row: Dict[str, Any] = {
        "case_id": case_id,
        "dataset_idx": dataset_idx,
        "candidate_id": candidate_id,
        "policy": scenario["policy"],
        "b_seq": [int(b) for b in b_seq],
        "b_seq_text": behavior_seq_to_text(b_seq),
        "tof_steps": int(scenario["tof_steps"]),
        "status_cvx": solved["status_cvx"],
        "status_scp": solved["status_scp"],
        "fuel_dv": np.nan,
        "transfer_time_sec": np.nan,
        "observation_score": np.nan,
        "min_separation_m": np.nan,
        "safety_margin_m": np.nan,
    }
    
    if solved["status_scp"] in OK_STATUS:
        metrics = compute_metrics(
            prob=solved["prob"],
            roe=solved["roe_scp"],
            actions=solved["actions_scp"],
            rtn_ct=solved["rtn_scp_ct"],
        )
        row.update(metrics)
    return row


_CASE_WORKER_DATA: Optional[Dict[str, torch.Tensor]] = None
_CASE_WORKER_META: Optional[Dict[str, Any]] = None
_CASE_WORKER_MODEL_BUNDLE: Optional[Dict[str, Any]] = None
_CASE_WORKER_INPUT_SLICES: Optional[Dict[str, slice]] = None
_CASE_WORKER_M_CANDIDATES: int = 0
_CASE_WORKER_LLM_MODEL: str = ""
_CASE_WORKER_API_KEY: str = ""


def _init_case_worker(
    ckpt_path: str,
    data_path: str,
    m_candidates: int,
    llm_model: str,
    llm_api_key: str,
) -> None:
    global _CASE_WORKER_DATA
    global _CASE_WORKER_META
    global _CASE_WORKER_MODEL_BUNDLE
    global _CASE_WORKER_INPUT_SLICES
    global _CASE_WORKER_M_CANDIDATES
    global _CASE_WORKER_LLM_MODEL
    global _CASE_WORKER_API_KEY

    # Load heavy objects once per worker process, then reuse for all assigned cases.
    _CASE_WORKER_MODEL_BUNDLE = load_model(Path(ckpt_path))
    _CASE_WORKER_DATA, _CASE_WORKER_META = load_dataset(Path(data_path))
    cfg = _CASE_WORKER_MODEL_BUNDLE["cfg"]
    _CASE_WORKER_INPUT_SLICES = build_input_slices(
        _CASE_WORKER_DATA,
        _CASE_WORKER_MODEL_BUNDLE["inputs_arg"],
        b_seq_encoding=getattr(cfg, "b_seq_encoding", DEFAULT_B_SEQ_ENCODING),
        b_seq_num_classes=int(getattr(cfg, "b_seq_num_classes", DEFAULT_B_SEQ_NUM_CLASSES)),
    )
    _CASE_WORKER_M_CANDIDATES = int(m_candidates)
    _CASE_WORKER_LLM_MODEL = str(llm_model)
    _CASE_WORKER_API_KEY = str(llm_api_key)


def _run_case_from_spec(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    if (
        _CASE_WORKER_DATA is None
        or _CASE_WORKER_META is None
        or _CASE_WORKER_MODEL_BUNDLE is None
        or _CASE_WORKER_INPUT_SLICES is None
    ):
        raise RuntimeError("Case worker context is not initialized.")

    # Run one full case in this process; candidate SCP remains sequential within the case.
    return run_case(
        case_id=int(case_spec["case_id"]),
        dataset_idx=int(case_spec["dataset_idx"]),
        data=_CASE_WORKER_DATA,
        meta=_CASE_WORKER_META,
        model_bundle=_CASE_WORKER_MODEL_BUNDLE,
        input_slices=_CASE_WORKER_INPUT_SLICES,
        M=_CASE_WORKER_M_CANDIDATES,
        intent_priority=case_spec["intent_priority"],
        llm_model=_CASE_WORKER_LLM_MODEL,
        llm_api_key=_CASE_WORKER_API_KEY,
        use_mean_w=True,
    )


def run_case(
    case_id: int,
    dataset_idx: int,
    data: Dict[str, torch.Tensor],
    meta: Dict[str, Any],
    model_bundle: Dict[str, Any],
    input_slices: Dict[str, slice],
    M: int,
    intent_priority: Sequence[str],
    llm_model: str,
    llm_api_key: str,
    use_mean_w: bool = True,
) -> Dict[str, Any]:
    x0 = data["x0"][dataset_idx].numpy()
    oec0_mod = data["oec0_modified"][dataset_idx].numpy()
    artms = data["artms_scale_range_1e3"][dataset_idx].numpy()
    koz_dim = data["koz_dim"][dataset_idx].numpy()
    dt_sec = float(meta.get("dt_sec", param.dt_sec))
    max_phase = model_bundle["cfg"].max_phase

    scenarios = generate_behavior_seq(
        x0=x0,
        M=M,
        max_phase=max_phase,
        seed=BASE_SEED + case_id,
    )
    candidates: List[Dict[str, Any]] = []
    for i, scenario in enumerate(scenarios):
        candidates.append(
            _build_candidate_row(
                case_id=case_id,
                dataset_idx=dataset_idx,
                candidate_id=i,
                scenario=scenario,
                model_bundle=model_bundle,
                input_slices=input_slices,
                x0=x0,
                oec0_mod=oec0_mod,
                artms=artms,
                koz_dim=koz_dim,
                dt_sec=dt_sec,
                use_mean_w=use_mean_w
            )
        )

    llm_pick = {"best_candidate_id": None, "one_line_reason": ""}
    try:
        llm_pick = rank_with_llm(
            candidates=candidates,
            intent_priority=intent_priority,
            model=llm_model,
            api_key=llm_api_key,
        )
    except Exception as e:
        llm_pick = {"best_candidate_id": None, "one_line_reason": f"LLM error: {e.__class__.__name__}"}

    for c in candidates:
        c["selected_llm"] = int(c["candidate_id"] == llm_pick["best_candidate_id"]) if llm_pick["best_candidate_id"] is not None else 0

    return {
        "case_id": case_id,
        "dataset_idx": dataset_idx,
        "x0": np.asarray(x0, dtype=float).tolist(),
        "oec0_modified": np.asarray(oec0_mod, dtype=float).tolist(),
        "koz_param": np.asarray(koz_dim, dtype=float).tolist(),
        "artms_scaling_1e3": np.asarray(artms, dtype=float).tolist(),
        "intent_priority": list(intent_priority),
        "num_candidates_requested": int(M),
        "num_candidates_generated": len(candidates),
        "num_candidates_solved": int(sum(1 for c in candidates if c["status_scp"] in OK_STATUS)),
        "llm_best_candidate_id": llm_pick["best_candidate_id"],
        "llm_one_line_reason": llm_pick.get("one_line_reason", ""),
        "candidates": candidates,
    }


def save_final_dataset(dataset_path: Path, cases: List[Dict[str, Any]]) -> Path:
    def _to_int_or_none(v: Any) -> Optional[int]:
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = dataset_path

    records: List[Dict[str, Any]] = []
    for case in cases:
        
        # Does the LLM pick a valid candidate? If not we skip this case for the reasoning dataset
        selected_id_int = _to_int_or_none(case.get("llm_best_candidate_id"))
        if selected_id_int is None:
            continue

        # Find the selected candidate details to include in the reasoning dataset. If not found or invalid, skip this case.
        selected = next(
            (c for c in case.get("candidates", []) if _to_int_or_none(c.get("candidate_id")) == selected_id_int),
            None,
        )
        if selected is None or selected.get("b_seq") is None:
            continue
        
        # final format for each record in the reasoning dataset json:
        records.append(
            {
                "input": {
                    "x0": np.round(case["x0"], 2).tolist(),
                    "oec0_modified": np.round(case["oec0_modified"], 6).tolist(),
                    "koz_param": np.round(case["koz_param"], 4).tolist(),
                    "artms_scaling_1e3": np.round(case["artms_scaling_1e3"], 3).tolist(),
                    "intent_priority": case["intent_priority"],
                },
                "output": {
                    "reasoning": case["llm_one_line_reason"],
                    "tf": int(selected["tof_steps"]),
                    "b_seq": [int(b) for b in selected["b_seq"]],
                },
            }
        )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return json_path


def main():
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set!")

    ckpt_path = ROOT_FOLDER / args.ckpt_path
    data_path = ROOT_FOLDER / args.data_path
    dataset_path = ROOT_FOLDER / args.dataset_path

    # Only needed here to sample dataset indices deterministically before dispatch.
    data, _ = load_dataset(data_path)
    n_dataset = int(data["x0"].shape[0])
    rng = np.random.default_rng()

    case_specs: List[Dict[str, Any]] = []
    for i in range(args.num_cases):
        case_specs.append(
            {
                "case_id": i,
                "dataset_idx": int(rng.integers(0, n_dataset)),
                "intent_priority": rng.permutation(DEFAULT_INTENT_PRIORITY).tolist(),
            }
        )

    cases: List[Dict[str, Any]] = []
    if case_specs:
        n_workers = min(args.num_process, len(case_specs))
        with mp.get_context("spawn").Pool(
            processes=n_workers,
            initializer=_init_case_worker,
            initargs=(
                str(ckpt_path),
                str(data_path),
                args.m_candidates,
                args.llm_model,
                api_key,
            ),
        ) as pool:
            iterator = pool.imap_unordered(_run_case_from_spec, case_specs)
            for case in tqdm(iterator, total=len(case_specs), desc="cases", unit="case"):
                cases.append(case)
                # print(
                #     f"[case {case['case_id']}] idx={case['dataset_idx']} "
                #     f"solved={case['num_candidates_solved']}/{case['num_candidates_generated']} "
                #     f"llm_best={case['llm_best_candidate_id']} "
                #     f"llm_reason: {case['llm_one_line_reason']}"
                # )
        cases.sort(key=lambda c: int(c["case_id"]))

    json_path = save_final_dataset(dataset_path, cases)
    print(f"saved reasoning dataset json: {json_path}")


if __name__ == "__main__":
    main()
