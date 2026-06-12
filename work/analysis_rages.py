from __future__ import annotations

import argparse
import multiprocessing as mp
import os, sys 
import re 
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from openai import OpenAI

def find_root_path(path: str, word: str) -> str:
    parts = path.split(word, 1)
    return parts[0] + word if len(parts) > 1 else path


ROOT_FOLDER = Path(__file__).resolve().parents[1]

from dynamics.dynamics_trans import propagate_ct, propagate_oe, restore_koe  
from optimization.optimization import NonConvexOCP  
from optimization.scvx import solve_scvx  
import optimization.parameters as param  
from datagen_reasoning import (
    generate_behavior_seq,
    load_dataset,
)
from parameters import DEFAULT_INTENT_PRIORITY, OK_STATUS
from rages_scoring import compute_metrics
from wyp_predictor import (
    build_input_slices,
    load_model as load_wyp_model,
    predict_wyp_seq,
)
from utils import (  
    ReasoningSampler,
    Scenario,
    append_jsonl_line,
    as_float_list,
    contiguous_train_eval_index_ranges,
    extract_first_json_object,
    randomize_intent_priority,
    recover_target_domains_any_policy,
    sample_scenario_from_dataset,
    sample_state_from_node,
    utc_now_iso,
    waypoint_times_from_dts,
)


PIPELINES: Dict[str, Tuple[str, str]] = {
    "A": ("random_feasible", "random_feasible"),
    "B": ("trained_reasoning", "random_feasible"),
    "C": ("random_feasible", "trained_waypoint"),
    "D": ("trained_reasoning", "trained_waypoint"),
}
PIPELINE_ORDER = ["A", "B", "C", "D"]
METRIC_NAMES = ["fuel_dv", "transfer_time_sec", "observation_score", "safety_margin_m"]
METRIC_OPT_DIRECTION = {
    "fuel_dv": "min",
    "transfer_time_sec": "min",
    "observation_score": "max",
    "safety_margin_m": "max",
}
INTENT_TO_METRIC = {
    "fuel": "fuel_dv",
    "time": "transfer_time_sec",
    "observation": "observation_score",
    "safety_margin": "safety_margin_m",
}
SCHEMA_VERSION = "analysis_rages_min_v1"


def _nan_metrics() -> Dict[str, float]:
    return {k: float("nan") for k in METRIC_NAMES}


def _dt_steps_from_orbits(dt_orbits: Sequence[float], tf_steps: int) -> Tuple[List[int], List[int]]:
    n_time = int(tf_steps) + 1
    t_idx_wyp = waypoint_times_from_dts(list(np.asarray(dt_orbits, dtype=float).reshape(-1)), n_time)
    times = [0] + [int(x) for x in t_idx_wyp] + [n_time - 1]
    dt_steps = np.diff(np.asarray(times, dtype=int)).astype(int).tolist()
    return dt_steps, [int(x) for x in t_idx_wyp]


def _build_metric_matrix_4xM(pipelines: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    values: List[List[float]] = []
    for pid in PIPELINE_ORDER:
        metric_row = []
        for m in METRIC_NAMES:
            metric_row.append(float(pipelines.get(pid, {}).get("metrics", {}).get(m, float("nan"))))
        values.append(metric_row)
    return {
        "pipelines": PIPELINE_ORDER,
        "metrics": METRIC_NAMES,
        "values": values,
    }


def _safe_default_dt_orbits(n_phase: int) -> List[float]:
    if n_phase <= 0:
        return []
    return [1.0] * int(n_phase)


def _normalize_metric_name(name: str) -> Optional[str]:
    s = str(name).strip().lower().replace("-", " ").replace("_", " ")
    aliases = {
        "fuel": "fuel_dv",
        "delta v": "fuel_dv",
        "dv": "fuel_dv",
        "fuel dv": "fuel_dv",
        "transfer time": "transfer_time_sec",
        "time": "transfer_time_sec",
        "tof": "transfer_time_sec",
        "observation": "observation_score",
        "observation score": "observation_score",
        "safety margin": "safety_margin_m",
        "safety": "safety_margin_m",
        "clearance": "safety_margin_m",
    }
    if s in aliases:
        return aliases[s]
    for metric in METRIC_NAMES:
        if s == metric or s.replace(" ", "_") == metric:
            return metric
    return None


def _extract_metrics_heuristic(reasoning_sentence: str) -> List[str]:
    text = str(reasoning_sentence or "").lower()
    patterns = [
        ("fuel_dv", r"\b(fuel|delta[- ]?v|dv)\b"),
        ("transfer_time_sec", r"\b(time|transfer time|tof)\b"),
        ("observation_score", r"\b(observation)\b"),
        ("safety_margin_m", r"\b(safety|safety margin|clearance)\b"),
    ]
    hits: List[Tuple[int, str]] = []
    for metric, pat in patterns:
        m = re.search(pat, text)
        if m is not None:
            hits.append((int(m.start()), metric))
    hits.sort(key=lambda x: x[0])
    out: List[str] = []
    for _, metric in hits:
        if metric not in out:
            out.append(metric)
        if len(out) >= 2:
            break
    return out


def _extract_metrics_chatgpt(
    reasoning_sentence: str,
    openai_client: OpenAI,
    openai_model: str,
) -> List[str]:
    sentence = str(reasoning_sentence or "").strip()
    if not sentence:
        return []

    user_prompt = (
        "From the reasoning sentence, select up to two metrics mentioned in order of appearance.\n"
        "Allowed metrics: fuel_dv, transfer_time_sec, observation_score, safety_margin_m.\n"
        "Some words to check for each metric: \n"
        "- fuel_dv: fuel, delta-v, control cost\n"
        "- transfer_time_sec: time, transfer time, tof\n"
        "- observation_score: observation\n"
        "- safety_margin_m: safety, safety margin, clearance\n"
        "Return strict JSON only: {\"focused_metrics\": [\"...\"]}\n"
        f"Reasoning sentence:\n{sentence}\n"
    )

    try:
        resp = openai_client.chat.completions.create(
            model=str(openai_model),
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Extract metric names exactly from the allowed list."},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content if resp.choices else ""
        payload = extract_first_json_object(content or "")
        focused_raw = payload.get("focused_metrics", [])
        if not isinstance(focused_raw, list):
            return _extract_metrics_heuristic(sentence)
        focused: List[str] = []
        for x in focused_raw:
            metric = _normalize_metric_name(str(x))
            if metric and metric not in focused:
                focused.append(metric)
            if len(focused) >= 2:
                break
        if focused:
            return focused
    except Exception:
        pass

    return _extract_metrics_heuristic(sentence)


def _is_metric_advantageous(metric: str, d_value: float, others: Sequence[float]) -> bool:
    if np.isnan(float(d_value)):
        return False
    valid_others = [float(v) for v in others if not np.isnan(float(v))]
    if len(valid_others) == 0:
        return False
    direction = METRIC_OPT_DIRECTION[metric]
    if direction == "min":
        return float(d_value) <= float(np.min(valid_others))
    return float(d_value) >= float(np.max(valid_others))


def _score_from_advantage_flags(advantageous: Sequence[bool]) -> int:
    flags = [bool(x) for x in list(advantageous)[:2]]
    if len(flags) == 0:
        return -2
    if len(flags) == 1:
        return 4 if flags[0] else 0
    if not flags[0] and not flags[1]:
        return 1
    if not flags[0] and flags[1]:
        return 2
    if flags[0] and not flags[1]:
        return 3
    return 5


def _compute_reasoning_eval(
    rec: Dict[str, Any],
    openai_client: OpenAI,
    openai_model: str,
) -> Dict[str, Any]:
    """
    Compute per-scenario reasoning evaluation for all pipelines A/B/C/D.

    Score codes:
      -2: no focused metric extracted (0 metrics)
      -1: candidate pipeline has NaN on any focused metric
       0: 1-of-1 focused metric, candidate is losing (not advantageous)
       1: 2-of-2 focused metrics, candidate is losing on both
       2: 2 metrics, only the first metric is losing
       3: 2 metrics, only the second metric is losing
       4: 1-of-1 focused metric, candidate is advantageous
       5: 2-of-2 focused metrics, candidate is advantageous on both

    Notes:
      - "Focused metrics" are capped to at most 2, in order of appearance.
      - "Losing" means not advantageous under METRIC_OPT_DIRECTION.
      - For each candidate p in {A,B,C,D}, advantage check uses the
        other three pipelines as references.
    """
    reasoning_sentence = str(rec.get("reasoning_sentence") or "")
    focused = _extract_metrics_chatgpt(
        reasoning_sentence=reasoning_sentence,
        openai_client=openai_client,
        openai_model=openai_model,
    )
    pipelines = rec.get("pipelines", {})
    focused_2 = focused[:2]
    out: Dict[str, Any] = {
        "focus_source": "reasoning_sentence",
        "focused_metrics": [str(m) for m in focused_2],
        "score_by_pipeline": {},
        "advantage_by_pipeline": {},
    }
    if len(focused_2) == 0:
        out["score_by_pipeline"] = {pid: -2 for pid in PIPELINE_ORDER}
        out["advantage_by_pipeline"] = {pid: {} for pid in PIPELINE_ORDER}
        return out

    for pid in PIPELINE_ORDER:
        metrics_pid = (pipelines.get(pid, {}) or {}).get("metrics", {}) or {}
        advantageous: List[bool] = []
        has_nan = False
        per_metric_advantage: Dict[str, bool] = {}

        for metric in focused_2:
            p_val = float(metrics_pid.get(metric, float("nan")))
            if np.isnan(p_val):
                has_nan = True
                per_metric_advantage[str(metric)] = False
                continue

            others = [
                float(((pipelines.get(other_pid, {}) or {}).get("metrics", {}) or {}).get(metric, float("nan")))
                for other_pid in PIPELINE_ORDER
                if other_pid != pid
            ]
            is_advantageous = _is_metric_advantageous(metric, p_val, others)
            advantageous.append(bool(is_advantageous))
            per_metric_advantage[str(metric)] = bool(is_advantageous)

        score = -1 if has_nan else _score_from_advantage_flags(advantageous)
        out["score_by_pipeline"][pid] = int(score)
        out["advantage_by_pipeline"][pid] = per_metric_advantage

    return out


def _compute_intent_eval(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute per-scenario intent evaluation for all pipelines A/B/C/D.

    Focused metrics come from top-2 entries of input.intent_priority.
    Uses the same score labels as reasoning_eval.
    """
    intent_priority = [str(x) for x in ((rec.get("input", {}) or {}).get("intent_priority", []) or [])]
    focused_2: List[str] = []
    for intent_key in intent_priority:
        metric = INTENT_TO_METRIC.get(str(intent_key))
        if metric and metric not in focused_2:
            focused_2.append(metric)
        if len(focused_2) >= 2:
            break

    pipelines = rec.get("pipelines", {})
    out: Dict[str, Any] = {
        "focus_source": "intent_priority_top2",
        "focused_metrics": [str(m) for m in focused_2],
        "score_by_pipeline": {},
        "advantage_by_pipeline": {},
    }
    if len(focused_2) == 0:
        out["score_by_pipeline"] = {pid: -2 for pid in PIPELINE_ORDER}
        out["advantage_by_pipeline"] = {pid: {} for pid in PIPELINE_ORDER}
        return out

    for pid in PIPELINE_ORDER:
        metrics_pid = (pipelines.get(pid, {}) or {}).get("metrics", {}) or {}
        advantageous: List[bool] = []
        has_nan = False
        per_metric_advantage: Dict[str, bool] = {}

        for metric in focused_2:
            p_val = float(metrics_pid.get(metric, float("nan")))
            if np.isnan(p_val):
                has_nan = True
                per_metric_advantage[str(metric)] = False
                continue

            others = [
                float(((pipelines.get(other_pid, {}) or {}).get("metrics", {}) or {}).get(metric, float("nan")))
                for other_pid in PIPELINE_ORDER
                if other_pid != pid
            ]
            is_advantageous = _is_metric_advantageous(metric, p_val, others)
            advantageous.append(bool(is_advantageous))
            per_metric_advantage[str(metric)] = bool(is_advantageous)

        score = -1 if has_nan else _score_from_advantage_flags(advantageous)
        out["score_by_pipeline"][pid] = int(score)
        out["advantage_by_pipeline"][pid] = per_metric_advantage

    return out

def _sample_random_behavior_candidate(
    scenario: Scenario,
    max_phase: int,
    seed: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "source": "random_feasible",
        "success": False,
        "policy": None,
        "b_seq": [],
        "tf_steps": None,
        "dt_orbits": [],
        "target_domains": None,
        "domain_policy": None,
        "reasoning_sentence": None,
    }

    try:
        scenarios = generate_behavior_seq(
            x0=scenario.x0,
            M=1,
            max_phase=int(max_phase),
            seed=int(seed),
        )
        if not scenarios:
            return out

        sampled = scenarios[0]
        b_seq = [int(x) for x in sampled["b_seq"]]
        policy = str(sampled["policy"])
        tf_steps = int(sampled["tof_steps"])
        dt_orbits = [float(x) for x in np.asarray(sampled["dt_orbits"], dtype=float).reshape(-1)]

        target_domains = None
        domain_policy = None
        if scenario.start_domain is not None:
            target_domains, domain_policy = recover_target_domains_any_policy(
                start_node=scenario.start_domain,
                b_seq=b_seq,
                policy_hint=policy,
                strict_policy_hint=True,
            )

        out.update(
            {
                "success": True,
                "policy": policy,
                "b_seq": b_seq,
                "tf_steps": tf_steps,
                "dt_orbits": dt_orbits,
                "target_domains": target_domains,
                "domain_policy": domain_policy,
            }
        )
        return out
    except Exception:
        return out


def _sample_trained_reasoning_candidate(
    scenario: Scenario,
    reasoning_sampler: Optional[ReasoningSampler],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "source": "trained_reasoning",
        "success": False,
        "policy": None,
        "b_seq": [],
        "tf_steps": None,
        "dt_orbits": [],
        "target_domains": None,
        "domain_policy": None,
        "reasoning_sentence": None,
        "bseq_feasible_graph": False,
    }

    if reasoning_sampler is None:
        return out

    try:
        parsed = reasoning_sampler.sample(scenario)
        b_seq = [int(x) for x in parsed["b_seq"]]
        tf_steps = int(parsed["tf"])
        dt_orbits = _safe_default_dt_orbits(len(b_seq))

        target_domains = None
        domain_policy = None
        if scenario.start_domain is not None:
            target_domains, domain_policy = recover_target_domains_any_policy(
                start_node=scenario.start_domain,
                b_seq=b_seq,
                policy_hint=None,
            )

        out.update(
            {
                "success": True,
                "policy": domain_policy,
                "b_seq": b_seq,
                "tf_steps": tf_steps,
                "dt_orbits": dt_orbits,
                "target_domains": target_domains,
                "domain_policy": domain_policy,
                "reasoning_sentence": str(parsed["reasoning"]),
                # Feasibility is tied directly to deterministic graph replay.
                "bseq_feasible_graph": bool(
                    target_domains is not None and len(target_domains) == len(b_seq)
                ),
            }
        )
        return out
    except Exception:
        return out


def _generate_random_waypoints(
    scenario: Scenario,
    behavior_candidate: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "success": False,
        "x_pred_roe": [],
        "dt_step": [],
        "dt_orbits_for_solver": [],
    }
    try:
        b_seq = [int(x) for x in behavior_candidate.get("b_seq", [])]
        tf_steps = behavior_candidate.get("tf_steps")
        if len(b_seq) == 0 or tf_steps is None:
            return out
        if scenario.start_domain is None:
            return out

        target_domains = behavior_candidate.get("target_domains")
        if target_domains is None or len(target_domains) != len(b_seq):
            target_domains, _ = recover_target_domains_any_policy(
                start_node=scenario.start_domain,
                b_seq=b_seq,
                policy_hint=behavior_candidate.get("policy"),
                strict_policy_hint=True,
            )
        if target_domains is None or len(target_domains) != len(b_seq):
            return out

        rng = np.random.default_rng(int(seed))
        x_pred = np.stack([sample_state_from_node(str(node), rng) for node in target_domains], axis=0)

        dt_orbits_raw = behavior_candidate.get("dt_orbits", [])
        if len(dt_orbits_raw) == len(b_seq):
            dt_orbits = as_float_list(dt_orbits_raw)
        else:
            dt_orbits = _safe_default_dt_orbits(len(b_seq))

        dt_step, _ = _dt_steps_from_orbits(dt_orbits, int(tf_steps))
        out.update(
            {
                "success": True,
                "x_pred_roe": np.asarray(x_pred, dtype=float).tolist(),
                "dt_step": [int(x) for x in dt_step],
                "dt_orbits_for_solver": dt_orbits,
            }
        )
        return out
    except Exception:
        return out


def _generate_trained_waypoints(
    scenario: Scenario,
    behavior_candidate: Dict[str, Any],
    model_bundle: Dict[str, Any],
    input_slices: Dict[str, slice],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "success": False,
        "x_pred_roe": [],
        "dt_step": [],
        "dt_orbits_for_solver": [],
    }
    try:
        b_seq = [int(x) for x in behavior_candidate.get("b_seq", [])]
        tf_steps = behavior_candidate.get("tf_steps")
        if len(b_seq) == 0 or tf_steps is None:
            return out

        x_pred, dt_pred = predict_wyp_seq(
            model_bundle=model_bundle,
            input_slices=input_slices,
            x0=np.asarray(scenario.x0, dtype=float),
            tof_steps=int(tf_steps),
            b_seq=b_seq,
            oec0_mod=np.asarray(scenario.oec0_modified, dtype=float),
            artms=np.asarray(scenario.artms_scaling_1e3, dtype=float),
            koz_dim=np.asarray(scenario.koz_param, dtype=float),
            use_mean_w=True,
        )

        x_pred = np.asarray(x_pred, dtype=float)
        dt_orbits = np.asarray(dt_pred, dtype=float).reshape(-1).tolist()
        if x_pred.ndim != 2 or x_pred.shape[0] != len(b_seq):
            return out
        if len(dt_orbits) != len(b_seq):
            return out

        dt_step, _ = _dt_steps_from_orbits(dt_orbits, int(tf_steps))
        out.update(
            {
                "success": True,
                "x_pred_roe": x_pred.tolist(),
                "dt_step": [int(x) for x in dt_step],
                "dt_orbits_for_solver": [float(x) for x in dt_orbits],
            }
        )
        return out
    except Exception:
        return out


def _restore_scenario_oe(scenario: Scenario) -> np.ndarray:
    oe = np.asarray(restore_koe(np.asarray(scenario.oec0_modified, dtype=float)), dtype=float).reshape(-1)
    if oe.shape != (6,):
        raise ValueError(f"Restored OE must have shape (6,), got {oe.shape}")
    return oe


def _solve_scp(
    scenario: Scenario,
    tf_steps: int,
    x_pred_roe: np.ndarray,
    dt_pred_orbits: Sequence[float],
    dt_sec: float,
    obj_type: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "scp_converged": False,
        "metrics": _nan_metrics(),
    }

    try:
        x_pred_arr = np.asarray(x_pred_roe, dtype=float)
        if x_pred_arr.ndim != 2 or x_pred_arr.shape[1] != 6:
            return out
        if x_pred_arr.shape[0] < 1:
            return out

        n_time = int(tf_steps) + 1
        if n_time < 2:
            return out

        dt_orbits = list(np.asarray(dt_pred_orbits, dtype=float).reshape(-1))
        if len(dt_orbits) != x_pred_arr.shape[0]:
            return out

        tvec_sec = np.arange(n_time, dtype=float) * float(dt_sec)
        t_idx_wyp = waypoint_times_from_dts(dt_orbits, n_time)

        goal = x_pred_arr[-1]
        waypoints = x_pred_arr[:-1] if x_pred_arr.shape[0] > 1 else np.empty((0, 6), dtype=float)
        scenario_oe = _restore_scenario_oe(scenario)

        current_obs = {
            "state": np.asarray(scenario.x0, dtype=float),
            "goal": np.asarray(goal, dtype=float),
            "ttg": float(tvec_sec[-1]),
            "dt": float(dt_sec),
            "oe": scenario_oe,
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
                "waypoint_type": "roe",
                "koz_dim": np.asarray(scenario.koz_param, dtype=float),
                "artms_scale_range_1e3": np.asarray(scenario.artms_scaling_1e3, dtype=float),
            }
        )

        sol_cvx = prob.ocp_cvx()
        status_cvx = str(sol_cvx.get("status", "cvx_unknown"))
        if status_cvx not in OK_STATUS:
            return out

        roe_cvx = np.asarray(sol_cvx["z"]["state"], dtype=float)
        actions_cvx = np.asarray(sol_cvx["z"]["action"], dtype=float)
        prob.zref = {"state": roe_cvx, "action": actions_cvx}
        prob.sol_0 = {"z": prob.zref}
        prob.generate_scaling(roe_cvx, actions_cvx)

        if str(obj_type) == "feasibility":
            prob.type = "feasibility"
            prob._cvx_built_AL = False
            prob.update_flag = True

        sol_scp, _ = solve_scvx(prob)
        status_scp = str(sol_scp.get("status", "scp_unknown"))
        if status_scp not in OK_STATUS:
            return out

        roe_scp = np.asarray(sol_scp["z"]["state"], dtype=float)
        actions_scp = np.asarray(sol_scp["z"]["action"], dtype=float)
        oec = propagate_oe(prob.oe_i, prob.tvec_sec)
        _, _, rtn_scp_ct = propagate_ct(roe_scp, actions_scp, oec, prob.tvec_sec, n=10)

        metrics = compute_metrics(
            prob=prob,
            roe=roe_scp,
            actions=actions_scp,
            rtn_ct=np.asarray(rtn_scp_ct, dtype=float),
        )

        out["scp_converged"] = True
        out["metrics"] = {k: float(metrics[k]) for k in METRIC_NAMES}
        return out
    except Exception:
        return out


def _evaluate_pipeline(
    pipeline_id: str,
    reasoning_mode: str,
    waypoint_mode: str,
    scenario: Scenario,
    behavior_candidate: Dict[str, Any],
    random_waypoint_seed: int,
    model_bundle: Dict[str, Any],
    input_slices: Dict[str, slice],
    dt_sec: float,
    obj_type: str,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "pipeline_id": str(pipeline_id),
        "scp_converged": False,
        "metrics": _nan_metrics(),
        "b_seq": [int(x) for x in behavior_candidate.get("b_seq", [])],
        "waypoint": {
            "x_roe": [],
            "dt_step": [],
        },
    }

    if not behavior_candidate.get("success", False):
        return rec

    tf_steps = behavior_candidate.get("tf_steps")
    if tf_steps is None:
        return rec

    if waypoint_mode == "trained_waypoint":
        wp = _generate_trained_waypoints(
            scenario=scenario,
            behavior_candidate=behavior_candidate,
            model_bundle=model_bundle,
            input_slices=input_slices,
        )
    else:
        wp = _generate_random_waypoints(
            scenario=scenario,
            behavior_candidate=behavior_candidate,
            seed=int(random_waypoint_seed),
        )

    if not wp.get("success", False):
        return rec

    rec["waypoint"] = {
        "x_roe": wp.get("x_pred_roe", []),
        "dt_step": [int(x) for x in wp.get("dt_step", [])],
    }

    scp = _solve_scp(
        scenario=scenario,
        tf_steps=int(tf_steps),
        x_pred_roe=np.asarray(wp["x_pred_roe"], dtype=float),
        dt_pred_orbits=wp["dt_orbits_for_solver"],
        dt_sec=float(dt_sec),
        obj_type=str(obj_type),
    )

    rec["scp_converged"] = bool(scp.get("scp_converged", False))
    rec["metrics"] = dict(scp.get("metrics", _nan_metrics()))
    return rec


def _build_task_record(
    scenario: Scenario,
    trained_behavior: Dict[str, Any],
    pipelines: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    compact_pipelines: Dict[str, Dict[str, Any]] = {}
    for pid in PIPELINE_ORDER:
        p = pipelines.get(pid, {})
        compact_pipelines[pid] = {
            "b_seq": p.get("b_seq", []),
            "waypoint": p.get("waypoint", {"x_roe": [], "dt_step": []}),
            "scp_converged": bool(p.get("scp_converged", False)),
            "metrics": p.get("metrics", _nan_metrics()),
        }

    return {
        "scenario_id": int(scenario.scenario_id),
        "input": {
            "dataset_idx": int(scenario.dataset_idx),
            "x0": np.asarray(scenario.x0, dtype=float).tolist(),
            "koz_param": np.asarray(scenario.koz_param, dtype=float).tolist(),
            "artms_scaling_1e3": np.asarray(scenario.artms_scaling_1e3, dtype=float).tolist(),
            "oec0_modified": np.asarray(scenario.oec0_modified, dtype=float).tolist(),
            "intent_priority": list(scenario.intent_priority),
        },
        "reasoning_sentence": trained_behavior.get("reasoning_sentence"),
        "reasoning_bseq_feas": bool(trained_behavior.get("bseq_feasible_graph", False)),
        "scp_convergence_4": {pid: bool(compact_pipelines[pid]["scp_converged"]) for pid in PIPELINE_ORDER},
        "pipelines": compact_pipelines,
        "metrics_4xM": _build_metric_matrix_4xM(compact_pipelines),
    }


def _build_failure_record(task_spec: Dict[str, Any]) -> Dict[str, Any]:
    pipelines = {
        pid: {
            "b_seq": [],
            "waypoint": {"x_roe": [], "dt_step": []},
            "scp_converged": False,
            "metrics": _nan_metrics(),
        }
        for pid in PIPELINE_ORDER
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": utc_now_iso(),
        "scenario_id": int(task_spec.get("scenario_id", -1)),
        "rollout_id": int(task_spec.get("rollout_id", -1)),
        "case_id": f"s{int(task_spec.get('scenario_id', -1)):06d}_k{int(task_spec.get('rollout_id', -1)):03d}",
        "input": {
            "dataset_idx": int(task_spec.get("dataset_idx", -1)),
            "x0": [],
            "koz_param": [],
            "artms_scaling_1e3": [],
            "oec0_modified": [],
            "intent_priority": [],
        },
        "reasoning_sentence": None,
        "reasoning_bseq_feas": False,
        "scp_convergence_4": {pid: False for pid in PIPELINE_ORDER},
        "pipelines": pipelines,
        "metrics_4xM": _build_metric_matrix_4xM(pipelines),
    }


def _run_single_task(
    task_spec: Dict[str, Any],
    data: Dict[str, torch.Tensor],
    meta: Dict[str, Any],
    model_bundle: Dict[str, Any],
    input_slices: Dict[str, slice],
    reasoning_sampler: Optional[ReasoningSampler],
    intent_priority: Sequence[str],
    obj_type: str,
) -> Dict[str, Any]:
    randomized_intent_priority = randomize_intent_priority(
        intent_priority=intent_priority,
        seed=int(task_spec.get("intent_priority_seed", task_spec.get("random_behavior_seed", 0))),
    )

    scenario = sample_scenario_from_dataset(
        scenario_id=int(task_spec["scenario_id"]),
        rollout_id=int(task_spec["rollout_id"]),
        dataset_idx=int(task_spec["dataset_idx"]),
        data=data,
        intent_priority=randomized_intent_priority,
    )

    random_behavior = _sample_random_behavior_candidate(
        scenario=scenario,
        max_phase=int(model_bundle["cfg"].max_phase),
        seed=int(task_spec["random_behavior_seed"]),
    )
    precomputed_trained_behavior = task_spec.get("trained_behavior")
    if isinstance(precomputed_trained_behavior, dict):
        trained_behavior = dict(precomputed_trained_behavior)
    else:
        trained_behavior = _sample_trained_reasoning_candidate(
            scenario=scenario,
            reasoning_sampler=reasoning_sampler,
        )
    behavior_candidates = {
        "random_feasible": random_behavior,
        "trained_reasoning": trained_behavior,
    }

    pipelines: Dict[str, Dict[str, Any]] = {}
    dt_sec = float(meta.get("dt_sec", param.dt_sec))
    for pid in PIPELINE_ORDER:
        reasoning_mode, waypoint_mode = PIPELINES[pid]
        behavior = behavior_candidates[reasoning_mode]
        pipeline_seed = int(task_spec["random_waypoint_seed_base"]) + int(ord(pid[0]))
        pipelines[pid] = _evaluate_pipeline(
            pipeline_id=pid,
            reasoning_mode=reasoning_mode,
            waypoint_mode=waypoint_mode,
            scenario=scenario,
            behavior_candidate=behavior,
            random_waypoint_seed=pipeline_seed,
            model_bundle=model_bundle,
            input_slices=input_slices,
            dt_sec=dt_sec,
            obj_type=obj_type,
        )

    return _build_task_record(
        scenario=scenario,
        trained_behavior=trained_behavior,
        pipelines=pipelines,
    )


_WORKER_DATA: Optional[Dict[str, torch.Tensor]] = None
_WORKER_META: Optional[Dict[str, Any]] = None
_WORKER_MODEL_BUNDLE: Optional[Dict[str, Any]] = None
_WORKER_INPUT_SLICES: Optional[Dict[str, slice]] = None
_WORKER_REASONING_SAMPLER: Optional[ReasoningSampler] = None
_WORKER_INTENT_PRIORITY: List[str] = []
_WORKER_OBJ_TYPE: str = "min_fuel"


def _init_worker(
    ckpt_path: str,
    data_path: str,
    intent_priority: Sequence[str],
    obj_type: str,
    reasoning_adapter_dir: str,
    reasoning_base_model: str,
    enable_reasoning_sampler: bool = True,
) -> None:
    global _WORKER_DATA
    global _WORKER_META
    global _WORKER_MODEL_BUNDLE
    global _WORKER_INPUT_SLICES
    global _WORKER_REASONING_SAMPLER
    global _WORKER_INTENT_PRIORITY
    global _WORKER_OBJ_TYPE

    _WORKER_MODEL_BUNDLE = load_wyp_model(Path(ckpt_path))
    _WORKER_DATA, _WORKER_META = load_dataset(Path(data_path))
    cfg = _WORKER_MODEL_BUNDLE["cfg"]
    _WORKER_INPUT_SLICES = build_input_slices(
        _WORKER_DATA,
        _WORKER_MODEL_BUNDLE["inputs_arg"],
        b_seq_encoding=getattr(cfg, "b_seq_encoding", "scalar"),
        b_seq_num_classes=int(getattr(cfg, "b_seq_num_classes", 11)),
    )

    _WORKER_INTENT_PRIORITY = [str(x) for x in intent_priority]
    _WORKER_OBJ_TYPE = str(obj_type)
    if enable_reasoning_sampler:
        _WORKER_REASONING_SAMPLER = ReasoningSampler(
            adapter_dir=Path(reasoning_adapter_dir),
            base_model=str(reasoning_base_model),
            max_phase=int(cfg.max_phase),
        )
    else:
        _WORKER_REASONING_SAMPLER = None


def _run_task_from_spec(task_spec: Dict[str, Any]) -> Dict[str, Any]:
    if (
        _WORKER_DATA is None
        or _WORKER_META is None
        or _WORKER_MODEL_BUNDLE is None
        or _WORKER_INPUT_SLICES is None
    ):
        raise RuntimeError("Worker context is not initialized.")

    try:
        return _run_single_task(
            task_spec=task_spec,
            data=_WORKER_DATA,
            meta=_WORKER_META,
            model_bundle=_WORKER_MODEL_BUNDLE,
            input_slices=_WORKER_INPUT_SLICES,
            reasoning_sampler=_WORKER_REASONING_SAMPLER,
            intent_priority=_WORKER_INTENT_PRIORITY,
            obj_type=_WORKER_OBJ_TYPE,
        )
    except Exception:
        return _build_failure_record(task_spec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect compact end-to-end factorial logs for "
            "R in {random_feasible, trained_reasoning}, "
            "W in {random_feasible, trained_waypoint}."
        )
    )
    # computation threading etc. 
    parser.add_argument("--num-scenarios", type=int, default=500)
    parser.add_argument("--k-rollouts", type=int, default=1)
    parser.add_argument("--num-process", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--obj-type", type=str, default="min_fuel", choices=["min_fuel", "feasibility"])
    # ouput file path 
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="rpod/rages/out/analysis_rages_test_v2.jsonl",
        help="Append-only JSONL output file.",
    )
    # waypoint generation model 
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="rpod/rages/wyp_model/model_gmm_v3_weighted_one_hot.pt",
        help="Waypoint model checkpoint.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="rpod/rages/wyp_data/data_v3_discrete.pth",
        help="Dataset used to sample initial scenarios.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation ratio; sample scenarios only from this contiguous eval tail split.",
    )
    # reasoning model (with LoRA) 
    parser.add_argument(
        "--reasoning-adapter-dir",
        type=str,
        default="rpod/rages/reasoning_model/v2/checkpoint-8400",
        help="LoRA adapter directory for the trained reasoning model.",
    )
    parser.add_argument(
        "--reasoning-base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF base model name for reasoning.",
    )
    # ChatGPT evaluation 
    parser.add_argument(
        "--enable-chatgpt-annotation",
        type=bool,
        default=True,
        help=(
            "Annotate each record with reasoning_eval and intent_eval "
            "(focused metrics + score_by_pipeline + advantage_by_pipeline)."
        ),
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model used to extract focused metrics from reasoning sentences.",
    )
    parser.add_argument(
        "--openai-api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name containing OpenAI API key.",
    )
    return parser.parse_args()


def _build_task_specs(
    num_scenarios: int,
    k_rollouts: int,
    eval_dataset_indices: Sequence[int],
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    eval_indices = [int(idx) for idx in eval_dataset_indices]
    if len(eval_indices) == 0:
        raise ValueError("Validation split is empty; cannot sample analysis scenarios.")
    dataset_indices = [
        int(eval_indices[int(rng.integers(0, len(eval_indices)))]) for _ in range(int(num_scenarios))
    ]

    specs: List[Dict[str, Any]] = []
    for s_idx in range(int(num_scenarios)):
        for k_idx in range(int(k_rollouts)):
            specs.append(
                {
                    "scenario_id": int(s_idx),
                    "rollout_id": int(k_idx),
                    "dataset_idx": int(dataset_indices[s_idx]),
                    "random_behavior_seed": int(rng.integers(0, 2**31 - 1)),
                    "random_waypoint_seed_base": int(rng.integers(0, 2**31 - 1)),
                    "intent_priority_seed": int(rng.integers(0, 2**31 - 1)),
                }
            )
    return specs


def _precompute_trained_reasoning_candidates(
    task_specs: List[Dict[str, Any]],
    data: Dict[str, torch.Tensor],
    intent_priority: Sequence[str],
    reasoning_adapter_dir: Path,
    reasoning_base_model: str,
    max_phase: int,
) -> List[Dict[str, Any]]:
    """Run reasoning once in the main process and attach outputs to each task spec."""
    if len(task_specs) == 0:
        return task_specs

    sampler = ReasoningSampler(
        adapter_dir=reasoning_adapter_dir,
        base_model=str(reasoning_base_model),
        max_phase=int(max_phase),
    )

    out_specs: List[Dict[str, Any]] = []
    iterator = tqdm(task_specs, total=len(task_specs), desc="precompute_reasoning", unit="case")
    for spec in iterator:
        randomized_intent_priority = randomize_intent_priority(
            intent_priority=intent_priority,
            seed=int(spec.get("intent_priority_seed", spec.get("random_behavior_seed", 0))),
        )
        scenario = sample_scenario_from_dataset(
            scenario_id=int(spec["scenario_id"]),
            rollout_id=int(spec["rollout_id"]),
            dataset_idx=int(spec["dataset_idx"]),
            data=data,
            intent_priority=randomized_intent_priority,
        )
        trained_behavior = _sample_trained_reasoning_candidate(
            scenario=scenario,
            reasoning_sampler=sampler,
        )

        spec_with_reasoning = dict(spec)
        spec_with_reasoning["trained_behavior"] = trained_behavior
        out_specs.append(spec_with_reasoning)

    return out_specs


def main() -> None:
    args = parse_args()
    if args.num_scenarios < 1:
        raise ValueError("--num-scenarios must be >= 1")
    if args.k_rollouts < 1:
        raise ValueError("--k-rollouts must be >= 1")
    if args.num_process < 1:
        raise ValueError("--num-process must be >= 1")

    ckpt_path = ROOT_FOLDER / args.ckpt_path
    data_path = ROOT_FOLDER / args.data_path
    output_jsonl = ROOT_FOLDER / args.output_jsonl
    intent_priority = [str(x) for x in DEFAULT_INTENT_PRIORITY]
    enable_chatgpt_annotation = bool(args.enable_chatgpt_annotation)
    effective_num_process = int(args.num_process)
    reasoning_score_counts_by_pipeline = {
        pid: {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0} for pid in PIPELINE_ORDER
    }
    intent_score_counts_by_pipeline = {
        pid: {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0} for pid in PIPELINE_ORDER
    }

    openai_client: Optional[OpenAI] = None
    if enable_chatgpt_annotation:
        api_key_env = str(args.openai_api_key_env)
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise ValueError(
                f"--enable-chatgpt-annotation requires env var {api_key_env} to be set."
            )
        openai_client = OpenAI(api_key=api_key)

    data_for_spec, _ = load_dataset(data_path)
    n_dataset = int(data_for_spec["x0"].shape[0])
    _, eval_dataset_indices = contiguous_train_eval_index_ranges(
        n_rows=n_dataset,
        val_ratio=float(args.val_ratio),
    )
    if eval_dataset_indices is None:
        raise ValueError("--val-ratio must be > 0 to sample from a validation split.")
    task_specs = _build_task_specs(
        num_scenarios=int(args.num_scenarios),
        k_rollouts=int(args.k_rollouts),
        eval_dataset_indices=eval_dataset_indices,
        seed=int(args.seed),
    )

    # Unified two-stage execution:
    # 1) single-process precompute of trained reasoning outputs (one LLM instance),
    # 2) worker-pool rollout/SCP evaluation using cached reasoning (num_process can be 1).
    max_phase = int(data_for_spec["b_seq"].shape[1])
    task_specs = _precompute_trained_reasoning_candidates(
        task_specs=task_specs,
        data=data_for_spec,
        intent_priority=intent_priority,
        reasoning_adapter_dir=ROOT_FOLDER / args.reasoning_adapter_dir,
        reasoning_base_model=str(args.reasoning_base_model),
        max_phase=max_phase,
    )
    precomputed_reasoning = True
    print(
        "[info] precomputed trained reasoning in main process once; "
        "workers will not load reasoning model replicas."
    )

    print(
        "[config] "
        f"num_scenarios={args.num_scenarios} "
        f"k_rollouts={args.k_rollouts} "
        f"val_ratio={args.val_ratio} "
        f"eval_pool={len(eval_dataset_indices)} "
        f"tasks={len(task_specs)} "
        f"num_process={effective_num_process} "
        f"precomputed_reasoning={precomputed_reasoning} "
        f"chatgpt_annotation={enable_chatgpt_annotation} "
        f"output={output_jsonl}"
    )

    n_total = 0
    n_scp_ok = {pid: 0 for pid in PIPELINE_ORDER}

    # Workers only do rollout/SCP; reasoning has already been precomputed above.
    n_workers = min(int(effective_num_process), len(task_specs))
    with mp.get_context("spawn").Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(
            str(ckpt_path),
            str(data_path),
            intent_priority,
            str(args.obj_type),
            str(ROOT_FOLDER / args.reasoning_adapter_dir),
            str(args.reasoning_base_model),
            False,
        ),
    ) as pool:
        iterator = pool.imap_unordered(_run_task_from_spec, task_specs)
        for rec in tqdm(iterator, total=len(task_specs), desc="analysis_rages", unit="case"):
            if enable_chatgpt_annotation and openai_client is not None:
                reasoning_eval = _compute_reasoning_eval(
                    rec=rec,
                    openai_client=openai_client,
                    openai_model=str(args.openai_model),
                )
                rec["reasoning_eval"] = reasoning_eval
                reasoning_score_by_pipeline = reasoning_eval.get("score_by_pipeline", {}) or {}
                for pid in PIPELINE_ORDER:
                    score = int(reasoning_score_by_pipeline.get(pid, -2))
                    if score in reasoning_score_counts_by_pipeline[pid]:
                        reasoning_score_counts_by_pipeline[pid][score] += 1

                intent_eval = _compute_intent_eval(rec)
                rec["intent_eval"] = intent_eval
                intent_score_by_pipeline = intent_eval.get("score_by_pipeline", {}) or {}
                for pid in PIPELINE_ORDER:
                    score = int(intent_score_by_pipeline.get(pid, -2))
                    if score in intent_score_counts_by_pipeline[pid]:
                        intent_score_counts_by_pipeline[pid][score] += 1
            append_jsonl_line(output_jsonl, rec)
            n_total += 1
            for pid in PIPELINE_ORDER:
                if bool(rec.get("scp_convergence_4", {}).get(pid, False)):
                    n_scp_ok[pid] += 1

    print(f"[done] appended_records={n_total} file={output_jsonl}")
    for pid in PIPELINE_ORDER:
        print(f"[done] {pid}_scp_converged={n_scp_ok[pid]}/{n_total}")
    if enable_chatgpt_annotation:
        print("[done] reasoning_score_counts_by_pipeline")
        for pid in PIPELINE_ORDER:
            counts = reasoning_score_counts_by_pipeline[pid]
            print(
                f"[done] {pid} "
                f"-2={counts[-2]} "
                f"-1={counts[-1]} "
                f"0={counts[0]} "
                f"1={counts[1]} "
                f"2={counts[2]} "
                f"3={counts[3]} "
                f"4={counts[4]} "
                f"5={counts[5]}"
            )
        print("[done] intent_score_counts_by_pipeline")
        for pid in PIPELINE_ORDER:
            counts = intent_score_counts_by_pipeline[pid]
            print(
                f"[done] {pid} "
                f"-2={counts[-2]} "
                f"-1={counts[-1]} "
                f"0={counts[0]} "
                f"1={counts[1]} "
                f"2={counts[2]} "
                f"3={counts[3]} "
                f"4={counts[4]} "
                f"5={counts[5]}"
            )


if __name__ == "__main__":
    main()
