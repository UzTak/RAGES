from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rages_sampling import (  # noqa: E402
    SCPVerifierConfig,
    SplitConfig,
    WaypointPlan,
    sample_candidate_actions,
    sample_scenario,
    verify_waypoint_plan,
)
from rages_scoring import VERIFIER_METRIC_KEYS  # noqa: E402
from wyp_predictor import build_input_slices, load_model, predict_wyp_seq  # noqa: E402


SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 2 Q labels.")
    parser.add_argument("--p-phi-ckpt", required=True, type=Path)
    parser.add_argument("--out-path", type=Path, default=ROOT / "data/q_data/stage2_q_v0.pth")
    parser.add_argument("--num-scenarios", type=int, default=1000)
    parser.add_argument("--candidates-per-scenario", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-phase", type=int, default=None)
    parser.add_argument("--dt-sec", type=float, default=None)
    parser.add_argument("--obj-type", type=str, default="min_fuel")
    return parser.parse_args()



def load_wyp_for_datagen(ckpt_path: Path) -> Tuple[Dict[str, Any], Dict[str, slice]]:
    bundle = load_model(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_slices = ckpt.get("input_slices")
    if raw_slices:
        input_slices = {k: slice(int(v[0]), int(v[1])) for k, v in raw_slices.items()}
        return bundle, input_slices
    input_slices = build_input_slices(
        {
            "x0": torch.zeros(1, 6),
            "tof": torch.zeros(1, 1),
            "oec0_modified": torch.zeros(1, 10),
            "artms_scale_range_1e3": torch.zeros(1, 6),
            "koz_dim": torch.zeros(1, 3),
            "b_seq": torch.zeros(1, bundle["cfg"].max_phase),
        },
        bundle["inputs_arg"],
        b_seq_encoding=getattr(bundle["cfg"], "b_seq_encoding", "one_hot"),
        b_seq_num_classes=int(getattr(bundle["cfg"], "b_seq_num_classes", 11)),
    )
    return bundle, input_slices


def make_tasks(
    num_scenarios: int,
    candidates_per_scenario: int,
    *,
    seed: int,
    max_phase: int,
    split_config: SplitConfig,
) -> List[Tuple[int, int, Any, Any]]:
    tasks: List[Tuple[int, int, Any, Any]] = []
    for scenario_id in range(int(num_scenarios)):
        split = split_config.split_for_index(scenario_id, int(num_scenarios))
        scenario = sample_scenario(scenario_id, seed=seed, split=split)
        candidates = sample_candidate_actions(
            scenario,
            n_candidates=candidates_per_scenario,
            seed=seed,
            max_phase=max_phase,
        )
        for candidate_id, curated_action in enumerate(candidates):
            row_index = scenario_id * int(candidates_per_scenario) + candidate_id
            tasks.append((row_index, candidate_id, scenario, curated_action))
    return tasks


def run_task(
    task: Tuple[int, int, Any, Any],
    *,
    model_bundle: Dict[str, Any],
    input_slices: Dict[str, slice],
    verifier_config: SCPVerifierConfig,
    seed: int,
    max_phase: int,
) -> Dict[str, Any]:
    row_index, candidate_id, scenario, curated_action = task
    action = curated_action.action
    x_seq = np.zeros((int(max_phase), 6), dtype=np.float32)
    dt_seq = np.zeros((int(max_phase),), dtype=np.float32)
    metrics = {k: float("nan") for k in VERIFIER_METRIC_KEYS}
    status_cvx = "not_run"
    status_scp = "not_run"

    try:
        x_pred, dt_pred = predict_wyp_seq(
            model_bundle=model_bundle,
            input_slices=input_slices,
            x0=np.asarray(scenario.x0, dtype=float),
            tof_steps=int(action.tof_steps),
            b_seq=action.b_seq,
            oec0_mod=np.asarray(scenario.oec0_modified, dtype=float),
            artms=np.asarray(scenario.artms_scale_range_1e3, dtype=float),
            koz_dim=np.asarray(scenario.koz_dim, dtype=float),
            use_mean_w=True,
        )
        n_phase = len(action.b_seq)
        x_seq[:n_phase] = np.asarray(x_pred, dtype=np.float32)
        dt_seq[:n_phase] = np.asarray(dt_pred, dtype=np.float32)
        result = verify_waypoint_plan(
            scenario,
            action,
            WaypointPlan(
                waypoint_states=tuple(np.asarray(x, dtype=float) for x in x_pred),
                dt_fractions=tuple(float(x) for x in dt_pred),
            ),
            config=verifier_config,
            seed=seed,
            candidate_id=candidate_id,
        )
        status_cvx = result.status_cvx
        status_scp = result.status_scp
        metrics = dict(result.metrics)
        converged = bool(result.converged)
    except Exception as exc:
        status_scp = f"error:{exc.__class__.__name__}"
        converged = False

    return {
        "row_index": int(row_index),
        "scenario_id": int(scenario.sample_id),
        "candidate_id": int(candidate_id),
        "split": scenario.split,
        "scenario": scenario,
        "curated_action": curated_action,
        "x_seq": x_seq,
        "dt_seq": dt_seq,
        "converged": bool(converged),
        "metrics": [float(metrics.get(k, float("nan"))) for k in VERIFIER_METRIC_KEYS],
        "status_cvx": status_cvx,
        "status_scp": status_scp,
    }


def allocate_dataset(num_rows: int, max_phase: int) -> Dict[str, torch.Tensor]:
    return {
        "scenario_id": torch.zeros(num_rows, dtype=torch.long),
        "candidate_id": torch.zeros(num_rows, dtype=torch.long),
        "split_id": torch.zeros(num_rows, dtype=torch.long),
        "x0": torch.zeros((num_rows, 6), dtype=torch.float32),
        "tof": torch.zeros((num_rows, 1), dtype=torch.float32),
        "b_seq": torch.zeros((num_rows, max_phase), dtype=torch.float32),
        "phase_valid": torch.zeros((num_rows, max_phase), dtype=torch.float32),
        "oec0_modified": torch.zeros((num_rows, 10), dtype=torch.float32),
        "koz_dim": torch.zeros((num_rows, 3), dtype=torch.float32),
        "artms_scale_range_1e3": torch.zeros((num_rows, 6), dtype=torch.float32),
        "x_seq": torch.zeros((num_rows, max_phase, 6), dtype=torch.float32),
        "dt_seq": torch.zeros((num_rows, max_phase), dtype=torch.float32),
        "converged": torch.zeros(num_rows, dtype=torch.float32),
        "metrics": torch.full((num_rows, len(VERIFIER_METRIC_KEYS)), float("nan"), dtype=torch.float32),
    }


def fill_row(data: Dict[str, torch.Tensor], row: Dict[str, Any], max_phase: int) -> Dict[str, Any]:
    idx = int(row["row_index"])
    scenario = row["scenario"]
    curated_action = row["curated_action"]
    action = curated_action.action
    n_phase = len(action.b_seq)

    data["scenario_id"][idx] = int(row["scenario_id"])
    data["candidate_id"][idx] = int(row["candidate_id"])
    data["split_id"][idx] = SPLIT_TO_ID[str(row["split"])]
    data["x0"][idx] = torch.as_tensor(np.asarray(scenario.x0, dtype=np.float32))
    data["tof"][idx, 0] = float(action.tof_steps)
    data["b_seq"][idx, :n_phase] = torch.as_tensor(action.b_seq, dtype=torch.float32)
    data["phase_valid"][idx, :n_phase] = 1.0
    data["oec0_modified"][idx] = torch.as_tensor(np.asarray(scenario.oec0_modified, dtype=np.float32))
    data["koz_dim"][idx] = torch.as_tensor(np.asarray(scenario.koz_dim, dtype=np.float32).reshape(-1))
    data["artms_scale_range_1e3"][idx] = torch.as_tensor(
        np.asarray(scenario.artms_scale_range_1e3, dtype=np.float32).reshape(-1)
    )
    data["x_seq"][idx] = torch.as_tensor(row["x_seq"], dtype=torch.float32)
    data["dt_seq"][idx] = torch.as_tensor(row["dt_seq"], dtype=torch.float32)
    data["converged"][idx] = float(row["converged"])
    data["metrics"][idx] = torch.as_tensor(row["metrics"], dtype=torch.float32)

    return {
        "row_index": idx,
        "scenario_id": int(row["scenario_id"]),
        "candidate_id": int(row["candidate_id"]),
        "split": str(row["split"]),
        "status_cvx": str(row["status_cvx"]),
        "status_scp": str(row["status_scp"]),
        "policy": curated_action.curation.policy,
        "b_seq": [int(x) for x in action.b_seq],
        "tof_steps": int(action.tof_steps),
        "target_domains": list(curated_action.curation.target_domains),
    }


def main() -> None:
    args = parse_args()
    if args.num_workers != 1:
        raise ValueError("Stage 2 Q V0 datagen currently supports --num-workers 1.")
    model_bundle, input_slices = load_wyp_for_datagen(args.p_phi_ckpt)
    max_phase = int(args.max_phase or model_bundle["cfg"].max_phase)
    default_verifier = SCPVerifierConfig()
    verifier_config = SCPVerifierConfig(
        dt_sec=float(args.dt_sec if args.dt_sec is not None else default_verifier.dt_sec),
        obj_type=str(args.obj_type),
    )

    tasks = make_tasks(
        args.num_scenarios,
        args.candidates_per_scenario,
        seed=args.seed,
        max_phase=max_phase,
        split_config=SplitConfig(),
    )
    data = allocate_dataset(len(tasks), max_phase)
    record_meta: List[Dict[str, Any]] = []

    for task in tqdm(tasks, desc="stage2_q_datagen", unit="candidate"):
        row = run_task(
            task,
            model_bundle=model_bundle,
            input_slices=input_slices,
            verifier_config=verifier_config,
            seed=args.seed,
            max_phase=max_phase,
        )
        record_meta.append(fill_row(data, row, max_phase))

    n_conv = int((data["converged"] > 0.5).sum().item())
    dataset = {
        "data": data,
        "records": sorted(record_meta, key=lambda r: r["row_index"]),
        "meta": {
            "num_scenarios": int(args.num_scenarios),
            "candidates_per_scenario": int(args.candidates_per_scenario),
            "num_rows": len(tasks),
            "max_phase": max_phase,
            "split_names": list(SPLIT_NAMES),
            "metric_keys": list(VERIFIER_METRIC_KEYS),
            "seed": int(args.seed),
            "p_phi_checkpoint": args.p_phi_ckpt.name,
            "deterministic_decode": "use_mean_w=True",
            "scp_config": verifier_config.to_dict(),
            "converged_rows": n_conv,
        },
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.out_path)
    print(f"saved {args.out_path}")
    print(f"rows={len(tasks)} converged={n_conv}/{len(tasks)} metric_keys={list(VERIFIER_METRIC_KEYS)}")


if __name__ == "__main__":
    main()
