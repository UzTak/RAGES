from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm


def find_root_path(path: str, word: str) -> str:
    parts = path.split(word, 1)
    return parts[0] + word if len(parts) > 1 else path


ROOT_FOLDER = Path(__file__).resolve().parents[1]

from datagen_reasoning import load_dataset  
from parameters import DEFAULT_INTENT_PRIORITY  
from utils import (  
    ReasoningSampler,
    append_jsonl_line,
    contiguous_train_eval_index_ranges,
    is_behavior_sequence_feasible_in_graph,
    randomize_intent_priority,
    recover_target_domains_any_policy,
    sample_scenario_from_dataset,
    utc_now_iso,
)


SCHEMA_VERSION = "analysis_rages_bseq_v1"


def _build_task_specs(
    num_scenarios: int,
    k_rollouts: int,
    eval_dataset_indices: Sequence[int],
    seed: int,
) -> List[Dict[str, int]]:
    rng = np.random.default_rng(int(seed))
    eval_indices = [int(idx) for idx in eval_dataset_indices]
    if len(eval_indices) == 0:
        raise ValueError("Validation split is empty; cannot sample analysis scenarios.")
    specs: List[Dict[str, int]] = []
    for s_idx in range(int(num_scenarios)):
        for k_idx in range(int(k_rollouts)):
            specs.append(
                {
                    "scenario_id": int(s_idx),
                    "rollout_id": int(k_idx),
                    "dataset_idx": int(eval_indices[int(rng.integers(0, len(eval_indices)))]),
                    "intent_priority_seed": int(rng.integers(0, 2**31 - 1)),
                }
            )
    return specs


def _evaluate_bseq_case(
    task_spec: Dict[str, int],
    data: Dict[str, torch.Tensor],
    reasoning_sampler: ReasoningSampler,
    base_intent_priority: Sequence[str],
) -> Dict[str, Any]:
    randomized_intent_priority = randomize_intent_priority(
        intent_priority=base_intent_priority,
        seed=int(task_spec["intent_priority_seed"]),
    )
    scenario = sample_scenario_from_dataset(
        scenario_id=int(task_spec["scenario_id"]),
        rollout_id=int(task_spec["rollout_id"]),
        dataset_idx=int(task_spec["dataset_idx"]),
        data=data,
        intent_priority=randomized_intent_priority,
    )

    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": utc_now_iso(),
        "scenario_id": int(scenario.scenario_id),
        "rollout_id": int(scenario.rollout_id),
        "case_id": f"s{int(scenario.scenario_id):06d}_k{int(scenario.rollout_id):03d}",
        "input": {
            "dataset_idx": int(scenario.dataset_idx),
            "x0": np.asarray(scenario.x0, dtype=float).tolist(),
            "koz_param": np.asarray(scenario.koz_param, dtype=float).tolist(),
            "artms_scaling_1e3": np.asarray(scenario.artms_scaling_1e3, dtype=float).tolist(),
            "oec0_modified": np.asarray(scenario.oec0_modified, dtype=float).tolist(),
            "start_domain": scenario.start_domain,
            "intent_priority": list(scenario.intent_priority),
        },
        "reasoning_sentence": None,
        "tf_steps": None,
        "b_seq": [],
        "checks": {
            "parse_success": False,
            "bseq_feasible_graph": False,
            "tf_positive": False,
            "b_seq_nonempty": False,
            "b_seq_len": 0,
            "matched_policy": None,
        },
        "error": None,
    }

    try:
        pred = reasoning_sampler.sample(scenario)
        b_seq = [int(x) for x in pred["b_seq"]]
        tf_steps = int(pred["tf"])
        target_domains, matched_policy = recover_target_domains_any_policy(
            start_node=str(scenario.start_domain),
            b_seq=b_seq,
            policy_hint=None,
        ) if scenario.start_domain is not None else (None, None)
        bseq_feasible_graph = is_behavior_sequence_feasible_in_graph(
            start_domain=scenario.start_domain,
            b_seq=b_seq,
            policy_hint=matched_policy,
        )

        out["reasoning_sentence"] = str(pred["reasoning"])
        out["tf_steps"] = tf_steps
        out["b_seq"] = b_seq
        out["checks"] = {
            "parse_success": True,
            "bseq_feasible_graph": bool(bseq_feasible_graph and target_domains is not None),
            "tf_positive": bool(tf_steps > 0),
            "b_seq_nonempty": bool(len(b_seq) > 0),
            "b_seq_len": int(len(b_seq)),
            "matched_policy": matched_policy,
        }
    except Exception as e:
        out["error"] = str(e)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained reasoning outputs (tf, b_seq) only. "
            "No waypoint generation and no SCP solve."
        )
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="rpod/rages/wyp_data/data_v3_discrete.pth",
        help="Dataset used to sample initial scenarios.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="rpod/rages/out/analysis_rages_bseq.jsonl",
        help="Append-only JSONL output file.",
    )
    parser.add_argument("--num-scenarios", type=int, default=10)
    parser.add_argument("--k-rollouts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation ratio; sample scenarios only from this contiguous eval tail split.",
    )
    parser.add_argument(
        "--intent-priority",
        type=str,
        nargs="+",
        default=DEFAULT_INTENT_PRIORITY,
        help="Base intent priority; shuffled per task before prompting.",
    )
    parser.add_argument(
        "--reasoning-adapter-dir",
        type=str,
        default="rpod/rages/reasoning_model/v1/checkpoint-5200",
        help="LoRA adapter directory for the trained reasoning model.",
    )
    parser.add_argument(
        "--reasoning-base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF base model name for reasoning.",
    )
    parser.add_argument(
        "--max-phase",
        type=int,
        default=None,
        help="Max allowed b_seq length. If omitted, inferred from dataset b_seq width.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_scenarios < 1:
        raise ValueError("--num-scenarios must be >= 1")
    if args.k_rollouts < 1:
        raise ValueError("--k-rollouts must be >= 1")

    data_path = ROOT_FOLDER / args.data_path
    output_jsonl = ROOT_FOLDER / args.output_jsonl
    base_intent_priority = [str(x) for x in args.intent_priority]

    data, _ = load_dataset(data_path)
    n_dataset = int(data["x0"].shape[0])
    _, eval_dataset_indices = contiguous_train_eval_index_ranges(
        n_rows=n_dataset,
        val_ratio=float(args.val_ratio),
    )
    if eval_dataset_indices is None:
        raise ValueError("--val-ratio must be > 0 to sample from a validation split.")
    if args.max_phase is None:
        max_phase = int(data["b_seq"].shape[1])
    else:
        max_phase = int(args.max_phase)
    if max_phase < 1:
        raise ValueError("--max-phase must be >= 1")

    task_specs = _build_task_specs(
        num_scenarios=int(args.num_scenarios),
        k_rollouts=int(args.k_rollouts),
        eval_dataset_indices=eval_dataset_indices,
        seed=int(args.seed),
    )

    print(
        "[config] "
        f"num_scenarios={args.num_scenarios} "
        f"k_rollouts={args.k_rollouts} "
        f"val_ratio={args.val_ratio} "
        f"eval_pool={len(eval_dataset_indices)} "
        f"tasks={len(task_specs)} "
        f"max_phase={max_phase} "
        f"output={output_jsonl}"
    )

    reasoning_sampler = ReasoningSampler(
        adapter_dir=ROOT_FOLDER / args.reasoning_adapter_dir,
        base_model=str(args.reasoning_base_model),
        max_phase=max_phase,
    )

    n_total = 0
    n_parse_ok = 0
    n_bseq_feasible = 0
    for spec in tqdm(task_specs, total=len(task_specs), desc="analysis_rages_bseq", unit="case"):
        rec = _evaluate_bseq_case(
            task_spec=spec,
            data=data,
            reasoning_sampler=reasoning_sampler,
            base_intent_priority=base_intent_priority,
        )
        append_jsonl_line(output_jsonl, rec)

        n_total += 1
        checks = rec.get("checks", {})
        if bool(checks.get("parse_success", False)):
            n_parse_ok += 1
            if bool(checks.get("bseq_feasible_graph", False)):
                n_bseq_feasible += 1

    denom_parse = max(n_parse_ok, 1)
    print(f"[done] appended_records={n_total} file={output_jsonl}")
    print(f"[done] parse_success={n_parse_ok}/{n_total}")
    print(f"[done] bseq_feasible_given_parse={n_bseq_feasible}/{denom_parse}")


if __name__ == "__main__":
    main()
