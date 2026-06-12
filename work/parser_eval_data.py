from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

ROOT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_FOLDER / "src"))

from parameters import BEHAVIOR_IDS
from rages_parser import ir_from_json
from rages_sampling import IR, sample_ir_batch


DOMAIN_TEXT = {
    "a_Safe_Orbit": "safe orbit",
    "b_Pos_EI": "+V EI zone",
    "c_Pos_Flat": "+V flat zone",
    "d_Neg_EI": "-V EI zone",
    "e_Neg_Flat": "-V flat zone",
}

TASK_TEXT = {
    "circumnav": ("circumnavigate the target", "orbit around the target"),
    "flyby": ("perform a flyby", "pass by the target"),
    "ducking": ("execute a ducking maneuver", "duck with a fast drift"),
    "hold": ("hold station", "station keep"),
    "approach": ("approach the target", "close in for rendezvous"),
    "retreat": ("retreat from the target", "back away from the target"),
}


def _metric_text(metric: str) -> str:
    return str(metric).replace("_", " ")


def _domain_text(domain: str) -> str:
    return DOMAIN_TEXT.get(domain, domain)


def _behavior_text(behavior_id: int) -> str:
    name = BEHAVIOR_IDS.get(int(behavior_id), "unknown")
    return f"behavior {int(behavior_id)} ({name})"


def render_ir_to_text(ir: IR, rng: random.Random) -> str:
    task_choices = TASK_TEXT.get(ir.g.task_class, (ir.g.task_class,))
    clauses: List[str] = [rng.choice(task_choices)]

    if ir.dz.terminal_domain is not None:
        clauses.append(f"finish in the {_domain_text(ir.dz.terminal_domain)}")
    elif ir.dz.direction is not None:
        clauses.append(f"finish toward {ir.dz.direction}")

    priority = list(ir.sigma.priority)
    if priority:
        priority_text = " then ".join(_metric_text(metric) for metric in priority)
        clauses.append(f"prioritize {priority_text}")

    for metric, threshold in sorted(ir.sigma.thresholds.items()):
        if metric == "time":
            clauses.append(f"keep time under {threshold:g} steps")
        elif metric == "fuel":
            clauses.append(f"keep fuel under {threshold:g}")
        elif metric == "safety_margin":
            clauses.append(f"keep safety margin above {threshold:g} m")
        else:
            clauses.append(f"respect {_metric_text(metric)} threshold {threshold:g}")

    filters = ir.filters
    if filters.max_tof_steps is not None:
        clauses.append(f"within {int(filters.max_tof_steps)} time steps")
    if filters.min_safety_margin_m is not None:
        clauses.append(f"at least {float(filters.min_safety_margin_m):g} m safety margin")
    if filters.max_num_phases is not None:
        clauses.append(f"no more than {int(filters.max_num_phases)} phases")
    for domain in filters.forbidden_domains:
        clauses.append(f"avoid the {_domain_text(domain)}")
    for behavior_id in filters.required_behaviors:
        clauses.append(f"must use {_behavior_text(int(behavior_id))}")
    for behavior_id in filters.forbidden_behaviors:
        clauses.append(f"do not use {_behavior_text(int(behavior_id))}")

    if len(clauses) == 1:
        return clauses[0].capitalize() + "."
    return clauses[0].capitalize() + ", " + ", ".join(clauses[1:]) + "."


def build_gold_dataset(n: int, seed: int) -> List[Tuple[str, IR]]:
    rng = random.Random(int(seed))
    samples = sample_ir_batch(int(n), seed=int(seed), profile="auto")
    return [(render_ir_to_text(sample.ir, rng), sample.ir) for sample in samples]


def _example_ir(
    task_class: str,
    priority: Sequence[str],
    *,
    terminal_domain: Optional[str] = None,
    direction: Optional[str] = None,
    max_tof_steps: Optional[int] = None,
    min_safety_margin_m: Optional[float] = None,
    forbidden_domains: Sequence[str] = (),
    required_behaviors: Sequence[int] = (),
    forbidden_behaviors: Sequence[int] = (),
    max_num_phases: Optional[int] = None,
) -> IR:
    return ir_from_json(
        {
            "sigma": {"priority": list(priority), "thresholds": {}},
            "dz": {"terminal_domain": terminal_domain, "direction": direction},
            "g": {"task_class": task_class},
            "filters": {
                "max_tof_steps": max_tof_steps,
                "min_safety_margin_m": min_safety_margin_m,
                "forbidden_domains": list(forbidden_domains),
                "required_behaviors": list(required_behaviors),
                "forbidden_behaviors": list(forbidden_behaviors),
                "max_num_phases": max_num_phases,
            },
        }
    )


HANDWRITTEN_EXAMPLES: List[Tuple[str, Optional[IR]]] = [
    (
        "Circumnavigate the target, prioritize fuel then safety, avoid the -V flat zone, "
        "no more than 2 phases.",
        _example_ir(
            "circumnav",
            ("fuel", "safety_margin"),
            forbidden_domains=("e_Neg_Flat",),
            max_num_phases=2,
        ),
    ),
    (
        "Approach the target and finish in the +V EI zone with at least 10 m safety margin.",
        _example_ir(
            "approach",
            ("safety_margin",),
            terminal_domain="b_Pos_EI",
            min_safety_margin_m=10.0,
        ),
    ),
    (
        "Hold station near center, prioritize safety before fuel, and do not use behavior 10.",
        _example_ir(
            "hold",
            ("safety_margin", "fuel"),
            terminal_domain="a_Safe_Orbit",
            forbidden_behaviors=(10,),
        ),
    ),
    (
        "Perform a flyby toward -V within 50 time steps and require behavior 3.",
        _example_ir(
            "flyby",
            (),
            direction="-V",
            max_tof_steps=50,
            required_behaviors=(3,),
        ),
    ),
    (
        "Write a haiku about orbital mechanics.",
        None,
    ),
    (
        "Move sort of nicely but do not specify any rendezvous task.",
        None,
    ),
    (
        "Use a banana-shaped orbit with sparkle mode enabled.",
        None,
    ),
]
