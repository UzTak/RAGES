from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import optimization.parameters as opt_param
from dynamics.dynamics_trans import modify_koe, mu_E, true_to_mean_anomaly
from parameters import (
    ARTMS_SCALE_FACTORS,
    BEHAVIOR_IDS,
    DEFAULT_INTENT_PRIORITY,
    DT_SEC,
    KOZ_DIMS,
    NODES,
    N_TIME_MAX,
    OK_STATUS,
    POLICY_REGISTRY,
    TRUE_ANOMALY_GRID_RAD,
    Action,
    ActionCurationMetadata,
    CuratedAction,
    Range,
)

METRIC_NAMES = tuple(DEFAULT_INTENT_PRIORITY)
TASK_CLASSES = ("circumnav", "flyby", "ducking", "hold", "approach", "retreat")
TERMINAL_DIRECTIONS = ("+V", "-V", "center")
IR_PROFILES = ("balanced", "fuel_first", "time_first", "observation_first", "safety_first", "hard_filters")
ROOT_FOLDER = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        vals = (float(self.train), float(self.val), float(self.test))
        if any(v < 0.0 for v in vals):
            raise ValueError("Split fractions must be non-negative.")
        total = sum(vals)
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split fractions must sum to 1.0, got {total}.")

    def index_ranges(self, n_rows: int) -> Dict[str, range]:
        n_rows = int(n_rows)
        if n_rows <= 0:
            raise ValueError("n_rows must be positive.")
        n_train = int(np.floor(n_rows * float(self.train)))
        n_val = int(np.floor(n_rows * float(self.val)))
        n_train = min(n_train, n_rows)
        n_val = min(n_val, n_rows - n_train)
        return {
            "train": range(0, n_train),
            "val": range(n_train, n_train + n_val),
            "test": range(n_train + n_val, n_rows),
        }

    def split_for_index(self, index: int, n_rows: int) -> str:
        index = int(index)
        for split, idx_range in self.index_ranges(n_rows).items():
            if index in idx_range:
                return split
        raise IndexError(f"index {index} is outside n_rows={n_rows}.")


@dataclass(frozen=True)
class ScenarioSample:
    sample_id: int
    split: str
    x0: np.ndarray
    oec0_modified: np.ndarray
    koz_dim: np.ndarray
    artms_scale_range_1e3: np.ndarray
    start_domain: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "sample_id": int(self.sample_id),
            "split": self.split,
            "x0": np.asarray(self.x0, dtype=float).tolist(),
            "oec0_modified": np.asarray(self.oec0_modified, dtype=float).tolist(),
            "koz_dim": np.asarray(self.koz_dim, dtype=float).tolist(),
            "artms_scale_range_1e3": np.asarray(self.artms_scale_range_1e3, dtype=float).tolist(),
            "start_domain": self.start_domain,
        }


@dataclass(frozen=True)
class ScenarioRolloutSample:
    scenario: ScenarioSample
    curated_action: CuratedAction
    waypoint_states: Tuple[np.ndarray, ...]
    dt_fractions: Tuple[float, ...]
    waypoint_time_indices: Tuple[int, ...]

    @property
    def action(self) -> Action:
        return self.curated_action.action

    @property
    def curation(self) -> ActionCurationMetadata:
        return self.curated_action.curation

    def to_dict(self) -> Dict[str, object]:
        return {
            "scenario": self.scenario.to_dict(),
            "action": self.action.to_dict(),
            "curation": self.curation.to_dict(),
            "waypoint_states": [np.asarray(x, dtype=float).tolist() for x in self.waypoint_states],
            "dt_fractions": [float(x) for x in self.dt_fractions],
            "waypoint_time_indices": [int(x) for x in self.waypoint_time_indices],
        }


@dataclass(frozen=True)
class WaypointPlan:
    waypoint_states: Tuple[np.ndarray, ...]
    dt_fractions: Tuple[float, ...]

    def __post_init__(self) -> None:
        waypoint_states = tuple(np.asarray(x, dtype=float).reshape(6) for x in self.waypoint_states)
        dt_fractions = tuple(float(x) for x in self.dt_fractions)
        if len(waypoint_states) == 0:
            raise ValueError("WaypointPlan requires at least one waypoint/terminal state.")
        if len(waypoint_states) != len(dt_fractions):
            raise ValueError(
                "WaypointPlan waypoint_states and dt_fractions must have the same length."
            )
        if any(x <= 0.0 for x in dt_fractions):
            raise ValueError("WaypointPlan dt_fractions must be positive.")
        object.__setattr__(self, "waypoint_states", waypoint_states)
        object.__setattr__(self, "dt_fractions", dt_fractions)

    @classmethod
    def from_rollout(cls, rollout: ScenarioRolloutSample) -> "WaypointPlan":
        return cls(
            waypoint_states=rollout.waypoint_states,
            dt_fractions=rollout.dt_fractions,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "waypoint_states": [np.asarray(x, dtype=float).tolist() for x in self.waypoint_states],
            "dt_fractions": [float(x) for x in self.dt_fractions],
        }


@dataclass(frozen=True)
class SCPVerifierConfig:
    dt_sec: float = DT_SEC
    obj_type: str = "min_fuel"
    verifier_version: str = "v1_datagen_reasoning"

    def to_dict(self) -> Dict[str, object]:
        return {
            "dt_sec": float(self.dt_sec),
            "obj_type": self.obj_type,
            "verifier_version": self.verifier_version,
        }


@dataclass(frozen=True)
class SCPVerifierResult:
    status_cvx: str
    status_scp: str
    metrics: Dict[str, float]
    log: Dict[str, object]
    raw: Dict[str, Any]

    @property
    def converged(self) -> bool:
        return self.status_scp in OK_STATUS

    def to_dict(self, include_raw: bool = False) -> Dict[str, object]:
        out: Dict[str, object] = {
            "status_cvx": self.status_cvx,
            "status_scp": self.status_scp,
            "converged": self.converged,
            "metrics": dict(self.metrics),
            "log": dict(self.log),
        }
        if include_raw:
            out["raw"] = self.raw
        return out


@dataclass(frozen=True)
class IRSigma:
    priority: Tuple[str, ...]
    weights: Dict[str, float]
    thresholds: Dict[str, float]

    def __post_init__(self) -> None:
        priority = tuple(str(x) for x in self.priority)
        if set(priority) != set(METRIC_NAMES) or len(priority) != len(METRIC_NAMES):
            raise ValueError(f"priority must be a permutation of {METRIC_NAMES}.")

        weights = {str(k): float(v) for k, v in self.weights.items()}
        unknown_weights = sorted(set(weights) - set(METRIC_NAMES))
        if unknown_weights:
            raise ValueError(f"Unknown sigma weight metric(s): {unknown_weights}")
        if any(v < 0.0 for v in weights.values()):
            raise ValueError("sigma weights must be non-negative.")

        thresholds = {str(k): float(v) for k, v in self.thresholds.items()}
        unknown_thresholds = sorted(set(thresholds) - set(METRIC_NAMES))
        if unknown_thresholds:
            raise ValueError(f"Unknown sigma threshold metric(s): {unknown_thresholds}")

        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "thresholds", thresholds)

    def to_dict(self) -> Dict[str, object]:
        return {
            "priority": list(self.priority),
            "weights": dict(self.weights),
            "thresholds": dict(self.thresholds),
        }


@dataclass(frozen=True)
class IRDeltaZ:
    terminal_domain: Optional[str]
    direction: Optional[str]

    def __post_init__(self) -> None:
        terminal_domain = None if self.terminal_domain is None else str(self.terminal_domain)
        direction = None if self.direction is None else str(self.direction)
        if terminal_domain is not None and terminal_domain not in NODES:
            raise ValueError(f"Unknown terminal_domain: {terminal_domain}")
        if direction is not None and direction not in TERMINAL_DIRECTIONS:
            raise ValueError(f"Unknown terminal direction: {direction}")
        object.__setattr__(self, "terminal_domain", terminal_domain)
        object.__setattr__(self, "direction", direction)

    def to_dict(self) -> Dict[str, object]:
        return {
            "terminal_domain": self.terminal_domain,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class IRGoal:
    task_class: str

    def __post_init__(self) -> None:
        task_class = str(self.task_class)
        if task_class not in TASK_CLASSES:
            raise ValueError(f"Unknown task_class: {task_class}")
        object.__setattr__(self, "task_class", task_class)

    def to_dict(self) -> Dict[str, object]:
        return {"task_class": self.task_class}


@dataclass(frozen=True)
class IRFilters:
    max_tof_steps: Optional[int] = None
    min_safety_margin_m: Optional[float] = None
    forbidden_domains: Tuple[str, ...] = ()
    required_behaviors: Tuple[int, ...] = ()
    forbidden_behaviors: Tuple[int, ...] = ()
    max_num_phases: Optional[int] = None

    def __post_init__(self) -> None:
        max_tof_steps = None if self.max_tof_steps is None else int(self.max_tof_steps)
        min_safety_margin_m = (
            None if self.min_safety_margin_m is None else float(self.min_safety_margin_m)
        )
        max_num_phases = None if self.max_num_phases is None else int(self.max_num_phases)
        if max_tof_steps is not None and max_tof_steps <= 0:
            raise ValueError("max_tof_steps must be positive.")
        if max_num_phases is not None and max_num_phases <= 0:
            raise ValueError("max_num_phases must be positive.")

        forbidden_domains = tuple(str(x) for x in self.forbidden_domains)
        unknown_domains = sorted(set(forbidden_domains) - set(NODES))
        if unknown_domains:
            raise ValueError(f"Unknown forbidden domain(s): {unknown_domains}")

        required_behaviors = tuple(int(x) for x in self.required_behaviors)
        forbidden_behaviors = tuple(int(x) for x in self.forbidden_behaviors)
        unknown_behaviors = sorted(
            (set(required_behaviors) | set(forbidden_behaviors)) - set(BEHAVIOR_IDS)
        )
        if unknown_behaviors:
            raise ValueError(f"Unknown behavior id(s): {unknown_behaviors}")
        overlap = sorted(set(required_behaviors) & set(forbidden_behaviors))
        if overlap:
            raise ValueError(f"Behavior(s) cannot be both required and forbidden: {overlap}")

        object.__setattr__(self, "max_tof_steps", max_tof_steps)
        object.__setattr__(self, "min_safety_margin_m", min_safety_margin_m)
        object.__setattr__(self, "forbidden_domains", forbidden_domains)
        object.__setattr__(self, "required_behaviors", required_behaviors)
        object.__setattr__(self, "forbidden_behaviors", forbidden_behaviors)
        object.__setattr__(self, "max_num_phases", max_num_phases)

    def to_dict(self) -> Dict[str, object]:
        return {
            "max_tof_steps": self.max_tof_steps,
            "min_safety_margin_m": self.min_safety_margin_m,
            "forbidden_domains": list(self.forbidden_domains),
            "required_behaviors": [int(x) for x in self.required_behaviors],
            "forbidden_behaviors": [int(x) for x in self.forbidden_behaviors],
            "max_num_phases": self.max_num_phases,
        }


@dataclass(frozen=True)
class IR:
    sigma: IRSigma
    dz: IRDeltaZ
    g: IRGoal
    filters: IRFilters

    def to_dict(self) -> Dict[str, object]:
        return {
            "sigma": self.sigma.to_dict(),
            "dz": self.dz.to_dict(),
            "g": self.g.to_dict(),
            "filters": self.filters.to_dict(),
        }


@dataclass(frozen=True)
class IRSample:
    ir: IR
    seed: int
    split: str
    profile: str
    expected_mask_nonempty: Optional[bool]
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "ir": self.ir.to_dict(),
            "metadata": {
                "seed": int(self.seed),
                "split": self.split,
                "profile": self.profile,
                "expected_mask_nonempty": self.expected_mask_nonempty,
                "notes": list(self.notes),
            },
        }


def _rng_for_index(seed: int, index: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) + 1009 * int(index))


def _sample_koz_dim(rng: np.random.Generator) -> np.ndarray:
    dim = float(KOZ_DIMS[int(rng.integers(len(KOZ_DIMS)))])
    return np.array([dim, dim, dim], dtype=float)


def _sample_artms_param(rng: np.random.Generator) -> np.ndarray:
    nominal = np.asarray(opt_param.artms_scale_range_1e3, dtype=float).copy()
    scale = float(ARTMS_SCALE_FACTORS[int(rng.integers(len(ARTMS_SCALE_FACTORS)))])
    return nominal * scale


def _sample_oec0_modified(rng: np.random.Generator) -> np.ndarray:
    oec0 = np.asarray(opt_param.oec0, dtype=float).copy()
    nu = float(TRUE_ANOMALY_GRID_RAD[int(rng.integers(len(TRUE_ANOMALY_GRID_RAD)))])
    oec0[5] = true_to_mean_anomaly(nu, oec0[1])
    return np.asarray(modify_koe(oec0), dtype=float)


def _sample_from_range(r: Range, rng: np.random.Generator, n: int = 3) -> float:
    lo, hi = float(r[0]), float(r[1])
    grid = np.linspace(lo, hi, int(n), dtype=float)
    return float(grid[int(rng.integers(len(grid)))])


def _time_grid_from_orbits(
    total_orbits: float,
    dt_sec: float,
    n_time_max: int,
    semi_major_axis_m: float,
) -> Tuple[int, np.ndarray]:
    period = 2.0 * np.pi * np.sqrt((semi_major_axis_m ** 3) / mu_E)
    total_sec = float(total_orbits) * period
    n_steps = int(np.round(total_sec / dt_sec))
    n_time = max(2, min(int(n_time_max), n_steps + 1))
    tvec_sec = np.arange(n_time, dtype=float) * float(dt_sec)
    return n_time, tvec_sec


def _waypoint_times_from_dts(dt_seq: Sequence[float], n_time: int) -> List[int]:
    total = float(np.sum(dt_seq))
    cum = np.cumsum(list(dt_seq)[:-1]) / total
    idx = np.rint(cum * (int(n_time) - 1)).astype(int)
    idx = np.clip(idx, 1, int(n_time) - 2)

    for i in range(1, len(idx)):
        idx[i] = max(idx[i], idx[i - 1] + 1)

    max_last = int(n_time) - 2
    if len(idx) > 0 and idx[-1] > max_last:
        overflow = idx[-1] - max_last
        idx = idx - overflow
        for i in range(len(idx)):
            if idx[i] < 1:
                idx[i] = 1
            if i > 0 and idx[i] <= idx[i - 1]:
                idx[i] = idx[i - 1] + 1
        idx[-1] = min(idx[-1], max_last)

    return idx.tolist()


def sample_curated_rollout(
    sample_id: int,
    *,
    seed: int,
    split: str = "unspecified",
    max_phase: int = 3,
    max_retries: int = 100,
) -> ScenarioRolloutSample:
    """
    Deterministically sample one scenario plus curation-derived action.

    The returned `Action` contains only `b_seq` and `tof_steps`. The campaign
    `policy`, sampled per-phase `dt_orbits`, transfer windows, and target
    domains are preserved only as curation metadata.
    """

    for retry in range(int(max_retries)):
        rng = _rng_for_index(seed=int(seed) + retry, index=int(sample_id))
        policy_names = sorted(POLICY_REGISTRY.keys())
        policy_name = str(policy_names[int(rng.integers(len(policy_names)))])
        policy = POLICY_REGISTRY[policy_name]

        valid_starts = policy.get_valid_start_nodes()
        start_domain = str(valid_starts[int(rng.integers(len(valid_starts)))])
        current_node = start_domain
        current_state = NODES[current_node].sample(rng=rng)

        states: List[np.ndarray] = [current_state]
        target_domains: List[str] = []
        behaviors: List[int] = []
        dt_orbits: List[float] = []
        dt_ranges: List[Range] = []

        step = 0
        while len(behaviors) < int(max_phase):
            next_step = policy.get_next_step(current_node, step, rng=rng)
            if next_step is None:
                break

            next_node, behavior_id, dt_range = next_step
            next_node = str(next_node)
            behavior_id = int(behavior_id)
            dt_range = (float(dt_range[0]), float(dt_range[1]))

            if behavior_id == 0:
                current_node = next_node
                step += 1
                continue

            if behavior_id == 1:
                next_state = current_state.copy()
            else:
                next_state = NODES[next_node].sample(rng=rng)
            dt_orbit = _sample_from_range(dt_range, rng=rng, n=3)

            states.append(next_state)
            target_domains.append(next_node)
            behaviors.append(behavior_id)
            dt_orbits.append(dt_orbit)
            dt_ranges.append(dt_range)

            current_node = next_node
            current_state = next_state
            step += 1

        if not behaviors:
            continue

        oec0_modified = _sample_oec0_modified(rng)
        oec0 = np.asarray(opt_param.oec0, dtype=float).copy()
        # Use nominal semi-major axis for the time-grid conversion, matching
        # the current waypoint data generator.
        n_time, _ = _time_grid_from_orbits(
            float(np.sum(dt_orbits)),
            DT_SEC,
            N_TIME_MAX,
            oec0[0],
        )
        tof_steps = int(n_time - 1)
        if tof_steps <= 0:
            continue

        t_idx_wyp = _waypoint_times_from_dts(dt_orbits, n_time)
        times = [0] + [int(x) for x in t_idx_wyp] + [n_time - 1]
        dt_steps = np.diff(np.asarray(times, dtype=int)).astype(int)
        dt_fractions = tuple(float(x) / float(tof_steps) for x in dt_steps)

        scenario = ScenarioSample(
            sample_id=int(sample_id),
            split=str(split),
            x0=np.asarray(states[0], dtype=float),
            oec0_modified=oec0_modified,
            koz_dim=_sample_koz_dim(rng),
            artms_scale_range_1e3=_sample_artms_param(rng),
            start_domain=start_domain,
        )
        action = Action.from_values(b_seq=behaviors, tof_steps=tof_steps)
        curation = ActionCurationMetadata(
            policy=policy_name,
            dt_orbits=tuple(float(x) for x in dt_orbits),
            dt_ranges=tuple(dt_ranges),
            target_domains=tuple(target_domains),
        )
        return ScenarioRolloutSample(
            scenario=scenario,
            curated_action=CuratedAction(action=action, curation=curation),
            waypoint_states=tuple(np.asarray(x, dtype=float) for x in states[1:]),
            dt_fractions=dt_fractions,
            waypoint_time_indices=tuple(int(x) for x in t_idx_wyp),
        )

    raise RuntimeError(
        f"Unable to sample a non-empty rollout for sample_id={sample_id} after {max_retries} retries."
    )


def sample_curated_rollouts(
    n_samples: int,
    *,
    seed: int,
    split_config: Optional[SplitConfig] = None,
    max_phase: int = 3,
) -> List[ScenarioRolloutSample]:
    split_config = split_config or SplitConfig()
    out: List[ScenarioRolloutSample] = []
    for sample_id in range(int(n_samples)):
        split = split_config.split_for_index(sample_id, int(n_samples))
        out.append(
            sample_curated_rollout(
                sample_id=sample_id,
                seed=seed,
                split=split,
                max_phase=max_phase,
            )
        )
    return out


def enumerate_priority_profiles() -> List[Tuple[str, ...]]:
    """Return all priority permutations in deterministic order."""

    priorities: List[Tuple[str, ...]] = [()]
    for metric in METRIC_NAMES:
        next_priorities: List[Tuple[str, ...]] = []
        for prefix in priorities:
            for i in range(len(prefix) + 1):
                next_priorities.append(prefix[:i] + (metric,) + prefix[i:])
        priorities = next_priorities
    return sorted(priorities)


def _profile_priority(profile: str, rng: np.random.Generator) -> Tuple[str, ...]:
    metric_by_profile = {
        "fuel_first": "fuel",
        "time_first": "time",
        "observation_first": "observation",
        "safety_first": "safety_margin",
    }
    if profile in metric_by_profile:
        first = metric_by_profile[profile]
        rest = [m for m in METRIC_NAMES if m != first]
        perm = rng.permutation(rest).tolist()
        return tuple([first] + [str(x) for x in perm])

    all_priorities = enumerate_priority_profiles()
    return tuple(all_priorities[int(rng.integers(len(all_priorities)))])


def _sample_sigma(profile: str, rng: np.random.Generator) -> IRSigma:
    priority = _profile_priority(profile, rng)
    weights = {
        metric: float(len(METRIC_NAMES) - rank)
        for rank, metric in enumerate(priority)
    }
    total = sum(weights.values())
    weights = {metric: val / total for metric, val in weights.items()}

    thresholds: Dict[str, float] = {}
    if profile in ("hard_filters", "time_first") and rng.random() < 0.7:
        thresholds["time"] = float(rng.choice([30, 45, 60, 75, 90]))
    if profile in ("hard_filters", "fuel_first") and rng.random() < 0.5:
        thresholds["fuel"] = float(rng.choice([0.03, 0.05, 0.08, 0.12]))
    if profile in ("hard_filters", "safety_first") and rng.random() < 0.5:
        thresholds["safety_margin"] = float(rng.choice([5.0, 10.0, 20.0]))

    return IRSigma(priority=priority, weights=weights, thresholds=thresholds)


def _domain_direction(domain: Optional[str]) -> Optional[str]:
    if domain is None:
        return None
    if domain in ("b_Pos_EI", "c_Pos_Flat"):
        return "+V"
    if domain in ("d_Neg_EI", "e_Neg_Flat"):
        return "-V"
    if domain == "a_Safe_Orbit":
        return "center"
    return None


def _sample_terminal_domain(
    scenario: Optional[ScenarioSample],
    rng: np.random.Generator,
    allow_none: bool,
) -> Optional[str]:
    domains = sorted(NODES.keys())
    if scenario is not None and scenario.start_domain in domains and len(domains) > 1:
        # Bias away from the current domain, while still allowing no terminal
        # constraint for broad-coverage examples.
        domains = [d for d in domains if d != scenario.start_domain]
    if allow_none and rng.random() < 0.25:
        return None
    return str(domains[int(rng.integers(len(domains)))])


def _sample_goal(profile: str, rng: np.random.Generator) -> IRGoal:
    if profile == "balanced":
        task_class = TASK_CLASSES[int(rng.integers(len(TASK_CLASSES)))]
    elif profile == "time_first":
        task_class = str(rng.choice(["flyby", "ducking", "approach"]))
    elif profile == "observation_first":
        task_class = str(rng.choice(["circumnav", "hold", "approach"]))
    elif profile == "safety_first":
        task_class = str(rng.choice(["retreat", "hold", "circumnav"]))
    else:
        task_class = str(TASK_CLASSES[int(rng.integers(len(TASK_CLASSES)))])
    return IRGoal(task_class=task_class)


def _sample_filters(
    profile: str,
    rng: np.random.Generator,
    *,
    max_phase: int,
) -> Tuple[IRFilters, List[str], Optional[bool]]:
    notes: List[str] = []
    expected_mask_nonempty: Optional[bool] = None

    max_tof_steps: Optional[int] = None
    min_safety_margin_m: Optional[float] = None
    forbidden_domains: Tuple[str, ...] = ()
    required_behaviors: Tuple[int, ...] = ()
    forbidden_behaviors: Tuple[int, ...] = ()
    max_num_phases: Optional[int] = None

    if profile in ("hard_filters", "time_first") or rng.random() < 0.25:
        max_tof_steps = int(rng.choice([35, 50, 65, 80, 95]))
        notes.append("sampled_max_tof_steps")

    if profile in ("hard_filters", "safety_first") and rng.random() < 0.5:
        min_safety_margin_m = float(rng.choice([5.0, 10.0, 20.0]))
        notes.append("sampled_min_safety_margin")

    if rng.random() < (0.45 if profile == "hard_filters" else 0.2):
        n_forbidden = int(rng.integers(1, min(3, len(NODES)) + 1))
        domains = rng.choice(sorted(NODES.keys()), size=n_forbidden, replace=False)
        forbidden_domains = tuple(str(x) for x in domains)
        notes.append("sampled_forbidden_domains")

    behavior_ids = sorted(BEHAVIOR_IDS.keys())
    if rng.random() < (0.35 if profile == "hard_filters" else 0.15):
        required_behaviors = (int(behavior_ids[int(rng.integers(len(behavior_ids)))]),)
        notes.append("sampled_required_behavior")

    if rng.random() < (0.45 if profile == "hard_filters" else 0.2):
        candidates = [b for b in behavior_ids if b not in required_behaviors]
        n_forbidden = int(rng.integers(1, min(3, len(candidates)) + 1))
        forbidden_behaviors = tuple(
            int(x) for x in rng.choice(candidates, size=n_forbidden, replace=False)
        )
        notes.append("sampled_forbidden_behaviors")

    if rng.random() < 0.35:
        max_num_phases = int(rng.integers(1, int(max_phase) + 1))
        notes.append("sampled_max_num_phases")

    if profile == "hard_filters":
        # Some hard-filter examples are intentionally tight; these are useful
        # later for re-query/abstention training once masks are implemented.
        expected_mask_nonempty = None
        notes.append("mask_effect_requires_item_4")

    return (
        IRFilters(
            max_tof_steps=max_tof_steps,
            min_safety_margin_m=min_safety_margin_m,
            forbidden_domains=forbidden_domains,
            required_behaviors=required_behaviors,
            forbidden_behaviors=forbidden_behaviors,
            max_num_phases=max_num_phases,
        ),
        notes,
        expected_mask_nonempty,
    )


def sample_ir(
    sample_id: int,
    *,
    seed: int,
    scenario: Optional[ScenarioSample] = None,
    split: str = "unspecified",
    profile: str = "balanced",
    max_phase: int = 3,
) -> IRSample:
    """
    Deterministically sample structured intent representation for training.

    This sampler intentionally does not call an LLM and does not consume raw
    text. LLM parsing should target this IR schema as a separate workstream.
    """

    profile = str(profile)
    if profile == "auto":
        # Keep auto deterministic for a given sample_id and seed.
        rng_profile = _rng_for_index(seed=int(seed), index=int(sample_id))
        profile = str(IR_PROFILES[int(rng_profile.integers(len(IR_PROFILES)))])
    if profile not in IR_PROFILES:
        raise ValueError(f"profile must be one of {IR_PROFILES} or 'auto', got {profile}.")

    rng = _rng_for_index(seed=int(seed) + 31, index=int(sample_id))
    sigma = _sample_sigma(profile, rng)
    terminal_domain = _sample_terminal_domain(
        scenario=scenario,
        rng=rng,
        allow_none=(profile == "balanced"),
    )
    dz = IRDeltaZ(
        terminal_domain=terminal_domain,
        direction=_domain_direction(terminal_domain),
    )
    goal = _sample_goal(profile, rng)
    filters, notes, expected_mask_nonempty = _sample_filters(profile, rng, max_phase=max_phase)

    if scenario is not None:
        notes.append(f"conditioned_on_start_domain={scenario.start_domain}")

    return IRSample(
        ir=IR(sigma=sigma, dz=dz, g=goal, filters=filters),
        seed=int(seed),
        split=str(split),
        profile=profile,
        expected_mask_nonempty=expected_mask_nonempty,
        notes=tuple(notes),
    )


def sample_ir_batch(
    n_samples: int,
    *,
    seed: int,
    scenarios: Optional[Sequence[ScenarioSample]] = None,
    split_config: Optional[SplitConfig] = None,
    profile: str = "auto",
    max_phase: int = 3,
) -> List[IRSample]:
    split_config = split_config or SplitConfig()
    out: List[IRSample] = []
    for sample_id in range(int(n_samples)):
        scenario = None if scenarios is None else scenarios[int(sample_id) % len(scenarios)]
        split = split_config.split_for_index(sample_id, int(n_samples))
        out.append(
            sample_ir(
                sample_id=sample_id,
                seed=seed,
                scenario=scenario,
                split=split,
                profile=profile,
                max_phase=max_phase,
            )
        )
    return out


def _load_scp_verifier_backend():
    work_folder = ROOT_FOLDER / "work"
    if str(work_folder) not in sys.path:
        sys.path.append(str(work_folder))
    from datagen_reasoning import generate_traj_with_wyp
    from rages_scoring import compute_metrics

    return generate_traj_with_wyp, compute_metrics


def verify_waypoint_plan(
    scenario: ScenarioSample,
    action: Action,
    waypoint_plan: WaypointPlan,
    *,
    config: Optional[SCPVerifierConfig] = None,
    seed: Optional[int] = None,
    candidate_id: Optional[int] = None,
) -> SCPVerifierResult:
    """
    Run the existing SCP verifier stack for one fixed waypoint plan.

    This is intentionally a thin wrapper around `work/datagen_reasoning.py`.
    It does not alter CVX/SCP settings; it only standardizes inputs and emits a
    deterministic log suitable for Stage 2 dataset generation.
    """

    config = config or SCPVerifierConfig()
    if len(action.b_seq) != len(waypoint_plan.waypoint_states):
        raise ValueError(
            "Action b_seq length must match WaypointPlan waypoint_states length."
        )

    generate_traj_with_wyp, compute_metrics = _load_scp_verifier_backend()
    solved = generate_traj_with_wyp(
        x0=np.asarray(scenario.x0, dtype=float),
        x_pred=np.asarray(waypoint_plan.waypoint_states, dtype=float),
        dt_pred=np.asarray(waypoint_plan.dt_fractions, dtype=float),
        tof_steps=int(action.tof_steps),
        koz_dim=np.asarray(scenario.koz_dim, dtype=float),
        artms=np.asarray(scenario.artms_scale_range_1e3, dtype=float),
        dt_sec=float(config.dt_sec),
        oec0_mod=np.asarray(scenario.oec0_modified, dtype=float),
        obj_type=config.obj_type,
    )

    status_cvx = str(solved.get("status_cvx", "missing_status"))
    status_scp = str(solved.get("status_scp", "missing_status"))
    metrics = {
        "fuel_dv": float("nan"),
        "transfer_time_sec": float("nan"),
        "observation_score": float("nan"),
        "safety_margin_m": float("nan"),
    }
    if status_scp in OK_STATUS:
        metrics.update(
            compute_metrics(
                prob=solved["prob"],
                roe=solved["roe_scp"],
                actions=solved["actions_scp"],
                rtn_ct=solved["rtn_scp_ct"],
            )
        )

    log: Dict[str, object] = {
        "verifier_version": config.verifier_version,
        "sample_id": int(scenario.sample_id),
        "split": scenario.split,
        "candidate_id": None if candidate_id is None else int(candidate_id),
        "seed": None if seed is None else int(seed),
        "action": action.to_dict(),
        "scenario": {
            "start_domain": scenario.start_domain,
            "koz_dim": np.asarray(scenario.koz_dim, dtype=float).tolist(),
            "artms_scale_range_1e3": np.asarray(
                scenario.artms_scale_range_1e3, dtype=float
            ).tolist(),
        },
        "waypoint_plan": waypoint_plan.to_dict(),
        "config": config.to_dict(),
        "status": {"cvx": status_cvx, "scp": status_scp},
    }
    return SCPVerifierResult(
        status_cvx=status_cvx,
        status_scp=status_scp,
        metrics=metrics,
        log=log,
        raw=solved,
    )
