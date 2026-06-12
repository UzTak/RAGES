from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


SYSTEM_PROMPT = (
    "You are an assistant that selects trajectory decisions from structured mission context. "
    "Follow intent priority and constraints, reason briefly, and return a strict JSON answer."
)
THINK_TAG = "<|think|>"
ANSWER_TAG = "<|answer|>"
END_TAG = "<|end|>"

DEFAULT_INTENT_PRIORITY = ["fuel", "time", "observation", "safety_margin"]
INTENT_TO_METRIC = {
    "fuel": "fuel_dv",
    "time": "transfer_time_sec",
    "observation": "observation_score",
    "safety_margin": "safety_margin_m",
}
METRIC_PREF = {
    "fuel_dv": "min",
    "transfer_time_sec": "min",
    "observation_score": "max",
    "safety_margin_m": "max",
}
OK_STATUS = {"optimal", "optimal_inaccurate"}
DEFAULT_B_SEQ_ENCODING = "one_hot"
DEFAULT_B_SEQ_NUM_CLASSES = 11

DT_SEC = 900.0
N_TIME_MAX = 100
KOZ_DIMS = np.array([20.0, 30.0, 40.0], dtype=float)
ARTMS_SCALE_FACTORS = np.array([0.75, 1.0, 1.25, 1.5, 2.0], dtype=float)
TRUE_ANOMALY_GRID_RAD = np.deg2rad(np.linspace(0.0, 360.0, 20, endpoint=False))

N_CENTER = "a_Safe_Orbit"
N_POS_EI = "b_Pos_EI"
N_POS_FLAT = "c_Pos_Flat"
N_NEG_EI = "d_Neg_EI"
N_NEG_FLAT = "e_Neg_Flat"

BEHAVIOR_IDS = {
    1: "Station-Keeping",
    2: "Drift +V-direction",
    3: "Drift -V-direction",
    4: "Expand R/N separation",
    5: "Shrink R/N separation",
    6: "Approach from -V-bar",
    7: "Retreat to +V-bar",
    8: "Approach from +V-bar",
    9: "Retreat to -V-bar",
    10: "Ducking (fast drift) +V-direction",
    11: "Ducking (fast drift) -V-direction",
}

Range = Tuple[float, float]
MultiRange = Union[Range, Sequence[Range]]


@dataclass(frozen=True)
class Action:
    """
    Inference-time behavior action.

    This is intentionally limited to the learned guidance action:
    behavior sequence and total time of flight in discrete steps.
    Campaign policy names and per-phase dt_orbits are data-curation
    metadata only; they must not be treated as inference-time action
    fields for RAGES+.
    """

    b_seq: Tuple[int, ...]
    tof_steps: int

    def __post_init__(self) -> None:
        b_seq = tuple(int(b) for b in self.b_seq)
        tof_steps = int(self.tof_steps)
        if len(b_seq) == 0:
            raise ValueError("Action.b_seq must be non-empty.")
        invalid = [b for b in b_seq if b not in BEHAVIOR_IDS]
        if invalid:
            raise ValueError(f"Unknown behavior id(s): {invalid}")
        if tof_steps <= 0:
            raise ValueError("Action.tof_steps must be positive.")
        object.__setattr__(self, "b_seq", b_seq)
        object.__setattr__(self, "tof_steps", tof_steps)

    @classmethod
    def from_values(cls, b_seq: Sequence[int], tof_steps: int) -> "Action":
        return cls(b_seq=tuple(int(b) for b in b_seq), tof_steps=int(tof_steps))

    def to_dict(self) -> Dict[str, object]:
        return {
            "b_seq": [int(b) for b in self.b_seq],
            "tof_steps": int(self.tof_steps),
        }


@dataclass(frozen=True)
class ActionCurationMetadata:
    """
    Data-curation metadata used to synthesize scalable supervision.

    `policy`, `dt_orbits`, transfer windows, and target domains explain how
    a training sample was generated. They are useful for audits and dataset
    regeneration, but they are deliberately not part of `Action`.
    """

    policy: str
    dt_orbits: Tuple[float, ...]
    dt_ranges: Tuple[Range, ...]
    target_domains: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "policy": self.policy,
            "dt_orbits": [float(x) for x in self.dt_orbits],
            "dt_ranges": [[float(lo), float(hi)] for lo, hi in self.dt_ranges],
            "target_domains": list(self.target_domains),
        }


@dataclass(frozen=True)
class CuratedAction:
    """Action plus training-only curation metadata."""

    action: Action
    curation: ActionCurationMetadata


@dataclass
class NodeVolume:
    name: str
    d_lambda_range: Range
    d_ex_range: MultiRange
    d_i_range: Optional[MultiRange] = None

    def _sample_range(
        self,
        r: MultiRange,
        n: int = 5,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        if isinstance(r[0], (list, tuple)):
            idx = int(np.random.randint(len(r))) if rng is None else int(rng.integers(len(r)))
            lo, hi = r[idx]
        else:
            lo, hi = r
        grid = np.linspace(lo, hi, n)
        if rng is None:
            return float(np.random.choice(grid))
        return float(grid[int(rng.integers(len(grid)))])

    @property
    def d_iy_range(self) -> MultiRange:
        return self.d_i_range if self.d_i_range is not None else self.d_ex_range

    def sample(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        da = 0.0
        dl = self._sample_range(self.d_lambda_range, rng=rng)
        dex = 0.0
        dey = self._sample_range(self.d_ex_range, rng=rng)
        dix = 0.0
        diy = self._sample_range(self.d_iy_range, rng=rng)
        return np.array([da, dl, dex, dey, dix, diy])


# Keep EI-separated samples away from the flat near-zero band while allowing
# independent E/I signs: (+,+), (+,-), (-,+), and (-,-).
EI_SEPARATION_RANGES = ((30.0, 70.0), (-70.0, -30.0))
FLAT_EI_RANGE = (-5.0, 5.0)

NODES = {
    N_CENTER: NodeVolume(N_CENTER, (-5, 5), EI_SEPARATION_RANGES, EI_SEPARATION_RANGES),
    N_POS_EI: NodeVolume(N_POS_EI, (100, 250), EI_SEPARATION_RANGES, EI_SEPARATION_RANGES),
    N_POS_FLAT: NodeVolume(N_POS_FLAT, (100, 250), FLAT_EI_RANGE, FLAT_EI_RANGE),
    N_NEG_EI: NodeVolume(N_NEG_EI, (-250, -100), EI_SEPARATION_RANGES, EI_SEPARATION_RANGES),
    N_NEG_FLAT: NodeVolume(N_NEG_FLAT, (-250, -100), FLAT_EI_RANGE, FLAT_EI_RANGE),
}


class MissionPolicy:
    def get_valid_start_nodes(self) -> List[str]:
        raise NotImplementedError

    def get_next_options(self, current_node: str, step_index: int) -> List[Tuple[str, int, Tuple[float, float]]]:
        raise NotImplementedError

    def get_next_step(
        self,
        current_node: str,
        step_index: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[Tuple[str, int, Tuple[float, float]]]:
        options = self.get_next_options(current_node, step_index)
        if not options:
            return None
        idx = int(np.random.randint(len(options))) if rng is None else int(rng.integers(len(options)))
        return options[idx]


class CircumnavPolicy(MissionPolicy):
    def get_valid_start_nodes(self):
        return [N_POS_EI, N_POS_FLAT, N_NEG_EI, N_NEG_FLAT]

    def get_next_options(self, current_node, step_index):
        if step_index == 0:
            if current_node == N_POS_FLAT:
                return [(N_CENTER, 8, (2, 5))]
            if current_node == N_NEG_FLAT:
                return [(N_CENTER, 6, (2, 5))]
            if current_node == N_POS_EI:
                return [(N_CENTER, 3, (2, 5))]
            if current_node == N_NEG_EI:
                return [(N_CENTER, 2, (2, 5))]

        if step_index == 1:
            if current_node == N_CENTER:
                return [(N_CENTER, 1, (3, 5))]
            return []

        if step_index == 2:
            if current_node != N_CENTER:
                return []
            return [
                (N_POS_FLAT, 7, (2, 5)),
                (N_NEG_FLAT, 9, (2, 5)),
                (N_POS_EI, 2, (2, 5)),
                (N_NEG_EI, 3, (2, 5)),
            ]

        return []


class FlybyPolicy(MissionPolicy):
    def get_valid_start_nodes(self):
        return [N_POS_EI, N_POS_FLAT, N_NEG_FLAT, N_NEG_EI]

    def get_next_options(self, current_node, step_index):
        if step_index == 0:
            if current_node == N_NEG_FLAT:
                return [(N_NEG_EI, 4, (2, 5))]
            if current_node == N_NEG_EI:
                return [(N_NEG_EI, 0, (0, 0))]
            if current_node == N_POS_FLAT:
                return [(N_POS_EI, 4, (2, 5))]
            if current_node == N_POS_EI:
                return [(N_POS_EI, 0, (0, 0))]

        if step_index == 1:
            if current_node == N_NEG_EI:
                return [(N_POS_EI, 2, (6, 10))]
            if current_node == N_POS_EI:
                return [(N_NEG_EI, 3, (6, 10))]

        if step_index == 2:
            if current_node == N_POS_EI:
                return [
                    (N_POS_EI, 1, (2, 5)),
                    (N_POS_FLAT, 5, (2, 5)),
                ]
            if current_node == N_NEG_EI:
                return [
                    (N_NEG_EI, 1, (2, 5)),
                    (N_NEG_FLAT, 5, (2, 5)),
                ]

        return []


class DuckingPolicy(MissionPolicy):
    def get_valid_start_nodes(self):
        return [N_POS_FLAT, N_POS_EI, N_NEG_FLAT, N_NEG_EI]

    def get_next_options(self, current_node, step_index):
        if step_index == 0:
            if current_node == N_NEG_EI:
                return [(N_NEG_FLAT, 5, (2, 5))]
            if current_node == N_NEG_FLAT:
                return [(N_NEG_FLAT, 0, (0, 0))]
            if current_node == N_POS_EI:
                return [(N_POS_FLAT, 5, (2, 5))]
            if current_node == N_POS_FLAT:
                return [(N_POS_FLAT, 0, (0, 0))]

        if step_index == 1:
            if current_node == N_NEG_FLAT:
                return [(N_POS_FLAT, 10, (1, 2))]
            if current_node == N_POS_FLAT:
                return [(N_NEG_FLAT, 11, (1, 2))]

        if step_index == 2:
            if current_node == N_POS_FLAT:
                return [
                    (N_POS_FLAT, 1, (2, 5)),
                    (N_POS_EI, 4, (2, 5)),
                ]
            if current_node == N_NEG_FLAT:
                return [
                    (N_NEG_FLAT, 1, (2, 5)),
                    (N_NEG_EI, 4, (2, 5)),
                ]

        return []


POLICY_REGISTRY = {
    "CIRCUMNAV": CircumnavPolicy(),
    "FLYBY": FlybyPolicy(),
    "DUCKING": DuckingPolicy(),
}
