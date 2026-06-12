from __future__ import annotations

"""
Soft-preference scoring (`rho_sigma`) and the IR hard-filter placeholder.

The initial `rho_sigma` is a tolerance-based lexicographic ordering rather
than a scalar score: metrics are compared in intent-priority order, and two
candidates are tied on a metric when they fall in the same epsilon-width
bucket. Pairwise tolerance ties (|m_i - m_j| <= eps) are not transitive, so
each metric is quantized as floor(value / eps) instead; this keeps the order
total, deterministic, and per-candidate (independent of the candidate group).
With eps = 0 the comparison is strict and recovers the v1 `rank_det` behavior
in work/datagen_reasoning.py.

Downstream use only needs an ordering: group-relative GRPO advantages, top-1
selection, and Kendall-tau evaluation all consume ranks, not scalars.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dynamics.dynamics_trans import propagate_oe
from parameters import DEFAULT_INTENT_PRIORITY, INTENT_TO_METRIC, METRIC_PREF, OK_STATUS


# Placeholder tolerances in metric units. Recalibrate from Stage 2 candidate
# groups via `epsilons_from_metric_spread` once frozen p_phi + SCP data exists.
DEFAULT_EPSILONS: Dict[str, float] = {
    "fuel": 0.01,  # [m/s]
    "time": 900.0,  # [s], one discrete time step
    "observation": 1.0,  # [-]
    "safety_margin": 5.0,  # [m]
}
VERIFIER_METRIC_KEYS: Tuple[str, ...] = tuple(INTENT_TO_METRIC[p] for p in DEFAULT_INTENT_PRIORITY)


def verifier_feasible(result: Any) -> bool:
    """
    Extract SCP feasibility from either `SCPVerifierResult` or a legacy row.
    """
    converged = getattr(result, "converged", None)
    if converged is not None:
        return bool(converged)
    status_scp = getattr(result, "status_scp", None)
    if status_scp is None and isinstance(result, Mapping):
        status_scp = result.get("status_scp")
    return str(status_scp) in OK_STATUS


def verifier_metric_row(result: Any) -> Dict[str, float]:
    """
    Extract the metric dict consumed by `rho_sigma`.

    Supports the new verifier wrapper, dicts with a nested `metrics` field, and
    the old flat candidate rows from `work/datagen_reasoning.py`.
    """
    raw_metrics = getattr(result, "metrics", None)
    if raw_metrics is None and isinstance(result, Mapping):
        raw_metrics = result.get("metrics", result)
    raw_metrics = raw_metrics or {}
    return {
        metric: float(raw_metrics.get(metric, np.nan))
        for metric in VERIFIER_METRIC_KEYS
    }


def verifier_scoring_inputs(results: Sequence[Any]) -> Tuple[List[Dict[str, float]], List[bool]]:
    """
    Convert verifier outputs to `(metric_rows, feasible)` for ranking helpers.
    """
    return [verifier_metric_row(r) for r in results], [verifier_feasible(r) for r in results]


def compute_obs_score(prob: Any, roe: np.ndarray) -> float:
    """
    Observation score used by the v1 SCP rollout labels.
    """
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


def compute_metrics(
    prob: Any,
    roe: np.ndarray,
    actions: np.ndarray,
    rtn_ct: np.ndarray,
) -> Dict[str, float]:
    """
    Compute verifier metrics for converged SCP rollouts.
    """
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


@dataclass(frozen=True)
class LexicographicPreference:
    """
    Soft preference `sigma` for tolerance-based lexicographic ranking.

    `priority` lists intent names (keys of INTENT_TO_METRIC) in decreasing
    importance. `epsilons` gives the tie tolerance per intent in metric units;
    a missing or non-positive epsilon means strict comparison for that metric.
    """

    priority: Tuple[str, ...] = tuple(DEFAULT_INTENT_PRIORITY)
    epsilons: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_EPSILONS))
    p_conv_threshold: float = 0.5

    def __post_init__(self) -> None:
        unknown = [p for p in self.priority if p not in INTENT_TO_METRIC]
        if unknown:
            raise ValueError(f"Unknown intent name(s) in priority: {unknown}")
        if len(set(self.priority)) != len(self.priority):
            raise ValueError(f"Duplicate intent names in priority: {self.priority}")
        threshold = float(self.p_conv_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("p_conv_threshold must be in [0, 1].")
        object.__setattr__(self, "priority", tuple(str(p) for p in self.priority))
        object.__setattr__(self, "p_conv_threshold", threshold)


def lexicographic_key(
    metrics: Mapping[str, float],
    preference: LexicographicPreference,
    feasible: bool = True,
) -> Tuple[float, ...]:
    """
    Per-candidate sort key; lexicographically smaller is better.

    Infeasible candidates rank after all feasible ones, and candidates with
    non-finite metric values rank after all fully valid ones. For Q outputs,
    pass `feasible = (p_conv >= threshold)` with metric means as `metrics`;
    the threshold is itself a sigma-level risk-tolerance parameter.
    """
    invalid = 0
    buckets: List[float] = []
    for intent in preference.priority:
        metric = INTENT_TO_METRIC[intent]
        val = float(metrics.get(metric, np.nan))
        if not np.isfinite(val):
            invalid = 1
            buckets.append(np.inf)
            continue
        if METRIC_PREF[metric] == "max":
            val = -val
        eps = float(preference.epsilons.get(intent, 0.0))
        buckets.append(float(np.floor(val / eps)) if eps > 0.0 else val)
    return tuple([0.0 if feasible else 1.0, float(invalid)] + buckets)


def rank_candidates(
    metric_rows: Sequence[Mapping[str, float]],
    preference: LexicographicPreference,
    feasible: Optional[Sequence[bool]] = None,
) -> List[int]:
    """
    Dense ranks (0 = best); candidates with equal keys share a rank.
    """
    n = len(metric_rows)
    flags = [True] * n if feasible is None else [bool(f) for f in feasible]
    if len(flags) != n:
        raise ValueError(f"feasible has length {len(flags)}, expected {n}.")
    keys = [lexicographic_key(m, preference, f) for m, f in zip(metric_rows, flags)]
    order = sorted(range(n), key=lambda i: keys[i])

    ranks = [0] * n
    rank = 0
    for pos, idx in enumerate(order):
        if pos > 0 and keys[idx] != keys[order[pos - 1]]:
            rank += 1
        ranks[idx] = rank
    return ranks


def select_best(
    metric_rows: Sequence[Mapping[str, float]],
    preference: LexicographicPreference,
    feasible: Optional[Sequence[bool]] = None,
) -> Optional[int]:
    """
    Index of the best feasible candidate (lowest index on ties); None if no
    candidate is feasible.
    """
    n = len(metric_rows)
    flags = [True] * n if feasible is None else [bool(f) for f in feasible]
    candidates = [
        (lexicographic_key(m, preference, True), i)
        for i, (m, f) in enumerate(zip(metric_rows, flags))
        if f
    ]
    if not candidates:
        return None
    return min(candidates)[1]


def q_p_conv(q_output: Any) -> float:
    p_conv = getattr(q_output, "p_conv", None)
    if p_conv is None and isinstance(q_output, Mapping):
        p_conv = q_output.get("p_conv")
    return float(p_conv)


def q_metric_row(q_output: Any) -> Dict[str, float]:
    raw_metrics = getattr(q_output, "metric_means", None)
    if raw_metrics is None and isinstance(q_output, Mapping):
        raw_metrics = q_output.get("metric_means", q_output.get("metrics", None))
    if isinstance(raw_metrics, Mapping):
        return {
            metric: float(raw_metrics.get(metric, np.nan))
            for metric in VERIFIER_METRIC_KEYS
        }
    vals = np.asarray(raw_metrics if raw_metrics is not None else [], dtype=float).reshape(-1)
    return {
        metric: float(vals[i]) if i < len(vals) else float("nan")
        for i, metric in enumerate(VERIFIER_METRIC_KEYS)
    }


def q_feasible(
    q_output: Any,
    preference: Optional[LexicographicPreference] = None,
    *,
    threshold: Optional[float] = None,
) -> bool:
    if threshold is None:
        threshold = 0.5 if preference is None else preference.p_conv_threshold
    return q_p_conv(q_output) >= float(threshold)


def q_scoring_inputs(
    q_outputs: Sequence[Any],
    preference: Optional[LexicographicPreference] = None,
    *,
    threshold: Optional[float] = None,
) -> Tuple[List[Dict[str, float]], List[bool]]:
    return (
        [q_metric_row(q) for q in q_outputs],
        [q_feasible(q, preference, threshold=threshold) for q in q_outputs],
    )


def rank_q_candidates(
    q_outputs: Sequence[Any],
    preference: LexicographicPreference,
) -> List[int]:
    metric_rows, feasible = q_scoring_inputs(q_outputs, preference)
    return rank_candidates(metric_rows, preference, feasible=feasible)


def select_best_q(
    q_outputs: Sequence[Any],
    preference: LexicographicPreference,
) -> Optional[int]:
    metric_rows, feasible = q_scoring_inputs(q_outputs, preference)
    return select_best(metric_rows, preference, feasible=feasible)


def hard_filter_pass_all(
    candidates: Sequence[object],
    scenario: object = None,
    ir: object = None,
) -> List[bool]:
    """
    v0 IR hard filter: admits every candidate.

    Grammar, precedence, and window validity are enforced by the behavior-graph
    sampler / action mask, not here. IR-derived hard constraints (forbidden
    windows, budgets, terminal-domain prohibitions) plug in later by replacing
    this function.
    """
    return [True] * len(candidates)


def epsilons_from_metric_spread(
    groups: Iterable[Sequence[Mapping[str, float]]],
    fraction: float = 0.05,
    intents: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Calibrate per-intent tolerances from within-group candidate spread.

    `groups` is an iterable of candidate groups, each holding metric dicts for
    candidates of the same scenario. The tolerance for each intent is
    `fraction` times the median within-group (max - min) of its metric; intents
    with no usable groups get 0.0 (strict comparison).
    """
    intents = list(intents) if intents is not None else list(DEFAULT_INTENT_PRIORITY)
    spreads: Dict[str, List[float]] = {p: [] for p in intents}
    for group in groups:
        for intent in intents:
            metric = INTENT_TO_METRIC[intent]
            vals = [float(m.get(metric, np.nan)) for m in group]
            vals = [v for v in vals if np.isfinite(v)]
            if len(vals) >= 2:
                spreads[intent].append(max(vals) - min(vals))
    return {
        intent: float(fraction) * float(np.median(s)) if s else 0.0
        for intent, s in spreads.items()
    }
