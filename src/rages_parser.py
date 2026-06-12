from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # Keep offline imports usable in lean Python envs.
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        return False

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None  # type: ignore[assignment]

from parameters import BEHAVIOR_IDS, DEFAULT_INTENT_PRIORITY, NODES
from rages_sampling import (
    IR,
    IRDeltaZ,
    IRFilters,
    IRGoal,
    IRSigma,
    METRIC_NAMES,
    TASK_CLASSES,
    TERMINAL_DIRECTIONS,
)

Messages = Sequence[Mapping[str, str]]
CompletionFn = Callable[[Messages, "ParserConfig"], str]


@dataclass(frozen=True)
class ParserConfig:
    backend: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 900
    parser_version: str = "v0_parser"
    schema_version: str = "rages_ir_v1"
    api_key_env: str = "OPENAI_API_KEY"


@dataclass(frozen=True)
class ParseResult:
    ir: Optional[IR]
    abstained: bool
    command_class: Optional[str]
    confidence: float
    raw_response: Optional[str] = None
    errors: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()
    parser_version: str = "v0_parser"
    schema_version: str = "rages_ir_v1"

    def to_dict(self) -> Dict[str, object]:
        return {
            "ir": None if self.ir is None else self.ir.to_dict(),
            "abstained": self.abstained,
            "command_class": self.command_class,
            "confidence": float(self.confidence),
            "raw_response": self.raw_response,
            "errors": list(self.errors),
            "notes": list(self.notes),
            "parser_version": self.parser_version,
            "schema_version": self.schema_version,
        }


PARSER_SYSTEM_PROMPT = (
    "Parse spacecraft rendezvous mission text into strict RAGES IR JSON. "
    "Use only allowed schema values. Output JSON only. If the text is malformed, "
    "out of scope, or lacks a task class, return {\"abstain\": true}. "
    "If no terminal target is stated, set both dz fields to null."
)


TASK_ALIASES = {
    "circumnav": ("circumnav", "circumnavigate", "circumnavigation", "circle", "orbit around"),
    "flyby": ("flyby", "fly-by", "fly by", "pass by"),
    "ducking": ("ducking", "duck", "fast drift"),
    "hold": ("hold", "station keep", "station-keeping", "keep station", "loiter"),
    "approach": ("approach", "rendezvous", "close in"),
    "retreat": ("retreat", "depart", "back away", "withdraw"),
}

DOMAIN_ALIASES = {
    "a_Safe_Orbit": ("a_safe_orbit", "safe orbit", "center", "centre", "safe zone"),
    "b_Pos_EI": ("b_pos_ei", "+v ei", "+v ei zone", "pos ei", "positive ei", "plus v ei"),
    "c_Pos_Flat": ("c_pos_flat", "+v flat", "+v flat zone", "pos flat", "positive flat", "plus v flat"),
    "d_Neg_EI": ("d_neg_ei", "-v ei", "-v ei zone", "neg ei", "negative ei", "minus v ei"),
    "e_Neg_Flat": ("e_neg_flat", "-v flat", "-v flat zone", "neg flat", "negative flat", "minus v flat"),
}

PRIORITY_WORDS = (
    "priority",
    "prioritize",
    "prioritise",
    "prefer",
    "first",
    "then",
    "before",
    "over",
    "minimize",
    "minimise",
    "maximize",
    "maximise",
)
NEGATION_WORDS = ("no", "not", "without", "do not", "does not", "not specify", "unspecified")
FORBID_WORDS = ("avoid", "forbid", "forbidden", "exclude", "without", "stay out", "no", "do not")
TERMINAL_WORDS = ("finish", "end", "terminal", "target", "to", "into", "back to", "return to", "arrive")
REQUIRE_WORDS = ("require", "required", "must use", "include", "use")


def _default_priority() -> Tuple[str, ...]:
    metric_set = set(str(x) for x in METRIC_NAMES)
    out: List[str] = []
    for metric in tuple(DEFAULT_INTENT_PRIORITY) + tuple(METRIC_NAMES):
        metric = str(metric)
        if metric in metric_set and metric not in out:
            out.append(metric)
    return tuple(out)


def _complete_priority(priority: Sequence[str]) -> Tuple[str, ...]:
    metric_set = set(str(x) for x in METRIC_NAMES)
    out: List[str] = []
    for metric in priority:
        metric = str(metric)
        if metric not in metric_set:
            raise ValueError(f"Unknown priority metric: {metric}")
        if metric not in out:
            out.append(metric)
    for metric in _default_priority():
        if metric not in out:
            out.append(metric)
    if set(out) != metric_set:
        raise ValueError("priority must cover all metrics")
    return tuple(out)


def _weights_for_priority(priority: Sequence[str]) -> Dict[str, float]:
    priority = _complete_priority(priority)
    raw = {metric: float(len(priority) - i) for i, metric in enumerate(priority)}
    total = sum(raw.values())
    return {metric: value / total for metric, value in raw.items()}


def _metric_aliases() -> Dict[str, Tuple[str, ...]]:
    aliases = {
        str(metric): (
            str(metric),
            str(metric).replace("_", " "),
            str(metric).replace("_", "-"),
        )
        for metric in METRIC_NAMES
    }
    extras = {
        "fuel": ("fuel", "delta-v", "delta v", "dv", "propellant"),
        "time": ("time", "duration", "transfer time", "tof", "fast", "quick"),
        "observation": ("observation", "observe", "coverage", "view", "inspect", "imaging"),
        "safety_margin": ("safety", "safety margin", "clearance", "keepout", "koz"),
    }
    return {
        metric: tuple(sorted(set(vals) | set(extras.get(metric, ()))))
        for metric, vals in aliases.items()
    }


def build_schema_doc() -> str:
    return "\n".join(
        [
            f"Allowed priority metrics: {list(METRIC_NAMES)}",
            f"Default priority tail: {list(_default_priority())}",
            f"Allowed task_class values: {list(TASK_CLASSES)}",
            f"Allowed terminal directions: {list(TERMINAL_DIRECTIONS)}",
            f"Allowed domains: {list(sorted(NODES.keys()))}",
            f"Allowed behavior ids: {list(sorted(BEHAVIOR_IDS.keys()))}",
            "Return either {\"abstain\": true, \"confidence\": 0.0, \"notes\": [...]}"
            " or an object with sigma, dz, g, filters, confidence, notes.",
            "Complete partial priority lists by appending the default tail.",
        ]
    )


def build_few_shot_examples() -> str:
    priority = _complete_priority(("fuel", "safety_margin"))
    good = {
        "sigma": {"priority": list(priority), "weights": _weights_for_priority(priority), "thresholds": {}},
        "dz": {"terminal_domain": None, "direction": None},
        "g": {"task_class": "circumnav"},
        "filters": {
            "max_tof_steps": None,
            "min_safety_margin_m": None,
            "forbidden_domains": ["e_Neg_Flat"],
            "required_behaviors": [],
            "forbidden_behaviors": [],
            "max_num_phases": 2,
        },
        "confidence": 0.8,
        "notes": ["terminal intent not specified"],
    }
    return "\n".join(
        [
            "Text: Circumnavigate the target, prioritize fuel then safety, avoid the -V flat zone, no more than 2 phases.",
            "JSON: " + json.dumps(good, sort_keys=True),
            'Text: Write a poem about lunch.\nJSON: {"abstain": true, "confidence": 0.0, "notes": ["out of scope"]}',
        ]
    )


def build_parser_messages(raw_text: str, config: Optional[ParserConfig] = None) -> List[Dict[str, str]]:
    del config
    return [
        {"role": "system", "content": PARSER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "\n\n".join([build_schema_doc(), build_few_shot_examples(), f"Text:\n{raw_text}"]),
        },
    ]


def openai_completion(messages: Messages, config: ParserConfig) -> str:
    if OpenAI is None:
        raise ValueError("The openai package is not installed in this Python environment.")
    load_dotenv()
    api_key = os.environ.get(config.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"{config.api_key_env} is not set.")
    client = OpenAI(api_key=api_key)
    rsp = client.chat.completions.create(
        model=config.model,
        messages=[dict(msg) for msg in messages],
        temperature=float(config.temperature),
        max_tokens=int(config.max_tokens),
    )
    return (rsp.choices[0].message.content or "").strip()


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text.strip())
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _direction_for_domain(domain: Optional[str]) -> Optional[str]:
    if domain in ("b_Pos_EI", "c_Pos_Flat"):
        return "+V"
    if domain in ("d_Neg_EI", "e_Neg_Flat"):
        return "-V"
    if domain == "a_Safe_Orbit":
        return "center"
    return None


def ir_from_json(obj: Mapping[str, Any]) -> IR:
    if isinstance(obj.get("ir"), Mapping):
        obj = obj["ir"]  # type: ignore[assignment]
    sigma_obj = obj.get("sigma", {}) or {}
    dz_obj = obj.get("dz", {}) or {}
    goal_obj = obj.get("g", {}) or {}
    filters_obj = obj.get("filters", {}) or {}
    if not all(isinstance(x, Mapping) for x in (sigma_obj, dz_obj, goal_obj, filters_obj)):
        raise ValueError("sigma, dz, g, and filters must be objects")

    priority = _complete_priority(sigma_obj.get("priority", ()))  # type: ignore[union-attr]
    weights = _weights_for_priority(priority)
    supplied_weights = sigma_obj.get("weights", {})  # type: ignore[union-attr]
    if isinstance(supplied_weights, Mapping):
        weights.update({str(k): float(v) for k, v in supplied_weights.items()})
    thresholds = sigma_obj.get("thresholds", {}) or {}  # type: ignore[union-attr]
    if not isinstance(thresholds, Mapping):
        raise ValueError("sigma.thresholds must be an object")

    terminal_domain = dz_obj.get("terminal_domain", None)  # type: ignore[union-attr]
    direction = dz_obj.get("direction", None)  # type: ignore[union-attr]
    terminal_domain = None if terminal_domain in ("", "null") else terminal_domain
    direction = None if direction in ("", "null") else direction
    if terminal_domain is not None and direction is None:
        direction = _direction_for_domain(str(terminal_domain))

    task_class = goal_obj.get("task_class", None)  # type: ignore[union-attr]
    if task_class is None:
        raise ValueError("g.task_class is required")

    return IR(
        sigma=IRSigma(
            priority=priority,
            weights=weights,
            thresholds={str(k): float(v) for k, v in thresholds.items()},
        ),
        dz=IRDeltaZ(
            terminal_domain=None if terminal_domain is None else str(terminal_domain),
            direction=None if direction is None else str(direction),
        ),
        g=IRGoal(task_class=str(task_class)),
        filters=IRFilters(
            max_tof_steps=filters_obj.get("max_tof_steps", None),  # type: ignore[union-attr]
            min_safety_margin_m=filters_obj.get("min_safety_margin_m", None),  # type: ignore[union-attr]
            forbidden_domains=tuple(filters_obj.get("forbidden_domains", ()) or ()),  # type: ignore[union-attr]
            required_behaviors=tuple(filters_obj.get("required_behaviors", ()) or ()),  # type: ignore[union-attr]
            forbidden_behaviors=tuple(filters_obj.get("forbidden_behaviors", ()) or ()),  # type: ignore[union-attr]
            max_num_phases=filters_obj.get("max_num_phases", None),  # type: ignore[union-attr]
        ),
    )


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _matches(text: str, aliases: Iterable[str]) -> Iterable[re.Match[str]]:
    for alias in aliases:
        pattern = r"(?<![a-z0-9])" + re.escape(alias.lower()) + r"(?![a-z0-9])"
        yield from re.finditer(pattern, text)


def _near(text: str, start: int, end: int, words: Sequence[str], radius: int = 36) -> bool:
    window = text[max(0, start - radius):min(len(text), end + radius)]
    return any(re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", window) for word in words)


def _parse_task(text: str) -> Optional[str]:
    found: List[Tuple[int, str]] = []
    for task, aliases in TASK_ALIASES.items():
        if task not in TASK_CLASSES:
            continue
        for match in _matches(text, aliases):
            if not _near(text, match.start(), match.end(), NEGATION_WORDS):
                found.append((match.start(), task))
    return sorted(found)[0][1] if found else None


def _parse_priority(text: str) -> Tuple[str, ...]:
    if not any(word in text for word in PRIORITY_WORDS):
        return _default_priority()
    found: List[Tuple[int, str]] = []
    for metric, aliases in _metric_aliases().items():
        positions = [match.start() for match in _matches(text, aliases)]
        if positions:
            found.append((min(positions), metric))
    return _complete_priority([metric for _, metric in sorted(found)])


def _parse_domains(text: str) -> Tuple[Optional[str], Optional[str], Tuple[str, ...]]:
    terminal_domain: Optional[str] = None
    forbidden: List[str] = []
    for domain, aliases in DOMAIN_ALIASES.items():
        if domain not in NODES:
            continue
        for match in _matches(text, aliases):
            if _near(text, match.start(), match.end(), FORBID_WORDS):
                if domain not in forbidden:
                    forbidden.append(domain)
            elif terminal_domain is None and _near(text, match.start(), match.end(), TERMINAL_WORDS):
                terminal_domain = domain

    direction = _direction_for_domain(terminal_domain)
    if terminal_domain is None:
        if re.search(r"\b(to|toward|towards|finish|end|terminal|target)\b.{0,20}\+v\b", text):
            direction = "+V"
        elif re.search(r"\b(to|toward|towards|finish|end|terminal|target)\b.{0,20}-v\b", text):
            direction = "-V"
        elif re.search(r"\b(to|finish|end|terminal|target|back to|return to)\b.{0,24}\b(center|centre)\b", text):
            direction = "center"
    return terminal_domain, direction, tuple(forbidden)


def _parse_int(text: str, patterns: Sequence[str]) -> Optional[int]:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _parse_float(text: str, patterns: Sequence[str]) -> Optional[float]:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def _parse_behaviors(text: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    required: List[int] = []
    forbidden: List[int] = []
    for match in re.finditer(r"\b(?:behavior|behaviour|b)\s*(\d+)\b", text):
        behavior_id = int(match.group(1))
        if behavior_id not in BEHAVIOR_IDS:
            continue
        if _near(text, match.start(), match.end(), FORBID_WORDS):
            forbidden.append(behavior_id)
        elif _near(text, match.start(), match.end(), REQUIRE_WORDS):
            required.append(behavior_id)
    return tuple(sorted(set(required))), tuple(sorted(set(forbidden)))


def heuristic_parse(raw_text: str, config: Optional[ParserConfig] = None) -> ParseResult:
    config = config or ParserConfig(backend="heuristic")
    text = _norm(raw_text)
    notes = ["heuristic backend"]
    if not text:
        return _abstain(config, "empty text", notes)

    task_class = _parse_task(text)
    if task_class is None:
        return _abstain(config, "task_class not found", notes)

    terminal_domain, direction, forbidden_domains = _parse_domains(text)
    if terminal_domain is None and direction is None:
        notes.append("terminal intent not specified; dz left unconstrained")
    required_behaviors, forbidden_behaviors = _parse_behaviors(text)

    try:
        ir = IR(
            sigma=IRSigma(
                priority=_parse_priority(text),
                weights=_weights_for_priority(_parse_priority(text)),
                thresholds={},
            ),
            dz=IRDeltaZ(terminal_domain=terminal_domain, direction=direction),
            g=IRGoal(task_class=task_class),
            filters=IRFilters(
                max_tof_steps=_parse_int(
                    text,
                    (
                        r"\b(?:max|maximum|under|within|up to|no more than|<=)\s+(\d+)\s+(?:tof\s+)?(?:time\s+)?steps?\b",
                        r"\b(\d+)\s+(?:tof\s+)?(?:time\s+)?steps?\s+(?:or less|max|maximum)\b",
                    ),
                ),
                min_safety_margin_m=_parse_float(
                    text,
                    (
                        r"\b(?:at least|minimum|min|>=)\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters)\s+(?:safety|clearance|margin)\b",
                        r"\b(?:safety|clearance|margin)\s+(?:at least|minimum|min|>=)\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters)\b",
                    ),
                ),
                forbidden_domains=forbidden_domains,
                required_behaviors=required_behaviors,
                forbidden_behaviors=forbidden_behaviors,
                max_num_phases=_parse_int(
                    text,
                    (
                        r"\b(?:max|maximum|under|within|up to|no more than|<=)\s+(\d+)\s+phases?\b",
                        r"\b(\d+)\s+phases?\s+(?:or less|max|maximum)\b",
                    ),
                ),
            ),
        )
    except ValueError as exc:
        return _abstain(config, str(exc), notes)

    confidence = 0.62
    confidence += 0.08 if terminal_domain is not None or direction is not None else 0.0
    confidence += 0.08 if forbidden_domains or required_behaviors or forbidden_behaviors else 0.0
    confidence += 0.08 if any(word in text for word in PRIORITY_WORDS) else 0.0
    return ParseResult(
        ir=ir,
        abstained=False,
        command_class=ir.g.task_class,
        confidence=min(confidence, 0.9),
        notes=tuple(notes),
        parser_version=config.parser_version,
        schema_version=config.schema_version,
    )


def _abstain(config: ParserConfig, error: str, notes: Sequence[str]) -> ParseResult:
    return ParseResult(
        ir=None,
        abstained=True,
        command_class=None,
        confidence=0.0,
        errors=(error,),
        notes=tuple(notes),
        parser_version=config.parser_version,
        schema_version=config.schema_version,
    )


def parse_intent_to_ir(
    raw_text: str,
    *,
    config: Optional[ParserConfig] = None,
    completion: Optional[CompletionFn] = None,
) -> ParseResult:
    config = config or ParserConfig()
    if config.backend == "heuristic":
        return heuristic_parse(raw_text, config)
    if config.backend != "openai":
        return _abstain(config, f"unknown backend: {config.backend}", ())
    if not raw_text.strip():
        return _abstain(config, "empty text", ())

    raw_response: Optional[str] = None
    try:
        raw_response = (completion or openai_completion)(build_parser_messages(raw_text, config), config)
        parsed = _extract_json_obj(raw_response)
        if parsed is None:
            raise ValueError("LLM response did not contain a JSON object")
        notes = tuple(str(x) for x in parsed.get("notes", ()) or ())
        confidence = float(parsed.get("confidence", 0.0))
        if bool(parsed.get("abstain", False)):
            return ParseResult(None, True, None, confidence, raw_response, notes=notes)
        ir = ir_from_json(parsed)
        return ParseResult(
            ir=ir,
            abstained=False,
            command_class=ir.g.task_class,
            confidence=float(parsed.get("confidence", max(confidence, 0.5))),
            raw_response=raw_response,
            notes=notes,
            parser_version=config.parser_version,
            schema_version=config.schema_version,
        )
    except Exception as exc:
        return ParseResult(
            ir=None,
            abstained=True,
            command_class=None,
            confidence=0.0,
            raw_response=raw_response,
            errors=(str(exc),),
            parser_version=config.parser_version,
            schema_version=config.schema_version,
        )


def parse_batch(
    raw_texts: Sequence[str],
    *,
    config: Optional[ParserConfig] = None,
    completion: Optional[CompletionFn] = None,
) -> List[ParseResult]:
    config = config or ParserConfig()
    return [parse_intent_to_ir(text, config=config, completion=completion) for text in raw_texts]


def ir_slots(ir: Optional[IR]) -> Dict[str, object]:
    if ir is None:
        return {
            "sigma.priority": None,
            "sigma.thresholds": tuple(),
            "dz.terminal_domain": None,
            "dz.direction": None,
            "g.task_class": None,
            "filters.max_tof_steps": None,
            "filters.min_safety_margin_m": None,
            "filters.forbidden_domains": tuple(),
            "filters.required_behaviors": tuple(),
            "filters.forbidden_behaviors": tuple(),
            "filters.max_num_phases": None,
        }
    return {
        "sigma.priority": tuple(ir.sigma.priority),
        "sigma.thresholds": tuple(sorted(ir.sigma.thresholds.items())),
        "dz.terminal_domain": ir.dz.terminal_domain,
        "dz.direction": ir.dz.direction,
        "g.task_class": ir.g.task_class,
        "filters.max_tof_steps": ir.filters.max_tof_steps,
        "filters.min_safety_margin_m": ir.filters.min_safety_margin_m,
        "filters.forbidden_domains": tuple(ir.filters.forbidden_domains),
        "filters.required_behaviors": tuple(ir.filters.required_behaviors),
        "filters.forbidden_behaviors": tuple(ir.filters.forbidden_behaviors),
        "filters.max_num_phases": ir.filters.max_num_phases,
    }


def _slot_group(slot: str) -> str:
    return slot.split(".", 1)[0]


def _score_value(slot: str, pred: object, gold: object, float_tol: float) -> Tuple[int, int, int]:
    if slot == "sigma.priority":
        return (1, 0, 0) if tuple(pred or ()) == tuple(gold or ()) else (0, 1, 1)
    if isinstance(pred, (tuple, list, set)) or isinstance(gold, (tuple, list, set)):
        pred_set = set(pred or ())
        gold_set = set(gold or ())
        if not pred_set and not gold_set:
            return 1, 0, 0
        return len(pred_set & gold_set), len(pred_set - gold_set), len(gold_set - pred_set)
    if pred is None or gold is None:
        return (1, 0, 0) if pred is gold else (0, int(pred is not None), int(gold is not None))
    if isinstance(pred, (float, int)) and isinstance(gold, (float, int)):
        return (1, 0, 0) if abs(float(pred) - float(gold)) <= float_tol else (0, 1, 1)
    return (1, 0, 0) if pred == gold else (0, 1, 1)


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score_slots(pred: Optional[IR], gold: Optional[IR], *, float_tol: float = 1e-6) -> Dict[str, object]:
    pred_slots = ir_slots(pred)
    gold_slots = ir_slots(gold)
    groups: Dict[str, Dict[str, int]] = {}
    slots: Dict[str, Dict[str, object]] = {}
    total = {"tp": 0, "fp": 0, "fn": 0}

    for slot in sorted(set(pred_slots) | set(gold_slots)):
        tp, fp, fn = _score_value(slot, pred_slots.get(slot), gold_slots.get(slot), float_tol)
        group = _slot_group(slot)
        groups.setdefault(group, {"tp": 0, "fp": 0, "fn": 0})
        for key, value in (("tp", tp), ("fp", fp), ("fn", fn)):
            groups[group][key] += value
            total[key] += value
        slots[slot] = {"group": group, "tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}

    return {
        "slots": slots,
        "groups": {g: {**c, **_prf(c["tp"], c["fp"], c["fn"])} for g, c in sorted(groups.items())},
        "overall": {**total, **_prf(total["tp"], total["fp"], total["fn"])},
    }
