from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset

from dynamics.dynamics_trans import mu_E
from parameters import (
    ANSWER_TAG,
    BEHAVIOR_IDS,
    END_TAG,
    NODES,
    POLICY_REGISTRY,
    SYSTEM_PROMPT,
)


def _contiguous_split_index(n_rows: int, val_ratio: float) -> int:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1). Got: {val_ratio}")
    if n_rows <= 0:
        raise ValueError("Cannot split an empty dataset.")
    if val_ratio == 0.0:
        return n_rows

    split_idx = int(n_rows * (1.0 - val_ratio))
    if split_idx <= 0 or split_idx >= n_rows:
        raise ValueError(
            f"Validation ratio {val_ratio} leaves an empty split for dataset size {n_rows}."
        )
    return split_idx


def contiguous_train_eval_split(
    dataset: Dataset,
    val_ratio: float,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Split dataset by order: first (1-val_ratio) for train, last val_ratio for eval."""
    n_rows = len(dataset)
    split_idx = _contiguous_split_index(n_rows=n_rows, val_ratio=val_ratio)
    if split_idx == n_rows:
        return dataset, None

    train_dataset = dataset.select(range(split_idx))
    eval_dataset = dataset.select(range(split_idx, n_rows))
    return train_dataset, eval_dataset


def contiguous_train_eval_index_ranges(
    n_rows: int,
    val_ratio: float,
) -> Tuple[range, Optional[range]]:
    """Return contiguous train/eval row index ranges using the same split rule."""
    split_idx = _contiguous_split_index(n_rows=n_rows, val_ratio=val_ratio)
    if split_idx == n_rows:
        return range(0, n_rows), None
    return range(0, split_idx), range(split_idx, n_rows)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def append_jsonl_line(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(payload), ensure_ascii=False, allow_nan=True))
        f.write("\n")


def as_float_list(x: Sequence[float]) -> List[float]:
    return [float(v) for v in np.asarray(x, dtype=float).reshape(-1)]


def sample_from_range(r: Any, rng: np.random.Generator) -> float:
    first = r[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        rr = r[int(rng.integers(len(r)))]
        lo, hi = float(rr[0]), float(rr[1])
    else:
        lo, hi = float(r[0]), float(r[1])
    grid = np.linspace(lo, hi, 5, dtype=float)
    return float(grid[int(rng.integers(len(grid)))])


def sample_state_from_node(node_name: str, rng: np.random.Generator) -> np.ndarray:
    vol = NODES[node_name]
    dl = sample_from_range(vol.d_lambda_range, rng)
    dey = sample_from_range(vol.d_ex_range, rng)
    return np.array([0.0, dl, 0.0, dey, 0.0, dey], dtype=float)


def in_range(x: float, r: Tuple[float, float], tol: float = 0.0) -> bool:
    return (r[0] - tol) <= x <= (r[1] + tol)


def in_multirange(x: float, r: Any, tol: float = 0.0) -> bool:
    if isinstance(r[0], (list, tuple, np.ndarray)):
        return any(in_range(x, rr, tol=tol) for rr in r)
    return in_range(x, r, tol=tol)


def classify_orbital_domain(state: np.ndarray, nodes=NODES, tol: float = 1e-6) -> List[str]:
    dl = float(state[1])
    dey = float(state[3])
    matches = []
    for name, vol in nodes.items():
        if in_range(dl, vol.d_lambda_range, tol=tol) and in_multirange(dey, vol.d_ex_range, tol=tol):
            matches.append(name)
    return matches


def behavior_seq_to_text(b_seq: Sequence[int]) -> str:
    if not b_seq:
        return "[]"
    return " -> ".join([BEHAVIOR_IDS.get(int(b), f"unknown({b})") for b in b_seq])


def time_grid_from_orbits(
    total_orbits: float,
    dt_sec: float,
    n_time_max: int,
    semi_major_axis_m: float,
) -> Tuple[int, np.ndarray]:
    period = 2.0 * np.pi * np.sqrt((semi_major_axis_m ** 3) / mu_E)
    total_sec = float(total_orbits) * period
    n_steps = int(np.round(total_sec / dt_sec))
    n_time = max(2, min(n_time_max, n_steps + 1))
    tvec_sec = np.arange(n_time, dtype=float) * dt_sec
    return n_time, tvec_sec


def waypoint_times_from_dts(dt_seq: Sequence[float], n_time: int) -> List[int]:
    total = float(np.sum(dt_seq))
    cum = np.cumsum(list(dt_seq)[:-1]) / total
    idx = np.rint(cum * (n_time - 1)).astype(int)
    idx = np.clip(idx, 1, n_time - 2)

    for i in range(1, len(idx)):
        idx[i] = max(idx[i], idx[i - 1] + 1)

    max_last = n_time - 2
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


def format_reasoning_user_prompt(inp: Dict[str, Any]) -> str:
    return (
        "You're an expert spacecraft operator for rendezvous missions.\n"
        "Task: choose (b_seq, tf) based on mission context and intent priority.\n"
        "Then provide one-line reasoning and justification.\n\n"
        f"x0_roe_m = {inp['x0']}\n"
        f"oec0_modified = {inp['oec0_modified']}\n"
        f"koz_param = {inp['koz_param']}\n"
        f"artms_scaling_1e3 = {inp['artms_scaling_1e3']}\n"
        f"intent_priority = {inp['intent_priority']}\n\n"
        "Return JSON with keys: reasoning, tf, b_seq.\n"
    )


def make_reasoning_answer_payload(out: Dict[str, Any]) -> str:
    payload = {
        "reasoning": out["reasoning"],
        "tf": int(out["tf"]),
        "b_seq": [int(x) for x in out["b_seq"]],
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def apply_chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    return (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"User:\n{user_prompt}\n"
        "Assistant:\n"
    )


def align_special_tokens(model: Any, tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has no pad/eos/unk token to use as pad_token.")

    token_ids = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    for key, value in token_ids.items():
        setattr(model.config, key, value)
        if getattr(model, "generation_config", None) is not None:
            setattr(model.generation_config, key, value)


def extract_first_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    candidates = [cleaned]
    first_brace = cleaned.find("{")
    while first_brace != -1:
        candidates.append(cleaned[first_brace:])
        first_brace = cleaned.find("{", first_brace + 1)

    for candidate in candidates:
        try:
            payload, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    preview = cleaned[:300].replace("\n", "\\n")
    raise ValueError(f"Unable to parse JSON object from model output. Preview: {preview!r}")


def parse_reasoning_output_strict(
    text: str,
    *,
    max_phase: Optional[int] = None,
) -> Dict[str, Any]:
    chunk = text
    if ANSWER_TAG in text:
        chunk = text.split(ANSWER_TAG, 1)[1]
    if END_TAG in chunk:
        chunk = chunk.split(END_TAG, 1)[0]
    payload_str = chunk.strip()
    if not payload_str:
        raise ValueError("Empty payload after answer tag.")

    payload = extract_first_json_object(payload_str)
    for key in ("reasoning", "tf", "b_seq"):
        if key not in payload:
            raise ValueError(f"Missing key in payload: {key}")

    reasoning = str(payload["reasoning"]).strip()
    tf = int(payload["tf"])
    b_seq = [int(x) for x in payload["b_seq"]]

    if tf <= 0:
        raise ValueError(f"Invalid tf={tf}; must be positive.")
    if len(b_seq) == 0:
        raise ValueError("b_seq is empty.")
    if max_phase is not None and len(b_seq) > int(max_phase):
        raise ValueError(f"b_seq length {len(b_seq)} exceeds model max_phase={max_phase}.")

    return {
        "reasoning": reasoning,
        "tf": tf,
        "b_seq": b_seq,
        "raw_text": text,
    }


def recover_target_domains_with_policy(
    policy_name: str,
    start_node: str,
    b_seq: Sequence[int],
) -> Optional[List[str]]:
    policy = POLICY_REGISTRY.get(policy_name)
    if policy is None:
        return None

    current_node = str(start_node)
    out_nodes: List[str] = []
    i = 0
    max_steps = max(8, int(len(b_seq)) + 6)

    for step_index in range(max_steps):
        if i >= len(b_seq):
            break

        options = policy.get_next_options(current_node, step_index)
        if not options:
            return None

        wanted = int(b_seq[i])
        picked = None
        for next_node, beh, _ in options:
            if int(beh) == wanted:
                picked = str(next_node)
                break

        if picked is not None:
            out_nodes.append(picked)
            current_node = picked
            i += 1
            continue

        dummy = None
        for next_node, beh, _ in options:
            if int(beh) == 0:
                dummy = str(next_node)
                break
        if dummy is None:
            return None
        current_node = dummy

    if i != len(b_seq):
        return None
    return out_nodes


def recover_target_domains_any_policy(
    start_node: str,
    b_seq: Sequence[int],
    policy_hint: Optional[str] = None,
    *,
    strict_policy_hint: bool = False,
) -> Tuple[Optional[List[str]], Optional[str]]:
    if policy_hint:
        out = recover_target_domains_with_policy(policy_hint, start_node, b_seq)
        if out is not None:
            return out, policy_hint
        if strict_policy_hint:
            return None, policy_hint

    for name in sorted(POLICY_REGISTRY.keys()):
        if policy_hint and name == policy_hint:
            continue
        out = recover_target_domains_with_policy(name, start_node, b_seq)
        if out is not None:
            return out, name
    return None, None


def is_behavior_sequence_feasible_in_graph(
    start_domain: Optional[str],
    b_seq: Sequence[int],
    policy_hint: Optional[str] = None,
    *,
    strict_policy_hint: bool = False,
) -> bool:
    if start_domain is None or len(b_seq) == 0:
        return False
    target_domains, _ = recover_target_domains_any_policy(
        start_node=start_domain,
        b_seq=b_seq,
        policy_hint=policy_hint,
        strict_policy_hint=strict_policy_hint,
    )
    return bool(target_domains is not None and len(target_domains) == len(b_seq))


def randomize_intent_priority(intent_priority: Sequence[str], seed: int) -> List[str]:
    items = [str(x) for x in intent_priority]
    if len(items) <= 1:
        return items
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(items))
    return [items[int(i)] for i in order]


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    rollout_id: int
    dataset_idx: int
    x0: np.ndarray
    koz_param: np.ndarray
    artms_scaling_1e3: np.ndarray
    oec0_modified: np.ndarray
    intent_priority: List[str]
    start_domain: Optional[str]


def sample_scenario_from_dataset(
    scenario_id: int,
    rollout_id: int,
    dataset_idx: int,
    data: Dict[str, torch.Tensor],
    intent_priority: Sequence[str],
) -> Scenario:
    x0 = data["x0"][dataset_idx].numpy().astype(float)
    koz_param = data["koz_dim"][dataset_idx].numpy().astype(float)
    artms = data["artms_scale_range_1e3"][dataset_idx].numpy().astype(float)
    oec0_mod = data["oec0_modified"][dataset_idx].numpy().astype(float)

    start_domains = classify_orbital_domain(x0)
    start_domain = str(start_domains[0]) if start_domains else None

    return Scenario(
        scenario_id=int(scenario_id),
        rollout_id=int(rollout_id),
        dataset_idx=int(dataset_idx),
        x0=x0,
        koz_param=koz_param,
        artms_scaling_1e3=artms,
        oec0_modified=oec0_mod,
        intent_priority=[str(x) for x in intent_priority],
        start_domain=start_domain,
    )


class ReasoningSampler:
    def __init__(
        self,
        adapter_dir: Path,
        base_model: str,
        max_phase: int,
        *,
        trust_remote_code: bool = True,
    ) -> None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_phase = int(max_phase)
        self.adapter_dir = Path(adapter_dir)
        self.base_model = str(base_model)
        self.trust_remote_code = bool(trust_remote_code)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_dir,
            use_fast=True,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
        )
        self.model = PeftModel.from_pretrained(base, str(self.adapter_dir))
        self.model.eval()
        align_special_tokens(self.model, self.tokenizer)

    def generate_text(self, inputs: Dict[str, Any]) -> str:
        user_prompt = format_reasoning_user_prompt(inputs)
        prompt_text = apply_chat_prompt(self.tokenizer, user_prompt)
        enc = self.tokenizer(prompt_text, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=220,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = out[0][enc["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=False)

    def sample_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raw = self.generate_text(inputs)
        return parse_reasoning_output_strict(raw, max_phase=self.max_phase)

    def sample(self, scenario: Scenario) -> Dict[str, Any]:
        llm_inputs = {
            "x0": np.round(scenario.x0, 2).tolist(),
            "oec0_modified": np.round(scenario.oec0_modified, 6).tolist(),
            "koz_param": np.round(scenario.koz_param, 4).tolist(),
            "artms_scaling_1e3": np.round(scenario.artms_scaling_1e3, 3).tolist(),
            "intent_priority": list(scenario.intent_priority),
        }
        return self.sample_inputs(llm_inputs)
