import os,sys 
from pathlib import Path
import numpy as np
import re
from typing import Dict, Any, List, Tuple, Optional
def find_root_path(path:str, word:str):
    parts = path.split(word, 1)
    return parts[0] + word if len(parts) > 1 else path 
root_folder = Path(__file__).resolve().parents[2]

import optimization.parameters as param
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")

def extract_features(
    template: str,
    cmd_text: str,
) -> Tuple[Optional[Dict[str, float]], List[str]]:
    """
    Given an unfilled template and a filled command string, extract
    placeholder → numeric value. Returns (cmd_params, used_placeholders).
    If no placeholders are used, cmd_params and used_placeholders are both [].
    """
    
    def _normalize_spaces(s: str) -> str:
        return " ".join(s.split())

    def _build_pattern(template: str) -> Tuple[re.Pattern, List[str]]:
        """
        Build a regex that matches a filled command based on an unfilled template
        with {var} placeholders, and returns the used placeholder names.
        """
        tmpl = _normalize_spaces(template)
        parts: List[str] = []
        used: List[str] = []
        last = 0

        for m in _PLACEHOLDER_RE.finditer(tmpl):
            # literal text before this placeholder
            parts.append(re.escape(tmpl[last:m.start()]))

            var = m.group(1)
            used.append(var)

            # numeric value (signed int/float)
            parts.append(rf"(?P<{var}>[-+]?\d+(?:\.\d+)?)")
            last = m.end()

        # trailing literal
        parts.append(re.escape(tmpl[last:]))

        pattern_str = "^" + "".join(parts) + "$"
        return re.compile(pattern_str), used
    
    pattern, used_placeholders = _build_pattern(template)
    cmd_norm = _normalize_spaces(cmd_text)
    m = pattern.fullmatch(cmd_norm)

    if not m:
        raise ValueError(
            f"Command text does not match template (except for numbers):\n"
            f"  template: {template}\n"
            f"  command:  {cmd_text}"
        )

    groups = m.groupdict()
    cmd_params = {k: float(v) for k, v in groups.items() if v is not None}
    return cmd_params, used_placeholders

def check_semantics(
    behavior_id: int,
    used_placeholders: List[str],
    cmd_params: Dict[str, float],
    traj: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Check whether a trajectory semantically matches the command placeholders.

    Returns a dict of absolute mismatches (e.g., d_roef, T_appr_idx, ...).
    """    
    def _vicinity_mask(
        roe: np.ndarray,
        target: np.ndarray,
        droe: np.ndarray,
        droe_close_factor: float,
    ) -> np.ndarray:
        """
        Return a boolean mask over time where ROE is within a scaled tolerance
        of a target ROE.
        """
        is_close = np.isclose(roe, target[None, :], atol=droe[None, :] * droe_close_factor)
        return is_close.all(axis=1)

    def _first_true(mask: np.ndarray, idx0: int = 0, default: int = None) -> int:
        """
        Return the first index >= idx0 where mask is True.
        If no such index exists, return `default`.
        """
        # Only search from idx0 onward
        idx = np.where(mask[idx0:])[0]
        if len(idx) == 0:
            return default
        
        # Add idx0 offset because idx is relative to mask[idx0:]
        return int(idx0 + idx[0])
    
    def _last_true(mask: np.ndarray, idx0: int = 0, default: int = None) -> int:
        """
        Return the last index >= idx0 where mask is True.
        If none exist, return `default`.
        """
        # Search only in mask[idx0:]
        sub = mask[idx0:]
        idx = np.where(sub)[0]
        if idx.size == 0:
            return default
        # last true index is the last element in idx
        return int(idx0 + idx[-1])

    def _last_false_plus_one(mask: np.ndarray) -> int:
        # last index where mask is False, then +1
        idx_false = np.flatnonzero(~mask)
        if idx_false.size == 0:
            # all True → return len(mask) 
            return len(mask)
        return int(idx_false[-1] + 1)
    
    ##### Hyperparameters #####
    droe_close_factor = 2
    ###########################

    if verbose:
        print(cmd_params)

    out: Dict[str, Any] = { "behavior": behavior_id, "d_roef": None,}

    # For all behaviors we need roe at least once
    roe = traj["roe"]

    if behavior_id == 0:  # approach and circumnavigate KOZ
        roef_targ = np.array([0, 0, 0, 32, 0, 32], dtype=float)
        droe = np.array([2, 5, 2, 2, 2, 2], dtype=float)

        if verbose: print(f"  Terminal state mismatch traj vs cmd : {roe[-1]} vs {roef_targ}")
        out["d_roef"] = roe[-1] - roef_targ

        if "T_appr_orbits" in used_placeholders:
            T_appr_idx_cmd = cmd_params["T_appr_orbits"] * param.period / param.dt_sec

            mask = _vicinity_mask(roe, roe[-1], droe, droe_close_factor)
            T_appr_idx = _last_false_plus_one(mask)

            if verbose: print(f"  T_appr_orbits mismatch traj vs cmd: {T_appr_idx} vs {T_appr_idx_cmd}")
            out["d_T_appr_idx"] = T_appr_idx - T_appr_idx_cmd

    elif behavior_id == 1:  # dock
        roef_targ = np.array([0, -35, 0, 0, 0, 0], dtype=float)
        droe = np.array([2, 5, 2, 2, 2, 2], dtype=float)

        if verbose: print(f"  Terminal state mismatch traj vs cmd : {roe[-1]} vs {roef_targ}")
        out["d_roef"] = roe[-1] - roef_targ

        if "T_appr_orbits" in used_placeholders:
            T_appr_idx_cmd = cmd_params["T_appr_orbits"] * param.period / param.dt_sec
            mask = _vicinity_mask(roe, roe[-1], droe, droe_close_factor)
            T_appr_idx = _last_false_plus_one(mask)

            if verbose: print(f"  T_appr_orbits mismatch traj vs cmd: {T_appr_idx} vs {T_appr_idx_cmd}")
            out["d_T_appr_idx"] = T_appr_idx - T_appr_idx_cmd

        if "d_lambda_meters" in used_placeholders:
            d_lambda_meters = cmd_params["d_lambda_meters"]
            if verbose: print(f"  d_lambda_meters mismatch traj vs cmd : {roe[-1, 1]} vs {d_lambda_meters}")
            out["d_d_lambda_meters"] = roe[-1, 1] - d_lambda_meters


    elif behavior_id == 2:  # flyby (under KOZ)
        roef_targ = np.array([0, 120, 0, 5, 0, 5], dtype=float)
        droe = np.array([2, 20, 2, 2, 2, 2], dtype=float)

        if verbose: print(f"  Terminal state mismatch traj vs cmd : {roe[-1]} vs {roef_targ}")
        out["d_roef"] = roe[-1] - roef_targ

        if "T_appr_orbits" in used_placeholders:
            T_appr_idx_cmd = cmd_params["T_appr_orbits"] * param.period / param.dt_sec
            mask = _vicinity_mask(roe, roe[-1], droe, droe_close_factor)
            T_appr_idx = _last_false_plus_one(mask)

            if verbose: print(f"  T_appr_orbits mismatch traj vs cmd: {T_appr_idx} vs {T_appr_idx_cmd}")
            out["d_T_appr_idx"] = T_appr_idx - T_appr_idx_cmd
            
        if "d_lambda_meters" in used_placeholders:
            d_lambda_meters = cmd_params["d_lambda_meters"]
            if verbose: print(f"  d_lambda_meters mismatch traj vs cmd : {roe[-1, 1]} vs {d_lambda_meters}")
            out["d_d_lambda_meters"] = roe[-1, 1] - d_lambda_meters

    elif behavior_id == 3:  # flyby (E/I-separated)
        roef_targ = np.array([0, 120, 0, 5, 0, 5], dtype=float)
        wyp0_targ = np.array([0, -120, 0, 25, 0, 25], dtype=float)
        wyp1_targ = np.array([0, 120, 0, 25, 0, 25], dtype=float)
        droe = np.array([2, 10, 2, 2, 2, 2], dtype=float)
        if verbose: print(f"  Terminal state mismatch traj vs cmd : {roe[-1]} vs {roef_targ}")
        out["d_roef"] = roe[-1] - roef_targ
        
        mask = _vicinity_mask(roe, wyp0_targ, droe, droe_close_factor)
        T_EI_sep_idx = _first_true(mask)
        if verbose: print(f'roe at T_EI_sep_idx: {roe[:, T_EI_sep_idx]} vs target {wyp0_targ}')
        if T_EI_sep_idx is None:
            out["d_T_EI_sep_idx"] = None 
            out["d_T_transfer_idx"] = None
            return out
        
        if "T_EI_sep_orbits" in used_placeholders:
            T_EI_sep_idx_cmd = cmd_params["T_EI_sep_orbits"] * param.period / param.dt_sec
            if verbose: print(f"  T_EI_sep_orbits mismatch traj vs cmd : {T_EI_sep_idx} vs {T_EI_sep_idx_cmd}")
            out["d_T_EI_sep_idx"] = T_EI_sep_idx - T_EI_sep_idx_cmd

        if "T_transfer_orbits" in used_placeholders:
            mask = _vicinity_mask(roe, wyp1_targ, droe, droe_close_factor)
            T_transfer_idx = _first_true(mask, idx0=T_EI_sep_idx)
            if T_transfer_idx is None:
                out["d_T_transfer_idx"] = None
                return out
            dT_transfer_idx = T_transfer_idx - T_EI_sep_idx
            dT_transfer_idx_cmd = (cmd_params["T_transfer_orbits"]) * param.period / param.dt_sec
            if verbose: print(f"  T_transfer_orbits mismatch traj vs cmd : {dT_transfer_idx} vs {dT_transfer_idx_cmd}")
            out["d_T_transfer_idx"] = dT_transfer_idx - dT_transfer_idx_cmd

    elif behavior_id == 4:  # approach, circumnavigate, and forward
        roef_targ = np.array([0, 120, 0, 35, 0, 35], dtype=float)
        wyp0_targ = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        wyp1_targ = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        droe = np.array([2, 10, 2, 2, 2, 2], dtype=float)
        
        if verbose: print(f"  Terminal state mismatch traj vs cmd : {roe[-1]} vs {roef_targ}")
        out["d_roef"] = roe[-1] - roef_targ
        
        mask = _vicinity_mask(roe, wyp0_targ, droe, droe_close_factor)
        T_appr_idx = _first_true(mask)
        if T_appr_idx is None:
            out["d_T_appr_idx"] = None
            out["d_T_circ_idx"] = None
            return out

        if "T_appr_orbits" in used_placeholders:
            T_appr_idx_cmd = cmd_params["T_appr_orbits"] * param.period / param.dt_sec
            if verbose: print(f"  T_appr_orbits mismatch: {T_appr_idx} vs {T_appr_idx_cmd}")
            out["d_T_appr_idx"] = T_appr_idx - T_appr_idx_cmd

        if "T_circ_orbits" in used_placeholders:
            mask = _vicinity_mask(roe, wyp1_targ, droe, droe_close_factor)
            T_circ_idx = _last_true(mask, idx0=T_appr_idx)
            if T_circ_idx is None:
                out["d_T_circ_idx"] = None
                return out
            dT_circ_idx = T_circ_idx - T_appr_idx
            dT_circ_idx_cmd = (cmd_params["T_circ_orbits"]) * param.period / param.dt_sec
            if verbose: print(f"  T_circ_orbits mismatch: {dT_circ_idx} vs {dT_circ_idx_cmd}")
            out["d_T_circ_idx"] = dT_circ_idx - dT_circ_idx_cmd

    elif behavior_id == 5:  # approach, circumnavigate, and retreat
        roef_targ = np.array([0, -120, 0, 35, 0, 35], dtype=float)
        wyp0_targ = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        wyp1_targ = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        droe = np.array([2, 10, 2, 2, 2, 2], dtype=float)
        
        if verbose: print(f"  Terminal state mismatch traj vs cmd : {roe[-1]} vs {roef_targ}")
        out["d_roef"] = roe[-1] - roef_targ

        mask = _vicinity_mask(roe, wyp0_targ, droe, droe_close_factor)
        T_appr_idx = _first_true(mask)
        if T_appr_idx is None:
            out["d_T_appr_idx"] = None
            out["d_T_circ_idx"] = None
            return out

        if "T_appr_orbits" in used_placeholders:
            T_appr_idx_cmd = cmd_params["T_appr_orbits"] * param.period / param.dt_sec
            if verbose: print(f"  T_appr_orbits mismatch: {T_appr_idx} vs {T_appr_idx_cmd}")
            out["d_T_appr_idx"] = T_appr_idx - T_appr_idx_cmd

        if "T_circ_orbits" in used_placeholders:            
            mask = _vicinity_mask(roe, wyp1_targ, droe, droe_close_factor)
            T_circ_idx = _last_true(mask, idx0=T_appr_idx)
            if T_circ_idx is None:
                out["d_T_circ_idx"] = None
                return out
            dT_circ_idx = T_circ_idx - T_appr_idx
            dT_circ_idx_cmd = (cmd_params["T_circ_orbits"]) * param.period / param.dt_sec
            if verbose: print(f"  T_circ_orbits mismatch: {dT_circ_idx} vs {dT_circ_idx_cmd}")
            out["d_T_circ_idx"] = dT_circ_idx - dT_circ_idx_cmd

    else:
        raise ValueError(f"Unknown behavior_id: {behavior_id}")

    return out


def check_semantics2(
    behavior_id: int,
    used_placeholders: List[str],
    cmd_params: Dict[str, float],
    traj: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Check whether a trajectory semantically matches the command placeholders.

    Returns a dict of absolute mismatches (e.g., d_roef, T_appr_idx, ...).
    """
    ##### Hyperparameters #####
    droe_close_factor = np.array([1, 2, 3], dtype=float)
    ###########################

    if verbose:
        print(cmd_params)

    out: Dict[str, Any] = { "behavior": behavior_id,
                           "is_correct": [True] * len(droe_close_factor),
                           "is_correct_with_terminal": [True] * len(droe_close_factor),
                           "d_roef": None,}

    # For all behaviors we need roe at least once
    roe = traj["roe"]

    if behavior_id == 0:  # approach and circumnavigate KOZ
        roef_targ = np.array([0, 0, 0, 32, 0, 32], dtype=float)
        droe = np.array([2, 5, 2, 2, 2, 2], dtype=float)

        if verbose: print(f"  Terminal state mismatch traj vs cmd : {roe[-1]} vs {roef_targ}")
        out["d_roef"] = roe[-1] - roef_targ

        for i, factor in enumerate(droe_close_factor):
            if not np.isclose(roe[-1], roef_targ, atol=droe * factor).all():
                out["is_correct_with_terminal"][i] = False

        if "T_appr_orbits" in used_placeholders:
            T_appr_idx_cmd = int(cmd_params["T_appr_orbits"] * param.period / param.dt_sec)
            out["d_T_appr_idx"] = roe[T_appr_idx_cmd] - roe[-1]
            for i, factor in enumerate(droe_close_factor):
                if not np.isclose(roe[T_appr_idx_cmd], roe[-1], atol=droe * factor).all():
                    out["is_correct"][i] = False
                    out["is_correct_with_terminal"][i] = False

    elif behavior_id == 1:  # dock
        roef_targ = np.array([0, -35, 0, 0, 0, 0], dtype=float)
        droe = np.array([2, 5, 2, 2, 2, 2], dtype=float)

        if verbose: print(f"  Terminal state mismatch traj vs cmd : {roe[-1]} vs {roef_targ}")
        out["d_roef"] = roe[-1] - roef_targ
        for i, factor in enumerate(droe_close_factor):
            if not np.isclose(roe[-1], roef_targ, atol=droe * factor).all():
                out["is_correct_with_terminal"][i] = False

        if "T_appr_orbits" in used_placeholders:
            T_appr_idx_cmd = int(cmd_params["T_appr_orbits"] * param.period / param.dt_sec)            
            out["d_T_appr_idx"] = roe[T_appr_idx_cmd] - roe[-1]
            for i, factor in enumerate(droe_close_factor):
                if not np.isclose(roe[T_appr_idx_cmd], roe[-1], atol=droe * factor).all():
                    out["is_correct"][i] = False
                    out["is_correct_with_terminal"][i] = False

        if "d_lambda_meters" in used_placeholders:
            d_lambda_meters = cmd_params["d_lambda_meters"]
            out["d_d_lambda_meters"] = roe[-1, 1] - d_lambda_meters
            for i, factor in enumerate(droe_close_factor):
                if not np.isclose(roe[-1, 1], d_lambda_meters, atol=droe[1] * factor):
                    out["is_correct"][i] = False
                    out["is_correct_with_terminal"][i] = False

    elif behavior_id == 2:  # flyby (under KOZ)
        roef_targ = np.array([0, 150, 0, 5, 0, 5], dtype=float)
        droe = np.array([2, 20, 2, 2, 2, 2], dtype=float)

        if verbose: print(f"  Terminal state mismatch traj vs cmd : {roe[-1]} vs {roef_targ}")
        out["d_roef"] = roe[-1] - roef_targ
        for i, factor in enumerate(droe_close_factor):
            if not np.isclose(roe[-1], roef_targ, atol=droe * factor).all():
                out["is_correct_with_terminal"][i] = False

        if "T_appr_orbits" in used_placeholders:
            T_appr_idx_cmd = int(cmd_params["T_appr_orbits"] * param.period / param.dt_sec)
            out["d_T_appr_idx"] = roe[T_appr_idx_cmd] - roe[-1]
            for i, factor in enumerate(droe_close_factor):
                if not np.isclose(roe[T_appr_idx_cmd], roe[-1], atol=droe * factor).all():
                    out["is_correct"][i] = False
                    out["is_correct_with_terminal"][i] = False
            
        if "d_lambda_meters" in used_placeholders:
            d_lambda_meters = cmd_params["d_lambda_meters"]
            out["d_d_lambda_meters"] = roe[-1, 1] - d_lambda_meters
            for i, factor in enumerate(droe_close_factor):
                if not np.isclose(roe[-1, 1], d_lambda_meters, atol=droe[1] * factor):
                    out["is_correct"][i] = False
                    out["is_correct_with_terminal"][i] = False

    elif behavior_id == 3:  # flyby (E/I-separated)
        roef_targ = np.array([0, 120, 0, 5, 0, 5], dtype=float)
        wyp0_targ = np.array([0, -120, 0, 25, 0, 25], dtype=float)
        wyp1_targ = np.array([0, 120, 0, 25, 0, 25], dtype=float)
        droe = np.array([2, 10, 2, 2, 2, 2], dtype=float)
        if verbose: print(f"  Terminal state mismatch traj vs cmd : {roe[-1]} vs {roef_targ}")
        out["d_roef"] = roe[-1] - roef_targ
        for i, factor in enumerate(droe_close_factor):
            if not np.isclose(roe[-1], roef_targ, atol=droe * factor).all():
                # print("mismatch at terminal! ", np.round(roe[:, -1], 3), " vs ", roef_targ)
                out["is_correct_with_terminal"][i] = False

        if "T_EI_sep_orbits" in used_placeholders:
            T_EI_sep_idx_cmd = int(cmd_params["T_EI_sep_orbits"] * param.period / param.dt_sec)
            out["d_T_EI_sep_idx"] = roe[T_EI_sep_idx_cmd] - wyp0_targ
            for i, factor in enumerate(droe_close_factor):
                if not np.isclose(roe[T_EI_sep_idx_cmd], wyp0_targ, atol=droe * factor).all():
                    # print("mismatch at T_EI_sep! ", np.round(roe[:, T_EI_sep_idx_cmd], 3), " vs ", wyp0_targ)
                    out["is_correct"][i] = False
                    out["is_correct_with_terminal"][i] = False

        if "T_transfer_orbits" in used_placeholders:
            T_transfer_idx_cmd = int(cmd_params["T_transfer_orbits"] * param.period / param.dt_sec)
            out["d_T_transfer_idx"] = roe[T_transfer_idx_cmd] - wyp1_targ
            for i, factor in enumerate(droe_close_factor):
                if not np.isclose(roe[T_transfer_idx_cmd], wyp1_targ, atol=droe * factor).all():
                    # print("mismatch at T_transfer! ", np.round(roe[T_transfer_idx_cmd], 3), " vs ", wyp1_targ)
                    out["is_correct"][i] = False
                    out["is_correct_with_terminal"][i] = False

    elif behavior_id == 4:  # approach, circumnavigate, and forward
        roef_targ = np.array([0, 120, 0, 35, 0, 35], dtype=float)
        wyp0_targ = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        wyp1_targ = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        droe = np.array([2, 10, 2, 2, 2, 2], dtype=float)
        
        if verbose: print(f"  Terminal state mismatch traj vs cmd : {roe[-1]} vs {roef_targ}")
        out["d_roef"] = roe[-1] - roef_targ
        for i, factor in enumerate(droe_close_factor):
            if not np.isclose(roe[-1], roef_targ, atol=droe * factor).all():
                out["is_correct_with_terminal"][i] = False

        if "T_appr_orbits" in used_placeholders:
            T_appr_idx_cmd = int(cmd_params["T_appr_orbits"] * param.period / param.dt_sec)
            out["d_T_appr_idx"] = roe[T_appr_idx_cmd] - wyp0_targ
            for i, factor in enumerate(droe_close_factor):
                if not np.isclose(roe[T_appr_idx_cmd], wyp0_targ, atol=droe * factor).all():
                    out["is_correct"][i] = False
                    out["is_correct_with_terminal"][i] = False
            
        if "T_circ_orbits" in used_placeholders:
            T_circ_idx_cmd = int(cmd_params["T_circ_orbits"] * param.period / param.dt_sec)
            out["d_T_circ_idx"] = roe[T_circ_idx_cmd] - wyp1_targ
            for i, factor in enumerate(droe_close_factor):
                if not np.isclose(roe[T_circ_idx_cmd], wyp1_targ, atol=droe * factor).all():
                    out["is_correct"][i] = False
                    out["is_correct_with_terminal"][i] = False

    elif behavior_id == 5:  # approach, circumnavigate, and retreat
        roef_targ = np.array([0, -120, 0, 35, 0, 35], dtype=float)
        wyp0_targ = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        wyp1_targ = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        droe = np.array([2, 10, 2, 2, 2, 2], dtype=float)
        
        if verbose: print(f"  Terminal state mismatch traj vs cmd : {roe[-1]} vs {roef_targ}")
        out["d_roef"] = roe[-1] - roef_targ

        for i, factor in enumerate(droe_close_factor):
            if not np.isclose(roe[-1], roef_targ, atol=droe * factor).all():
                out["is_correct_with_terminal"][i] = False

        if "T_appr_orbits" in used_placeholders:
            T_appr_idx_cmd = int(cmd_params["T_appr_orbits"] * param.period / param.dt_sec)
            out["d_T_appr_idx"] = roe[T_appr_idx_cmd] - wyp0_targ
            for i, factor in enumerate(droe_close_factor):
                if not np.isclose(roe[T_appr_idx_cmd], wyp0_targ, atol=droe * factor).all():
                    out["is_correct"][i] = False
                    out["is_correct_with_terminal"][i] = False

        if "T_circ_orbits" in used_placeholders:
            T_circ_idx_cmd = int(cmd_params["T_circ_orbits"] * param.period / param.dt_sec)
            out["d_T_circ_idx"] = roe[T_circ_idx_cmd] - wyp1_targ
            for i, factor in enumerate(droe_close_factor):
                if not np.isclose(roe[T_circ_idx_cmd], wyp1_targ, atol=droe * factor).all():
                    out["is_correct"][i] = False
                    out["is_correct_with_terminal"][i] = False
    else:
        raise ValueError(f"Unknown behavior_id: {behavior_id}")

    return out

def eval_semantics_traj(
    behavior_id: int,
    cmd_text: str,
    template: str,
    traj: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    High-level wrapper.

    template is the unfilled string, e.g.
      templates_index[behavior_id]["description"][command_id]

    Binary score: True iff ALL defined constraints are satisfied.
    If no constraints are defined, treat as trivially satisfied.
    """
    if verbose: print(f"behavior {behavior_id} / command: {cmd_text}")
    cmd_params, used_placeholders = extract_features(template, cmd_text)   
    out_dict = check_semantics2(
        behavior_id=behavior_id,
        used_placeholders=used_placeholders,
        cmd_params=cmd_params,
        traj=traj,
        verbose=verbose,
    )

    return out_dict
