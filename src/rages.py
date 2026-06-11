from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

def find_root_path(path: str, word: str) -> str:
    parts = path.split(word, 1)
    return parts[0] + word if len(parts) > 1 else path

ROOT_FOLDER = Path(__file__).resolve().parents[1]

from datagen_reasoning import (
    generate_traj_with_wyp,
    load_model as load_wyp_model,
    load_dataset,
    predict_wyp_seq,
)
from train_wyp_predictor import build_input_slices
from utils import ReasoningSampler, to_jsonable


class RAGES:
    """
    Reasoning-based Autonomous Guidance Engine for Space.

    Primary API:
        out = engine.intent_to_traj(inputs)

    Required inputs:
        {
            "x0": [...6...],
            "koz_param": [...3...] or "koz_dim",
            "artms_scaling_1e3": [...6...],
            "intent_priority": ["fuel", "time", ...],
            "oec0_modified": [...10...]
        }
    """

    def __init__(
        self,
        adapter_dir: Path = ROOT_FOLDER / "rpod/rages/reasoning_model/v2/checkpoint-8400",
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        wyp_ckpt_path: Path = ROOT_FOLDER / "rpod/rages/wyp_model/model_gmm_v3_weighted_one_hot.pt",
        data_path: Path = ROOT_FOLDER / "rpod/rages/wyp_data/data_v3_discrete.pth",
    ) -> None:
        self.adapter_dir = Path(adapter_dir)
        self.base_model_name = base_model
        self.wyp_ckpt_path = Path(wyp_ckpt_path)
        self.data_path = Path(data_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) Load waypoint generator and dataset metadata/statistics
        self.wyp_model_bundle = load_wyp_model(self.wyp_ckpt_path)
        self.data, self.meta = load_dataset(self.data_path)
        cfg = self.wyp_model_bundle["cfg"]
        self.input_slices = build_input_slices(
            self.data,
            self.wyp_model_bundle["inputs_arg"],
            b_seq_encoding=getattr(cfg, "b_seq_encoding", "scalar"),
            b_seq_num_classes=int(getattr(cfg, "b_seq_num_classes", 11)),
        )
        self.dt_sec = float(self.meta["dt_sec"])
        self.reasoning_sampler = ReasoningSampler(
            adapter_dir=self.adapter_dir,
            base_model=self.base_model_name,
            max_phase=int(cfg.max_phase),
        )

    def _to_np(self, x: Any, key: str) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if key == "x0" and arr.shape != (6,):
            raise ValueError(f"inputs['x0'] must have shape (6,), got {arr.shape}")
        if key in ("koz_param", "koz_dim") and arr.shape != (3,):
            raise ValueError(f"inputs['{key}'] must have shape (3,), got {arr.shape}")
        if key == "artms_scaling_1e3" and arr.shape != (6,):
            raise ValueError(f"inputs['artms_scaling_1e3'] must have shape (6,), got {arr.shape}")
        if key == "oec0_modified" and arr.shape != (10,):
            raise ValueError(f"inputs['oec0_modified'] must have shape (10,), got {arr.shape}")
        return arr

    def intent_to_traj(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Validate and normalize inputs
        if "x0" not in inputs:
            raise ValueError("inputs must contain key 'x0'.")
        if "intent_priority" not in inputs:
            raise ValueError("inputs must contain key 'intent_priority'.")

        x0 = self._to_np(inputs["x0"], "x0")
        koz_key = "koz_param" if "koz_param" in inputs else "koz_dim"
        if koz_key not in inputs:
            raise ValueError("inputs must contain key 'koz_param' or 'koz_dim'.")
        koz_dim = self._to_np(inputs[koz_key], koz_key)
        if "artms_scaling_1e3" not in inputs:
            raise ValueError("inputs must contain key 'artms_scaling_1e3'.")
        artms = self._to_np(inputs["artms_scaling_1e3"], "artms_scaling_1e3")
        if "oec0_modified" not in inputs:
            raise ValueError("inputs must contain key 'oec0_modified'.")
        oec0_mod = self._to_np(inputs["oec0_modified"], "oec0_modified")

        intent_priority = list(inputs["intent_priority"])
        if len(intent_priority) == 0:
            raise ValueError("inputs['intent_priority'] must be non-empty.")

        # 1) intent -> (reasoning, tf, b_seq) via LoRA-augmented reasoning model
        print(f"[RAGES] Generating reasoning and trajectory decisions from intent...")
        llm_out = self.reasoning_sampler.sample_inputs(
            {
                "x0": np.round(x0, 2).tolist(),
                "oec0_modified": np.round(oec0_mod, 6).tolist(),
                "koz_param": np.round(koz_dim, 4).tolist(),
                "artms_scaling_1e3": np.round(artms, 3).tolist(),
                "intent_priority": intent_priority,
            }
        )

        # 2) behavior sequence + tf -> stochastic waypoint prediction
        print(f"[RAGES] Reasoning output parsed. tf={llm_out['tf']} b_seq={llm_out['b_seq']}")
        x_pred, dt_pred = predict_wyp_seq(
            model_bundle=self.wyp_model_bundle,
            input_slices=self.input_slices,
            x0=x0,
            tof_steps=int(llm_out["tf"]),
            b_seq=llm_out["b_seq"],
            oec0_mod=oec0_mod,
            artms=artms,
            koz_dim=koz_dim,
        )

        # 3) waypoint constraints -> SCP trajectory (chance=True, ct=False in implementation)
        print(f"[RAGES] Waypoints are generated. SCP is being solved...")
        solved = generate_traj_with_wyp(
            x0=x0,
            x_pred=x_pred,
            dt_pred=dt_pred,
            tof_steps=int(llm_out["tf"]),
            koz_dim=koz_dim,
            artms=artms,
            dt_sec=self.dt_sec,
            oec0_mod=oec0_mod,
        )        

        return {
            "input": {
                "x0": x0,
                "koz_param": koz_dim,
                "artms_scaling_1e3": artms,
                "intent_priority": intent_priority,
                "oec0_modified": oec0_mod,
            },
            "reasoning_output": {
                "reasoning": llm_out["reasoning"],
                "tf": int(llm_out["tf"]),
                "b_seq": llm_out["b_seq"],
                "raw_text": llm_out["raw_text"],
            },
            "status": {
                "cvx": solved["status_cvx"],
                "scp": solved["status_scp"],
            },
            "trajectory": {
                "wyp": solved["wyp"],
                "t_idx_wyp": solved["t_idx_wyp"],
                "roe": solved.get("roe_scp", solved.get("roe_cvx", None)),
                "rtn": solved.get("rtn_scp", solved.get("rtn_cvx", None)),
                "rtn_ct": solved.get("rtn_scp_ct", solved.get("rtn_cvx_ct", None)),
                "actions": solved.get("actions_scp", solved.get("actions_cvx", None)),
            },
        }


def _jsonable(x: Any) -> Any:
    return to_jsonable(x)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGES end-to-end smoke test.")
    parser.add_argument("--idx", type=int, default=10, help="Dataset row index for smoke input.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="rpod/rages/out/rages_smoke_output.json",
        help="Where to save full smoke output JSON. Set empty string to disable saving.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    save_path = args.save_path.strip() if args.save_path is not None else ""    
    
    # Run the smoke test
    engine = RAGES()

    # Build one deterministic smoke-test input from the existing training dataset.
    n = int(engine.data["x0"].shape[0])
    if n <= 0:
        raise ValueError("Dataset is empty; cannot build smoke-test input.")
    i = int(args.idx) % n
    inputs = {
        "x0": engine.data["x0"][i].cpu().numpy(),
        "koz_param": engine.data["koz_dim"][i].cpu().numpy(),
        "artms_scaling_1e3": engine.data["artms_scale_range_1e3"][i].cpu().numpy(),
        "intent_priority": ["fuel", "time", "observation", "safety_margin"],
        "oec0_modified": engine.data["oec0_modified"][i].cpu().numpy(),
    }

    # main run (Rasoning -> Waypoint Prediction -> SCP)
    out = engine.intent_to_traj(inputs)

    # unpack the output 
    status_cvx = out["status"]["cvx"]
    status_scp = out["status"]["scp"]
    tf = out["reasoning_output"]["tf"]
    b_seq = out["reasoning_output"]["b_seq"]
    print(f"[smoke] status_cvx={status_cvx} status_scp={status_scp}")
    print(f"[smoke] tf={tf} num_phase={len(b_seq)} b_seq={b_seq}")
    print(f"[smoke] reasoning={out['reasoning_output']['reasoning']}")

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(_jsonable(out), f, indent=2, ensure_ascii=False)
        print(f"[smoke] saved full output to: {p}")
