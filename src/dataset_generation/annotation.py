"""
Individual dataset annotation pipeline.
Load the file first, and then call the annotation functions.
"""

import os, sys
from pathlib import Path
import numpy as np 
import torch
import json 
from typing import Dict, Any, List
import re 

def find_root_path(path:str, word:str):
    parts = path.split(word, 1)    
    return parts[0] + word if len(parts) > 1 else path 
root_folder = str(Path(__file__).resolve().parents[2])

from optimization.parameters import dim_koz, tvec_sec, period, dt_sec, ALLOWED_PLACEHOLDERS

def annotate(module_path: str, master_fpath: str, save_file=True):
    """
    associate random text commands to each data point in the dataset. 
    Used for the preliminary annotation work (trajectory-agnostic command).
    """
    data_param = np.load(module_path + '/dataset-rpod-param.npz', allow_pickle=True)
    behav = data_param['behavior']
    command_id = data_param['command_id']
    n_data = behav.shape[0]
    
    # load command file 
    with open(master_fpath, 'r') as f:
        command_list = [json.loads(line) for line in f]
    
    text_filled_list = []
    
    for i in range(n_data):
        text_i = command_list[behav[i]]['description'][command_id[i]]
        text_filled_list.append(text_i)
    
    if save_file:
        dataset_fpath = module_path + '/annotation_texts.pth'
        torch.save(text_filled_list, dataset_fpath)
        print("done!")

    return text_filled_list


class SafeDict(dict):
    def __missing__(self, k):
        return "{" + k + "}"  # keep unknown placeholders as-is

def fill_partial(template: str, values: dict) -> str:
    # normalize {{var}} → {var}
    template = re.sub(r"\{\{(\w+)\}\}", r"{\1}", template)
    present = set(re.findall(r"\{(\w+)\}", template))
    sub = {k: values[k] for k in present & values.keys()}
    return template.format_map(SafeDict(sub))

def annotate_number(module_path: str, master_fpath: str, save_file=True):
    
    data_param = np.load(module_path + '/dataset-rpod-param.npz', allow_pickle=True)
    roe_cvx = torch.load(module_path + '/torch_states_roe_cvx.pth').numpy()

    behav = data_param['behavior']
    wyp = data_param["waypoints"]
    wyp_times = data_param["waypoint_times"]    
    n_data = behav.shape[0]
    state_f = data_param["target_state"]
    
    # load command file 
    with open(master_fpath, 'r') as f:
        command_list = [json.loads(line) for line in f]

    n_command = len(command_list[0]['templates'])
    decimal = 1
    
    text_filled_list = []
    
    print("total data size:", n_data)

    for i in range(n_data):    
        
        behav_i = behav[i]
        command_dict_i = command_list[behav_i]
        text_unfilled_i = command_dict_i['templates'][np.random.randint(0, n_command)]
        
        wyp_i = wyp[i]
        wyp_times_i = wyp_times[i]
        
        xf_i = state_f[i]
        x0_i = roe_cvx[i,0]

        if behav_i == 0:
            vals = {
                "T_appr_orbits": np.round((tvec_sec[wyp_times_i[0]] - tvec_sec[0]) / period, decimal),
            }
        elif behav_i == 1:
            vals = {
                "wyp_Tf": np.round(xf_i[1], decimal),   # delta-lambda
                "T_hold_orbits": np.round((tvec_sec[-1] - tvec_sec[wyp_times_i[0]]) / period, decimal),
            }
        elif behav_i == 2:
            vals = {
                "wyp_Ti": np.round(x0_i[1], decimal),   # delta-lambda
                "wyp_Tf": np.round(xf_i[1], decimal),   # delta-lambda
            }
        elif behav_i == 3:
            vals = {
                "T_EI_sep_orbits": np.round((tvec_sec[wyp_times_i[0]] - tvec_sec[0]) / period, decimal),
                "T_transfer_orbits": np.round((tvec_sec[wyp_times_i[1]] - tvec_sec[wyp_times_i[0]]) / period, decimal),
                "T_settle_orbits": np.round((tvec_sec[-1] - tvec_sec[wyp_times_i[1]]) / period, decimal),
            }
        elif behav_i == 4:
            vals = {
                "T_appr_orbits": np.round((tvec_sec[wyp_times_i[0]] - tvec_sec[0]) / period, decimal),
                "T_circ_orbits": np.round((tvec_sec[wyp_times_i[1]] - tvec_sec[wyp_times_i[0]]) / period, decimal),
                "T_evac_orbits": np.round((tvec_sec[-1] - tvec_sec[wyp_times_i[1]]) / period, decimal),
            }
        elif behav_i == 5:
            vals = {
                "T_appr_orbits": np.round((tvec_sec[wyp_times_i[0]] - tvec_sec[0]) / period, decimal),
                "T_circ_orbits": np.round((tvec_sec[wyp_times_i[1]] - tvec_sec[wyp_times_i[0]]) / period, decimal),
                "T_evac_orbits": np.round((tvec_sec[-1] - tvec_sec[wyp_times_i[1]]) / period, decimal),
            }
        else:
            raise ValueError("Unknown behavior id.")

        text_filled_i = fill_partial(text_unfilled_i, vals)
        text_filled_list.append(text_filled_i)

    if save_file:
        # save to dataset folder as .pth file
        dataset_fpath = module_path + '/annotation_texts.pth'
        torch.save(text_filled_list, dataset_fpath)
        print("done!")

    return text_filled_list


def annotate_number2old(data_param: Dict[str, Any], command_list: List[Dict[str, Any]], command_id=None, save_dir=False) -> List[str]:
    """
    Ad-hoc function to annotate dataset with numerical values filled in the text commands.
    WARNING: the convention of the waypoint times is different from annotate_number() 
    as this includes start and goal times.
    Used for the evaluation process. Will be merged to annotate_number() later. (2025/11/12)
    """

    behav = data_param['behavior']
    wyp = data_param["waypoints"]    # this includes start and goal states
    wyp_times = data_param["waypoint_times"]   # this includes start and goal times
    roe_cvx = torch.load(module_path + '/torch_states_roe_cvx.pth').numpy()
    state_f = data_param["target_state"]

    n_command = len(command_list[0]['templates'])
    decimal = 1
    n_data = behav.shape[0]
    
    text_filled_list = []
    
    if command_id is None:
        command_id = np.random.randint(0, n_command, size=n_data)

    for i in range(n_data):
                
        behav_i = behav[i]
        command_dict_i = command_list[behav_i]
        text_unfilled_i = command_dict_i['templates'][command_id[i]]
        
        wyp_times_i = wyp_times[i]
        
        xf_i = state_f[i]

        if behav_i == 0:
            # 1 waypoint + start / goal 
            vals = {
                "T_appr_orbits": np.round((tvec_sec[wyp_times_i[0]] - tvec_sec[0]) / period, decimal),
            }
        elif behav_i == 1:
            # 1 waypoint + start / goal
            vals = {
                "T_appr_orbits": np.round((tvec_sec[wyp_times_i[1]] - tvec_sec[wyp_times_i[0]]) / period, decimal),
                "d_lambda_meters": np.round(xf_i[1], decimal),   # delta-lambda
            }
        elif behav_i == 2:
            # 0 waypoint + start / goal
            vals = {
                "T_appr_orbits": np.round((tvec_sec[wyp_times_i[1]] - tvec_sec[wyp_times_i[0]]) / period, decimal),
                "d_lambda_meters": np.round(xf_i[1], decimal),   # delta-lambda
            }
        elif behav_i == 3:
            # 2 waypoints + start / goal
            vals = {
                "T_EI_sep_orbits": np.round((tvec_sec[wyp_times_i[0]] - tvec_sec[0]) / period, decimal),
                "T_transfer_orbits": np.round((tvec_sec[wyp_times_i[1]] - tvec_sec[wyp_times_i[0]]) / period, decimal),
                "T_settle_orbits": np.round((tvec_sec[-1] - tvec_sec[wyp_times_i[1]]) / period, decimal),
            }
        elif behav_i == 4:
            # 2 waypoints + start / goal
            vals = {
                "T_appr_orbits": np.round((tvec_sec[wyp_times_i[0]] - tvec_sec[0]) / period, decimal),
                "T_circ_orbits": np.round((tvec_sec[wyp_times_i[1]] - tvec_sec[wyp_times_i[0]]) / period, decimal),
                "T_evac_orbits": np.round((tvec_sec[-1] - tvec_sec[wyp_times_i[1]]) / period, decimal),
            }
        elif behav_i == 5:
            # 2 waypoints + start / goal
            vals = {
                "T_appr_orbits": np.round((tvec_sec[wyp_times_i[0]] - tvec_sec[0]) / period, decimal),
                "T_circ_orbits": np.round((tvec_sec[wyp_times_i[1]] - tvec_sec[wyp_times_i[0]]) / period, decimal),
                "T_evac_orbits": np.round((tvec_sec[-1] - tvec_sec[wyp_times_i[1]]) / period, decimal),
            }
        else:
            raise ValueError("Unknown behavior id.")

        text_filled_i = fill_partial(text_unfilled_i, vals)
        text_filled_list.append(text_filled_i)
        
    if save_dir is not None:
        # save to dataset folder as .pth file
        dataset_fpath = save_dir + '/' + annotation_fname
        torch.save(text_filled_list, dataset_fpath)
        print("done!")

    return text_filled_list


def annotate_number2(data_param: Dict[str, Any], 
                     command_list: List[Dict[str, Any]], 
                     command_id=None, 
                     save_dir=False, 
                     annotation_fname='annotation_texts.pth') -> List[str]:
    """
    Ad-hoc function to annotate dataset with numerical values filled in the text commands.
    WARNING: the convention of the waypoint times is different from annotate_number() 
    as this includes start and goal times.
    Used for the evaluation process. Will be merged to annotate_number() later. (2025/11/12)
    """

    behav = data_param['behavior']
    wyp = data_param["waypoints"]    # this includes start and goal states
    wyp_times = data_param["waypoint_times"]   # this includes start and goal times
    
    n_command = len(command_list[0]['templates'])
    decimal = 1
    n_data = behav.shape[0]
    
    text_filled_list = []
    
    if command_id is None:
        command_id = np.random.randint(0, n_command, size=n_data)

    for i in range(n_data):
                
        behav_i = behav[i]
        command_dict_i = command_list[behav_i]
        command_id_i = command_id[i]
        
        if command_id[i] >= len(command_dict_i['templates']):
            command_id_i = np.mod(command_id[i], len(command_dict_i['templates']))
            
        text_unfilled_i = command_dict_i['templates'][command_id_i]
        
        wyp_i = wyp[i]
        wyp_times_i = wyp_times[i]

        if behav_i == 0:
            # 1 waypoint + start / goal 
            vals = {
                "T_appr_orbits": np.round((tvec_sec[wyp_times_i[1]]) / period, decimal),
            }
        elif behav_i == 1:
            # 1 waypoint + start / goal
            vals = {
                "T_appr_orbits": np.round((tvec_sec[wyp_times_i[1]]) / period, decimal),
                "d_lambda_meters": np.round(wyp_i[1,1], decimal),   # delta-lambda
            }
        elif behav_i == 2:
            # 0 waypoint + start / goal
            vals = {
               "T_appr_orbits": np.round((tvec_sec[wyp_times_i[1]]) / period, decimal),
               "d_lambda_meters": np.round(wyp_i[1,1], decimal),   # delta-lambda
            }
        elif behav_i == 3:
            # 2 waypoints + start / goal
            vals = {
                "T_EI_sep_orbits": np.round((tvec_sec[wyp_times_i[1]]) / period, decimal),
                "T_transfer_orbits": np.round((tvec_sec[wyp_times_i[2]]) / period, decimal),
            }
        elif behav_i == 4:
            # 2 waypoints + start / goal
            vals = {
                "T_appr_orbits": np.round((tvec_sec[wyp_times_i[1]]) / period, decimal),
                "T_circ_orbits": np.round((tvec_sec[wyp_times_i[2]]) / period, decimal),
            }
        elif behav_i == 5:
            # 2 waypoints + start / goal
            vals = {
                "T_appr_orbits": np.round((tvec_sec[wyp_times_i[1]]) / period, decimal),
                "T_circ_orbits": np.round((tvec_sec[wyp_times_i[2]]) / period, decimal),
            }
        else:
            raise ValueError("Unknown behavior id.")

        text_filled_i = fill_partial(text_unfilled_i, vals)
        text_filled_list.append(text_filled_i)
        
    if save_dir is not None:
        # save to dataset folder as .pth file
        dataset_fpath = save_dir + '/' + annotation_fname
        torch.save(text_filled_list, dataset_fpath)
        print("done!")

    return text_filled_list


if __name__ == "__main__":
    
    dataset_name = 'test'
    master_file = "w3"
    annotation_fname = 'annotation_texts_val.pth'
    
    module_path = root_folder + '/rpod/dataset/torch/' + dataset_name 
    # master_fpath = root_folder + '/rpod/dataset/commands_summary_' + master_file + '_train.jsonl'
    master_fpath   = root_folder + '/rpod/dataset/commands_summary_' + master_file + '_val.jsonl'
    
    data_param = np.load(module_path + '/dataset-rpod-param.npz', allow_pickle=True)
    command_id = data_param['command_id']
    with open(master_fpath, 'r') as f:
            command_list = [json.loads(line) for line in f]

    # _ = annotate(module_path, master_fpath_train, save_file=True)
    _ = annotate_number2(data_param, command_list, command_id=command_id, save_dir=module_path, annotation_fname=annotation_fname)
