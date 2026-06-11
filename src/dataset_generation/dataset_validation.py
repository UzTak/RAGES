import sys
import os
from pathlib import Path
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

def find_root_path(path:str, word:str):
    parts = path.split(word, 1)
    return parts[0] + word if len(parts) > 1 else path 
root_folder = Path(__file__).resolve().parents[2]

from optimization.parameters import n_time, dim_koz # , t_switch

def extract_data_ART(file_path):
    """Extract ROE trajectories, actions, and behavior labels from ART-generated data."""
    data = np.load(file_path, allow_pickle=True)
    
    roe_DT = data['roe_DT']  # (N_data, 6, n_time)
    a_DT = data['a_DT']  # (N_data, 3, n_time)
    behavior = data['behavior']  # (N_data,)
    state_init = data['state_init']  # (N_data, 6)
    state_final = data['state_final']  # (N_data, 6)
    rtn_DT = data.get('rtn_DT', None)  # (N_data, 6, n_time) or None
    
    return roe_DT, a_DT, behavior, state_init, state_final, rtn_DT


def extract_data_train(dataset_dir):
    """Extract ROE trajectories, actions, and behavior labels from training data."""
    import torch
    
    roe_scp = torch.load(f"{dataset_dir}/torch_states_roe_scp.pth").numpy()  # (N_data, n_time, 6)
    a_scp = torch.load(f"{dataset_dir}/torch_actions_scp.pth").numpy()  # (N_data, n_time, 3)
    behavior = torch.load(f"{dataset_dir}/torch_behavior_mode.pth").numpy()  # (N_data,)
    data_param = np.load(f"{dataset_dir}/dataset-rpod-param.npz")
    state_final = data_param['target_state']  # (N_data, 6)
    
    roe_DT = np.transpose(roe_scp, (0, 2, 1))  # (N_data, 6, n_time)
    a_DT = np.transpose(a_scp, (0, 2, 1))  # (N_data, 3, n_time)
    state_init = roe_DT[:, :, 0]  # (N_data, 6)
    
    rtn_scp = torch.load(f"{dataset_dir}/torch_states_rtn_scp.pth").numpy()  # (N_data, n_time, 6)
    rtn_DT = np.transpose(rtn_scp, (0, 2, 1))  # (N_data, 6, n_time)
    
    return roe_DT, a_DT, behavior, state_init, state_final, rtn_DT


def check_roe_in_range(roe, roe_base, noise_ranges):
    """Check if ROE state falls within expected noise ranges."""
    for i, (base_val, noise_range) in enumerate(zip(roe_base, noise_ranges)):
        if noise_range is None:
            if not np.isclose(roe[i], base_val, atol=1e1): # 1e1 is technically a hyperparam to be tuned based on ART
                return False
        else:
            noise_range = [x * 1.0001 for x in noise_range] # 1.0001 is technically a hyperparam to be tuned based on ART
            min_val = base_val + noise_range[0]
            max_val = base_val + noise_range[1]
            if not (min_val <= roe[i] <= max_val):
                return False
    return True


def check_waypoint(roe_traj, t_range, wyp_base, noise_ranges):
    """Check if trajectory passes through waypoint in time range."""
    for t_idx in range(t_range[0], t_range[1] + 1):
        if check_roe_in_range(roe_traj[:, t_idx], wyp_base, noise_ranges):
            return True
    return False


# def classify_training_metric(roe_DT, a_DT, state_init):
#     """Classify behaviors using training data generation criteria."""
#     N_data = roe_DT.shape[0]
#     predicted = np.full(N_data, 7, dtype=int)  # 7 = unclassified
#     # cim_final = get_cim_final()
    
#     for i in range(N_data):
#         roe_traj = roe_DT[i]  # (6, n_time)
#         a_traj = a_DT[i]  # (3, n_time)
#         roe_0 = state_init[i]  # (6,)
        
#         # roe_final = roe_traj[:, -1] + cim_final @ a_traj[:, -1]
#         roe_final = state_final[i]
        
#         # Behavior 0: circumnavigate, no waypoints
#         roe_f_base = np.array([0, 0, 0, 32, 0, 32])
#         noise_ranges = [
#             None, None,
#             (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)
#         ]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             predicted[i] = 0
#             continue
        
#         # Behavior 1: fast circumnavigate, waypoint at ~70%
#         roe_f_base = np.array([0, 0, 0, 32, 0, 32])
#         noise_ranges = [
#             None, None,
#             (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)
#         ]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             wyp_base = roe_f_base.copy()
#             if check_waypoint(roe_traj, [30, 40], wyp_base, noise_ranges):
#                 predicted[i] = 1
#                 continue
        
#         # Behavior 2: hold at -35m, no waypoints
#         roe_f_base = np.array([0, -35, 0, 0, 0, 0])
#         noise_ranges = [
#             None, (-5.0, 5.0),
#             (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)
#         ]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             predicted[i] = 2
#             continue
        
#         # Behavior 3: fast hold at -35m, waypoint at ~70%
#         roe_f_base = np.array([0, -35, 0, 0, 0, 0])
#         noise_ranges = [
#             None, (-5.0, 5.0),
#             (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)
#         ]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             wyp_base = roe_f_base.copy()
#             if check_waypoint(roe_traj, [30, 40], wyp_base, noise_ranges):
#                 predicted[i] = 3
#                 continue
        
#         # Behavior 4: fuel-optimal flyby, no waypoints
#         roe_f_base = np.array([0, 120, 0, 5, 0, 5])
#         noise_ranges = [
#             None, (-20.0, 20.0),
#             (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)
#         ]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             predicted[i] = 4
#             continue
        
#         # Behavior 5: E/I separated flyby, two waypoints
#         roe_f_base = np.array([0, 120, 0, 10, 0, 10])
#         noise_ranges = [
#             None, (-20.0, 20.0),
#             (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)
#         ]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             wyp0_base = np.array([0, roe_0[1], 0, 25, 0, 25])
#             wyp1_base = np.array([0, roe_final[1], 0, 25, 0, 25])
#             wyp_noise = [None, None, None, (-2.0, 2.0), None, (-2.0, 2.0)]
            
#             if (check_waypoint(roe_traj, [3, 7], wyp0_base, wyp_noise) and
#                 check_waypoint(roe_traj, [43, 47], wyp1_base, wyp_noise)):
#                 predicted[i] = 5
#                 continue
        
#         # Behavior 6: recede with waypoints
#         roe_f_base = np.array([0, 120, 0, 10, 0, 10])
#         noise_ranges = [
#             None, (-20.0, 20.0),
#             (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)
#         ]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             roe_mid_base = np.array([0, 0, 0, 32, 0, 32])
#             wyp_noise = [None, None, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#             if (check_waypoint(roe_traj, [15, 25], roe_mid_base, wyp_noise) and
#                 check_waypoint(roe_traj, [30, 40], roe_mid_base, wyp_noise)):
#                 predicted[i] = 6
    
#     return predicted


# def classify_training_metric_repeat(roe_DT, a_DT, state_init, state_final):
#     """Classify behaviors allowing multiple matches per trajectory."""
#     N_data = roe_DT.shape[0]
#     predicted = []
#     # cim_final = get_cim_final()
    
#     for i in range(N_data):
#         roe_traj = roe_DT[i]
#         a_traj = a_DT[i]
#         roe_0 = state_init[i]
        
#         # roe_final = roe_traj[:, -1] + cim_final @ a_traj[:, -1]
#         roe_final = state_final[i]
#         matches = []
        
#         # Behavior 0: circumnavigate, no waypoints
#         roe_f_base = np.array([0, 0, 0, 32, 0, 32])
#         noise_ranges = [None, None, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             matches.append(0)
        
#         # Behavior 1: fast circumnavigate, waypoint at ~70%
#         roe_f_base = np.array([0, 0, 0, 32, 0, 32])
#         noise_ranges = [None, None, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             wyp_base = roe_f_base.copy()
#             if check_waypoint(roe_traj, [30, 40], wyp_base, noise_ranges):
#                 matches.append(1)
        
#         # Behavior 2: hold at -35m, no waypoints
#         roe_f_base = np.array([0, -35, 0, 0, 0, 0])
#         noise_ranges = [None, (-5.0, 5.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             matches.append(2)
        
#         # Behavior 3: fast hold at -35m, waypoint at ~70%
#         roe_f_base = np.array([0, -35, 0, 0, 0, 0])
#         noise_ranges = [None, (-5.0, 5.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             wyp_base = roe_f_base.copy()
#             if check_waypoint(roe_traj, [30, 40], wyp_base, noise_ranges):
#                 matches.append(3)
        
#         # Behavior 4: fuel-optimal flyby, no waypoints
#         roe_f_base = np.array([0, 120, 0, 5, 0, 5])
#         noise_ranges = [None, (-20.0, 20.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             matches.append(4)
        
#         # Behavior 5: E/I separated flyby, two waypoints
#         roe_f_base = np.array([0, 120, 0, 10, 0, 10])
#         noise_ranges = [None, (-20.0, 20.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             wyp0_base = np.array([0, roe_0[1], 0, 25, 0, 25])
#             wyp1_base = np.array([0, roe_final[1], 0, 25, 0, 25])
#             wyp_noise = [None, None, None, (-2.0, 2.0), None, (-2.0, 2.0)]
#             if (check_waypoint(roe_traj, [3, 7], wyp0_base, wyp_noise) and
#                 check_waypoint(roe_traj, [43, 47], wyp1_base, wyp_noise)):
#                 matches.append(5)
        
#         # Behavior 6: recede with waypoints
#         roe_f_base = np.array([0, 120, 0, 10, 0, 10])
#         noise_ranges = [None, (-20.0, 20.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             roe_mid_base = np.array([0, 0, 0, 32, 0, 32])
#             wyp_noise = [None, None, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#             if (check_waypoint(roe_traj, [15, 25], roe_mid_base, wyp_noise) and
#                 check_waypoint(roe_traj, [30, 40], roe_mid_base, wyp_noise)):
#                 matches.append(6)
        
#         predicted.append(matches if matches else [7])
    
#     return predicted


# def classify_training_speed_repeat(roe_DT, a_DT, state_init, state_final, rtn_DT, H01, H23, H45):
#     """Classify behaviors with speed differentiation using variance and distance metrics."""
#     N_data = roe_DT.shape[0]
#     predicted = []
#     # cim_final = get_cim_final()
    
#     for i in range(N_data):
#         roe_traj = roe_DT[i]
#         a_traj = a_DT[i]
#         rtn_traj = rtn_DT[i]
#         roe_0 = state_init[i]
        
#         # roe_final = roe_traj[:, -1] + cim_final @ a_traj[:, -1]
#         roe_final = state_final[i]
#         matches = []
        
#         # Behavior 0: circumnavigate, no waypoints
#         roe_f_base = np.array([0, 0, 0, 32, 0, 32])
#         noise_ranges = [None, None, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             matches.append(0)
        
#         # Behavior 1: fast circumnavigate, waypoint at ~70%
#         roe_f_base = np.array([0, 0, 0, 32, 0, 32])
#         noise_ranges = [None, None, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             wyp_base = roe_f_base.copy()
#             if check_waypoint(roe_traj, [30, 40], wyp_base, noise_ranges):
#                 matches.append(1)
        
#         # Behavior 2: hold at -35m, no waypoints
#         roe_f_base = np.array([0, -35, 0, 0, 0, 0])
#         noise_ranges = [None, (-5.0, 5.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             matches.append(2)
        
#         # Behavior 3: fast hold at -35m, waypoint at ~70%
#         roe_f_base = np.array([0, -35, 0, 0, 0, 0])
#         noise_ranges = [None, (-5.0, 5.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             wyp_base = roe_f_base.copy()
#             if check_waypoint(roe_traj, [30, 40], wyp_base, noise_ranges):
#                 matches.append(3)
        
#         # Behavior 4: fuel-optimal flyby, no waypoints
#         roe_f_base = np.array([0, 120, 0, 5, 0, 5])
#         noise_ranges = [None, (-20.0, 20.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             matches.append(4)
        
#         # Behavior 5: E/I separated flyby, two waypoints
#         roe_f_base = np.array([0, 120, 0, 10, 0, 10])
#         noise_ranges = [None, (-20.0, 20.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             wyp0_base = np.array([0, roe_0[1], 0, 25, 0, 25])
#             wyp1_base = np.array([0, roe_final[1], 0, 25, 0, 25])
#             wyp_noise = [None, None, None, (-2.0, 2.0), None, (-2.0, 2.0)]
#             if (check_waypoint(roe_traj, [3, 7], wyp0_base, wyp_noise) and
#                 check_waypoint(roe_traj, [43, 47], wyp1_base, wyp_noise)):
#                 matches.append(5)
        
#         # Behavior 6: recede with waypoints
#         roe_f_base = np.array([0, 120, 0, 10, 0, 10])
#         noise_ranges = [None, (-20.0, 20.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#         if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
#             roe_mid_base = np.array([0, 0, 0, 32, 0, 32])
#             wyp_noise = [None, None, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
#             if (check_waypoint(roe_traj, [15, 25], roe_mid_base, wyp_noise) and
#                 check_waypoint(roe_traj, [30, 40], roe_mid_base, wyp_noise)):
#                 matches.append(6)
        
#         # Apply further differentiating constraints
#         # if 0 in matches and 1 in matches:
#         #     var_total = np.var(roe_traj[1, :])
#         #     var_late = np.var(roe_traj[1, 35:50])
#         #     ratio = var_late / var_total if var_total > 0 else 0
#         #     matches.remove(0 if ratio > H01 else 1)
        
#         # if 2 in matches and 3 in matches:
#         #     var_total = np.var(roe_traj[1, :])
#         #     var_late = np.var(roe_traj[1, 35:50])
#         #     ratio = var_late / var_total if var_total > 0 else 0
#         #     matches.remove(2 if ratio > H23 else 3)
        
#         if 4 in matches and 5 in matches:
#             # dist_nr = np.sqrt(rtn_traj[0, :]**2 + rtn_traj[2, :]**2)
#             # avg_dist = np.mean(dist_nr)
#             dist_r = rtn_traj[0, :]
#             avg_dist = np.mean(dist_r)
#             matches.remove(5 if avg_dist > H45 else 4)
        
#         predicted.append(matches if matches else [7])
    
#     return predicted


def accuracy_training_metric(roe_DT, a_DT, state_init, state_final, rtn_DT, behavior_true, H45, output_csv_path):
    """Check if trajectories satisfy constraints for their given behaviors."""
    N_data = roe_DT.shape[0]
    is_correct = np.zeros(N_data, dtype=bool)
    
    for i in range(N_data):
        roe_traj = roe_DT[i]
        a_traj = a_DT[i]
        rtn_traj = rtn_DT[i]
        roe_0 = state_init[i]
        behavior = behavior_true[i]
        
        roe_final = state_final[i]
        
        if behavior == 0:
            roe_f_base = np.array([0, 0, 0, 32, 0, 32])
            noise_ranges = [None, None, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
            is_correct[i] = check_roe_in_range(roe_final, roe_f_base, noise_ranges)
        
        elif behavior == 1:
            roe_f_base = np.array([0, 0, 0, 32, 0, 32])
            noise_ranges = [None, None, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
            if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
                wyp_base = roe_f_base.copy()
                t_wyp = int(0.7 * n_time)
                is_correct[i] = check_waypoint(roe_traj, [t_wyp - 5, t_wyp + 5], wyp_base, noise_ranges)
        
        elif behavior == 2:
            roe_f_base = np.array([0, -35, 0, 0, 0, 0])
            noise_ranges = [None, (-5.0, 5.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
            is_correct[i] = check_roe_in_range(roe_final, roe_f_base, noise_ranges)
        
        elif behavior == 3:
            roe_f_base = np.array([0, -35, 0, 0, 0, 0])
            noise_ranges = [None, (-5.0, 5.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
            if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
                wyp_base = roe_f_base.copy()
                t_wyp = int(0.7 * n_time)
                is_correct[i] = check_waypoint(roe_traj, [t_wyp - 5, t_wyp + 5], wyp_base, noise_ranges)
        
        elif behavior == 4:
            roe_f_base = np.array([0, 120, 0, 5, 0, 5])
            noise_ranges = [None, (-20.0, 20.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
            if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
                max_r = np.max(np.abs(rtn_traj[0, :]))
                is_correct[i] = max_r <= H45
        
        elif behavior == 5:
            roe_f_base = np.array([0, 120, 0, 10, 0, 10])
            noise_ranges = [None, (-20.0, 20.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
            if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
                wyp0_base = np.array([0, roe_0[1], 0, 25, 0, 25])
                wyp1_base = np.array([0, roe_final[1], 0, 25, 0, 25])
                wyp_noise = [None, None, None, (-2.0, 2.0), None, (-2.0, 2.0)]
                t_wyp0 = int(0.1 * n_time)
                t_wyp1 = int(0.9 * n_time)
                if (check_waypoint(roe_traj, [t_wyp0 - 2, t_wyp0 + 2], wyp0_base, wyp_noise) and
                    check_waypoint(roe_traj, [t_wyp1 - 2, t_wyp1 + 2], wyp1_base, wyp_noise)):
                    max_r = np.max(np.abs(rtn_traj[0, :]))
                    is_correct[i] = max_r > H45
        
        elif behavior == 6:
            roe_f_base = np.array([0, 120, 0, 10, 0, 10])
            noise_ranges = [None, (-20.0, 20.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
            if check_roe_in_range(roe_final, roe_f_base, noise_ranges):
                roe_mid_base = np.array([0, 0, 0, 32, 0, 32])
                wyp_noise = [None, None, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
                t_wyp0 = int(0.4 * n_time)
                t_wyp1 = int(0.7 * n_time)
                is_correct[i] = (check_waypoint(roe_traj, [t_wyp0 - 5, t_wyp0 + 5], roe_mid_base, wyp_noise) and
                                 check_waypoint(roe_traj, [t_wyp1 - 5, t_wyp1 + 5], roe_mid_base, wyp_noise))
    
    stats = []
    for b in range(7):
        mask = behavior_true == b
        total = np.sum(mask)
        successes = np.sum(is_correct[mask])
        success_rate = successes / total if total > 0 else 0.0
        stats.append([b, total, successes, success_rate])
    
    header = ['Behavior', 'Total', 'Successes', 'Success-Rate']
    np.savetxt(output_csv_path, stats, delimiter=',', fmt=['%d', '%d', '%d', '%.4f'], 
               header=','.join(header), comments='')
    
    return is_correct


# def plot_confusion_matrix(y_true, y_pred, output_path):
#     """Plot and save confusion matrix. Handles both single and multiple predictions."""
#     cm = np.zeros((8, 8), dtype=int)
#     for true, pred in zip(y_true, y_pred):
#         if isinstance(pred, (list, np.ndarray)):
#             for p in pred:
#                 cm[int(true), int(p)] += 1
#         else:
#             cm[int(true), int(pred)] += 1
    
#     labels = [f"B{i}" for i in range(7)] + ["Unclassified"]
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=labels, yticklabels=labels,
#                 cbar_kws={'label': 'Count'})
#     plt.xlabel('Predicted Behavior', fontsize=12)
#     plt.ylabel('True Behavior', fontsize=12)
#     plt.title('Behavior Classification Confusion Matrix', fontsize=14)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()

# For classification task, use this main function
# if __name__ == '__main__':
    
#     # Training data
#     # input_path = root_folder / 'rpod/dataset_generation/training_data'
#     input_path = root_folder / 'rpod/dataset/torch/test'
#     roe_DT, a_DT, behavior_true, state_init, state_final, rtn_DT = extract_data_train(input_path)

#     # ART data
#     # input_path = root_folder / 'rpod/optimization/saved_files/warmstarting/ws_analysis_test.npz'
#     # input_path = root_folder / 'rpod/optimization/saved_files/warmstarting/ws_analysis_v02_ct_v4.npz'
#     # roe_DT, a_DT, behavior_true, state_init, state_final, rtn_DT = extract_data_ART(input_path)

#     # output_path = root_folder / 'rpod/dataset_generation/CM_TSR_same.png'

#     # Classify
#     # classification_method = 'TRAINING-SPEED-REPEAT'
#     # H01, H23, H45 = 0.0000005, 0.0000005, 15.0

#     # Accuracy threshold
#     H45 = 30.0
    
#     # if classification_method == 'TRAINING-METRIC':
#     #     behavior_pred = classify_training_metric(roe_DT, a_DT, state_init, state_final)
#     # elif classification_method == 'TRAINING-METRIC-REPEAT':
#     #     behavior_pred = classify_training_metric_repeat(roe_DT, a_DT, state_init, state_final)
#     # elif classification_method == 'TRAINING-SPEED-REPEAT':
#     #     behavior_pred = classify_training_speed_repeat(roe_DT, a_DT, state_init, state_final, rtn_DT, H01, H23, H45)
#     # else:
#     #     raise ValueError(f"Unknown classification method: {classification_method}")
    
#     # # Plot confusion matrix
#     # plot_confusion_matrix(behavior_true, behavior_pred, output_path)
    
#     # print(f"Confusion matrix saved to {output_path}")
    
#     # Check accuracy
#     accuracy_csv_path = root_folder / 'rpod/dataset_generation/accuracy_stats.csv'
#     is_correct = accuracy_training_metric(roe_DT, a_DT, state_init, state_final, rtn_DT, behavior_true, H45, accuracy_csv_path)
#     print(f"Accuracy statistics saved to {accuracy_csv_path}")

if __name__ == '__main__':
    
    # Training data
    input_path = root_folder / 'rpod/dataset/torch/test'
    roe_DT, a_DT, behavior_true, state_init, state_final, rtn_DT = extract_data_train(input_path)

    # ART data
    # input_path = root_folder / 'rpod/optimization/saved_files/warmstarting/ws_analysis_v02_ct_v4.npz'
    # roe_DT, a_DT, behavior_true, state_init, state_final, rtn_DT = extract_data_ART(input_path)

    # Accuracy threshold for behavior 4 and 5
    H45 = dim_koz[0][0] + 5.0
    
    # Check accuracy
    accuracy_csv_path = root_folder / 'rpod/dataset_generation/accuracy_stats.csv'
    is_correct = accuracy_training_metric(roe_DT, a_DT, state_init, state_final, rtn_DT, behavior_true, H45, accuracy_csv_path)
    print(f"Accuracy statistics saved to {accuracy_csv_path}")
