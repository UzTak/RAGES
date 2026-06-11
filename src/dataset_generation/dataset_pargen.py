import os, sys
from pathlib import Path
def find_root_path(path:str, word:str):
    parts = path.split(word, 1)
    return parts[0] + word if len(parts) > 1 else path 
root_folder = Path(__file__).resolve().parents[2]

import numpy as np
import torch 
from multiprocessing import Pool, get_context
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()   

from optimization.parameters import *
from dynamics.dynamics_trans import roe_to_rtn
from optimization.scvx import solve_scvx
from optimization.optimization import NonConvexOCP
from dataset_generation.annotation import annotate_number2
# HACK 
root_folder = str(root_folder)

def as_time_series_column(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr.T
        if arr.shape[1] == 1:
            return arr
    if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 1:
        return arr[0]
    raise ValueError(f"Unsupported time-series shape: {arr.shape}")

def for_computation(current_data_index):
    rng = np.random.default_rng(int(current_data_index))

    (n_time_i, tf_i_sec, tvec_sec_i, oec0_i, 
     roe0, roef, artms_param_1e3, koz_dim_i) = sample_problem_context(rng=rng)
    behavior_mode = -1
    t_idx_wyp = []
    wyp = []
    waypoint_times = np.array([0, n_time_i - 1], dtype=int)

    # Output dictionary initialization
    out = {'feasible' : True,
           'scp_feasible' : False,
           'states_roe_cvx' : [],
           'states_rtn_cvx' : [],
           'actions_cvx' : [],
           'states_roe_scp': [],
           'states_rtn_scp' : [],
           'actions_scp' : [],
           'target_state' : [],
           'horizons' : [],
           'dtime' : [],
           'time' : [],
           'oe' : [],
           'traj_n_time' : n_time_i,
           'tf_sec' : tf_i_sec,
           'rtgs_cvx' : [], 
           'rtgs_scp' : [],
           'ctgs_cvx' : [],
           'ctgs_scp' : [], 
           'behavior' : behavior_mode,
           'waypoints' : np.vstack((roe0, roef)),
           'waypoint_times' : waypoint_times,
           'artms_param_1e3' : artms_param_1e3,
           'koz_dim' : koz_dim_i,
           }   

    # Define current observation
    current_obs = {'state' : roe0, 'goal' : roef, 'ttg' : tf_i_sec, 'dt' : dt_sec, 'oe' : oec0_i}
    
    chance = False
    ct = True 

    prob = NonConvexOCP(
        prob_definition={
            't_i' : 0,
            't_f' : n_time_i,
            'tvec_sec' : tvec_sec_i,
            'chance' : chance,
            'current_obs' : current_obs,
            'waypoint_times' : t_idx_wyp,
            'waypoints' : wyp,
            'artms_param_1e3' : artms_param_1e3,
            'koz_dim' : koz_dim_i,
            'ct' : ct,
        }
    )

    sol_cvx = prob.ocp_cvx()

    states_roe_cvx_i, actions_cvx_i = sol_cvx['z']['state'], sol_cvx['z']['action']
    feas_cvx_i = sol_cvx['status']

    if feas_cvx_i in ['optimal', 'optimal_inaccurate']:
        
        # Mapping done after the feasibility check to avoid NoneType errors
        # roe_to_rtn expects (6, n_times) format, transpose if needed and use full OE history
        oe = propagate_oe(prob.oe_i, prob.tvec_sec)  # (n_time_i, 6)
        states_rtn_cvx_i = roe_to_rtn(states_roe_cvx_i, oe)  # Transpose back to (n_time, 6)

        ext_len = n_time_max - n_time_i
        time_full = np.arange(n_time_max, dtype=float) * dt_sec
        oe_full = propagate_oe(oec0_i, time_full)

        states_roe_cvx = np.vstack((states_roe_cvx_i, np.repeat(states_roe_cvx_i[-1][None, :], ext_len, axis=0)))
        actions_cvx = np.vstack((actions_cvx_i, np.zeros((ext_len, 3), dtype=float)))
        states_rtn_cvx = roe_to_rtn(states_roe_cvx, oe_full)

        prob_pad = NonConvexOCP(
            prob_definition={
                't_i' : 0,
                't_f' : n_time_max,
                'tvec_sec' : time_full,
                'chance' : chance,
                'current_obs' : {'state' : roe0, 'goal' : roef, 'ttg' : time_full[-1], 'dt' : dt_sec, 'oe' : oec0_i},
                'waypoint_times' : t_idx_wyp,
                'waypoints' : wyp,
                'artms_param_1e3' : artms_param_1e3,
                'koz_dim' : koz_dim_i,
                'ct' : ct,
            }
        )

        rtgs_cvx = prob_pad.compute_rtg(actions_cvx)
        ctgs_cvx, _ = prob_pad.compute_ctg(states_roe_cvx, actions_cvx, time_full, chance=chance, ct=ct, n_time=n_time_max)

        out['states_roe_cvx'] = states_roe_cvx
        out['states_rtn_cvx'] = states_rtn_cvx
        out['actions_cvx'] = actions_cvx
        out['target_state'] = roef
        out['horizons'] = prob.horizon
        out['dtime'] = dt_sec
        out['time'] = time_full
        out['oe'] = oe_full
        out['rtgs_cvx'] = rtgs_cvx
        out['ctgs_cvx'] = ctgs_cvx
        
        #  Solve transfer scp
        prob.zref = {'state': states_roe_cvx_i, 'action': actions_cvx_i}
        prob.sol_0 = {"z": prob.zref}
        prob.generate_scaling(states_roe_cvx_i, actions_cvx_i)
        sol_scp, log_scp = solve_scvx(prob)
        feas_scp_i = sol_scp['status']

        if feas_scp_i in ['optimal', 'optimal_inaccurate']:
            # Mapping done after feasibility check to avoid NoneType errors
            # roe_to_rtn expects (6, n_times) format, transpose if needed and use full OE history
            states_roe_scp_i = sol_scp['z']['state']
            actions_scp_i = sol_scp['z']['action']
            states_roe_scp = np.vstack((states_roe_scp_i, np.repeat(states_roe_scp_i[-1][None, :], ext_len, axis=0)))
            actions_scp = np.vstack((actions_scp_i, np.zeros((ext_len, 3), dtype=float)))
            states_rtn_scp = roe_to_rtn(states_roe_scp, oe_full)
            rtgs_scp = prob_pad.compute_rtg(actions_scp)
            ctgs_scp, _ = prob_pad.compute_ctg(states_roe_scp, actions_scp, time_full, chance=chance, ct=ct, n_time=n_time_max)

            out['states_roe_scp'] = states_roe_scp
            out['states_rtn_scp'] = states_rtn_scp
            out['actions_scp']    = actions_scp
            out['rtgs_scp'] = rtgs_scp
            out['ctgs_scp'] = ctgs_scp
            out['scp_feasible'] = True
        else:
            out['feasible'] = False

    else:
        out['feasible'] = False
    
    return out

if __name__ == '__main__':

    N_data = 100000
    N_proc = 25
    ver_name = 'art_02'
    
    n_S = 6 # state size
    n_A = 3 # action size

    dataset_dir = root_folder + '/rpod/dataset/torch/' + ver_name   
    os.makedirs(dataset_dir, exist_ok=True)   # ensures all subfolders exist
    dataset_path = Path(dataset_dir) / "dataset-rpod-param.npz"
    assert not dataset_path.exists() or ver_name == 'test', f"Error: Seems like dataset already exists in {dataset_dir}. Please remove it manually if you want to regenerate it with this name."

    states_roe_cvx = np.empty(shape=(N_data, n_time_max, n_S), dtype=float) # [m]
    states_rtn_cvx = np.empty(shape=(N_data, n_time_max, n_S), dtype=float) # [m,m,m,m/s,m/s,m/s]
    actions_cvx = np.empty(shape=(N_data, n_time_max, n_A), dtype=float) # [m/s]
    rtgs_cvx = np.empty(shape=(N_data, n_time_max, 1), dtype=float) 
    ctgs_cvx = np.empty(shape=(N_data, n_time_max, 1), dtype=float) 
    rtgs_cvx_unfiltered = np.full(shape=(N_data, n_time_max, 1), fill_value=np.nan, dtype=float)
    ctgs_cvx_unfiltered = np.full(shape=(N_data, n_time_max, 1), fill_value=np.nan, dtype=float)

    states_roe_scp = np.empty(shape=(N_data, n_time_max, n_S), dtype=float) # [m]
    states_rtn_scp = np.empty(shape=(N_data, n_time_max, n_S), dtype=float) # [m,m,m,m/s,m/s,m/s]
    actions_scp = np.empty(shape=(N_data, n_time_max, n_A), dtype=float) # [m/s]
    rtgs_scp = np.empty(shape=(N_data, n_time_max, 1), dtype=float)
    ctgs_scp = np.empty(shape=(N_data, n_time_max, 1), dtype=float) 
    ctg_cvx_full = np.empty(shape=(N_data, ), dtype=float)   # final ctg value for each data point (including infeasible ones) 

    target_state = np.empty(shape=(N_data, n_S), dtype=float)
    horizons = np.empty(shape=(N_data, ), dtype=float)
    dtime = np.empty(shape=(N_data, ), dtype=float)
    time = np.empty(shape=(N_data, n_time_max), dtype=float)
    oe = np.empty(shape=(N_data, n_time_max, n_S), dtype=float)
    traj_n_time = np.empty(shape=(N_data, ), dtype=int)
    tf_sec_arr = np.empty(shape=(N_data, ), dtype=float)
    artms_param_arr = np.empty(shape=(N_data, ), dtype=float)
    koz_dim_arr = np.empty(shape=(N_data, 3), dtype=float)
    
    behavior_mode = np.empty(shape=(N_data, ), dtype=int)
    
    wyp = np.full((N_data, 5, n_S), np.nan, dtype=float)
    t_idx_wyp = np.full((N_data, 5), -1, dtype=int)
    scp_feasible_full = []
    behavior_mode_full = []
    rtgs_cvx_unfiltered = []
    ctgs_cvx_unfiltered = []
    ctg_cvx_full = []

    i_unfeas = []
    n_success = 0
    n_attempts = 0
    next_seed = 0

    # Pool creation --> keep sampling until N_data SCP-feasible trajectories are collected
    ctx = get_context("spawn")  # Windows-safe
    with ctx.Pool(processes=N_proc) as p:  # avoiding the pool shutdown issue on Windows
        with tqdm(total=N_data, desc='Collected feasible SCP trajectories') as pbar:
            while n_success < N_data:
                n_remaining = N_data - n_success
                batch_indices = np.arange(next_seed, next_seed + n_remaining)
                next_seed += n_remaining

                for res in p.imap(for_computation, batch_indices):
                    behavior_mode_full.append(res['behavior'])
                    scp_feasible_full.append(bool(res['scp_feasible']))

                    if len(res['rtgs_cvx']) > 0:
                        rtgs_cvx_arr = as_time_series_column(res['rtgs_cvx'])
                        rtgs_cvx_unfiltered.append(rtgs_cvx_arr)
                    else:
                        rtgs_cvx_unfiltered.append(np.full((n_time_max, 1), fill_value=np.nan, dtype=float))

                    if len(res['ctgs_cvx']) > 0:
                        ctgs_cvx_arr = as_time_series_column(res['ctgs_cvx'])
                        ctgs_cvx_unfiltered.append(ctgs_cvx_arr)
                        ctg_cvx_full.append(float(np.nanmean(ctgs_cvx_arr[:, 0])))
                    else:
                        ctgs_cvx_unfiltered.append(np.full((n_time_max, 1), fill_value=np.nan, dtype=float))
                        ctg_cvx_full.append(np.nan)

                    if res['feasible']:
                        i = n_success
                        behavior_mode[i] = res['behavior']

                        states_roe_cvx[i,:,:] = res['states_roe_cvx']
                        states_rtn_cvx[i,:,:] = res['states_rtn_cvx']
                        actions_cvx[i,:,:] = res['actions_cvx']

                        states_roe_scp[i,:,:] = res['states_roe_scp']
                        states_rtn_scp[i,:,:] = res['states_rtn_scp']
                        actions_scp[i,:,:] = res['actions_scp']

                        target_state[i,:] = res['target_state']
                        horizons[i] = res['horizons']
                        dtime[i] = res['dtime']
                        time[i,:] = res['time']
                        oe[i,:,:] = res['oe']
                        traj_n_time[i] = res['traj_n_time']
                        tf_sec_arr[i] = res['tf_sec']
                        artms_param_arr[i] = res['artms_param_1e3']
                        koz_dim_arr[i,:] = res['koz_dim']

                        rtgs_scp[i,:,:] = res['rtgs_scp'].T
                        ctgs_scp[i,:,:] = res['ctgs_scp'].T
                        
                        rtgs_cvx[i,:,:] = res['rtgs_cvx'].T
                        ctgs_cvx[i,:,:] = res['ctgs_cvx'].T
                        
                        n_wyp = res["waypoints"].shape[0]  # > 0 because start and goal are included
                        wyp[i, :n_wyp, :] = res["waypoints"]
                        t_idx_wyp[i, :n_wyp] = res["waypoint_times"]

                        n_success += 1
                        pbar.update(1)
                    else:
                        i_unfeas.append(n_attempts)

                    n_attempts += 1

    behavior_mode_full = np.asarray(behavior_mode_full, dtype=int)
    scp_feasible_full = np.asarray(scp_feasible_full, dtype=bool)
    rtgs_cvx_unfiltered = np.asarray(rtgs_cvx_unfiltered, dtype=float)
    ctgs_cvx_unfiltered = np.asarray(ctgs_cvx_unfiltered, dtype=float)
    ctg_cvx_full = np.asarray(ctg_cvx_full, dtype=float)

    # create random command ids for each behavior entry (integers in [0,100))
    command_id = np.random.randint(0, 100, size=behavior_mode.shape[0])
    np.savez_compressed(dataset_dir + '/dataset-rpod-param', target_state=target_state, time=time, oe=oe, dtime=dtime, horizons=horizons, 
                                                            behavior=behavior_mode, command_id=command_id, 
                                                            traj_n_time=traj_n_time, tf_sec=tf_sec_arr,
                                                            artms_param_1e3=artms_param_arr, koz_dim=koz_dim_arr,
                                                            i_unfeas=i_unfeas, behavior_full=behavior_mode_full, ctg_all=ctg_cvx_full,
                                                            scp_feasible_full=scp_feasible_full,
                                                            rtgs_cvx_unfiltered=rtgs_cvx_unfiltered,
                                                            ctgs_cvx_unfiltered=ctgs_cvx_unfiltered, 
                                                            waypoints=wyp, waypoint_times=t_idx_wyp,)

    # save torch file directly
    torch_states_roe_cvx = torch.from_numpy(states_roe_cvx)
    torch_states_rtn_cvx = torch.from_numpy(states_rtn_cvx)
    torch_actions_cvx = torch.from_numpy(actions_cvx)
    torch_states_roe_scp = torch.from_numpy(states_roe_scp)
    torch_states_rtn_scp = torch.from_numpy(states_rtn_scp)
    torch_actions_scp = torch.from_numpy(actions_scp)
    torch_behavior_mode = torch.from_numpy(behavior_mode)
    torch_command_id = torch.from_numpy(command_id)

    torch.save(torch_states_roe_cvx, dataset_dir + '/torch_states_roe_cvx.pth')
    torch.save(torch_states_rtn_cvx, dataset_dir + '/torch_states_rtn_cvx.pth')    
    torch.save(torch_states_roe_scp, dataset_dir + '/torch_states_roe_scp.pth')
    torch.save(torch_states_rtn_scp, dataset_dir + '/torch_states_rtn_scp.pth')
    torch.save(torch_actions_scp, dataset_dir + '/torch_actions_scp.pth')
    torch.save(torch_actions_cvx, dataset_dir + '/torch_actions_cvx.pth')
    torch.save(torch_behavior_mode, dataset_dir + '/torch_behavior_mode.pth')
    torch.save(torch_command_id, dataset_dir + '/torch_command_id.pth')

    torch_rtgs_cvx = torch.from_numpy(rtgs_cvx)
    torch_rtgs_scp = torch.from_numpy(rtgs_scp)
    torch_ctgs_cvx = torch.from_numpy(ctgs_cvx)
    torch_ctgs_scp = torch.from_numpy(ctgs_scp)

    torch.save(torch_rtgs_scp, dataset_dir + '/torch_rtgs_scp.pth')
    torch.save(torch_rtgs_cvx, dataset_dir + '/torch_rtgs_cvx.pth')
    torch.save(torch_ctgs_scp, dataset_dir + '/torch_ctgs_scp.pth')
    torch.save(torch_ctgs_cvx, dataset_dir + '/torch_ctgs_cvx.pth')

    # Permutation
    if states_rtn_cvx.shape[0] != states_rtn_scp.shape[0]:
        raise RuntimeError('Different dimensions of cvx and scp datasets.')
    perm = np.random.permutation(states_rtn_cvx.shape[0]*2)
    np.save(dataset_dir + '/permutation.npy', perm)

    print(f'dataset generation completed successfully after {n_attempts} attempts ({len(i_unfeas)} rejected).')
