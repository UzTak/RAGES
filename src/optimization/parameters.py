################ Low Earth Orbit RPO Design Reference Mission Parameters
import sys
import numpy as np
import scipy.stats as stats

from pathlib import Path
root_folder = Path(__file__).resolve().parent.parent.parent  # /art_lang/

from optimization.scvx import SCVxParams 
from dynamics.dynamics_trans import generate_koz, mu_E, propagate_oe

##########################################################################################
################################### PARAMETERS ###########################################
##########################################################################################

# Problem dimensions #####################################################################
N_STATE = 6
N_ACTION = 3

########## HELPER FUNCTIONS ##############

def sample_problem_context(rng=None, n_time_choices=None):
    """
    Sample per-trajectory mission context while keeping the dataset-wide dt fixed.
    Returns a local horizon, time grid, chief orbit at t0, and uniformly sampled
    boundary conditions for a two-point ROE transfer, plus case-specific
    uncertainty and KOZ context.
    Used for the generalized ART construction (for RAGES, not SAGES). 
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_time_choices is None:
        n_time_choices = np.arange(5, 50, 5, dtype=int)
        
    n_time_i = int(n_time_choices[rng.integers(0, len(n_time_choices))])
    tf_i_sec = dt_sec * (n_time_i - 1)
    tvec_sec_i = np.arange(n_time_i, dtype=float) * dt_sec

    oec0_i = oec0.copy()

    sma_vec = oec0[0] + np.array([-100.0, 0.0, 100, 200, 300, 400, 500], dtype=float)
    ecc_vec = np.array([2.0e-4, 4.0e-4, oec0[1], 8.0e-4, 1.2e-3], dtype=float)
    inc_vec = np.linspace(1e-4, np.pi, 15, endpoint=False)
    raan_vec = np.linspace(1e-4, 2.0 * np.pi, 15, endpoint=False)
    omega_vec = np.linspace(1e-4, 2.0 * np.pi, 15, endpoint=False)
    ma_vec = np.linspace(0.0, 2.0 * np.pi, 15, endpoint=False)

    oec0_i[0] = sma_vec[rng.integers(0, len(sma_vec))]
    oec0_i[1] = ecc_vec[rng.integers(0, len(ecc_vec))]
    oec0_i[2] = inc_vec[rng.integers(0, len(inc_vec))]
    oec0_i[3] = raan_vec[rng.integers(0, len(raan_vec))]
    oec0_i[4] = omega_vec[rng.integers(0, len(omega_vec))]
    oec0_i[5] = ma_vec[rng.integers(0, len(ma_vec))]

    roe_lb = np.array([-1.5, -150.0, -4.0, -40.0, -4.0, -40.0], dtype=float)
    roe_ub = np.array([1.5, 150.0, 4.0, 40.0, 4.0, 40.0], dtype=float)
    roe_0 = rng.uniform(low=roe_lb, high=roe_ub)
    roe_f = rng.uniform(low=roe_lb, high=roe_ub)
    artms_factor_choices = np.array([0.75, 1.0, 1.25, 1.5, 2.0], dtype=float)
    artms_param_1e3 = float(artms_factor_choices[rng.integers(0, len(artms_factor_choices))])

    koz_base = np.asarray(dim_koz, dtype=float).reshape(-1, 3)[0]
    koz_factor_choices = np.array([0.8, 1.0, 1.2, 1.4, 1.6], dtype=float)
    koz_factor = float(koz_factor_choices[rng.integers(0, len(koz_factor_choices))])
    koz_dim_i = koz_base * koz_factor

    return n_time_i, tf_i_sec, tvec_sec_i, oec0_i, roe_0, roe_f, artms_param_1e3, koz_dim_i


def sample_reset_condition(rng=None, behavior=None, det=False, n_time_local=None):
    """
    Sample initial/final conditions and waypoints with reproducible randomness.
    Pass in a numpy.random.Generator (rng) to avoid global RNG state.
    Used for the waypoint-behavior (text) matching for SAGES. 
    """

    if rng is None:
        rng = np.random.default_rng()
    n_time_use = n_time if n_time_local is None else int(n_time_local)

    # initial condition
    roe_0 = np.array([0, -120, 0, 5, 0, 5], dtype=float)
    if not det: 
        roe_0[1] += rng.integers(-10, 10) / 10 * 20
        roe_0[2] += rng.integers(-10, 10) / 10 * 4
        roe_0[3] += rng.integers(-10, 10) / 10 * 4
        roe_0[4] += rng.integers(-10, 10) / 10 * 4
        roe_0[5] += rng.integers(-10, 10) / 10 * 4

    if behavior is None:
        behavior = rng.integers(0, 6)

    if behavior == 0:  # approach and circumnavigate KOZ
        roe_f = np.array([0, 0, 0, 32, 0, 32], dtype=float)
        t_idx_wyp = [int(0.8 * n_time_use)]
        if not det:
            roe_f[1] += rng.integers(-10, 10) / 10 * 5
            roe_f[2] += rng.integers(-10, 10) / 10 * 2
            roe_f[3] += rng.integers(-10, 10) / 10 * 2
            roe_f[4] += rng.integers(-10, 10) / 10 * 2
            roe_f[5] += rng.integers(-10, 10) / 10 * 2
            t_idx_wyp[0] += rng.integers(-10, 9)
        wyp = [roe_f.copy()]

    elif behavior == 1:  # dock
        roe_f = np.array([0, -35, 0, 0, 0, 0], dtype=float)
        t_idx_wyp = [int(0.8 * n_time_use)]
        if not det:
            roe_f[1] += rng.integers(-10, 10) / 10 * 5
            roe_f[2] += rng.integers(-10, 10) / 10 * 2
            roe_f[3] += rng.integers(-10, 10) / 10 * 2
            roe_f[4] += rng.integers(-10, 10) / 10 * 2
            roe_f[5] += rng.integers(-10, 10) / 10 * 2
            t_idx_wyp[0] += rng.integers(-10, 9)
        wyp = [roe_f.copy()]

    elif behavior == 2:  # flyby (under KOZ)
        roe_f = np.array([0, 150, 0, 5, 0, 5], dtype=float)
        t_idx_wyp = [int(0.9 * n_time_use)]
        if not det:
            roe_f[1] += rng.integers(-10, 10) / 10 * 10
            roe_f[2] += rng.integers(-10, 10) / 10 * 2
            roe_f[3] += rng.integers(-10, 10) / 10 * 2
            roe_f[4] += rng.integers(-10, 10) / 10 * 2
            roe_f[5] += rng.integers(-10, 10) / 10 * 2
            t_idx_wyp[0] += rng.integers(-10, 4)
        wyp = [roe_f.copy()]
        
    elif behavior == 3:  # flyby (E/I-separated)
        roe_f = np.array([0, 120, 0, 5, 0, 5], dtype=float)
        wyp0 = np.array([0, roe_0[1], 0, 25, 0, 25], dtype=float)
        wyp1 = np.array([0, roe_f[1], 0, 25, 0, 25], dtype=float)
        t_idx_wyp = [int(0.2 * n_time_use), int(0.8 * n_time_use)]
        if not det:
            roe_f[1] += rng.integers(-10, 10) / 10 * 10
            roe_f[2] += rng.integers(-10, 10) / 10 * 2
            roe_f[3] += rng.integers(-10, 10) / 10 * 2
            roe_f[4] += rng.integers(-10, 10) / 10 * 2
            roe_f[5] += rng.integers(-10, 10) / 10 * 2
            t_idx_wyp[0] += rng.integers(-5, 5)
            t_idx_wyp[1] += rng.integers(-5, 5)
            wyp0 = np.array([0, roe_0[1], 0, 25 + rng.integers(-10, 10) / 10 * 2, 0, 25 + rng.integers(-10, 10) / 10 * 2], dtype=float)
            wyp1 = np.array([0, roe_f[1], 0, 25 + rng.integers(-10, 10) / 10 * 2, 0, 25 + rng.integers(-10, 10) / 10 * 2], dtype=float)
        wyp = [wyp0, wyp1]

    elif behavior == 4:  # approach, circumnavigate, and forward 
        roe_f = np.array([0, 120, 0, 35, 0, 35], dtype=float)
        wyp0 = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        wyp1 = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        t_idx_wyp = [int(0.5 * n_time_use), int(0.7 * n_time_use)]
        if not det:
            roe_f[1] += rng.integers(-10, 10) / 10 * 10
            roe_f[2] += rng.integers(-10, 10) / 10 * 2
            roe_f[3] += rng.integers(-10, 10) / 10 * 2
            roe_f[4] += rng.integers(-10, 10) / 10 * 2
            roe_f[5] += rng.integers(-10, 10) / 10 * 2
            t_idx_wyp[0] += rng.integers(-5, 0)
            t_idx_wyp[1] += rng.integers(0, 5)
            wyp0 = np.array([0, 0, 0, 30 + rng.integers(-10, 10) / 10 * 2, 0, 30 + rng.integers(-10, 10) / 10 * 2], dtype=float)
            wyp1 = np.array([0, 0, 0, 30 + rng.integers(-10, 10) / 10 * 2, 0, 30 + rng.integers(-10, 10) / 10 * 2], dtype=float)
        wyp = [wyp0, wyp1]

    elif behavior == 5:  # approach, circumnavigate, and retreat 
        roe_f = np.array([0, -120, 0, 35, 0, 35], dtype=float)
        wyp0 = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        wyp1 = np.array([0, 0, 0, 30, 0, 30], dtype=float)
        t_idx_wyp = [int(0.5 * n_time_use), int(0.7 * n_time_use)]
        if not det:
            roe_f[1] += rng.integers(-10, 10) / 10 * 10
            roe_f[2] += rng.integers(-10, 10) / 10 * 2
            roe_f[3] += rng.integers(-10, 10) / 10 * 2
            roe_f[4] += rng.integers(-10, 10) / 10 * 2
            roe_f[5] += rng.integers(-10, 10) / 10 * 2
            t_idx_wyp[0] += rng.integers(-5, 0)
            t_idx_wyp[1] += rng.integers(0, 5)
            wyp0 = np.array([0, 0, 0, 30 + rng.integers(-10, 10) / 10 * 2, 0, 30 + rng.integers(-10, 10) / 10 * 2], dtype=float)
            wyp1 = np.array([0, 0, 0, 30 + rng.integers(-10, 10) / 10 * 2, 0, 30 + rng.integers(-10, 10) / 10 * 2], dtype=float)
        wyp = [wyp0, wyp1]

    else:
        raise ValueError("behavior not recognized")

    if len(t_idx_wyp) > 0:
        t_idx_wyp = np.clip(np.asarray(t_idx_wyp, dtype=int), 1, n_time_use - 2).tolist()

    return behavior, roe_0, roe_f, t_idx_wyp, wyp

# RPOD scenario specification #############################################################

scpparam = SCVxParams()

# Canonical command map (6 modalities)
COMMAND_LIST = {
    0: "Approach to the relative orbit around the target, and circumnavigate",
    1: "Go to -V-bar waypoint, and hold",
    2: "Fast flyby under KOZ, from -V-bar (anti-velocity direction) to +V-bar (velocity direction)",
    3: "Flyby (slow, using E/I separation), from -V (ant-velocity) to +V-bar (velocity direction)",
    4: "approach to the target from -V-bar (anti-velocity direction), circumnavigate, then move to the +V-bar direction (abort maneuver)",
    5: "approach to the target from -V-bar (anti-velocity direction), circumnavigate, then move back to the -V-bar direction (abort maneuver) with RN-plane separation",
}

# per behavior/mode, list the ONLY placeholders allowed in templates
ALLOWED_PLACEHOLDERS = {
    0: ["T_appr_orbits"],
    1: ["T_appr_orbits", "d_lambda_meters"],
    2: ["T_appr_orbits", "d_lambda_meters"],
    3: ["T_EI_sep_orbits", "T_transfer_orbits"],  # "T_settle_orbits"
    4: ["T_appr_orbits", "T_circ_orbits"],   # "T_evac_orbits"
    5: ["T_appr_orbits", "T_circ_orbits"],   # "T_evac_orbits"
}

# for dummy command version
# COMMAND_LIST = {
#     0: "Abort the mission and escape",
#     1: "grasp the target satellite",
# }
# ALLOWED_PLACEHOLDERS = {
#     0: ["T_appr_orbits"],
#     1: ["T_appr_orbits"],
# }


# shared specification 
n_time = 50
n_time_max = 50
n_safe = 50
state = 'roe'

# time dilation (for test) 
# n_time = 8 
# n_time_max = 8
# n_safe = 20
# state = 'roe'

# time discretization
oec0 = np.array([6738.14, 0.0005581, np.deg2rad(51.6418), np.deg2rad(301.0371), np.deg2rad(26.1813), np.deg2rad(68.2333)]) 
n = np.sqrt(mu_E/oec0[0]**3)
period = 2*np.pi/n   # seconds

t0_sec = 0
tf_sec = 5 * period
t_safe_sec = 1 * period  # seconds
tvec_sec = np.linspace(t0_sec, tf_sec, n_time)  # nominal dt
dt_sec = tvec_sec[1] - tvec_sec[0]   # seconds
dt_safe_sec = t_safe_sec / n_safe    # seconds

oec = propagate_oe(oec0, tvec_sec)

# Waypoints [m, m, m, m/s, m/s, m/s]
rtn0 = np.array([-4e3, -17.5e3, 0, 0, 6.849, 0])
rtnf = np.array([0, 750, 0, 0, 0, 0])

# dim_koz = np.array([[1000, 1000, 1000], [500, 500, 500]])  
# t_switch = [int(n_time*0.36)]
dim_koz = np.array([[25, 25, 25]])  
t_switch = []
DEED, r_ell = generate_koz(dim_koz, n_time, t_switch=t_switch) 

# Chance constraining,  invICDF = stats.norm.ppf(1-delta_chance)
invICDF = 3.0

# variable scaling (put a rough order of your variables)
Ds = np.eye(N_STATE) * 1.0e3  # [m] for the state
Da = np.diag([1.0,1.0,1.0])

# Navigation
use_nav_artms = True
# Digital
digital_relative_std = np.array([1e-2, 1e-2, 1e-2, 25e-6, 25e-6, 25e-6]) #[m,m,m,m/s,m/s,m/s]
digital_absolute_std = np.array([10., 10., 10., 0.5, 0.5, 0.5]) #[m,m,m,m/s,m/s,m/s]
S_digital_rel = np.diag(digital_relative_std)
Sigma_nav_digital_rtn = S_digital_rel @ S_digital_rel.T
# ARTMS (Kruger Ph.D. thesis)
artms_scale_range_1e5 = np.array([4e-5, 4e-3, 4e-5, 2e-5, 2e-5, 4e-5])
artms_scale_range_1e3 = np.array([1e-4, 4e-3, 2e-3, 2e-3, 2e-3, 2e-3]) 

# Process noise
Q = np.diag([1e-3,1e-3,1e-3,1e-3,1e-3,1e-3])   # per each dt_safe. assume only actuation error / unmodeled process noise for now (Note : this should be just process noise along the uncontrolled trajectory)
QQ = Q @ Q.T

u_max = 5 # [m/s^2], this is not really effective in the current scenario

# Actuation
use_gates_model = True
# Proportional
actuation_noise_std = [0.05, 0.05, 0.05] # [%] Note : this should be improved see model used by BLUE
# Gates model [simga_s, sigma_p, sigma_r, sigma_a], reference: Berning Jr. et al. 2023
sigma_gates = np.array([2e-3, 0.3e-3, 3e-4, 0.3e-3])  

scp_iter_max = scpparam.iter_max 
