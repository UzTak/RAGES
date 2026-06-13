################ Low Earth Orbit RPO Design Reference Mission Parameters
import numpy as np

from pathlib import Path
root_folder = Path(__file__).resolve().parent.parent.parent  # /RAGES/

from dynamics.dynamics_trans import generate_koz, mu_E

##########################################################################################
################################### PARAMETERS ###########################################
##########################################################################################

# Problem dimensions #####################################################################
N_STATE = 6
N_ACTION = 3

# Time discretization ####################################################################
oec0 = np.array([6738.14, 0.0005581, np.deg2rad(51.6418), np.deg2rad(301.0371), np.deg2rad(26.1813), np.deg2rad(68.2333)])
n = np.sqrt(mu_E / oec0[0]**3)
period = 2 * np.pi / n   # seconds

# Passive safety integration #############################################################
n_safe = 50
# Reference values only — dt_safe_sec is recomputed per-problem from the actual oec0
t_safe_sec = 1 * period
dt_safe_sec = t_safe_sec / n_safe

# Navigation #############################################################################
use_nav_artms = True
# ARTMS (Kruger Ph.D. thesis)
artms_scale_range_1e5 = np.array([4e-5, 4e-3, 4e-5, 2e-5, 2e-5, 4e-5])
artms_scale_range_1e3 = np.array([1e-4, 4e-3, 2e-3, 2e-3, 2e-3, 2e-3])
# Digital
digital_relative_std = np.array([1e-2, 1e-2, 1e-2, 25e-6, 25e-6, 25e-6])
S_digital_rel = np.diag(digital_relative_std)
Sigma_nav_digital_rtn = S_digital_rel @ S_digital_rel.T

# Process noise ##########################################################################
Q = np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
QQ = Q @ Q.T

# Actuation ##############################################################################
u_max = 5   # [m/s^2]
use_gates_model = True
# Gates model [sigma_s, sigma_p, sigma_r, sigma_a], reference: Berning Jr. et al. 2023
sigma_gates = np.array([2e-3, 0.3e-3, 3e-4, 0.3e-3])

# Chance constraining,  invICDF = stats.norm.ppf(1-delta_chance) ###########################
invICDF = 3.0
