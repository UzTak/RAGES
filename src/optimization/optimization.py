import os
import numpy as np
import scipy as sp 
import cvxpy as cp
import torch 
import warnings
from multiprocessing import Pool, cpu_count
import time
import sys
from numpy.polynomial.legendre import leggauss
import pickle
import importlib

# /rpod/
from pathlib import Path

root_folder = Path(__file__).resolve().parent.parent.parent  # /art_lang/
import dynamics.dynamics_trans as dyn
from optimization.scvx import *
import optimization.parameters as param

class NonConvexOCP():
    """
    Nonconvex Optimal Control Problem (OCP) class. Tailored for the SCVx* implementation.
    """
    def __init__(self,
                 prob_definition : dict,
                 zref=None):

        self.nx = param.N_STATE
        self.nu = param.N_ACTION

        # scp parameters
        self.verbose = prob_definition.get('verbose', False)  # verbose for cvxpy
        self.verbose_scvx = False  # verbose for scvx
        self.ignore_cvxpy_warnings = prob_definition.get('ignore_cvxpy_warnings', True)

        # SVCx parameters (fixed)
        self.scvx_param = SCVxParams()
        self.iter_max = self.scvx_param.iter_max  # param.scp_iter_max
        # parameters (variable) : it will be reset in solve_scvx() anyways so don't worry too much 
        self.pen_λ, self.pen_μ, self.pen_w, self.rk = None, None, None, None
        
        self.__setup_problem(prob_definition)
        
        out = self.generate_dynamics(self.oe_i, self.tvec_sec)
        self.oe = out["oe"]
        self.stm, self.cim, self.psi = out["stm"], out["cim"], out["psi"]
        
        self.f_stm, self.f_cim, self.f_psi, self.f_2rtn = dyn.stm_roe, dyn.cim_roe, dyn.mtx_roe_to_rtn, dyn.roe_to_rtn   
        
        self.flag_ps_efficient = True
        self.update_flag = True   # flag to update the reference trajectory 
        
        # initial solution
        self.zref = zref if zref is not None else {'state': None, 'action': None}
        self.sol_0 = {"z": self.zref}
        self.t_star = None
        
        # variable scaling
        self.generate_scaling(self.zref['state'], self.zref['action'])
        
        # feasibility problem weight (action_error + feas_w * state_error)
        self.feas_w = 0.0003   # FIXME: this should be tuned a bit! 
        
        self.J_cvx = 0.0   # convex lower bound 
            
    def __setup_problem(self, prob_definition:dict):
        """
        Method to define the optimization problem based on the current observation.

        Args:
            - prob_definition (`dict`):
                Dictionary containing the information to correctly setup the optimization problem, as specified for the object constructor.
        """
        # Define problem
        current_obs      = prob_definition['current_obs']
        self.state_init  = current_obs['state']
        self.state_final = current_obs['goal']
        self.final_time  = current_obs['ttg']   # memo; terminal time in seconds 
        self.dt          = current_obs['dt']   # seconds 
        self.oe_i        = current_obs['oe']   # INITIAL chief OE, not the history of OEs) 
        self.period = 2*np.pi*np.sqrt((self.oe_i[0]**3)/dyn.mu_E)  # seconds
                
        self.t_i = int(prob_definition['t_i'])   # index, not the actual time in seconds.
        self.t_f = int(prob_definition['t_f'])   # index, not the actual time in seconds.
        self.t_i_sec = prob_definition['tvec_sec'][self.t_i] 
        self.t_f_sec = prob_definition['tvec_sec'][self.t_f-1]
        self.tvec_sec = prob_definition['tvec_sec'][self.t_i:self.t_f]
        self.n_time = self.t_f - self.t_i 
        self.horizon = self.t_f_sec / self.period
        
        self.chance = prob_definition.get('chance', False)
        self.ct = prob_definition.get('ct', False)

        # default: load the safety paraemters for passive safety 
        self.n_safe = param.n_safe
        self.dt_safe_sec = param.dt_safe_sec   
        self.t_safe_sec = self.n_safe * self.dt_safe_sec 
        self.tvec_safe_sec = np.linspace(0, self.t_safe_sec, self.n_safe)
        self.t_switch = prob_definition.get('t_switch', [])  # list of switch times for KOZ

        # Cache KOZ matrices for this time grid
        self.koz_dim = prob_definition.get('koz_dim', param.dim_koz)
        self.DEED, _ = param.generate_koz(self.koz_dim, self.n_time, t_switch=self.t_switch)
        
        # ARTMS Navigation parameter 
        self.artms_scale_range_1e5 = prob_definition.get('artms_scale_range_1e5', param.artms_scale_range_1e5)
        base_artms_scale_range_1e3 = np.asarray(param.artms_scale_range_1e3, dtype=float)
        if 'artms_param_1e3' in prob_definition:
            self.artms_param_1e3 = float(prob_definition['artms_param_1e3'])
            self.artms_scale_range_1e3 = base_artms_scale_range_1e3 * self.artms_param_1e3
        elif 'artms_scale_range_1e3' in prob_definition:
            self.artms_scale_range_1e3 = np.asarray(prob_definition['artms_scale_range_1e3'], dtype=float)
            self.artms_param_1e3 = float(np.mean(self.artms_scale_range_1e3 / base_artms_scale_range_1e3))
        else:
            self.artms_param_1e3 = 1.0
            self.artms_scale_range_1e3 = base_artms_scale_range_1e3.copy()
        
        # scaling factors. Each element of Ds and Da is expected magnitudes of values of the variables.
        self.Ds = prob_definition['Ds'] if 'Ds' in prob_definition else np.eye(self.nx)
        self.Da = prob_definition['Da'] if 'Da' in prob_definition else np.eye(self.nu)
        self.invDs = np.diag(1/np.diag(self.Ds))
        self.invDa = np.diag(1/np.diag(self.Da))
        self.cs = None 
        self.ca = None 
        
        # waypoints: optional waypoint constraints
        self.waypoints      = prob_definition.get('waypoints', np.empty(0))  # List of waypoint states
        self.waypoint_times = prob_definition.get('waypoint_times', np.empty(0))  # List of waypoint time indices
        self.waypoint_type  = prob_definition.get('waypoint_type', 'roe')  # Type of waypoints: "roe" or "rtn"

        # problem type: "feasibility", "min_fuel"
        self.type = prob_definition.get('type', 'min_fuel')
        self.behavior = prob_definition.get('behavior', None)
        
        self._cvx_built_AL = False  # flag for lazy building of the CVX problem

        # Gauss–Legendre nodes/weights on [0, τ_s]
        if self.ct:
            # tau discretization 
            self.n_tau = 30  
            xi, wi = leggauss(self.n_tau)          # nodes/weights on [-1,1]
            self.tau_nodes = 0.5 * self.t_safe_sec * (xi + 1.0)
            self.tau_weights = 0.5 * self.t_safe_sec * wi
            
            # alpha discretization 
            self.n_alpha = 10
            xi_a, wi_a = leggauss(self.n_alpha)     # nodes/weights on [-1,1]
            self.alpha_nodes = 0.5 * (xi_a + 1.0)   # nodes on [0,1]
            self.alpha_weights = 0.5 * wi_a
        
        # Validate waypoints if provided
        if len(self.waypoints) > 0 or len(self.waypoint_times) > 0:
            self._validate_waypoints()

    def _validate_waypoints(self):
        """
        Validate waypoint constraints for both EROE and RTN types.
        """
    
        assert len(self.waypoints) == len(self.waypoint_times), f"Mismatch: {len(self.waypoints)} waypoints but {len(self.waypoint_times)} waypoint_times"
        assert self.waypoint_type in ['roe', 'rtn'], f"waypoint_type must be 'roe' or 'rtn', got '{self.waypoint_type}'"
        
            
    @staticmethod
    def generate_dynamics(oe_0, tvec_sec, state='roe', J2=dyn.J2):
        """
        linearize the nonlinear dynamics of 6DoF relative motion 
        input: 
            oe: OE (chief);        6 x (n_time+1)
            t_0: initial time [s]
            t_f: final time [s]
            n_time: # time steps 
        return:
            stm: state transition matrix;                   6 x 6 x n_time
            cim: control input matrix;                      6 x 3 x n_time
            psi: ROE -> RTN map;                            6 x 6 x (n_time+1)
            oe: Keplarian orbital elements (chief) history  6 x n_time
            dt: time step
        """
        
        n_time = len(tvec_sec)

        if state == 'eroe':
            f_stm, f_cim, f_psi = dyn.stm_eroe, dyn.cim_eroe, dyn.mtx_eroe_to_rtn
        elif state == 'roe':
            f_stm, f_cim, f_psi = dyn.stm_roe, dyn.cim_roe, dyn.mtx_roe_to_rtn
        else:
            raise ValueError(f"State '{state}' not recognized. Use 'eroe' or 'roe'.")

        stm = np.empty(shape=(n_time-1, 6, 6), dtype=float)
        cim = np.empty(shape=(n_time, 6, 3), dtype=float)
        psi = np.empty(shape=(n_time, 6, 6), dtype=float)
        oe = np.empty(shape=(n_time, 6), dtype=float)

        oe[0] = dyn.propagate_oe(oe_0, tvec_sec[0], J2=J2)
        cim[0] = f_cim(oe[0])
        psi[0] = f_psi(oe[0])

        for iter in range(n_time-1):
            stm[iter] = f_stm(oe[iter], tvec_sec[iter+1]-tvec_sec[iter], J2=J2) 
            oe[iter+1] = dyn.propagate_oe(oe[iter], tvec_sec[iter+1]-tvec_sec[iter], J2=J2)
            cim[iter+1] = f_cim(oe[iter+1])
            psi[iter+1] = f_psi(oe[iter+1])  

        out  = {'stm': stm, 'cim': cim, 'psi': psi, 'oe': oe,}
        return out

    def generate_scaling(self, sref, aref, scale=(-1.0, 1.0), eps=1e-4):
        """
        Build an 'unscale' map centered at the mean:
            s = cs + Ds @ s_scl,   a = ca + Da @ a_scl

        Args:
            sref, aref: arrays of shape (N, nx), (N, nu) reference states/actions
            scale: (lo, hi) target interval for the scaled variables (default: [-1, 1])
            eps: small floor to avoid zero/inf scales
        """
        lo, hi = scale
        assert hi > lo, "scale must satisfy hi > lo"

        def _compute(ref):
            mu = np.mean(ref, axis=0)
            r = np.max(np.abs(ref - mu[None, :]), axis=0)   # symmetric radius
            d = np.maximum(r / max(hi, -lo), eps)           # ensure coverage
            return mu, d

        if sref is None or aref is None:
            self.Ds, self.invDs = np.eye(self.nx), np.eye(self.nx)
            self.Da, self.invDa = np.eye(self.nu), np.eye(self.nu)
            self.cs, self.ca = np.zeros(self.nx), np.zeros(self.nu)
            self.Dt, self.invDt = 1.0, 1.0
        else:
            mu_s, d_s = _compute(sref)
            mu_a, d_a = _compute(aref)
            # store diagonal matrices (and diagonals for convenience)
            self.Ds = np.diag(d_s)
            self.Da = np.diag(d_a)
            self.invDs = np.diag(1/d_s)
            self.invDa = np.diag(1/d_a)
            self.cs = mu_s
            self.ca = mu_a
    
    def ocp_cvx(self):
        # def ocp_cvx_AL(prob, verbose=False):
        """
        Classic Two point boundary value problem (TPBVP) formulation of the OCP.    
        """
    
        n_time = self.n_time  
        s0  = self.state_init
        sf  = self.state_final
        umax = param.u_max  # maximum actuation in non-dimensional units
        stm, cim = self.stm, self.cim

        # non-dimensional variables
        s_scl = cp.Variable((n_time, self.nx))
        a_scl = cp.Variable((n_time, self.nu))       
        
        # "real" (physical) values 
        s = cp.multiply(s_scl, np.diag(self.Ds)) 
        a = cp.multiply(a_scl, np.diag(self.Da))
        z = {"state": s, "action": a,}    
        
        con = []    
        con += [s[0] == s0]
        con += [s[i+1] == stm[i] @ (s[i] + cim[i] @ a[i]) for i in range(n_time-1)]
        con += [s[-1] + cim[-1] @ a[-1] == sf]
        con += [cp.norm(a[i]) <= umax for i in range(n_time)]  # actuation constraint

        for _, (waypoint_time, waypoint_state) in enumerate(zip(self.waypoint_times, self.waypoints)):
            if self.waypoint_type == 'roe':
                con += [s[waypoint_time] == waypoint_state]
            elif self.waypoint_type == 'rtn':  # Only position components for RTN
                wyp = self.psi[:,:,waypoint_time] @ waypoint_state
                con += [s[waypoint_time, :3] == wyp[:3]]  

        f0 = self.compute_f0(z)
        cost = f0 
        
        # Compute Cost    
        p = cp.Problem(cp.Minimize(cost), con)
                
        try:
            self._solve_problem(p, solver=cp.CLARABEL, verbose=self.verbose)
            status = p.status
        except:
            status = 'infeasible'
        z_opt  = {"state": s.value, "action": a.value} 
        f0_opt = f0.value
        L_opt  = p.value
        
        sol = {"z": z_opt, "status": status, "L": L_opt, "f0": f0_opt, "s_opt": s.value, "a_opt": a.value, "cost": f0_opt}
        
        return sol

    def build_ocp_cvx_AL(self):
        """
        Build the CVX problem once, with Parameters for all
        quantities that change across SCP iterations.
        """

        n_time = self.n_time        

        # scaled (non-dimensional) variables
        self.s_scl = cp.Variable((n_time, self.nx), name='s_scl')
        self.a_scl = cp.Variable((n_time, self.nu), name='a_scl')
        self.xi   = cp.Variable((self.nx,), name="xi")
        self.zeta = cp.Variable((n_time,), nonneg=True, name="zeta")

        # Parameters (store on self)
        self.s_sclref_param   = cp.Parameter((n_time, self.nx), name="s_sclref")
        self.a_sclref_param   = cp.Parameter((n_time, self.nu), name="a_sclref")
        self.pen_lambda_param = cp.Parameter((self.nx,), name="pen_lambda")
        self.pen_mu_param     = cp.Parameter((n_time,), nonneg=True, name="pen_mu")
        self.pen_w_param      = cp.Parameter(nonneg=True, name="pen_w")
        self.jac_ps_param     = cp.Parameter((n_time - 1, self.nx + self.nu), name="jac_ps")
        self.con_ref_ps_param = cp.Parameter((n_time - 1,), name="con_ref_ps")
        self.rk_param         = cp.Parameter(nonneg=True, name="rk")
        self.Da_param         = cp.Parameter((self.nu, self.nu), name="Da")
        self.Ds_param         = cp.Parameter((self.nx, self.nx), name="Ds")
        self.cs_param         = cp.Parameter((self.nx,), name="cs")
        self.ca_param         = cp.Parameter((self.nu,), name="ca")
        self.s0_param = cp.Parameter((self.nx,), name="s0")
        self.sf_param = cp.Parameter((self.nx,), name="sf")

        # physical variables
        self.s = self.s_scl @ self.Ds_param + self.cs_param[None, :]
        self.a = self.a_scl @ self.Da_param + self.ca_param[None, :]

        z = {"state": self.s, "action": self.a}

        con = []
        con += [self.s[0] == self.s0_param]
        con += [self.s[i+1] == self.stm[i] @ (self.s[i] + self.cim[i] @ self.a[i]) for i in range(n_time-1)]
        con += [cp.norm(self.a[i]) <= param.u_max for i in range(n_time)]
        con += [self.s[-1] + self.cim[-1] @ self.a[-1] - self.sf_param == self.xi]

        # Passive safety (DPP-safe: linearized in scaled variables)
        con += [self.con_ref_ps_param
                + cp.sum(cp.multiply(self.jac_ps_param, cp.hstack([self.s_scl[:-1], self.a_scl[:-1]])), axis=1)
                <= self.zeta[:-1]]
        
        for _, (waypoint_time, waypoint_state) in enumerate(zip(self.waypoint_times, self.waypoints)):
            if self.waypoint_type == 'roe':
                con += [self.s[waypoint_time] == waypoint_state]
            elif self.waypoint_type == 'rtn':  # Only position components for RTN
                wyp = self.psi[:,:,waypoint_time] @ waypoint_state
                con += [self.s[waypoint_time, :3] == wyp[:3]]  

        # Trust regions (scaled variables)
        con += [cp.norm(cp.reshape((self.s_scl - self.s_sclref_param), (-1,) , order='F'), "inf") <= self.rk_param]
        con += [cp.norm(cp.reshape((self.a_scl - self.a_sclref_param), (-1,) , order='F'), "inf") <= self.rk_param]

        f0 = self.compute_f0(z)
        P  = self.compute_P(self.xi, self.zeta,
                            self.pen_lambda_param, self.pen_mu_param, self.pen_w_param)

        self._f0_expr, self._P_expr = f0, P
        self.prob_AL = cp.Problem(cp.Minimize(f0 + P), con)
        
        self._cvx_built_AL = True

    @staticmethod
    def set_param(param, val):
        arr = np.asarray(val, dtype=np.float64)
        if arr.shape != param.shape:
            raise ValueError(f"{param.name()} shape mismatch {arr.shape} vs {param.shape}")
        if arr.ndim == 2:
            arr = np.asfortranarray(arr)
        elif arr.ndim == 1:
            arr = np.ascontiguousarray(arr)
        param.value = arr

    def _solve_problem(self, problem, **solve_kwargs):
        if not self.ignore_cvxpy_warnings:
            problem.solve(**solve_kwargs)
            return

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The problem includes expressions that don't support CPP backend\..*",
                category=UserWarning,
                module=r"cvxpy\..*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Solution may be inaccurate\..*",
                category=UserWarning,
                module=r"cvxpy\..*",
            )
            problem.solve(**solve_kwargs)

    def solve_cvx_AL(self):
        
        if not self._cvx_built_AL:
            self.build_ocp_cvx_AL()

        # --- Update references / linearizations if flagged ---
        if self.update_flag:
            self.set_param(self.s_sclref_param, (self.zref["state"] - self.cs[None, :]) @ self.invDs)
            self.set_param(self.a_sclref_param, (self.zref["action"] - self.ca[None, :]) @ self.invDa)

            # these do not need to be updated every time, but it's easier this way
            self.set_param(self.s0_param, self.state_init)
            self.set_param(self.sf_param, self.state_final)
            self.set_param(self.Ds_param, self.Ds)
            self.set_param(self.Da_param, self.Da)
            self.set_param(self.cs_param, self.cs)
            self.set_param(self.ca_param, self.ca)

            # Linearization uses scaled variables: z_scl = [(s-cs)@invDs, (a-ca)@invDa]
            zref_scl = np.hstack([
                (self.zref["state"][:-1] - self.cs[None, :]) @ self.invDs,
                (self.zref["action"][:-1] - self.ca[None, :]) @ self.invDa,
            ])
            self.set_param(self.jac_ps_param, self.jac_ps_tmp[:-1])
            self.set_param(self.con_ref_ps_param,
                           self.con_ref_ps_tmp[:-1] - np.sum(self.jac_ps_tmp[:-1] * zref_scl, axis=1))

        self.set_param(self.pen_lambda_param, self.pen_λ)
        self.set_param(self.pen_mu_param, np.maximum(self.pen_μ, 0.0))
        self.pen_w_param.value = float(self.pen_w)
        self.rk_param.value    = float(self.rk)

        try:
            self.prob_AL._solver_cache.clear()   # important: reset the solver cache to avoid CLARABLE failure
            self._solve_problem(self.prob_AL, solver=cp.CLARABEL, verbose=self.verbose)

            status = self.prob_AL.status
            if status not in ("optimal", "optimal_inaccurate"):
                print("status:", status)
                z_opt = None
            else:
                # Retrieve raw values from the decision variables
                # print("cvx subproblem solved successfully.")
                
                soln = self.prob_AL.solution
                pv = soln.primal_vars
                var_by_name = {v.name(): v for v in self.prob_AL.variables()}
                s_scl_id = var_by_name["s_scl"].id
                a_scl_id = var_by_name["a_scl"].id
                xi_id    = var_by_name["xi"].id
                zeta_id  = var_by_name["zeta"].id

                s_scl_val = pv[s_scl_id].reshape(var_by_name["s_scl"].shape, order="F")
                a_scl_val = pv[a_scl_id].reshape(var_by_name["a_scl"].shape, order="F")
                s_opt = s_scl_val @ self.Ds + self.cs[None, :] 
                a_opt = a_scl_val @ self.Da + self.ca[None, :]
                z_opt = {"state": s_opt, "action": a_opt}
                ξ_opt    = pv[xi_id].reshape(var_by_name["xi"].shape, order="F")
                ζ_opt  = pv[zeta_id].reshape(var_by_name["zeta"].shape, order="F")
                
                # s_scl_val = self.s_scl.value
                # a_scl_val = self.a_scl.value
                # s_opt = s_scl_val @ self.Ds + self.cs[None, :] 
                # a_opt = a_scl_val @ self.Da + self.ca[None, :]
                # z_opt = {"state": s_opt, "action": a_opt}
                # print(z_opt['state'][10,0])
                
                f0_opt = self.compute_f0(z_opt).value
                P_opt  = self.compute_P(self.xi.value, self.zeta.value).value
                L_opt  = f0_opt + P_opt
                # ξ_opt  = self.xi.value    
                # ζ_opt  = self.zeta.value
                
        except Exception as e:
            # raise ValueError(f"CVX solver failed: {e}")
            status = 'infeasible'
            z_opt = None

        if z_opt is None:
            f0_opt, P_opt, L_opt = None, None, None
            ξ_opt, ζ_opt, s_opt, a_opt = None, None, None, None

        sol = {
            "z": z_opt, "ξ": ξ_opt, "ζ": ζ_opt, "status": status,
            "L": L_opt, "f0": f0_opt, "P": P_opt, 
            "s_opt": s_opt, "a_opt": a_opt, "cost": f0_opt,
        }

        # reset the update flag
        self.update_flag = False
            
        return sol
    
    def compute_f0(self, z):
        if self.type == "min_fuel":
            return cp.sum(cp.norm(z["action"], 2, axis=1))
        elif self.type == "feasibility":
            # reference states/actions of the initial guess (NOT UPDATED)             
            dx = cp.sum(cp.norm(z["state"]  - self.sol_0["z"]["state"], 2, axis=1))
            du = cp.sum(cp.norm(z["action"] - self.sol_0["z"]["action"], 2, axis=1))
            obj = self.feas_w * dx + du
            return obj
        else: 
            raise ValueError(f"Unknown problem type '{self.type}'")
    
    def compute_P(self, g, h, lam=None, mu=None, w=None):
        """
        Warning! Make sure hp >= 0 when calling this function.
        It should be enforced in the definition of ζ and also in solve_scvx().
        """
        if lam is None:
            lam, mu, w = self.pen_λ, self.pen_μ, self.pen_w
        hp = cp.pos(h) 
        # use parameters instead of self.pen_λ, self.pen_μ, self.pen_w
        P = lam.T @ g + mu.T @ hp + 0.5 * w * (cp.sum_squares(g) + cp.sum_squares(hp))
        return P

    ##############################################################

    def compute_g(self, z):
        """ 
        Returning nonconvex constraint value g and its gradient dg (written in NUMPY)
        """
        
        s = z['state']
        a = z['action']
        
        # terminal constraint 
        g = s[-1] + self.cim[-1] @ a[-1] - self.state_final
        dg = None  
        
        return g, dg

    def compute_h(self, z):
        """
        Return inequality h and dh (written in NUMPY)
        """
        
        s = z['state']
        a = z['action']
        _,_,_,_, jac_ps, con_ref_ps = self.eval_ps(s, a, self.tvec_sec, chance=self.chance, ct=self.ct)
        # Scale Jacobian so linearization is affine in scaled variables (DPP-safe)
        jac_scl = np.empty_like(jac_ps)
        jac_scl[:, :self.nx] = jac_ps[:, :self.nx] @ self.Ds
        jac_scl[:, self.nx:] = jac_ps[:, self.nx:] @ self.Da
        self.jac_ps_tmp = jac_scl
        self.con_ref_ps_tmp = con_ref_ps
        df = None

        return self.con_ref_ps_tmp, df
    
    ##############################################################
    ##### Constraint / Reward Evaluation #########################
    ##############################################################    
    
    def _ps_step_torch(self, s_k, a_k, oe_k, DEED_k_torch, active_mask=None, chance=False, lib=torch):
        """
        Single passive safety step constraint evaluation (batch-capable).
        Processes a single trajectory at a time.

        Args:
            s_k: [B, 6] or [6,] - batch of states at time k
            a_k: [B, 3] or [3,] - batch of actions at time k
            oe_k: [6] - orbital elements at time k (same for all batch)
            DEED_k_torch: [6, 6] - DEED matrix at time k
            active_mask: [B] - boolean mask for active elements
            chance: bool - whether to use chance constraints
            lib: torch - tensor library
            
        Returns:
            max_constr_ps: [B] - worst constraint values for each batch element
        """
        
        is_batch = len(s_k.shape) == 2
        if not is_batch:
            s_k = s_k.unsqueeze(0)   # [1, 6]
            a_k = a_k.unsqueeze(0)   # [1, 3]
        
        if active_mask is None:
            active_mask = torch.ones(1, dtype=torch.bool, device=s_k.device)
        
        B = s_k.shape[0]
        device = s_k.device
        dt_safe, n_safe = self.dt_safe_sec, self.n_safe

        # Following original structure exactly
        QQ_torch = torch.as_tensor(param.QQ, dtype=torch.float32, device=device)
        stm_kj = torch.eye(6, dtype=torch.float32, device=device)
        psi_kj = torch.as_tensor(self.f_psi(oe_k), dtype=torch.float32, device=device)
        stmpsi_kj = psi_kj @ stm_kj

        # Control influence (torch) and post-burn state in ROE 
        cim_k = torch.as_tensor(self.f_cim(oe_k), dtype=torch.float32, device=device)
        s_roe0 = s_k + torch.matmul(cim_k, a_k.T).T  # [B, 6]
        s_roe_kj = s_roe0.clone() 
        s_rtn_kj = torch.matmul(psi_kj, s_roe_kj.T).T  

        # Actuation error covariance in ROE 
        if param.use_gates_model:
            UU_j = torch.stack([dyn.dv_cov_gates(a_k[n], param.sigma_gates, lib=torch) 
                                     for n in range(B)], dim=0)  # [B, 3, 3]
        else:
            UU_j = torch.stack([dyn.dv_cov_simple(a_k[n], lib=torch) 
                                     for n in range(B)], dim=0)  # [B, 3, 3]

        # Navigation covariance 
        if param.use_nav_artms:
            Σ_nav_roe = torch.stack([dyn.rel_nav_artms_roe(torch.linalg.norm(s_rtn_kj[n, :3]), 
                                                            artms_scale_range_1e5 = self.artms_scale_range_1e5,
                                                            artms_scale_range_1e3 = self.artms_scale_range_1e3,
                                                            lib=torch, device=device) for n in range(B)], dim=0)  # [B, 6, 6]
        else:
            Sigma_nav_rtn = torch.as_tensor(param.Sigma_nav_digital_rtn, dtype=torch.float32, device=device)
            invpsi_j = torch.linalg.inv(psi_kj)
            Σ_nav_roe_single = invpsi_j @ Sigma_nav_rtn @ invpsi_j.T  # [6, 6]
            Σ_nav_roe = Σ_nav_roe_single.unsqueeze(0).repeat(B, 1, 1)  # [B, 6, 6]

        # Initial covariance 
        cim_k_expanded = cim_k.unsqueeze(0).expand(B, -1, -1)  # [B, 6, 3]
        Σ_roe0 = Σ_nav_roe + torch.bmm(torch.bmm(cim_k_expanded, UU_j), 
                                                   cim_k_expanded.transpose(-2, -1))  # [B, 6, 6] freeze for chance variance
        Σ_roe_kj = Σ_roe0.clone()  # [B, 6, 6]
        Σ_rtn_kj = torch.bmm(torch.bmm(psi_kj.unsqueeze(0).expand(B, -1, -1), Σ_roe_kj), 
                                   psi_kj.T.unsqueeze(0).expand(B, -1, -1))  # [B, 6, 6]

        # Storage arrays  (following original structure)
        s_roe_ps = torch.zeros((B, n_safe, 6),    dtype=torch.float32, device=device)  # [B, n_safe, 6]
        s_rtn_ps = torch.zeros((B, n_safe, 6),    dtype=torch.float32, device=device)  # [B, n_safe, 6]
        Σ_roe_ps = torch.zeros((B, n_safe, 6, 6), dtype=torch.float32, device=device)  # [B, n_safe, 6, 6]
        Σ_rtn_ps = torch.zeros((B, n_safe, 6, 6), dtype=torch.float32, device=device)  # [B, n_safe, 6, 6]
        constr_k = torch.empty((B, n_safe),       dtype=torch.float32, device=device)  # [B, n_safe]
        invICDF = torch.tensor(float(param.invICDF), dtype=torch.float32, device=device)

        # Main loop 
        for j in range(n_safe):
            s_roe_ps[:, j] = s_roe_kj  # [B, 6]
            s_rtn_ps[:, j] = s_rtn_kj  # [B, 6]
            Σ_roe_ps[:, j] = Σ_roe_kj  # [B, 6, 6]
            Σ_rtn_ps[:, j] = Σ_rtn_kj  # [B, 6, 6]

            # quadratic form on post-burn reference ξ = s_roe0 
            M_kj = stmpsi_kj.T @ DEED_k_torch @ stmpsi_kj  # [6, 6] - same for all B
            q_j = torch.sum(s_roe0 * (M_kj @ s_roe0.T).T, dim=1)  # [B]
            g_j = 1.0 - q_j  # [B]

            if chance:
                # Chance constraint computation
                grad_s = -2.0 * (M_kj @ s_roe0.T).T  # [B, 6]
                var_g = torch.einsum('bi,bij,bj->b', grad_s, Σ_roe0, grad_s)  # [B]
                stdev = torch.sqrt(torch.clamp(var_g, min=0.0))  # [B]
                g_j = g_j + invICDF * stdev  # [B]
                
            constr_k[:, j] = g_j  # [B]

            # propagate one dt_safe - SHARED for all B trajectories
            stm_step = torch.as_tensor(self.f_stm(oe_k, dt_safe), dtype=torch.float32, device=device)
            stm_kj = stm_step @ stm_kj
            oe_k = dyn.propagate_oe(oe_k, dt_safe, J2=dyn.J2)
            psi_kj = torch.as_tensor(self.f_psi(oe_k), dtype=torch.float32, device=device)
            stmpsi_kj = psi_kj @ stm_kj

            # State and covariance propagation 
            s_roe_kj = torch.matmul(stm_step, s_roe_kj.T).T  # [B, 6]
            s_rtn_kj = torch.matmul(psi_kj, s_roe_kj.T).T  # [B, 6]
            
            # Batch covariance updates
            stm_exp = stm_step.unsqueeze(0).expand(B, -1, -1)  # [B, 6, 6]
            stm_T_exp = stm_step.T.unsqueeze(0).expand(B, -1, -1)  # [B, 6, 6]
            QQ_exp = QQ_torch.unsqueeze(0).expand(B, -1, -1)  # [B, 6, 6]
            Σ_roe_kj = torch.bmm(torch.bmm(stm_exp, Σ_roe_kj), stm_T_exp) + QQ_exp  # [B, 6, 6]
            
            psi_exp = psi_kj.unsqueeze(0).expand(B, -1, -1)  # [B, 6, 6]
            psi_T_exp = psi_kj.T.unsqueeze(0).expand(B, -1, -1)  # [B, 6, 6]
            Σ_rtn_kj = torch.bmm(torch.bmm(psi_exp, Σ_roe_kj), psi_T_exp)  # [B, 6, 6]

        # worst slice (largest g is most violating; >0 indicates violation) 
        t_star_j = torch.argmax(constr_k, dim=1)  # [B]
        max_constr_ps = constr_k.gather(1, t_star_j.unsqueeze(1)).squeeze(1)  # [B]

        # Following original structure
        if not chance:
            Σ_roe_ps.zero_()
            Σ_rtn_ps.zero_()

        # Apply active mask (inactive trajectories get 0.0 constraint value)
        max_constr_ps = torch.where(active_mask, max_constr_ps, torch.zeros_like(max_constr_ps))

        if not is_batch:   # return to original non-batch shape
            return (s_roe_ps[0], s_rtn_ps[0], Σ_roe_ps[0], Σ_rtn_ps[0], max_constr_ps[0])
        else: 
            return s_roe_ps, s_rtn_ps, Σ_roe_ps, Σ_rtn_ps, max_constr_ps
    
    
    # NEW implementation; everything all in once 
    def _actuation_cov(self, s_k, a_k, cim_k, psi_k):
        """
        Compute the actuation error covariance Σ_roe in ROE space.
        Args:
            a_k    : (3,) array of actuation at time k
            cim_k  : (6,3) array of control influence matrix at time k
            psi_k  : (6,6) array of rotation matrix from ROE to RTN frame at time k
        Returns:
            Σ_roe  : (6,6) covariance matrix in ROE space
        """
        if param.use_gates_model:
            UU_j = dyn.dv_cov_gates(a_k, param.sigma_gates)  # Gates model 
        else:
            UU_j = dyn.dv_cov_simple(a_k)
        
        if param.use_nav_artms:
            s_rtn_k = psi_k @ s_k
            Σ_nav_roe = dyn.rel_nav_artms_roe(np.linalg.norm(s_rtn_k[:3]), 
                                              artms_scale_range_1e5=self.artms_scale_range_1e5,
                                              artms_scale_range_1e3=self.artms_scale_range_1e3 )
        else:
            Sigma_nav_rtn = param.Sigma_nav_digital_rtn
            invpsi_j = np.linalg.inv(psi_k)
            Σ_nav_roe = invpsi_j @ Sigma_nav_rtn @ invpsi_j.T
            
        Σ_roe = Σ_nav_roe + cim_k @ UU_j @ cim_k.T
        return Σ_roe 
    
    def _ps_step(self, s_k, a_k, oe_k, DEED_k, chance=False):
        """Propagate one dt_safe hop and return everything needed downstream."""
        dt_safe, n_safe = self.dt_safe_sec, self.n_safe
        cim_k  = self.f_cim(oe_k)
        psi_k  = self.f_psi(oe_k)

        # initial state & covariance (ROE / RTN)
        s_roe, stm = s_k + cim_k @ a_k, np.eye(6)
        Σ_roe  = self._actuation_cov(s_k, a_k, cim_k, psi_k)  
        s_roe0 = s_roe.copy() 
        Σ_roe0 = Σ_roe.copy()
        s_rtn  = psi_k @ s_roe
        Σ_rtn  = psi_k @ Σ_roe @ psi_k.T

        # memory blocks ───────────────────────────────────────────────────────────
        s_roe_ps  = np.empty((n_safe, 6))
        s_rtn_ps  = np.empty((n_safe, 6))
        Σ_roe_ps  = np.empty((n_safe, 6, 6)) if chance else np.zeros((n_safe, 6, 6))
        Σ_rtn_ps  = np.empty((n_safe, 6, 6)) if chance else np.zeros((n_safe, 6, 6))
        jac_k     = np.empty((n_safe,9)) 
        con_ref_k = np.empty(n_safe ) 

        # propagate through each safe slice ───────────────────────────────────────
        for j in range(n_safe):
            s_roe_ps[j], s_rtn_ps[j] = s_roe, s_rtn
            if chance:
                Σ_roe_ps[j], Σ_rtn_ps[j] = Σ_roe, Σ_rtn

            # quadratic form & chance margin
            psistm_kj = psi_k @ stm
            M_kj = psistm_kj.T @ DEED_k @ psistm_kj
            con_ref_k[j] = 1.0 - (s_roe0.T @ M_kj @ s_roe0).item()

            # save linearisation terms (only once per j-step)
            jac_k[j] = np.hstack((
                    -2 * M_kj @ s_roe0, 
                    -2 * cim_k.T @ M_kj @ s_roe0,
                ))

            if chance:
                # variance_h = jac_k[:6,j].T @ Σ_roe0 @ jac_k[:6,j]  # this was wrong implementation (true if Q = 0)
                v = np.linalg.solve(stm.T, jac_k[j,:6].T)
                variance_h = v.T @ Σ_roe @ v
                stdev_h = np.sqrt(max(0, variance_h))
                con_ref_k[j] += param.invICDF * stdev_h

            # propagate one dt_safe
            stm_step = self.f_stm(oe_k, dt_safe)
            stm      = stm_step @ stm
            oe_k     = dyn.propagate_oe(oe_k, dt_safe, J2=dyn.J2)
            psi_k    = self.f_psi(oe_k)
            s_roe    = stm_step @ s_roe
            Σ_roe    = stm_step @ Σ_roe @ stm_step.T + param.QQ
            s_rtn    = psi_k @ s_roe
            Σ_rtn    = psi_k @ Σ_roe @ psi_k.T

        # select worst safe‑slice
        j_star  = np.argmax(con_ref_k)

        return s_roe_ps, s_rtn_ps, Σ_roe_ps, Σ_rtn_ps, jac_k[j_star], con_ref_k[j_star] 

    def _ps_step_ct(self, s_k, a_k, oe_k, DEED_k, chance=False):
        """
        Continuous-time passive safety
        """
        
        def _alpha_interval_and_moments(a0, b, c, eps=1e-14):
            """
            Determine hinge-active set S(τ) = [α1, α2] ⊂ [0,1] where ḡ(α) >= 0 for
            ḡ(α) = 1 - a0 - 2 b α - c α^2 (Eq. 173), and return (I0, I1, I2, J) moments.

            Returns
            -------
            (I0, I1, I2, J): all zeros if empty set.
            """
            # Handle near-linear case robustly
            if abs(c) < eps:
                # print("c ~= 0")
                # Linear: ḡ(α) = 1 - a0 - 2 b α
                g0 = 1.0 - a0
                g1 = 1.0 - a0 - 2.0 * b
                if g0 <= 0.0 and g1 <= 0.0:
                    return 0.0, 0.0, 0.0, 0.0
                if g0 >= 0.0 and g1 >= 0.0:
                    a1, a2 = 0.0, 1.0
                else:
                    # one crossing in (0,1)
                    alpha_star = (1.0 - a0) / (2.0 * b)  # where ḡ=0
                    if g0 >= 0.0:
                        a1, a2 = 0.0, np.clip(alpha_star, 0.0, 1.0)
                    else:
                        a1, a2 = np.clip(alpha_star, 0.0, 1.0), 1.0
                if a2 <= a1 + 1e-16:
                    return 0.0, 0.0, 0.0, 0.0
                # Moments with c=0 (use Eqs. 174–176 with c→0); J reduces accordingly (Eq. 180 with c=0)
                I0 = (1.0 - a0) * (a2 - a1) - b * (a2**2 - a1**2)
                I1 = 0.5 * (1.0 - a0) * (a2**2 - a1**2) - (2.0/3.0) * b * (a2**3 - a1**3)
                I2 = (1.0/3.0) * (1.0 - a0) * (a2**3 - a1**3) - 0.5 * b * (a2**4 - a1**4)
                d0 = (1.0 - a0)   # shorthand
                J = (d0**2) * (a2 - a1) - 2*d0*b * (a2**2 - a1**2) + (4.0/3.0) * (b**2) * (a2**3 - a1**3)
                return I0, I1, I2, max(J, 0.0)
            
            # Concave quadratic (c>0): roots of c α^2 + 2 b α + (a0 - 1) = 0
            D = b**2 + c * (1.0 - a0)
            if D <= 0.0:
                return 0.0, 0.0, 0.0, 0.0  # never positive
            
            sqrt_disc = np.sqrt(D)
            r1 = (-b - sqrt_disc) / c
            r2 = (-b + sqrt_disc) / c
            a1 = np.clip(min(r1, r2), 0.0, 1.0)
            a2 = np.clip(max(r1, r2), 0.0, 1.0)
            if a2 <= a1 + 1e-16:
                return 0.0, 0.0, 0.0, 0.0

            # Eqs. (174)–(176)
            d0 = (1.0 - a0)
            I0 = d0 * (a2 - a1) - b * (a2**2 - a1**2) - (c/3.0) * (a2**3 - a1**3)
            I1 = 0.5 * d0 * (a2**2 - a1**2) - (2.0/3.0) * b * (a2**3 - a1**3) - (c/4.0) * (a2**4 - a1**4)
            I2 = (1.0/3.0) * d0 * (a2**3 - a1**3) - 0.5 * b * (a2**4 - a1**4) - (c/5.0) * (a2**5 - a1**5)

            # Eq. (180) for J(τ)
            J = (d0**2) * (a2 - a1) \
                - 2.0*d0*b * (a2**2 - a1**2) \
                - (2.0/3.0)*d0*c * (a2**3 - a1**3) \
                + (4.0/3.0)*(b**2) * (a2**3 - a1**3) \
                + (b*c) * (a2**4 - a1**4) \
                + (1.0/5.0)*(c**2) * (a2**5 - a1**5)

            return I0, I1, I2, max(J, 0.0)

        B_k = self.f_cim(oe_k)     # (6,3), constant across τ
        psi_k  = self.f_psi(oe_k)
        Gx_acc, Gu_acc = np.zeros(6), np.zeros(3)
        gtilde_acc = 0.0

        # τ-loop (as in your current code)
        for w_tau, tau in zip(self.tau_weights, self.tau_nodes):
            Phi    = self.f_stm(oe_k, tau)  
            oe_tau = dyn.propagate_oe(oe_k, tau, J2=dyn.J2)
            psi    = self.f_psi(oe_tau)          
            psiPhi = psi @ Phi
            Gamma  = psiPhi.T @ DEED_k @ psiPhi; Gamma = 0.5*(Gamma + Gamma.T)  # ensure symmetry

            v0 = s_k
            v1 = B_k @ a_k

            if not chance:
                # ----- Deterministic (analytic α) — your original code -----
                I0, I1, I2, j_local = _alpha_interval_and_moments(v0@(Gamma@v0), v0@(Gamma@v1), v1@(Gamma@v1))

                gx_local = (Gamma @ (v0 * I0 + v1 * I1))
                gu_local = B_k.T @ (Gamma @ (v0 * I1 + v1 * I2))
            
            else: 
                # ----- Chance-constrained path: numeric α quadrature -----
                # initial covariances 
                Σ_nav, Σ_exe   = self._init_cov(s_k, a_k, B_k, psi_k)  
                gx_local, gu_local = np.zeros(6), np.zeros(3)
                j_local  = 0.0

                for w_alpha, alpha in zip(self.alpha_weights, self.alpha_nodes):
                    m = v0 + alpha * v1   

                    # Σ_y(α) = Σx + α^2 B Σu B^T + α (Σxu B^T + B Σxu^T)
                    Σy = Σ_nav + (alpha**2) * Σ_exe; Σy = 0.5 * (Σy + Σy.T)  # ensure symmetry

                    # μ_q, σ_q
                    mu  = m @ (Gamma @ m) + np.trace(Gamma @ Σy)
                    ΓΣy = Gamma @ Σy
                    # σ^2 = 2 tr((ΓΣy)^2) + 4 m^T Γ Σy Γ m
                    sigma2 = 2.0 * np.trace(ΓΣy @ ΓΣy) + 4.0 * (m @ (Gamma @ (Σy @ (Gamma @ m)))); sigma2 = max(sigma2, 0.0)
                    sigma  = max(np.sqrt(sigma2), 1e-12)  # avoid divide-by-zero in gradient

                    h = 1.0 - mu + param.invICDF * sigma
                    if h <= 0.0:   # hinge inactive; only accumulate residual (0)
                        continue
                    j_local += w_alpha * (h * h)

                    # derivatives 
                    # ∂(h^2)/∂m = -4 h Γ m - (8 κ h / σ) Γ Σy Γ m
                    grad_m = h * (Gamma @ m) + (8.0*param.invICDF*h / sigma) * (Gamma @ (Σy @ (Gamma @ m)))
                    gx_local += w_alpha * grad_m
                    gu_local += w_alpha * (B_k.T @ (alpha * grad_m))

            # τ-weight accumulation                
            Gx_acc     += w_tau * gx_local  
            Gu_acc     += w_tau * gu_local
            gtilde_acc += w_tau * j_local

        # Final scaling/sign (kept identical to your deterministic implementation)
        scl = 20.0
        Gx_k   = -4.0 * Gx_acc / scl
        Gu_k   = -4.0 * Gu_acc / scl
        gtilde =  gtilde_acc / scl

        # Return layout preserved for eval_ps unpacking
        # (s_roe_ps, s_rtn_ps, Σ_roe_ps, Σ_rtn_ps, max_constr_ps, jac_ps, con_ref_ps)
        return None, None, None, None, np.hstack((Gx_k, Gu_k)), gtilde

    def eval_ps(self, s_ref, a_ref, t_ref, chance=False, ct=False):
        """
        One‑stop passive‑safety analysis.
        IMPORTANT: to avoid the infeasibility at the pre-defined terminal states, 
        the last element stays at zero (just padded). 
        Args
        ----
            s_ref, a_ref : shape (n_time, 6) and (n_time, 3)
            chance       : include chance‑constraint margin?
        Returns
        -------
            s_roe_ps, s_rtn_ps  : (n_time, n_safe, 6)
            Σ_roe_ps, Σ_rtn_ps  : (n_time, n_safe, 6, 6)   (zeros if chance==False)
            max_constr_ps       : (n_time,)
            c_ps, b_ps          : (n_time, 9), (n_time,)  (None if flag False)
        """
        
        n_time = len(t_ref)
        oe_ref = dyn.propagate_oe(self.oe_i, t_ref, J2=dyn.J2)
        DEED = self.DEED
        
        n_safe = self.n_safe
        # Allocate result tensors ---------------------------------------------------
        s_roe_ps = np.zeros((n_time, n_safe, 6))
        s_rtn_ps = np.zeros((n_time, n_safe, 6))
        Σ_roe_ps = np.zeros((n_time, n_safe, 6, 6))
        Σ_rtn_ps = np.zeros((n_time, n_safe, 6, 6))
        jac_ps        = np.zeros((n_time, 9)) 
        con_ref_ps    = np.zeros(n_time)      
        
        # Loop over control instants ------------------------------------------------
        for k in range(n_time-1):
            if ct:
                step_out = self._ps_step_ct(s_ref[k], a_ref[k], oe_ref[k], DEED[k], chance=chance)
            else:
                step_out = self._ps_step(s_ref[k], a_ref[k], oe_ref[k], DEED[k], chance=chance)
            
            (s_roe_ps[k], s_rtn_ps[k], Σ_roe_ps[k], Σ_rtn_ps[k], jac_ps[k], con_ref_ps[k]) = step_out

        if not chance:  # compress if user didn’t request covariances
            Σ_roe_ps.fill(0.0); Σ_rtn_ps.fill(0.0)

        return s_roe_ps, s_rtn_ps, Σ_roe_ps, Σ_rtn_ps, jac_ps, con_ref_ps
    
    
    def propagate_ct(self, s_ref, a_ref, t_ref, n=50):
        """
        Continuous-time state propagation with finer grids than the original optimization variables (s_ref, a_ref)
        """

        roe_out = np.zeros(((self.n_time-1)*n + 1, 6))
        rtn_out = np.zeros(((self.n_time-1)*n + 1, 6))
        tout = np.zeros(((self.n_time-1)*n + 1,))
        
        oe_ref = dyn.propagate_oe(self.oe_i, t_ref, J2=dyn.J2)

        for k in range(self.n_time-1):

            roe_out[k*n] = s_ref[k]
            rtn_out[k*n] = self.f_psi(oe_ref[k]) @ roe_out[k*n]
            tout[k*n] = t_ref[k]
            sout_kp1 = s_ref[k] + self.f_cim(oe_ref[k]) @ a_ref[k]
            dt_k = (t_ref[k+1] - t_ref[k]) / n
            
            for i in range(1, n):
                stm_i = self.f_stm(oe_ref[k], i * dt_k)
                roe_out[k*n+i] = stm_i @ sout_kp1
                oe_ki = dyn.propagate_oe(oe_ref[k], i * dt_k, J2=dyn.J2)
                rtn_out[k*n+i] = self.f_psi(oe_ki) @ roe_out[k*n+i]
                tout[k*n+i] = t_ref[k] + i * dt_k

        # last step
        roe_out[-1] = s_ref[-1]
        rtn_out[-1] = self.f_psi(oe_ref[-1]) @ roe_out[-1]
        tout[-1] = t_ref[-1]    

        return tout, roe_out, rtn_out 

    def compute_cost(self, action, lib=np):
        """
        Computation of cost function. (np/torch) 
        """
        
        if lib == np:
            if np.shape(action)[0] == 3:
                cost = np.sum(np.linalg.norm(action, axis=0))
            else:
                cost = np.sum(np.linalg.norm(action, axis=1))
        elif lib == torch:
            if action.shape[0] == 3:
                cost = torch.norm(action, dim=0)
            else:
                cost = torch.norm(action, dim=1)            
        else:
            raise ValueError("lib must be either np or torch")
        return cost

    def propagate_rtg(self, rtgs_t, states, actions, t, dt_sec=None, lib=np):
        """
        numpy/torch-friendly RTG propagation.
        Arguments:
            - rtgs_t: (1,) or (B, 1)
            - actions:  (n_u, T) or (B, n_u, T)
        Returns shape matches rtgs_t (batched or unbatched).
        """
        
        torch_mode = getattr(lib, "__name__", "") == "torch"

        # --- pick a_t and rtgs_t as (B, n_dim) tensor/array ---
        # print("rtgs_t: ", rtgs_t, " actions: ", actions)
        assert rtgs_t.ndim + 1 == actions.ndim, "rtgs and actions shape mismatch"
        if actions.ndim == 3:   # (B, T, n_u)
            a_t = actions[:, t]
        else: 
            a_t = actions[None, t]
            rtgs_t = rtgs_t[None, ...]  # (1, 1)
                
        # control delta: per-row via self.compute_cost
        if torch_mode:
            ctrl_delta = -lib.linalg.norm(a_t, dim=1)   # (B,)
        else:
            ctrl_delta = -lib.linalg.norm(a_t, axis=1)   # (B,)

        # --- assemble objective(s) and update ---
        out = rtgs_t - ctrl_delta[:, None]  # (B, 1)

        # --- restore original shape ---
        if actions.ndim == 2:
            return out[0]        # (1,)
        return out               # (B, 1)


    def propagate_ctg(self, ctgs_t, states, actions, current_obs, t, ctg_clipped=True, lib=np):
        """
        (2025/12/14 ... kind of deprecated because we do not need this, as 
         there is no situation we feed ctg > 0 to the transformer...) 
        Propagate ctg by one step.
        Args:
            ctgs_t (B, n_con) or (n_con,) : float ... ctgs at current time step
            states (B, t+1, n_x) or (t+1, n_x) ... state history (one element more than action history) 
            actions (B, t, n_u) or (t, n_u) ... action history  
            env_observation: dict ... current environment observation
            t: int ... current time step
        """     
        
        k = int(t)   # current time step
        device = ctgs_t.device if lib == torch else None
        
        # state/action (B, n_dim) 
        if states.ndim == 2: 
            s_k = states[None,k]  
            a_k = actions[None,k]
        elif states.ndim == 3:
            s_k = states[:, k]
            a_k = actions[:, k]
        else:
            raise ValueError("states/actions must be 2D or 3D arrays")
        
        oe_k = self.oe[:,k] 
        deed_k = self.DEED[k]
        
        if lib == np:
            assert states.ndim == 2, "numpy version only supports single trajectory (as of now) "
            _, _, _, _, worst_constr_k, _, _ = self._ps_step(s_k, a_k, oe_k, deed_k, chance=self.chance)
        elif lib == torch:
            DEED_k_torch = torch.from_numpy(deed_k).float().to(device)        
            _, _, _, _, worst_constr_k = self._ps_step_torch(s_k, a_k, oe_k, DEED_k_torch, chance=self.chance)
        
        if states.ndim == 2: # unbatched 
            viol_t = 1.0 if (worst_constr_k > 0) else 0.0  
            return ctgs_t - (viol_t if (not ctg_clipped) else 0)

        else: # batched
            viol_t = (worst_constr_k > 0).float()       
            if ctg_clipped:
                viol_t = torch.zeros_like(viol_t)
                   
            return ctgs_t - viol_t.unsqueeze(1)   # (B, n_con)
    
    # NOTE; computation of CTG / RTG does not requires np/torch implementation
    def compute_ctg(self, states, action, time, chance=None,  ct=None, n_time=None,):
        """
        Computing CTG, given the trajectory data (states, action).
        Args: 
            states; (n_data) x (n_x) x (n_time)  or  (n_data) x (n_time) x (n_x) 
            action; (n_data) x (n_u) x (n_time)  or  (n_data) x (n_time) x (n_u)
            n_data; int 
            n_time; int 
            chanace; (bool) True if we are considering chance constraint. Default is False (i.e., deterministic constraint )
        Return: 
            ctg        ; (n_data, n_time) ... value of ctg at each time step (for n_data trajectories)
            constr_val ; (n_data, n_time) ... constraint value at each time step (<=0 if satysfying the constraint)
        """
        if n_time is None:
            n_time = self.n_time
        
        if states.ndim == 2:  # (n_data) x (n_time) x (n_x)
            states = states[None, :, :]
            action = action[None, :, :]
            time = time[None, :]
            
        chance = self.chance if chance is None else chance
        ct = self.ct if ct is None else ct
        
        n_data = np.shape(states)[0]    
        ctg        = np.zeros(shape=(n_data, n_time), dtype=float)
        constr_val = np.zeros(shape=(n_data, n_time), dtype=float)
        
        for n in range(n_data):
            # change the array shape to (n_x) x (n_time) and (n_u) x (n_time) 
            if np.shape(states)[1] == 6:    # (n_data, n_x, n_time)
                s = states[n, :, :].T
                a = action[n, :, :].T
            elif np.shape(states)[2] == 6:  # (n_data, n_time, n_x) 
                s = states[n, :, :]
                a = action[n, :, :]
                t = time[n, :]
            else:
                raise ValueError("check the dimension of states/action. Something is wrong")
            
            _, _, _, _, _, h = self.eval_ps(s, a, t, chance=chance, ct=ct)
            flag_violation = np.where(h > 1e-3, 1, 0)  # 1 for positive (i.g., constraint violation), 0 for <= 0 (constraint satisfaction)
            for t in range(n_time):
                ctg[n, t] = np.sum(flag_violation[t:])
            constr_val[n] = h

        return ctg, constr_val
    
    def compute_rtg(self, actions: np.ndarray) -> np.ndarray:
        """
        Compute reward‑to‑go (RTG) and optionally append a time‑to‑go (TTG) row.
        WARNING; only supported for 2D array (single trajectory) for now. 
        Arguments: 
            (n_data, T,  n_action) — single action history,  time‑major
        Returns:
            rtg  — shape (n_data, T) 
        """
        a = np.asarray(actions)
        if a.ndim == 2:
            a = a[None,:,:]  # add batch dim
        if np.shape(a)[1] == 3:    # if (n_data, n_x, n_time), then transpose to (n_data, n_time, n_x)
            a = a[:,:,:].transpose(0, 2, 1)  # (n_data, T, n_u)
        
        # reward‑to‑go 
        norms = np.linalg.norm(a, axis=2)                    # (n_data, T)
        rtg = -np.cumsum(norms[:, ::-1], axis=1)[:, ::-1]    # (n_data, T)
        
        return rtg


def _restore_oec0_modified(oec0_mod: np.ndarray) -> np.ndarray:
    oec0 = np.asarray(dyn.restore_koe(np.asarray(oec0_mod, dtype=float)), dtype=float).reshape(-1)
    if oec0.shape != (6,):
        raise ValueError(f"Restored OE must have shape (6,), got {oec0.shape}")
    return oec0


def _waypoint_times_from_dts(dt_seq, n_time: int):
    total = float(np.sum(dt_seq))
    cum = np.cumsum(list(dt_seq)[:-1]) / total
    idx = np.rint(cum * (int(n_time) - 1)).astype(int)
    idx = np.clip(idx, 1, int(n_time) - 2)

    for i in range(1, len(idx)):
        idx[i] = max(idx[i], idx[i - 1] + 1)

    max_last = int(n_time) - 2
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


def generate_traj_with_wyp(
    x0: np.ndarray,
    x_pred: np.ndarray,
    dt_pred: np.ndarray,
    tof_steps: int,
    koz_dim: np.ndarray,
    artms: np.ndarray,
    dt_sec: float,
    oec0_mod: np.ndarray | None = None,
    obj_type: str = "min_fuel",
):
    n_time = int(tof_steps) + 1
    if n_time < 2:
        return {"status_cvx": "invalid_tof", "status_scp": "invalid_tof"}

    tvec_sec = np.arange(n_time, dtype=float) * float(dt_sec)
    oec0 = _restore_oec0_modified(oec0_mod) if oec0_mod is not None else np.asarray(param.oec0, dtype=float)
    t_idx_wyp = _waypoint_times_from_dts(list(dt_pred), n_time)
    wyp = x_pred[:-1] if len(x_pred) > 1 else np.empty((0, 6))
    goal = x_pred[-1] if len(x_pred) > 0 else x0

    current_obs = {"state": x0, "goal": goal, "ttg": tvec_sec[-1], "dt": dt_sec, "oe": oec0}
    prob = NonConvexOCP(
        prob_definition={
            "t_i": 0,
            "t_f": n_time,
            "tvec_sec": tvec_sec,
            "chance": True,
            "ct": False,
            "current_obs": current_obs,
            "waypoint_times": t_idx_wyp,
            "waypoints": wyp,
            "waypoint_type": "roe",
            "koz_dim": koz_dim,
            "artms_scale_range_1e3": artms,
        }
    )

    sol_dict = {
        "wyp": x_pred,
        "t_idx_wyp": t_idx_wyp,
    }

    sol_cvx = prob.ocp_cvx()
    status_cvx = sol_cvx["status"]
    if status_cvx not in {"optimal", "optimal_inaccurate"}:
        sol_dict.update({"status_cvx": status_cvx, "status_scp": "cvx_failed"})
        return sol_dict
    roe_cvx = sol_cvx["z"]["state"]
    actions_cvx = sol_cvx["z"]["action"]

    prob.zref = {"state": roe_cvx, "action": actions_cvx}
    prob.sol_0 = {"z": prob.zref}
    prob.generate_scaling(roe_cvx, actions_cvx)

    if obj_type == "feasibility":
        prob.type = "feasibility"
        prob._cvx_built_AL = False
        prob.update_flag = True

    sol_scp, _ = solve_scvx(prob)
    status_scp = sol_scp["status"]
    if status_scp not in {"optimal", "optimal_inaccurate"}:
        rtn_cvx = prob.f_2rtn(roe_cvx, dyn.propagate_oe(oec0, tvec_sec))
        rtn_cvx_ct = dyn.propagate_ct(roe_cvx, actions_cvx, dyn.propagate_oe(oec0, tvec_sec), tvec_sec, n=10)
        sol_dict.update(
            {
                "status_cvx": status_cvx,
                "status_scp": status_scp,
                "prob": prob,
                "roe_cvx": roe_cvx,
                "actions_cvx": actions_cvx,
                "rtn_cvx": rtn_cvx,
                "rtn_cvx_ct": rtn_cvx_ct,
            }
        )
        return sol_dict

    roe_scp = sol_scp["z"]["state"]
    actions_scp = sol_scp["z"]["action"]
    oec = dyn.propagate_oe(oec0, tvec_sec)
    rtn_scp = prob.f_2rtn(roe_scp, oec)
    _, _, rtn_scp_ct = dyn.propagate_ct(roe_scp, actions_scp, oec, tvec_sec, n=10)

    sol_dict.update(
        {
            "status_cvx": status_cvx,
            "status_scp": status_scp,
            "prob": prob,
            "roe_scp": roe_scp,
            "actions_scp": actions_scp,
            "rtn_scp": rtn_scp,
            "rtn_scp_ct": rtn_scp_ct,
        }
    )
    return sol_dict
