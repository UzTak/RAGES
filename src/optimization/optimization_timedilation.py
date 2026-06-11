import numpy as np
import scipy as sp 
import cvxpy as cp
import torch 
from multiprocessing import Pool, cpu_count
import time
import sys
from numpy.polynomial.legendre import leggauss

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
        
        self.n_state = param.N_STATE
        self.n_action = param.N_ACTION

        # scp parameters
        self.verbose = prob_definition.get('verbose', False)  # verbose for cvxpy
        self.verbose_scvx = False  # verbose for scvx

        # SVCx parameters (fixed)
        self.scvx_param = SCVxParams()
        self.iter_max = self.scvx_param.iter_max  # param.scp_iter_max
        # parameters (variable) : it will be reset in solve_scvx() anyways so don't worry too much 
        self.pen_λ, self.pen_μ, self.pen_w, self.rk = None, None, None, None
                
        # initial solution
        self.zref = zref
        self.sol_0 = {"z": self.zref}
        self.t_star = None 
        
        self.__setup_problem(prob_definition)
       
        # variable scaling
        if zref is not None:  # (over-riding previous Ds, Da assignment)
            self.generate_scaling(zref['state'], zref['action'])
        
        # in __setup_problem(), we imported self.dt, so we can regenerate the 
        if 'zref' in prob_definition:
            self.tvec_sec_ref = np.cumsum(np.hstack((0, self.zref['dt']))) 
        else:
            self.tvec_sec_ref = np.linspace(self.ti_sec, self.tf_sec, self.n_time)
          
        # in __setup_problem(), we imported self.dt, so we can regenerate the 
        out = self.generate_dynamics(self.oe_i, self.tvec_sec_ref, self.state_def)
        self.stm, self.cim, self.psi = out["stm"], out["cim"], out["psi"]
        self.jac, self.dstm,  self.dcim = out["jac"], out["dstm"], out["dcim"]
        self.oe = out["oe"]  # history of chief oe 
        
        if self.state_def == 'eroe':
            self.f_stm, self.f_cim, self.f_psi, self.f_2rtn = dyn.stm_eroe, dyn.cim_eroe, dyn.mtx_eroe_to_rtn, dyn.eroe_to_rtn
            self.f_jac = dyn.jac_eroe
            self.f_dcim, self.f_dpsi = dyn.dcim_eroe, dyn.dpsi_eroe
        elif self.state_def == 'roe':
            self.f_stm, self.f_cim, self.f_psi, self.f_2rtn = dyn.stm_roe, dyn.cim_roe, dyn.mtx_roe_to_rtn, dyn.roe_to_rtn   
            self.f_jac = dyn.jac_roe
            self.f_dcim, self.f_dpsi = dyn.dcim_roe, dyn.dpsi_roe
                
        self.update_flag = True        
                

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

        self.state_def = prob_definition.get('state_def', 'roe')  # 'roe' or 'eroe'    
        
        self.ti = prob_definition['t_i']   # index, not the actual time in seconds.
        self.tf = prob_definition['t_f']
        self.ti_sec = prob_definition['tvec_sec'][self.ti] 
        self.tf_sec = prob_definition['tvec_sec'][self.tf-1]
        self.n_time = prob_definition['t_f'] - prob_definition['t_i']
        
        self.chance = prob_definition.get('chance', False)
        self.ct = prob_definition.get('ct', 0)  # 0: no ct, 1: ct for full-burn only, 2: ct for alpha \in [0,1] (imperfect burns) 
                
        self.λ_bin = prob_definition['λ_bin'] if 'λ_bin' in prob_definition else 1000.0  # penalty weight for binary relaxation
        self.large_burn_idx = prob_definition['large_burn_idx'] if 'large_burn_idx' in prob_definition else None
        
        # default: load the safety paraemters for passive safety 
        self.n_safe = param.n_safe
        self.dt_safe_sec = param.dt_safe_sec
        self.t_safe_sec = self.n_safe * self.dt_safe_sec 
        self.tvec_safe_sec = np.linspace(0, self.t_safe_sec, self.n_safe)

        # scaling factors. Each element of Ds and Da is expected magnitudes of values of the variables.
        self.Ds = prob_definition['Ds'] if 'Ds' in prob_definition else np.eye(param.N_STATE)
        self.Da = prob_definition['Da'] if 'Da' in prob_definition else np.eye(param.N_ACTION)
        self.invDs = np.diag(1/np.diag(self.Ds))
        self.invDa = np.diag(1/np.diag(self.Da))
        self.cs = None 
        self.ca = None 
        self.Dt = self.tf_sec
        
        # waypoints: optional waypoint constraints
        self.waypoints = prob_definition.get('waypoints', np.empty(0))  # List of waypoint states
        self.waypoint_times = prob_definition.get('waypoint_times', np.empty(0))  # List of waypoint time indices
        self.waypoint_type = prob_definition.get('waypoint_type', 'roe')  # Type of waypoints: "roe" or "rtn"        
        
        # self.free_time = prob_definition['free_time'] if 'free_time' in prob_definition else True
        self.passive_safety = prob_definition.get('passive_safety', True)
        self.flag_ps_efficient = prob_definition.get('flag_ps_efficient', True)
        
        # Gauss–Legendre nodes/weights on [0, τ_s]
        if self.ct:
            # tau discretization 
            self.n_tau = 25  
            xi, wi = leggauss(self.n_tau)          # nodes/weights on [-1,1]
            self.tau_nodes = 0.5 * self.t_safe_sec * (xi + 1.0)
            self.tau_weights = 0.5 * self.t_safe_sec * wi
            
            # alpha discretization 
            if self.ct == 2:
                self.n_alpha = 10
                xi_a, wi_a = leggauss(self.n_alpha)     # nodes/weights on [-1,1]
                self.alpha_nodes = 0.5 * (xi_a + 1.0)   # nodes on [0,1]
                self.alpha_weights = 0.5 * wi_a
    
    
    @staticmethod
    def generate_dynamics(oe_0, tvec_sec, state='roe', J2=dyn.J2):
        """
        linearize the nonlinear dynamics of 6DoF relative motion 
        input: 
            oec0: intial OE (chief); (6,)
            tvec_sec: time vector; (n_time,)
        return:
            stm: state transition matrix;                  n_time x 6 x 6 
            cim: control input matrix;                      n_time x 6 x 3
            psi: ROE -> RTN map;                            (n_time+1) x 6 x 6
            oe: Keplarian orbital elements (chief) history  n_time x 6
            dt: time step
        """
        
        n_time = len(tvec_sec)
        
        if state == 'eroe':
            f_stm, f_cim, f_psi, f_jac = dyn.stm_eroe, dyn.cim_eroe, dyn.mtx_eroe_to_rtn, dyn.jac_eroe
            f_dcim = dyn.dcim_eroe
        elif state == 'roe':
            f_stm, f_cim, f_psi, f_jac = dyn.stm_roe, dyn.cim_roe, dyn.mtx_roe_to_rtn, dyn.jac_roe
            f_dcim = dyn.dcim_roe 

        stm = np.empty(shape=(n_time-1, 6, 6), dtype=float)  
        jac = np.empty(shape=(n_time, 6, 6), dtype=float)
        cim = np.empty(shape=(n_time, 6, 3), dtype=float)
        psi = np.empty(shape=(n_time, 6, 6), dtype=float)
        oe  = np.empty(shape=(n_time, 6), dtype=float)
        dcim = np.empty(shape=(n_time, 6, 3), dtype=float) if f_dcim is not None else None
        dstm = np.empty(shape=(n_time-1, 6, 6), dtype=float)  # derivative of the state transition matrix

        oe[0] = oe_0
        cim[0] = f_cim(oe[0])
        psi[0] = f_psi(oe[0])
        jac[0] = f_jac(oe[0], 0.0, J2=J2)  # instanteneous Jacobian at t=tvec_sec[i]
        dcim[0] = f_dcim(oe[0])
    
        # finite difference for the state transition matrix
        eps = 1e-10
        for iter in range(n_time-1):
            dt = tvec_sec[iter+1]-tvec_sec[iter]
            stm[iter] = f_stm(oe[iter], dt, J2=J2) 
            stm_fwd  = f_stm(oe[iter], dt+eps, J2=J2)  # forward state transition matrix
            stm_back = f_stm(oe[iter], dt-eps, J2=J2)  # backward state transition matrix
            dstm[iter] = (stm_fwd - stm_back) / (2*eps)  # 6x6

            oe[iter+1] = dyn.propagate_oe(oe[iter], dt, J2=J2)
            cim[iter+1] = f_cim(oe[iter+1])        
            psi[iter+1] = f_psi(oe[iter+1])  
            jac[iter+1] = f_jac(oe[iter+1], 0.0, J2=J2)            
            dcim[iter+1] = f_dcim(oe[iter+1])
            
        out = {'stm': stm, 'cim': cim, 'psi': psi, 'oe': oe, 'jac': jac, 'dcim': dcim, 'dstm': dstm}
        return out 
    

    def generate_scaling(self, sref, aref, scale=(-1.0, 1.0), eps=1e-4):
        """
        Build an 'unscale' map centered at the mean:
            s = cs + s_scl @ Ds,  a = ca + a_scl @ Da

        Args:
            sref, aref: arrays of shape (N_time, nx)
            scale: (lo, hi) target interval for the scaled variables (default: [-1, 1])
            eps: small floor to avoid zero/inf scales
        """
        lo, hi = scale
        assert hi > lo, "scale must satisfy hi > lo"

        def _compute(ref):
            mu = np.mean(ref, axis=0)
            r = np.max(np.abs(ref - mu[None]), axis=0)  # symmetric radius
            d = np.maximum(r / max(hi, -lo), eps)       # ensure coverage
            return mu, d

        mu_s, d_s = _compute(sref)
        mu_a, d_a = _compute(aref)

        # store diagonal matrices (and diagonals for convenience)
        self.Ds = np.diag(d_s)
        self.Da = np.diag(d_a)
        self.invDs = np.diag(1/d_s)
        self.invDa = np.diag(1/d_a)
        self.cs = mu_s
        self.ca = mu_a
        self.Dt = self.tf_sec 
        self.invDt = 1/self.Dt

    def ocp_cvx(self):
        # def ocp_cvx_AL(prob, verbose=False):
        """
        Classic Two point boundary value problem (TPBVP) formulation of the OCP.    
        """
        
        n_time = self.n_time  
        s0  = self.state_init
        sf  = self.state_final
        umax = param.u_max  # maximum actuation in non-dimensional units
            
        # non-dimensional variables
        s_scl = cp.Variable((n_time, param.N_STATE))
        a_scl = cp.Variable((n_time, param.N_ACTION))       
        
        # "real" (physical) values 
        s = cp.multiply(s_scl, np.diag(self.Ds)) 
        a = cp.multiply(a_scl, np.diag(self.Da))
        z = {"state": s, "action": a,}    
         
        con = []    
        con += [s[0] == s0]
        con += [s[i+1] == self.stm[i] @ (s[i] + self.cim[i] @ a[i]) for i in range(n_time-1)]
        con += [s[-1] + self.cim[-1] @ a[-1] == sf]
        con += [cp.norm(a[i]) <= umax for i in range(n_time)]  # actuation constraint

        for _, (waypoint_time, waypoint_state) in enumerate(zip(self.waypoint_times, self.waypoints)):
            if self.waypoint_type == 'roe':
                con += [s[waypoint_time] == waypoint_state]
            elif self.waypoint_type == 'rtn':  # Only position components for RTN
                wyp = self.psi[waypoint_time] @ waypoint_state
                con += [s[waypoint_time][:3] == wyp[:3]]  
                
        f0 = self.compute_f0(z)
        cost = f0 
        
        # Compute Cost    
        p = cp.Problem(cp.Minimize(cost), con)
        try:
            p.solve(solver=cp.CLARABEL, verbose=self.verbose)
            status = p.status
            # print("Physical optimal final:", s[:,-1].value + self.cim[:,:,-1] @ a[:,-1].value)        
        except:
            status = 'infeasible'
        z_opt  = {"state": s.value, "action": a.value, "dt": np.array([self.tf_sec/(n_time-1)] * (n_time-1))}  # dt is not used in this problem formulation
        f0_opt = f0.value
        L_opt  = p.value
        
        sol = {"z": z_opt, "status": status, "L": L_opt, "f0": f0_opt, "cost": f0_opt}
        
        return sol
    
    def ocp_cvx_AL(self):
        """
        Passive safety with time dilation
        Using t_k as a decision variables
        """
        n_time = self.n_time
        sref, aref, τref = self.zref['state'], self.zref['action'], self.zref['dt']
        tref = np.hstack((0, np.cumsum(τref)))
        
        s0 = self.state_init
        sf = self.state_final

        # non-dimensional variables
        s_scl = cp.Variable((n_time, param.N_STATE))
        a_scl = cp.Variable((n_time, param.N_ACTION))       
        τ_scl = cp.Variable((n_time-1,), nonneg=True)  
        t_scl = cp.cumsum(cp.hstack([0, τ_scl]))  # cumulative time dilation in non-dimensional units

        # "real" (physical) values 
        s = cp.multiply(s_scl, np.diag(self.Ds)) 
        a = cp.multiply(a_scl, np.diag(self.Da))
        τ = τ_scl * self.Dt
        t = t_scl * self.Dt
        z = {"state": s, "action": a, "dt": τ}      
        slack_eq = cp.Variable((self.n_time-1, param.N_STATE))  # slack variable for nonconvex equality constraints
        
        con = []
        # convex equality constraints
        con += [s[0] == s0 ]
        con += [sf == (s[-1] + self.cim[-1] @ a[-1])]  # terminal state with equality
        con += [cp.sum(τ) == self.tf_sec]  # terminal time (if necessary, you may try inequality <= 1 as well)

        # convex inequality constraints
        # con += [τ[i] >= 0.1 * self.tf_sec for i in range(n_time-1)]  
        
        # nonconvex equality constraints
        for i in range(n_time - 1):
            x_i_postburn = sref[i] + self.cim[i] @ aref[i]
            # jacobian w.r.t. [x_k, x_k+1, u_k, t_k, t_k+1]
            jac_i = np.hstack((
                -self.stm[i], 
                np.eye(6), 
                -self.stm[i] @ self.cim[i], 
                np.reshape(self.stm[i] @ (self.At[i] @ x_i_postburn - self.dcim[i] @ aref[i]), (6,1)), 
                np.reshape(-self.At[i+1] @ self.stm[i] @ x_i_postburn, (6,1))
            ))
            var_i = cp.hstack([
                s[i]   - sref[i], 
                s[i+1] - sref[i+1], 
                a[i]   - aref[i],
                t[i]   - tref[i], 
                t[i+1] - tref[i+1]
            ])
            con_ref_i = sref[i+1] - self.stm[i] @ x_i_postburn 
            con += [con_ref_i + jac_i @ var_i == slack_eq[i]]  
        
        # nonconvex inequality constraints 
        if self.passive_safety:
            slack_ieq = cp.Variable((n_time,), nonneg=True)
            for i in range(self.n_time-1):  # do not consider the last one
                jac_i = self.jac_ps[i]  
                var_i = cp.hstack([s[i] - sref[i], 
                                   a[i] - aref[i], 
                                   t[i] - tref[i]])
                con_ref_i = self.con_ref_ps[i]
                con  += [con_ref_i + jac_i @ var_i <= slack_ieq[i]]  # nonconvex inequality constraints
        else:
            slack_ieq = cp.Variable(n_time, nonneg=True)
            con += [slack_ieq[i] == 0.0 for i in range(n_time)]

        # ---------- trust‑region constraints -----------------------------------
        con += [cp.norm(cp.reshape((s - sref) @ self.invDs, (-1,), 'F'), 'inf') <= self.rk]
        con += [cp.norm(cp.reshape((a - aref) @ self.invDa, (-1,), 'F'), 'inf') <= self.rk]
        con += [cp.norm(self.invDt * (τ - τref), 'inf') <= self.rk / 20]
        
        f0 = self.compute_f0(z)
        P = self.compute_P(cp.vec(slack_eq,'F'), cp.vec(slack_ieq,'F'))  
        p = cp.Problem(cp.Minimize(f0 + P), con)
        p.solve(solver=cp.CLARABEL, verbose=self.verbose)
        z_opt  = {"state": s.value, "action": a.value, "dt": τ.value}

        # print("max slack_eq_opt: ", np.max(slack_eq.value), "max slack_ieq_opt: ", np.max(slack_ieq.value))
        sol = {"z": z_opt, "ξ": cp.vec(slack_eq,'F').value, "ζ": cp.vec(slack_ieq,'F').value,  
               "status": p.status, "f0": f0.value, "P": P.value, "L": f0.value + P.value,
               "cost": f0.value}
                
        return sol  
    
    
    #### SCVX ROUTINE FUNCTIONS ##########################################

    def solve_cvx_AL(self):  
        """
        Solving the convexified problem. 
        You may add any convexification process here (e.g., comptuation of state transition matrix in the nonlinear dynamics...).
        """
        sref = self.zref['state']
        aref = self.zref['action']
        tref = np.cumsum(np.hstack((0,self.zref['dt'])))
        out = self.generate_dynamics(self.oe_i, tref, state=self.state_def, J2=dyn.J2)
        self.stm, self.cim, self.At, self.psi = out["stm"], out["cim"], out['jac'], out["psi"]
        self.jac, self.dstm, self.dcim = out["jac"], out["dstm"], out["dcim"]
        
        # Option 2; use t_k as a decision variable 
        if self.update_flag:
            self.jac_ps, self.con_ref_ps = self.jac_ps_tmp, self.con_ref_ps_tmp
        sol = self.ocp_cvx_AL()  
        
        return sol    
    
    def compute_f0(self, z):
        """
        Objective function (written in CVXPY)
        """
        
        a = z["action"]
        f = cp.sum(cp.norm(a, 2, axis=0)) 
        
        return f
    
    def compute_P(self, g, h):
        """
        Compute the (convex) penalty term for the argumented Lagrangian (written in CVXPY).
        NO NEED TO CHANGE THIS FUNCTION.
        """
        
        zero = cp.Constant((np.zeros(h.shape)))
        hp = cp.maximum(zero, h)
    
        P = self.pen_λ.T @ g + self.pen_μ.T @ h + self.pen_w/2 * (cp.norm(g)**2 + cp.norm(hp)**2)
        
        return P

    def compute_g(self, z):
        """ 
        Returning nonconvex equality g and its gradient dg (written in NUMPY)
        """
        
        s, a = z['state'], z['action']
        tvec_sec = np.concatenate(([0], np.cumsum(z['dt'])))
        out = self.generate_dynamics(self.oe_i, tvec_sec, state=self.state_def, J2=dyn.J2)
        # self.stm, self.cim, self.At, self.psi = out["stm"], out["cim"], out['jac'], out["psi"]
        # self.jac, self.dstm, self.dcim = out["jac"], out["dstm"], out["dcim"]
        stm, cim = out["stm"], out["cim"]

        g = np.zeros((self.n_time-1, 6))
        for i in range(self.n_time-1):
            g[i] = s[i+1] - stm[i] @ (s[i] + cim[i] @ a[i])
        g = g.flatten(order='F')
                
        dg = None 
        return g, dg

    def compute_h(self, z):
        """
        Return inequality h and dh (written in NUMPY)
        """
        s, a, dt = z['state'], z['action'], z['dt']
        t = np.concatenate(([0], np.cumsum(dt)))
        _,_,_,_, self.jac_ps_tmp, self.con_ref_ps_tmp = self.eval_ps(s, a, t, chance=self.chance, ct=self.ct)
        h = self.con_ref_ps_tmp.flatten(order='F')  
        
        if not self.passive_safety:
            h = np.zeros_like(h)  # no passive safety constraint
        
        dh = None  
                    
        return h, dh
    
    ##############################################################
    ##### Constraint / Reward Evaluation #########################
    ##############################################################
    

    def _ps_step(self, s_k, a_k, oe_k, DEED_k, chance=False):
        """
        Single-step passive safety propagation + linearization support
        for a given (s_k, a_k, oe_k, DEED_k).
        """

        dt_safe = self.dt_safe_sec
        n_safe  = self.n_safe

        # init STM, orbit, transforms
        stm_kj = np.identity(6)

        psi_kj = self.f_psi(oe_k)
        cim_k  = self.f_cim(oe_k)
        dcim_k = self.f_dcim(oe_k)

        # post-burn state
        s_roe_kj = s_k + cim_k @ a_k
        s_roe_k0 = s_roe_kj.copy()
        s_rtn_kj = self.f_2rtn(s_roe_kj, oe_k)

        # actuation error covariance
        if param.use_gates_model:
            UU_j = dyn.dv_cov_gates(a_k, param.sigma_gates)
        else:
            UU_j = dyn.dv_cov_simple(a_k)

        if param.use_nav_artms:
            Σ_nav_roe = dyn.rel_nav_artms_roe(np.linalg.norm(s_rtn_kj[:3]))
        else:
            Sigma_nav_rtn = param.Sigma_nav_digital_rtn
            invpsi_j = np.linalg.inv(psi_kj)
            Σ_nav_roe = invpsi_j @ Sigma_nav_rtn @ invpsi_j.T

        Σ_roe_kj = Σ_nav_roe + cim_k @ UU_j @ cim_k.T
        Σ_roe_k0 = Σ_roe_kj.copy()

        psi_stm_kj = psi_kj @ stm_kj
        A0 = self.f_jac(oe_k)

        # storage
        s_roe_ps = np.zeros((n_safe, 6), dtype=float)
        s_rtn_ps = np.zeros((n_safe, 6), dtype=float)
        Σ_roe_ps = np.zeros((n_safe, 6, 6), dtype=float)
        Σ_rtn_ps = np.zeros((n_safe, 6, 6), dtype=float)
        con_ref_k = np.empty((n_safe,), dtype=float)
        jac_k = np.empty((n_safe, 10), dtype=float)  # [s (6), a (3), t (1)]

        for j in range(n_safe):
            # store current step
            s_roe_ps[j] = s_roe_kj
            s_rtn_ps[j] = s_rtn_kj
            Σ_roe_ps[j] = Σ_roe_kj
            Σ_rtn_ps[j] = psi_kj @ Σ_roe_kj @ psi_kj.T

            # quadratic form
            M_kj = psi_stm_kj.T @ DEED_k @ psi_stm_kj
            q_val = (s_roe_k0.T @ M_kj @ s_roe_k0).item()
            h_det = 1.0 - q_val

            # linearization w.r.t. (s_k, a_k, t_k)
            dpsi_kj = self.f_dpsi(oe_k)
            A_kj    = self.f_jac(oe_k)
            dpsi_stm_kj = dpsi_kj @ stm_kj + psi_kj @ (A_kj @ stm_kj - stm_kj @ A0)
            dM_kj = psi_stm_kj.T @ DEED_k @ dpsi_stm_kj + dpsi_stm_kj.T @ DEED_k @ psi_stm_kj

            d_h_ds = -2.0 * (M_kj @ s_roe_k0)
            d_h_da = -2.0 * (cim_k.T @ M_kj @ s_roe_k0)

            v_t = dcim_k @ a_k
            dt_term = -2.0 * v_t.T @ (M_kj @ s_roe_k0) - s_roe_k0.T @ dM_kj @ s_roe_k0

            jac_k[j] = np.hstack((d_h_ds, d_h_da, np.array([dt_term])))

            # chance constraint margin
            if chance:
                var_h = jac_k[j, :6].T @ Σ_roe_k0 @ jac_k[j, :6]
                stdev_h = np.sqrt(max(0.0, var_h))
                h_det += param.invICDF * stdev_h

            con_ref_k[j] = h_det

            # propagate to next safeguard step
            stm_step = self.f_stm(oe_k, dt_safe)
            stm_kj   = stm_step @ stm_kj
            oe_k    = dyn.propagate_oe(oe_k, dt_safe, J2=dyn.J2)
            psi_kj   = self.f_psi(oe_k)
            psi_stm_kj = psi_kj @ stm_kj

            s_roe_kj = stm_step.dot(s_roe_kj)
            s_rtn_kj = psi_kj @ s_roe_kj
            Σ_roe_kj = stm_step @ Σ_roe_kj @ stm_step.T + param.QQ

        if not chance:
            Σ_roe_ps[...] = 0.0
            Σ_rtn_ps[...] = 0.0

        j_star = int(np.argmax(con_ref_k))

        return s_roe_ps, s_rtn_ps, Σ_roe_ps, Σ_rtn_ps, jac_k[j_star], con_ref_k[j_star]
    
    def _ps_step_ct(self, s_k, a_k, oe_k, DEED_k, chance=False):
        """
        Continuous-time passive safety at a given node (s_k, a_k, oe_k). 
        """
        
        B_k = self.f_cim(oe_k)     # (6,3), constant across τ
        psi_k  = self.f_psi(oe_k)
        Gx_acc, Gu_acc, Gt_acc = np.zeros(6), np.zeros(3), 0.0
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
            z = v0 + v1

            if self.ct == 1: # ct-passive safety for α = 1 only
                if not chance:
                    # only check if alpha = 1 is hinge-active
                    g_tau = 1.0 - z @ (Gamma @ z)
                    if g_tau > 0: 
                        g_plus  = g_tau            # hinge
                        j_local = g_plus**2        # |g|_+^2 integrand

                        Gamma_z  = Gamma @ z
                        gx_local = g_plus * Gamma_z
                        gu_local = g_plus * (B_k.T @ Gamma_z)
                        
                        # dz/dt_k = dB/dt_k @ a_k (evaluate at t_k, not oe_tau)
                        dz_dt = self.f_dcim(oe_k) @ a_k    # (6,)

                        # dΦ/dt_k = A(t_k+τ) Φ - Φ A(t_k)
                        A_tk     = self.f_jac(oe_k)
                        A_tk_tau = self.f_jac(oe_tau)
                        dPhi_dtk = A_tk_tau @ Phi - Phi @ A_tk

                        # dψ/dt at t_k+τ
                        dpsi_dt = self.f_dpsi(oe_tau)
                        # y = ψ Φ z;  dy/dt_k = (dψ/dt Φ + ψ dΦ/dt_k) z + ψ Φ dz/dt_k
                        dy_dt = (dpsi_dt @ Phi + psi @ dPhi_dtk) @ z + psiPhi @ dz_dt
                        gt_local = g_plus * dy_dt @ DEED_k @ psiPhi @ z
                        
                    else:
                        j_local = 0.0
                        gx_local = np.zeros(6)
                        gu_local = np.zeros(3)
                        gt_local = 0.0
                else:
                    NotImplementedError("Chance-constrained continuous-time passive safety for α=1 not implemented yet.")
                        
            elif self.ct == 2:  # ct-passive safety for all α ∈ [0,1]

                NotImplementedError("Continuous-time passive safety for all α ∈ [0,1] not implemented yet.")

                # if not chance:
                #     # still you can use this to get gx and gu, but gt is not implemented yet
                #     I0, I1, I2, j_local = _alpha_interval_and_moments(v0@(Gamma@v0), v0@(Gamma@v1), v1@(Gamma@v1))
                #     gx_local = (Gamma @ (v0 * I0 + v1 * I1))
                #     gu_local = B_k.T @ (Gamma @ (v0 * I1 + v1 * I2))

                # else: 
                #     # ----- Chance-constrained path: numeric α quadrature -----
                #     # initial covariances 
                #     Σ_nav, Σ_exe   = self._init_cov(s_k, a_k, B_k, psi_k)  
                #     gx_local, gu_local = np.zeros(6), np.zeros(3)
                #     j_local  = 0.0

                #     for w_alpha, alpha in zip(self.alpha_weights, self.alpha_nodes):
                #         m = v0 + alpha * v1   

                #         # Σ_y(α) = Σx + α^2 B Σu B^T + α (Σxu B^T + B Σxu^T)
                #         Σy = Σ_nav + (alpha**2) * Σ_exe; Σy = 0.5 * (Σy + Σy.T)  # ensure symmetry

                #         # μ_q, σ_q
                #         mu  = m @ (Gamma @ m) + np.trace(Gamma @ Σy)
                #         ΓΣy = Gamma @ Σy
                #         # σ^2 = 2 tr((ΓΣy)^2) + 4 m^T Γ Σy Γ m
                #         sigma2 = 2.0 * np.trace(ΓΣy @ ΓΣy) + 4.0 * (m @ (Gamma @ (Σy @ (Gamma @ m)))); sigma2 = max(sigma2, 0.0)
                #         sigma  = max(np.sqrt(sigma2), 1e-12)  # avoid divide-by-zero in gradient

                #         h = 1.0 - mu + param.invICDF * sigma
                #         if h <= 0.0:   # hinge inactive; only accumulate residual (0)
                #             continue
                #         j_local += w_alpha * (h * h)

                #         # derivatives 
                #         # ∂(h^2)/∂m = -4 h Γ m - (8 κ h / σ) Γ Σy Γ m
                #         grad_m = h * (Gamma @ m) + (8.0*param.invICDF*h / sigma) * (Gamma @ (Σy @ (Gamma @ m)))
                #         gx_local += w_alpha * grad_m
                #         gu_local += w_alpha * (B_k.T @ (alpha * grad_m))
        
            # τ-weight accumulation                
            Gx_acc     += w_tau * gx_local  
            Gu_acc     += w_tau * gu_local
            Gt_acc     += w_tau * gt_local
            gtilde_acc += w_tau * j_local

        # Final scaling/sign (kept identical to your deterministic implementation)
        scl = 20.0
        Gx_k   = -4.0 * Gx_acc / scl
        Gu_k   = -4.0 * Gu_acc / scl
        Gt_k   = -4.0 * Gt_acc / scl
        gtilde = gtilde_acc / scl

        return None, None, None, None, np.hstack((Gx_k, Gu_k, Gt_k)), gtilde
    
    def eval_ps(self, s_ref, a_ref, t_ref, chance=False, ct=False):
        """
        The evaluation of passive safety constraint / computation of ONE SINGLE trajectory.
        Args: 
            s_ref: (n_x, n_time) array
            a_ref: (n_u, n_time) array  
            n_time: int
            horizon_safe: 
            n_safe: int 
            chance: True if chance constraint is considered. 
        Return; 
            states_lvlh_ps : (6, n_time, n_safe)     ... hisotry of the mean of the free-drift trajectory at each controlled state 
            Sigma_lvlh_ps  : (6, 6, n_time, n_safe) ... hisotry of Covaraince of the free-drift trajectory at each constrained state (returns zero matrix if chance=False)
            min_constr_ps  : n_time ... hisotry of constraint values 
        """
        
        n_safe = self.n_safe  
        N = self.n_time
        oe_ref = dyn.propagate_oe(self.oe_i, t_ref, J2=dyn.J2)  # chief orbit elements at each time step
         
        s_roe_ps = np.zeros(shape=(N, n_safe, 6), dtype=float)
        s_rtn_ps = np.zeros(shape=(N, n_safe, 6), dtype=float)
        Σ_roe_ps = np.zeros(shape=(N, n_safe, 6, 6), dtype=float)
        Σ_rtn_ps = np.zeros(shape=(N, n_safe, 6, 6), dtype=float)
        jac_ps        = np.zeros((N, 10)) 
        con_ref_ps    = np.zeros(N)  
            
        for k in range(self.n_time-1):
            if ct: 
                s_roe_ps_k, s_rtn_ps_k, Σ_roe_ps_k, Σ_rtn_ps_k, jac_ps_k, con_ref_ps_k =\
                    self._ps_step_ct(s_ref[k], a_ref[k], oe_ref[k], param.DEED[k], chance=chance)
            else:
                s_roe_ps_k, s_rtn_ps_k, Σ_roe_ps_k, Σ_rtn_ps_k, jac_ps_k, con_ref_ps_k =\
                    self._ps_step(s_ref[k], a_ref[k], oe_ref[k], param.DEED[k], chance=chance)
                    
            s_roe_ps[k] = s_roe_ps_k
            s_rtn_ps[k] = s_rtn_ps_k
            Σ_roe_ps[k] = Σ_roe_ps_k
            Σ_rtn_ps[k] = Σ_rtn_ps_k
            jac_ps[k], con_ref_ps[k] = jac_ps_k, con_ref_ps_k
        
        return s_roe_ps, s_rtn_ps, Σ_roe_ps, Σ_rtn_ps, jac_ps, con_ref_ps
    
    def propagate_ct(self, s_ref, a_ref, t_ref, n=50):
        """
        Continuous-time state propagation with finer grids than the original optimization variables (s_ref, a_ref)
        """
        roe_out = np.zeros(((self.n_time-1)*n + 1, 6))
        rtn_out = np.zeros(((self.n_time-1)*n + 1, 6))
        tout = np.zeros(((self.n_time-1)*n + 1,))

        oe_ref = dyn.propagate_oe(self.oe_i, t_ref, J2=dyn.J2)  # chief orbit elements at each time step

        for k in range(self.n_time-1):

            roe_out[k*n] = s_ref[k]
            rtn_out[k*n] = self.f_psi(oe_ref[k]) @ roe_out[k*n]
            tout[k*n] = t_ref[k]
            sout_kp1 = s_ref[k] + self.f_cim(oe_ref[k]) @ a_ref[k]
            dt_k = (t_ref[k+1] - t_ref[k]) / n
            
            oe_k0 = oe_ref[k].copy()

            for i in range(1, n):
                stm_i = self.f_stm(oe_k0, i * dt_k)
                roe_out[k*n+i] = stm_i @ sout_kp1
                oe_ki = dyn.propagate_oe(oe_k0, i * dt_k, J2=dyn.J2)
                rtn_out[k*n+i] = self.f_psi(oe_ki) @ roe_out[k*n+i]
                tout[k*n+i] = t_ref[k] + i * dt_k
            
            oe_curr = dyn.propagate_oe(oe_k0, t_ref[k+1]-t_ref[k], J2=dyn.J2)
        # last step
        roe_out[-1] = s_ref[-1]
        rtn_out[-1] = self.f_psi(oe_curr) @ roe_out[-1]
        tout[-1] = t_ref[-1]

        return tout, roe_out, rtn_out
