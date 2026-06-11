"""
SCVx* algorithm, a convergence-guaranteed SCP algorithm for non-convex optimal control problems.
Reference: Oguri, "Successive Convexification with Feasibility Guarantee via Augmented Lagrangian for Non-Convex Optimal Control Problems" (2024)
Through a rigorous defintion of the trust region and penalty weight, the algorithm guarantees the convergence to a stationary point of the original problem.
"""

import numpy as np 
import scipy as sp 
import time 

# hyperparameter set 
class SCVxParams: 
    def __init__(self):
        self.iter_max = 100
        self.α = np.array([2, 2])
        self.β = 1.5
        self.γ = 0.9
        self.ρ = np.array([0.0, 0.25, 0.7])
        self.r_minmax = np.array([1e-6, 10])
        self.r0 = 0.5
        self.w0 = 10
        self.ϵopt  = 1e-3 
        self.ϵfeas = 1e-3 
        self.wmax = 1e9

# def scvx_update_weights(z, prob,β):
def scvx_update_weights(prob,β):
        
    prob.pen_λ += prob.pen_w * prob.gref
    prob.pen_μ += prob.pen_w * prob.href
    prob.pen_μ[prob.pen_μ < 0.0] = 0
    prob.pen_w *= β
    
    return prob 


def scvx_update_delta(δ, γ, ΔJ):
    return γ * δ if δ < 1e10 else abs(ΔJ) 


def scvx_update_r(r, α, ρ, ρk, r_minmax):
    α1, α2 = α
    _, ρ1, ρ2 = ρ   # ρ0 < ρ1 < ρ2
    r_min, r_max = r_minmax
     
    if ρk < ρ1:
        return np.max([r/α1, r_min])
    elif ρk < ρ2:
        return r    # no change in the trust region 
    else:
        return np.min([α2*r, r_max])


def scvx_compute_dJdL(sol, sol_prev, prob, iter):
    """
    a dictionary "z" will contain all regualr and slack variables that are optimzied in the problem. 
    """
        
    g_sol, dg_sol = prob.compute_g(sol["z"])
    h_sol, dh_sol = prob.compute_h(sol["z"])
        
    J0 = prob.compute_f0(sol["z"]).value      + prob.compute_P(g_sol,     h_sol).value  
    J1 = prob.compute_f0(sol_prev["z"]).value + prob.compute_P(prob.gref, prob.href).value  
    L  = prob.compute_f0(sol["z"]).value      + prob.compute_P(sol["ξ"],   sol["ζ"]).value 
    
    ΔJ = J1 - J0 
    ΔL = J1 - L 
    
    h_p = np.array([ele if ele >= 0 else 0 for ele in h_sol]) 
    χ = np.linalg.norm(np.hstack((g_sol, h_p)))
    
    # assert np.abs(L - sol["L"]) < 1e-6, "L must be equal to the one computed in cvxpy! "    
    
    if ΔL < 0 and iter!=0 and prob.verbose_scvx:
        # raise ValueError("ΔL must be positive! ")
        print("WARNING: ΔL must be positive! ")
        
    return ΔJ, ΔL, χ, g_sol, dg_sol, h_sol, dh_sol


# main routine 
def solve_scvx(prob):
    """
    General routine for SCvx*.
    Oguri, "Successive Convexification with Feasibility Guarantee via Augmented Lagrangian for Non-Convex Optimal Control Problems" (2023).  

    input variables: 
        z: optimization variables
        r: trust region
        w: penalty weight
        ϵopt: optimality tolerance
        ϵfeas: feasibility tolerance
        ρ0, ρ1, ρ2: trust region thresholds   
        α1, α2, β, γ: trust region / weight update parameters
        sol_0: initial solution dict
    return: 
        zopt: optimal solution
        log: log dictionary 
    """

    param: SCVxParams = SCVxParams() if not hasattr(prob, "scvx_param") else prob.scvx_param
    prob.rk, prob.pen_w = param.r0, param.w0           

    # make a initial solution dictionary    
    sol_prev = prob.sol_0 
    g, dg = prob.compute_g(sol_prev["z"])
    h, dh = prob.compute_h(sol_prev["z"])
    prob.gref, prob.href = g, h
    prob.dgref, prob.dhref = dg, dh
    prob.pen_λ, prob.pen_μ = np.full(np.shape(g), 0.0), np.full(np.shape(h), 0.0) # give a small value to avoid zero multiplier
    # print("initial h:", h[0])
    # if not np.any(h>0):
    #     print("WARNING: initial solution is feasible, so this SCP is likely not necessary...")
    # else:
    #     print("initial solution is infeasible... proceeding with SCP... ")    


    # initialization 
    k = 0 
    ΔJ, χ, δ = np.inf, np.inf, np.inf
    sol = {"L": 1.0}
    log = {"ϵopt":[], "ϵfeas":[], "f0":[], "P":[],  "ΔL":[]}
    sol_feas_subopt = {"z": None, "f0": np.inf}
    
    # Header for the table (run this once before your loop)
    header = f"{'Iter':^5} || {'χ':^9} | {'ΔJ':^9} || {'L':^9} | {'f0':^9} | {'P':^9} | {'r':^9} | {'w':^9} | {'ρk':^9} | {'ΔL':^9} | {'δ':^9}"
    
    while abs(ΔJ) > param.ϵopt or χ > param.ϵfeas:  # both optimality and feasibility must converge 
        
        if prob.verbose_scvx and np.mod(k, 10) == 0: 
            print("-" * len(header))  
            print(header)
            print("-" * len(header))  

        sol = prob.solve_cvx_AL()
        
        if sol["status"] not in ("optimal", "optimal_inaccurate") and "0" not in sol["status"]:
            status = sol["status"]
            # print(f"status is not optimal (cvxpy status: {status})! terminating...")
            break
        
        # t0 = time.time()
        ΔJ, ΔL, χ, g_sol, dg_sol, h_sol, dh_sol = scvx_compute_dJdL(sol, sol_prev, prob, k)

        if np.abs(ΔL) == 0:
            ρk = 1
        else:
            ρk = ΔJ / ΔL
        
        if prob.verbose_scvx:
            log_line = f"{k:^5} || {χ:9.2e} | {ΔJ:+9.2e} || {sol['L']:9.2e} | {sol['f0']:9.2e} | {sol['P']:9.2e} | {prob.rk:+9.2e} | {prob.pen_w:+9.2e} | {ρk:+9.2e} | {ΔL:+9.2e} | {δ:9.2e}"
            print(log_line)

        log["ϵopt"].append(ΔJ)
        log["ϵfeas"].append(χ)
        log["f0"].append(sol["f0"])
        log["P"].append(sol["P"])
        log["ΔL"].append(ΔL)
        
        if ρk >= param.ρ[0]:
            sol_prev = sol 
            prob.zref = sol["z"]
            prob.update_flag = True 
            
            prob.gref, prob.dgref = g_sol, dg_sol
            prob.href, prob.dhref = h_sol, dh_sol

            
            if abs(ΔJ) < δ:
                # print('======= weight update! =========')
                prob = scvx_update_weights(prob, param.β)
                δ    = scvx_update_delta(δ, param.γ, ΔJ) 
                
        prob.rk = scvx_update_r(prob.rk, param.α, param.ρ, ρk, param.r_minmax)  
        
        # store feasible solution 
        if χ < param.ϵfeas and sol['f0'] < sol_feas_subopt['f0']:
            sol_feas_subopt["z"] = sol["z"] 
            sol_feas_subopt["f0"] = sol["f0"]

        k += 1
        if k >= param.iter_max:
            if prob.verbose_scvx:
                print("SCVx* did not converge... terminating...")

            sol["status"] = "max_iter"
                        
            # if sol_feas_subopt["z"] is not None:
            #     sol["z"] = sol_feas_subopt["z"]
            #     sol["status"] = "subopt_max_iter"
            # else:
            #     sol["status"] = "max_iter"
                
            return sol, log 

        if prob.pen_w > param.wmax:
            
            if prob.verbose_scvx:
                print("penalty weight is too large... terminating...")

            sol["status"] = "max_w"

            # if sol_feas_subopt["z"] is not None:
            #     sol["z"] = sol_feas_subopt["z"]
            #     sol["status"] = "subopt_max_w"
            # else:
            #     sol["status"] = "max_w"
                
            return sol, log
            
            
        # EXTRA HACK; additional condition for the convergence
        # if χ < prob.ϵfeas and np.abs(log["f0"][-1] - log["f0"][-2]) < prob.ϵopt:
        #     print("optimality not guaranteed, but solution is converged to a feasible solution...")
        #     break
        
        # EXTRA HACK; also, if the solution is stuck, update the solution... 
        # if len(log["ΔL"]) > 1 and np.abs(log["ΔL"][-1] - log["ΔL"][-2]) < prob.ϵfeas:
        #     print("solution is stuck, so the reference is (forcefully) updated ... ")  
        #     sol_prev = sol 
        #     prob.zref = sol["z"]
        
        # print("update steps time", time.time() - t0 )
        
    return sol, log 

