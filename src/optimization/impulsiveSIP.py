import numpy as np 
import cvxpy as cp 

"""
KD Solver for impulsive control problems.
"""

class impulsiveSIP():
    
    def __init__(
        self,
        Φ,
        Γ,
        x0,
        xf,
        t0,
        tf,
        nu=3,
        p=2,
        N=3,
        Q=None,
        ϵcost=1e-6,
        ϵrmv=1e-6,
    ):
        
        self.Φ = Φ
        self.Γ = Γ
        self.x0 = x0 
        self.xf = xf
        self.t0 = t0
        self.tf = tf
        
        self.nx = np.shape(x0)[0]
        self.nu = nu 
        self.N = N  # initial guess of burn number 

        self.ϵcost = ϵcost
        self.ϵrmv = ϵrmv
        self.p = p
        self.q = 1 if p == np.inf else int(1/(1-1/p)) 
        self.Q = Q if Q is not None else np.eye(self.nx)        
        self.last_solver = None
        
        self.w = self.xf - self.Φ(self.tf,self.t0) @ self.x0
    
    def _compute_gU(self, t, λ):
        t_arr = np.array(t, ndmin=1)
        values = [np.linalg.norm(self.Γ(ti).T @ λ, self.q) for ti in t_arr]
        values = np.array(values)
        if np.isscalar(t):
            return values.item()
        return values
    
    def compute_sU(self, λ):
        """
        Suppport function 
        """
        if self.p == 2: 
            return λ / np.linalg.norm(λ, 2)
        elif self.p == 1:
            n = len(λ)
            W = np.concatenate([np.eye(n), -np.eye(n)])
            idx = np.argmax(W.T @ λ)
            return W.T[idx]
        else: 
            raise ValueError("p must be 1 or 2. Others are not implemented yet.")

    def _initialize(self):
        Td = np.linspace(self.t0, self.tf, 2*self.N)
        λest = self.w / np.linalg.norm(self.w)
        gU = [self._compute_gU(t, λest) for t in Td]
        # pair (gU, time), sort descending by gU, and get the top N times
        Test = [t for _, t in sorted(zip(gU, Td), reverse=True)[:self.N+1]]
        return Test
    
    def _refine(self, Test):
        
        iteration = 0
        while True:
            
            # solve optimization for lambda 
            λ = cp.Variable(self.nx)
            obj = cp.Maximize(λ.T @ self.w)
            constraints = [cp.norm(self.Γ(t).T @ λ, self.q) <= 1 for t in Test]
            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.CLARABEL, verbose=False)

            if not prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise RuntimeError("Unable to solve dual problem. Problem might be infeasible.")

            λest = λ.value
            
            # Discussion needed; shall we not remove non-critical times?
            # for t in Test.copy():
            #     if self._compute_gU(t, λest) < 1 - self.ϵrmv:
            #         Test.remove(t)
                
            t_grid = np.linspace(self.t0, self.tf, 1000)
            gU      = self._compute_gU(t_grid, λest)
            peaks = []
            # extract local maxima of gU to find critical times (candidate times for adding to Test)
            if len(gU) > 0:
                if len(gU) == 1 or gU[0] > gU[1]:
                    peaks.append(0)
                peaks.extend(
                    i for i in range(1, len(gU) - 1)
                    if gU[i] > gU[i - 1] and gU[i] > gU[i + 1]
                )
                if len(gU) > 1 and gU[-1] > gU[-2]:
                    peaks.append(len(gU) - 1)
            gU_localmax = gU[peaks]
            t_localmax  = t_grid[peaks]
            
            # add if t seems like a critical time 
            for (i,t) in enumerate(t_localmax):
                if gU_localmax[i] > 1:
                    Test.append(t)
                # Test.append(t)

            # terminate condition (dual's optimality)
            if max(gU) < 1 + self.ϵcost: 
                break 
            
            print(f"Iteration {iteration}: max gU = {max(gU)}")
            iteration += 1
            if iteration > 100:
                print("Too many iterations. Problem might be infeasible.")
                return Test, λest, gU
            
        return Test, λest, gU
    
    def _extract(self, Topt, λopt):
        
        uhat = np.zeros((len(Topt),self.nu))
        y = np.zeros((len(Topt), self.nx))
        for (j,tj) in enumerate(Topt):
            sU_tj = self.compute_sU(self.Γ(tj).T @ λopt)
            uhat[j] = sU_tj
            y[j] = self.Γ(tj) @ sU_tj

        # solve for coefficients (magnitudes) of thrusts 
        α = cp.Variable(len(Topt)) 
        w_err = cp.Variable(self.nx)
        obj = cp.Minimize(cp.quad_form(w_err, self.Q))
        con = [w_err == self.w - y.T @ α, α >= 0, cp.sum(α) <= λopt.T @ self.w]
        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if not prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError("Unable to solve for control magnitudes. Problem might be infeasible.")
        αopt = α.value
        
        # generate control histories
        uopt = np.zeros((len(Topt),self.nu))
        for j in range(len(Topt)):
            uopt[j] = αopt[j] * uhat[j]
            
        return uopt 
            
    def solve(self):
        
        Test = self._initialize()
        Topt, λopt, _ = self._refine(Test) 

        uopt = self._extract(Topt, λopt) 
        # Topt, uopt = zip(*sorted(zip(Topt, uopt), key=lambda x: x[0]))

        Topt, uopt = np.array(Topt), np.array(uopt)
        idx = np.argsort(np.array(Topt)) 
        return Topt[idx], uopt[idx], λopt
