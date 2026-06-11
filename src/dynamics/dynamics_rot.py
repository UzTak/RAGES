#### rotational dynamics library #### 
import numpy as np
import numpy.linalg as la
from scipy.linalg import expm
from scipy.integrate import solve_ivp, odeint
import scipy as sp 
# from sympy.functions.special.elliptic_integrals import elliptic_pi
import mpmath as mp

### Quaternion setup 
# q = [q0, q1, q2, q3] = [scalar, vector]

def skw(v): 
    return np.array([[0,     -v[2],  v[1]],
                     [v[2],   0,    -v[0]],
                     [-v[1],  v[0],  0   ]])
    
def get_phi(t, A, p=5):
    """
        numerically computing the matrix exp(A*t)
        p: order of the approximation
    """
    phi = np.eye(A.shape[0])
    for i in range(1, p):
        phi += np.linalg.matrix_power(A*t, i) / np.math.factorial(i)
    return phi    

def q_mul(q0, q1):
    return np.array([
        q0[0]*q1[0] - q0[1]*q1[1] - q0[2]*q1[2] - q0[3]*q1[3],
        q0[0]*q1[1] + q0[1]*q1[0] + q0[2]*q1[3] - q0[3]*q1[2],
        q0[0]*q1[2] - q0[1]*q1[3] + q0[2]*q1[0] + q0[3]*q1[1],
        q0[0]*q1[3] + q0[1]*q1[2] - q0[2]*q1[1] + q0[3]*q1[0]
    ])    

def q_cross(q0, q1):
    return np.array([
        q0[0]*q1[1] - q0[1]*q1[0] - q0[2]*q1[3] + q0[3]*q1[2],
        q0[0]*q1[2] + q0[1]*q1[3] - q0[2]*q1[0] - q0[3]*q1[1],
        q0[0]*q1[3] - q0[1]*q1[2] + q0[2]*q1[1] - q0[3]*q1[0]
    ])

def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_inv(q):
    """
        For unit quaternion, q_inv(q) = q_conj(q)
    """
    return q_conj(q) / np.dot(q, q)

def q_rot(q, v):
    # Rotate vector v by quaternion q (passive rotation)
    qv = np.array([0, v[0], v[1], v[2]])
    qv = q_mul(q_mul(q_inv(q), qv), q)
    return qv[1:]

def q2dcm(q):
    """
        Convert quaternion to rotation matrix
        input: q_a2b
        returns: R_a2b (i.e., v_b = R_a2b @ v_a) 
    """
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3,     2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3,     1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2,     2*q2*q3 + 2*q0*q1,     1 - 2*q1**2 - 2*q2**2]
    ])
    
def dcm2q(R): 
     # assuming R is rotation from A->B, return q_a2b
    w = np.sqrt(1 + np.trace(R)) / 2
    x = (R[2,1] - R[1,2]) / (4*w)
    y = (R[0,2] - R[2,0]) / (4*w)
    z = (R[1,0] - R[0,1]) / (4*w)
    return np.array([w,x,y,z])
    
def q_unit(q):
    # assert la.norm(q) > 1e-4 , "The size of the quaternion is too small"
    return q / la.norm(q)

def mrp2q(mrp, f=4, a=1):
    """
    Convert MRP to quaternion
    """
    assert len(mrp) == 3, "mrp should be a 3-element vector"
    pnorm2 = (mrp ** 2).sum()
    q0 = (-a * pnorm2 + f * np.sqrt(f**2 + (1-a**2))) / (f**2 + pnorm2)
    return np.concatenate(([q0], mrp*(a + q0)/f))     

def q2mrp(q, f=4, a=1): 
    assert len(q) == 4, "q should be a 4-element vector [q0, q1, q2, q3]"
    return q[1:] * f / (a + q[0])
    
def pw2qw(pw, qref, f=4, a=1):
    """
    Converting a [MRP, w] to [q, w], with a reference quaternion qref
    """
    assert len(pw) == 6, "pw should be a 6-element vector [p,w]"
    mrp = pw[:3]
    dq = mrp2q(mrp, f, a)     
    q  = q_mul(dq, qref)
    return np.concatenate((q, pw[3:]))  
    
def pw2qw_traj(pw, qref, n_time, f=4, a=1):
    """
    Converting pw -> qw along the reference quaternion trajectory (qref)
    Args: 
        - pw: [delta_p, w]^T ... MRP and angular velcoity history 
        - qref: reference quaternion trajectory
    Returns: 
        - qw: angular velocity in the inertial frame
    """
    qw = np.zeros((n_time, 7))
    for i in range(n_time):
        qw[i] = pw2qw(pw[i], qref[i], f, a)
    return qw

### EoM ###
def q_kin(q, omega):
    # qdot = 0.5 * q * omega
    return 0.5 * q_mul(q, np.array([0, omega[0], omega[1], omega[2]]))

def euler_dyn(w, I, tau):
    w = np.asarray(w).reshape(3,)
    tau = np.asarray(tau).reshape(3,)
    rhs = tau - np.cross(w, I @ w)
    return la.solve(I, rhs)

def euler_dyn_rel(w_dc_d, q_dc, w_cI_c, wdot_cI_c, I, tau):
    """
    Assuming the angular velocity and acceleration of the cheif (no torque applied),
    compute the relative angular acceleration of the deputy. 
    See "6-DOF robust adaptive terminal sliding mode control for spacecraft formation flying" (Wang et al., 2012, Acta) Eq. 22-d 
    inpus: 
        - w_cI_c : absolute ang. vel. of the chief (c) in the chief body frame 
        - w_dc_d : relative ang. vel. of deputy (d) w.r.t. c in d body frame 
        - q_dc   : quaternion of d w.r.t. c
        - w_cI_c : absolute ang. vel. of c in c body frame 
        - wdot_cI_c: absolute ang. acc. of c in c body frame
        - I      : inertia matrix of d
        - tau    : torque applied to d
    return: 
        - wdot_dc_d : relative ang. acc. of d w.r.t. c in d body frame
    """
    w_cI_d    = q_conj(q_dc) @ w_cI_c    @ q_dc
    wdot_cI_d = q_conj(q_dc) @ wdot_cI_c @ q_dc
    
    return la.inv(I) @ (tau - np.cross(w_dc_d + w_cI_d, (I @ (w_dc_d + w_cI_d)))) - wdot_cI_d + np.sross(w_dc_d, w_cI_d)

def mrp_kin(mrp, w):
    return 0.25 * ( (1 - np.dot(mrp,mrp)) * np.eye(3) + 2 * skw(mrp) + np.outer(mrp,mrp) ) @ w 

def dyn_qw(t, qw, J, T=np.zeros(3)):
    q  = qw[0:4]
    w  = qw[4:7]
    return np.concatenate((q_kin(q, w),  euler_dyn(w, J, T)))

def ode_qw(qw,t,J,T=np.zeros(3)):
    q = qw[0:4]
    w = qw[4:7]
    return np.concatenate((q_kin(q, w),  euler_dyn(w, J, T)))

#### Linearized dynamics #### 
def dyn_qw_lin(qw, J):
    """
        Obtain the linearized dynamics of the quaternion and angular velocity (continuous time)
    """
    q = qw[:4]
    w = qw[4:7]
    Aqq = 0.5 * np.array([[0, -w[0], -w[1], -w[2]],
                          [w[0], 0, w[2], -w[1]],
                          [w[1], -w[2], 0, w[0]],
                          [w[2], w[1], -w[0], 0]])
    Aqw = 0.5 * np.array([[-q[1], -q[2], -q[3]],
                          [q[0], -q[3], q[2]],
                          [q[3], q[0], -q[1]],
                          [-q[2], q[1], q[0]]])
    Aww = -np.linalg.inv(J) @ (skw(w) @ J - skw(J @ w))
    return Aqq, Aqw, Aww

def dyn_pw_lin(pw, J):
    """
    Based on Sam Low's AA279C paper. Check Zotero 
    However, the definition of the MRP here is mrp = 4*q_vec/(1+q0), so 
    Apw is multiplied by 4. Remaining parts remains the same. 
    """
    p = pw[:3]
    w = pw[3:]
    # App = 1/2 * ( np.outer(p, w) - np.outer(w, p) - skw(w) + np.dot(w, p) * np.eye(3) ) 
    # Apw = 1 * ( (1-np.dot(p,p))*np.eye(3) + 2*skw(p) + 2*np.outer(p,p) ).T
    App = -1/2 * skw(w) + 1/8 * (np.outer(p, w) - np.outer(w, p) + np.dot(w,p)*np.eye(3))
    Apw = 1/2 * skw(p) + 1/8 * np.outer(p, p) + (1-np.dot(p,p))*np.eye(3)
    Aww = -np.linalg.inv(J) @ (skw(w) @ J - skw(J @ w))
    return App, Apw, Aww 

def get_stm_qw(qw, dt, J):
    """
    Retrieve the STM of the [q,w] dynamics 
    """
    Aqq, Aqw, Aww = dyn_qw_lin(qw, J)

    A = np.zeros((7, 7))
    A[0:4, 0:4] = Aqq
    A[0:4, 4:7] = Aqw
    A[4:7, 4:7] = Aww

    # cim_qw = np.zeros((7, 3))
    # cim_qw[4:7, 0:3] = np.eye(3)  # if the input is angular velocity

    return get_phi(dt, A, 5)


def get_stm_pw(pw,dt,J):
    """
    Retrieve the STM of the [mrp,w] dynamics 
    """
    App, Apw, Aww = dyn_pw_lin(pw, J)
    A = np.zeros((6, 6))
    A[0:3, 0:3] = App
    A[0:3, 3:6] = Apw
    A[3:6, 3:6] = Aww
    
    cim_pw = np.zeros((6, 3))
    cim_pw[3:6, 0:3] = np.eye(3)  # if the input is angular velocity

    return get_phi(dt, A, 5)


def get_stm_pw2(pw,dt,J):
    """
    "Relative Computer Vision-Based Navigation for Small Inspection Spacecraft" (Tweddle, 2015)
    Eq. 17 - 21
    Assuming (1) small angular velocity and (2) small MRP (true for MEKF because of the frequent reset)
    """
    A = np.block([[-1/2*skw(pw[3:]), np.eye(3), np.zeros((3,3))],  
                  [np.zeros((3,6)), np.linalg.inv(J)], 
                  [np.zeros((3,9))]])
    
    Φ   = expm(A*dt)
    Φ12 = Φ[3:6, :3]
    Phi = np.block([[-expm(-dt/2*skw(pw[3:])), Φ12      ], 
                    [np.zeros((3,3)),          np.eye(3)]])
    
    return Phi
    

def dyn_qw_lin_discrete(qw, dt, J):
    Aqq, Aqw, Aww = dyn_qw_lin(qw, J)

    A = np.zeros((7, 7))
    A[0:4, 0:4] = Aqq
    A[0:4, 4:7] = Aqw
    A[4:7, 4:7] = Aww

    cim_qw = np.zeros((7, 3))
    cim_qw[4:7, 0:3] = np.eye(3)  # if the input is angular velocity

    phi0 = np.eye(7)
    A_ = get_phi(dt, A, 5)

    # Define the function to be integrated
    fun = lambda t, y: get_phi(t, A, 5).reshape(49, 1).flatten()

    # Solve the differential equation
    sol = solve_ivp(fun, [0, dt], phi0.flatten(), method='RK45')

    D_ = sol.y[:, -1].reshape(7, 7)
    B_ = D_ @ cim_qw

    return A_, B_, D_


def get_cim_qw(qw, J):
    """
    Assumption: control input is the angular velocity (delta_w), not the torque
    """
    cim_qw = np.zeros(shape=(7,3), dtype=float)
    cim_qw[4:7, 0:3] = np.eye(3) 
    return cim_qw


def get_cim_pw(pw, J):
    """
    Assumption: control input is the angular velocity (delta_w), not the torque
    """
    cim_pw = np.zeros(shape=(6,3), dtype=float)
    cim_pw[3:6, 0:3] = np.eye(3) 
    return cim_pw


### miscellaneous 

def sph_interp_angle(qw0, qwf, n_time, dt):
    """
    WARNING: Looks like it's broken. Need to fix.... (2024.06.17)
    
    Spherical interpolation in the quaternion sphere.
    Args:
        - qw0: initial quaternion
        - qwf: final quaternion
        - n_time: number of time steps
        - dt: time step
    Returns:
        - qw: quaternion and angular velocity history  n_time x 7
        - dw: angular acceleration history             (n_time-1) x 3
    """

    qhat = q_mul(q_conj(qw0[:4]), qwf[:4])
    sin_theta0f = np.linalg.norm(qhat[1:4])
    u0f = qhat[1:4] / sin_theta0f
    theta0f = np.arctan2(sin_theta0f, qhat[0]) * 2
    theta0f = np.mod(theta0f + np.pi, 2 * np.pi) - np.pi  # wrap to -pi ~ pi

    qw = np.zeros((n_time, 7))
    dw = np.zeros((n_time, 3))
    for k in range(n_time):
        ang = k * theta0f / (2 * n_time)
        q_k = np.hstack(([np.cos(ang)], u0f * np.sin(ang)))
        qw[k,  :4] = q_mul(qw0[:4], q_k)
        qw[k, 4:7] = theta0f / n_time / dt * u0f
        # dw[k]      = theta0f / n_time * u0f
        dw[k] = np.zeros(3) 
        
        assert np.abs(np.linalg.norm(qw[k, :4]) - 1) < 1e-3, "quaternion is not normalized"

    # reset the angular velocity
    qw[0] = qw0
    qw[-1] = qwf
    
    return qw, dw[:-1]


def track_target(r_rtn, t, r_target=np.zeros(3), los=np.array([1, 0, 0])):
    """
    Analytical formulation of quaternion to track the target
    "Intelligent Autonomous Control of Spacecraft with Multiple Constraints" (Hu et al., 2023) P.49 (Eq. 2.37)
    Be mindful about the singularities! 
    Args: 
        - r_rtn: relative position (of deputy w.r.t. target in the RTN frame) n_time x 3 
        - t: time vector
        - r_target: target position  (default: origin)
        - los : Body frame line of sight vector (default; body frame x-axis)
    Returns:
        - qw_deputy2rtn: quaternion and angular velocity history   n_time x 7
        - dM: angular acceleration history              n_time x 3
    """

    # if len(t) != r_rtn.shape[0]:
    #     raise ValueError("dimension of t and r_rtn is different. check the variable sizes.")
    
    n_time = r_rtn.shape[0]
    qw = np.zeros((n_time,7))
    dM = np.zeros((n_time-1,  3))
    dt = t[1] - t[0]
    
    for i in range(n_time):
        r = r_target - r_rtn[i, :3] # Adjust target location if necessary
        x_rho = r / np.linalg.norm(r)
        q_v = skw(los) @ x_rho / np.sqrt(2 * (1 + np.dot(los, x_rho)))
        q_0 = np.sqrt(2 * (1 + np.dot(los, x_rho))) / 2
        qw[i, :4] = np.concatenate((np.array([q_0]), q_v)).flatten()
        
        assert np.abs(np.linalg.norm(qw[i, :4]) - 1) < 1e-2, "quaternion is not normalized"
        
        # Check for singularity
        if np.any(np.isnan(q_v)):
            print(f'warning; singularity at timestep {i+1}/{n_time}')
            if i > 0:
                qw[i, 0:4] = qw[i-1, 0:4]
        
        if i > 0:
            # Compute the angular velocity
            q1 = qw[i-1, 0:4]
            q2 = qw[i,   0:4]
            dq = q_mul(q_inv(q1), q2)
            θ  = 2 * np.arccos(dq[0])
            uvec = dq[1:] / np.sin(θ/2)
            qw[i-1, 4:7] = θ/dt * uvec 
            
            # w = 2/dt * np.array([
            #     q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
            #     q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
            #     q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]
            # ])
            
            if i > 1:
                dM[i-2] = qw[i-1, 4:7] - qw[i-2, 4:7]
    
    # Set the angular velocity at the end to the previous value
    qw[-1, 4:7] = qw[-2, 4:7]
    dM[-1] = np.zeros(3)
    return qw, dM


def qw_fwd(qw,t0,tf,J,λ=None):
    """
    Analytical solution of torque-free motion
    assumimg I1 >= I2 >= I3
    "PERTURBATION FORMULATIONS FOR SATELLITE ATTITUDE DYNAMICS" (Kraige and Junkins, 1974) 
    Important quaterinons; 
        q: body frame (b) w.r.t. inertial frame (n)
        β: body frame (b) w.r.t. h-frame (h, h2 is aligned to the angular momentum vector)
        γ: h-frame    (h) w.r.t. inertial frame (n)
    """
    q, w = qw[:4], qw[4:]
    I1, I2, I3 = J[0,0], J[1,1], J[2,2]
    T = 1/2 * (I1 * w[0]**2 + I2 * w[1]**2 + I3 * w[2]**2)
    H2 = ((I1*w[0])**2 + (I2*w[1])**2 + (I3*w[2])**2)  # H**2
    H  = np.sqrt(H2)

    # max. values of ω1, ω2, ω3
    ω1m = np.sqrt((H2 - 2*I3*T) / (I1*(I1-I3)))
    ω3m = np.sqrt((2*I1*T - H2) / (I3*(I1-I3)))
    
    if H2 >= 2*I2*T:
        ω2m = np.sqrt((2*I1*T - H2) / (I2*(I1-I2))) 
        k2 = ((I2-I3)*(2*I1*T-H2) / ((I1-I2)*(H2-2*I3*T)))  # = m 
        Ω = ((I1-I2)*(H2 - 2*I3*T) / (I1*I2*I3))      # frequency of ω2(t) = body nutation rate (ω_p)
    else: 
        ω2m = np.sqrt((H2 - 2*I3*T) / (I2*(I2-I3)))
        k2 = ((I1-I2)*(H2-2*I3*T) / ((I2-I3)*(2*I1*T-H2)))
        Ω = np.sqrt((I2-I3)*(2*I1*T - H2) / (I1*I2*I3)) 

    # imcomplete elliptic integral of the first kind
    # K = sp.special.ellipk(k**2)   # = sp.special.ellipkinc(np.pi/2, k**2)   
    K = float(mp.ellipf(np.pi/2, k2))
    Θ = np.arcsin(np.clip(w[1] / ω2m, -1, 1))  # 0 < Θ < π/2
    
    if λ is not None:
        t1 = t0 + 1/Ω * (float(mp.ellipf(Θ, k2)) + λ*K ) 
        τ  = Ω * (tf-t1)
        τ0 = Ω * (t0-t1)
        
        sn, cn, dn, am = sp.special.ellipj(τ, k2)
        _,  _,  _, am0 = sp.special.ellipj(τ0, k2)
        # sn = float(mp.ellipfun('sn', τ, k2))
        # cn = float(mp.ellipfun('cn', τ, k2))
        # dn = float(mp.ellipfun('dn', τ, k2))

        s1 = np.sign(w[0])
        s3 = np.sign(w[2]) 
        
        if H2 >= 2*I2*T:
            f1 =  s1 * dn
            f3 = -s1 * cn
        else:
            f1 = -s3 * cn 
            f3 =  s3 * dn
        
        ω1 = ω1m * f1
        ω2 = ω2m * sn
        ω3 = ω3m * f3
    
    else:
        for λ in range(100): 
            # t1 = most recent instant (t1 < t0) s.t. ω2(t1) = 0 and dot{ω2}(t1) > 0
            # t1 = t0 - 1/Ω * (sp.special.ellipkinc(Θ, k**2) + λ*K)
            t1 = t0 + 1/Ω * (float(mp.ellipf(Θ, k2)) + λ*K ) 
            τ  = Ω * (tf-t1)
            τ0 = Ω * (t0-t1)
            
            sn, cn, dn, am = sp.special.ellipj(τ, k2)
            _, _, _, am0 = sp.special.ellipj(τ0, k2)
            # sn = float(mp.ellipfun('sn', τ, k2))
            # cn = float(mp.ellipfun('cn', τ, k2))
            # dn = float(mp.ellipfun('dn', τ, k2))

            s1 = np.sign(w[0])
            s3 = np.sign(w[2]) 
            
            if H2 >= 2*I2*T:
                f1 =  s1 * dn
                f3 = -s1 * cn
            else:
                f1 = -s3 * cn 
                f3 =  s3 * dn
            
            ω1 = ω1m * f1
            ω2 = ω2m * sn
            ω3 = ω3m * f3
            
            if ω1 * w[0] > 0 and ω2 * w[1] > 0 and ω3 * w[2] > 0:
                break
    
    a = I2 * ω2m / H 
    Hn = q2dcm(q).T @ np.array([I1*w[0], I2*w[1], I3*w[2]])  # angular momentum in the inertial frame 
    Hn1, Hn2, Hn3 = Hn[0], Hn[1], Hn[2]
    
    γ0 = np.sqrt((H + Hn2) / (2*H))
    γ1 = Hn3 * np.sqrt((H - Hn2) / (2*H*(Hn1**2 + Hn3**2))) 
    γ2 = 0
    γ3 = - Hn1 * np.sqrt((H - Hn2) / (2*H*(Hn1**2 + Hn3**2)))
    γ = np.array([γ0, γ1, γ2, γ3])   # quaternion of h-frame w.r.t. inertial frame
    
    β0 = q_mul(q_conj(γ), q)    # q_{b/h} = (q_{h/n}^*) * q_{b/n}
    Φ0_0 = np.arctan2(β0[2], β0[0])
    Φ0_1 = np.arctan2(β0[3], β0[1])
    
    Φs = np.arctan2(I1*ω1, -I3*ω3)   # -pi < Φs < pi
    Φs = Φs if Φs >= 0 else Φs + 2*np.pi
    Φd = Φ0_0 - Φ0_1 + H/I2*(tf-t0) + (2*I2*T - H2)/(I2*Ω*H) * float(mp.ellippi(a**2, am, k2) - mp.ellippi(a**2, am0, k2))   
    
    Φ0 = 0.5 * (Φs + Φd) 
    Φ1 = 0.5 * (Φs - Φd) 

    h2 = I2 * ω2 / H   
    β0 = np.sqrt(0.5*(1+h2)) * np.cos(Φ0) 
    β1 = np.sqrt(0.5*(1-h2)) * np.cos(Φ1) 
    β2 = np.sqrt(0.5*(1+h2)) * np.sin(Φ0) 
    β3 = np.sqrt(0.5*(1-h2)) * np.sin(Φ1) 
    β = np.array([β0, β1, β2, β3])
    
    q_ = q_mul(γ, β)   # q_{b/n} = (q_{n/h}^*) * q_{b/h}
    ω_ = np.array([ω1, ω2, ω3])
    
    return np.concatenate((q_, ω_)) , λ, Ω


### nonlinear propagation with control 

def nl_ppgt_qw(μ_qw, dw, t, J, K=None, dw_max=np.inf, burn_then_ppgt=False):
    
    n_time = len(t)

    if burn_then_ppgt:
        
        qw_list = np.empty(shape=(n_time+1, 7), dtype=float)
        qw_list[0]  = μ_qw[0] 
        # first burn (no disturbance)
        qw_list[1] = qw_list[0]
        qw_list[1, 4:] += dw[0] 

        for i in range(1,n_time):
            time = np.array([0, t[i]-t[i-1]])            
            qw_list[i+1] = odeint(ode_qw, qw_list[i], time, args=(J, np.zeros(3)))[1]
            
            if K is not None:
                dq  = q_mul(q_conj(μ_qw[i,:4]), qw_list[i,:4])           # error quaternion of the realized quaternion to the nominal quaternion
                dpw = np.hstack((q2mrp(dq), qw_list[i,4:] - μ_qw[i,-3:]))  # [error mrp, error w] 
                dw_feedback = K[i] @ dpw    # feedback control 
                dw[i] += dw_feedback   # in the deputy's body frame 
            
            # cap the control input 
            if np.linalg.norm(dw[i]) > dw_max:
                dw[i] = dw[i] / np.linalg.norm(dw[i]) * dw_max
        
            qw_list[i+1, 4:] += dw[i]  # impulsive control 

    else:
        
        qw_list  = np.empty(shape=(n_time, 7), dtype=float)
        qw_list[0]  = μ_qw[0]
    
        for i in range(n_time-1):
            time = np.array([0, t[i+1]-t[i]])
            
            # option 1; quasi-continuous control 
            # dm = dw[i]   # in the deputy's body frame
            # T = dm / (t[i+1] - t[i])   # in deputy's body frame        
            # qw_list[i+1]    = odeint(ode_qw, qw_list[:,i].flatten(), time, args=(J, T))[1]

            # option 2; impulsive control 
            qw_list[i+1] = odeint(ode_qw, qw_list[i], time, args=(J, np.zeros(3)))[1]
            qw_list[i+1, 4:] += dw[i]   # impulsive control 

    return qw_list

def nl_ppgt_pw(pw0, dw, t, J, qref):
    """
    Lock the initial condition using the qref[0] and pw0.
    After that, propagate the [q,w] dynamics along with the control input dw 
    """
    
    qw_list  = np.empty(shape=(len(t), 7), dtype=float)
    qw_list[0] = pw2qw(pw0[0], qref[0])     
    
    for i in range(len(t)-1):
        time = np.array([0, t[i+1]-t[i]])
        
        # option 1; quasi-continuous control 
        # dm = dw[i]   # in the deputy's body frame
        # T = dm / (t[i+1] - t[i])   # in deputy's body frame        
        # qw_list[i+1]    = odeint(ode_qw, qw_list[:,i].flatten(), time, args=(J, T))[1]

        # option 2; impulsive control 
        qw_list[i+1] = odeint(ode_qw, qw_list[i], time, args=(J, np.zeros(3)))[1]
        qw_list[i+1, 4:] += dw[i]   # impulsive control 

    return qw_list

