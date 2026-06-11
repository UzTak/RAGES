import numpy as np

from dynamics.dynamics_trans import (
    J2,
    R_E,
    mean_to_true_anomaly,
    mu_E,
    ecc_to_mean_anomaly,
    ecc_to_true_anomaly,
    mean_to_ecc_anomaly,
    true_to_ecc_anomaly,
    true_to_mean_anomaly,
    true_to_mean_anomaly,
)

__all__ = ["theta2lambda", "meanoscclosed", "mean2osc", "osc2mean"]

_SINGULAR_TOL = 1e-6

def _wrap_to_pi(x):
    wrapped = (np.asarray(x, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
    if wrapped.ndim == 0:
        return float(wrapped)
    return wrapped


def _normalize_oe_input(arr):
    arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Orbital elements must be finite")
    if arr.ndim == 1:
        if arr.size != 6:
            raise ValueError("Orbital elements must have shape (6,) or (N, 6)")
        return arr.reshape(1, 6), True
    if arr.ndim == 2 and arr.shape[1] == 6:
        return arr.copy(), False
    raise ValueError("Orbital elements must have shape (6,) or (N, 6)")


def _restore_oe_output(arr, scalar_input):
    if scalar_input:
        return arr[0]
    return arr


def _validate_keplerian_state(oe):
    if oe.shape != (6,):
        raise ValueError("Keplerian orbital element state must have shape (6,)")
    if oe[1] < 0.0 or oe[1] >= 1.0:
        raise ValueError("Eccentricity must satisfy 0 <= e < 1")


def _validate_nonsingular_state(state):
    state = np.asarray(state, dtype=float).reshape(-1)
    if state.size != 6:
        raise ValueError("Nonsingular orbital element state must have shape (6,)")
    ecc = np.hypot(state[3], state[4])
    if ecc >= 1.0:
        raise ValueError("Nonsingular elements imply eccentricity >= 1")
    return state


def _koe_to_nonsingular(oe):
    oe = np.asarray(oe, dtype=float).reshape(-1)
    _validate_keplerian_state(oe)

    a, e, inc, raan, omega, mean_anomaly = oe
    true_anomaly = mean_to_true_anomaly(mean_anomaly, e)

    state = np.array(
        [
            a,
            omega + true_anomaly,
            inc,
            e * np.cos(omega),
            e * np.sin(omega),
            raan,
        ],
        dtype=float,
    )
    state[1] = _wrap_to_pi(state[1])
    state[2] = _wrap_to_pi(state[2])
    state[5] = _wrap_to_pi(state[5])
    return state


def _nonsingular_to_koe(state):
    state = _validate_nonsingular_state(state)

    a, theta, inc, q1, q2, raan = state
    ecc = np.hypot(q1, q2)
    omega = np.arctan2(q2, q1)
    true_anomaly = theta - omega
    mean_anomaly = true_to_mean_anomaly(true_anomaly, ecc)

    oe = np.array([a, ecc, inc, raan, omega, mean_anomaly], dtype=float)
    oe[2] = _wrap_to_pi(oe[2])
    oe[3] = _wrap_to_pi(oe[3])
    oe[4] = _wrap_to_pi(oe[4])
    oe[5] = _wrap_to_pi(oe[5])
    return oe


def theta2lambda(a, theta, q1, q2):
    eta = np.sqrt(1.0 - q1**2 - q2**2)
    if eta <= _SINGULAR_TOL:
        raise ValueError("theta2lambda is singular for eta near zero")

    beta = 1.0 / (eta * (1.0 + eta))
    denom = 1.0 + q1 * np.cos(theta) + q2 * np.sin(theta)
    if abs(denom) <= _SINGULAR_TOL:
        raise ValueError("theta2lambda is singular for 1 + q1*cos(theta) + q2*sin(theta) near zero")

    radius = (a * eta**2) / denom
    num = radius * (1.0 + beta * q1**2) * np.sin(theta) - beta * radius * q1 * q2 * np.cos(theta) + a * q2
    den = radius * (1.0 + beta * q2**2) * np.cos(theta) - beta * radius * q1 * q2 * np.sin(theta) + a * q1

    F = np.arctan2(num, den)
    lambda_ = F - q1 * np.sin(F) + q2 * np.cos(F)

    while lambda_ < 0.0:
        lambda_ += 2.0 * np.pi
    while lambda_ >= 2.0 * np.pi:
        lambda_ -= 2.0 * np.pi

    theta_work = float(theta)
    if theta_work < 0.0:
        kk_plus = 0
        quad_plus = 0
        while theta_work < 0.0:
            kk_plus += 1
            theta_work += 2.0 * np.pi
        if theta_work < (np.pi / 2.0) and lambda_ > np.pi:
            quad_plus = 1
        elif lambda_ < (np.pi / 2.0) and theta_work > np.pi:
            quad_plus = -1
        lambda_ = lambda_ - (kk_plus + quad_plus) * (2.0 * np.pi)
    else:
        kk_minus = 0
        quad_minus = 0
        while theta_work >= 2.0 * np.pi:
            kk_minus += 1
            theta_work -= 2.0 * np.pi
        if theta_work < (np.pi / 2.0) and lambda_ > np.pi:
            quad_minus = -1
        elif lambda_ < (np.pi / 2.0) and theta_work > np.pi:
            quad_minus = 1
        lambda_ = lambda_ + (kk_minus + quad_minus) * (2.0 * np.pi)
    return float(lambda_)


def meanoscclosed(mean_c, J2=J2, Re=R_E, mu=mu_E):
    mean_c = _validate_nonsingular_state(mean_c)

    coef = -J2 * Re**2

    a = mean_c[0]
    theta = mean_c[1]
    i = mean_c[2]
    q1 = mean_c[3]
    q2 = mean_c[4]
    Omega = mean_c[5]

    s_i = np.sin(i)
    c_i = np.cos(i)
    s_2i = np.sin(2 * i)
    c_2i = np.cos(2 * i)
    s_th = np.sin(theta)
    c_th = np.cos(theta)
    s_2th = np.sin(2 * theta)
    c_2th = np.cos(2 * theta)
    s_3th = np.sin(3 * theta)
    c_3th = np.cos(3 * theta)
    s_4th = np.sin(4 * theta)
    c_4th = np.cos(4 * theta)
    s_5th = np.sin(5 * theta)
    c_5th = np.cos(5 * theta)

    total_E = -mu / (2 * a)
    n = np.sqrt(mu / a**3)
    p = a * (1 - (q1**2 + q2**2))
    denom = 1 + q1 * c_th + q2 * s_th
    ttheta_denom = 1 - 5 * c_i**2
    eta = np.sqrt(1 - (q1**2 + q2**2))
    eps1 = np.sqrt(q1**2 + q2**2)
    eps2 = q1 * c_th + q2 * s_th
    eps3 = q1 * s_th - q2 * c_th

    if eta <= _SINGULAR_TOL:
        raise ValueError("meanoscclosed is singular for eta near zero")
    if abs(ttheta_denom) <= _SINGULAR_TOL:
        raise ValueError("meanoscclosed is singular for 1 - 5*cos(i)^2 near zero")
    if abs(denom) <= _SINGULAR_TOL:
        raise ValueError("meanoscclosed is singular for 1 + q1*cos(theta) + q2*sin(theta) near zero")
    if abs(p) <= _SINGULAR_TOL:
        raise ValueError("meanoscclosed is singular for semilatus rectum near zero")
    if abs(1 + eps2) <= _SINGULAR_TOL:
        raise ValueError("meanoscclosed is singular for 1 + eps2 near zero")

    R = p / denom
    Vr = np.sqrt(mu / p) * (q1 * s_th - q2 * c_th)
    Vt = np.sqrt(mu / p) * (1 + q1 * c_th + q2 * s_th)

    Ttheta = 1 / ttheta_denom

    lambda_ = theta2lambda(a, theta, q1, q2)
    the_lam = theta - lambda_
    lam_q1 = (q1 * Vr) / (eta * Vt) + q2 / (eta * (1 + eta)) - eta * R * (a + R) * (q2 + np.sin(theta)) / (p**2)
    lam_q2 = (q2 * Vr) / (eta * Vt) - q1 / (eta * (1 + eta)) + eta * R * (a + R) * (q1 + np.cos(theta)) / (p**2)

    DI = np.eye(6)

    lam_lp = (s_i**2 / (8 * a**2 * eta**2 * (1 + eta))) * (1 - 10 * Ttheta * c_i**2) * q1 * q2 + (
        q1 * q2 / (16 * a**2 * eta**4)
    ) * (3 - 55 * c_i**2 - 280 * Ttheta * c_i**4 - 400 * Ttheta**2 * c_i**6)
    a_lp = 0
    theta_lp = lam_lp - (s_i**2 / (16 * a**2 * eta**4)) * (1 - 10 * Ttheta * c_i**2) * (
        (3 + 2 * eta**2 / (1 + eta)) * q1 * q2 + 2 * q1 * s_th + 2 * q2 * c_th + 0.5 * (q1**2 + q2**2) * s_2th
    )
    i_lp = (s_2i / (32 * a**2 * eta**4)) * (1 - 10 * Ttheta * c_i**2) * (q1**2 - q2**2)
    q1_lp = -(q1 * s_i**2 / (16 * a**2 * eta**2)) * (1 - 10 * Ttheta * c_i**2) - (
        q1 * q2**2 / (16 * a**2 * eta**4)
    ) * (3 - 55 * c_i**2 - 280 * Ttheta * c_i**4 - 400 * Ttheta**2 * c_i**6)
    q2_lp = (q2 * s_i**2 / (16 * a**2 * eta**2)) * (1 - 10 * Ttheta * c_i**2) + (
        q1**2 * q2 / (16 * a**2 * eta**4)
    ) * (3 - 55 * c_i**2 - 280 * Ttheta * c_i**4 - 400 * Ttheta**2 * c_i**6)
    Omega_lp = (q1 * q2 * c_i / (8 * a**2 * eta**4)) * (11 + 80 * Ttheta * c_i**2 + 200 * Ttheta**2 * c_i**4)

    D_lp_11 = -(1 / a) * a_lp
    D_lp_12 = 0
    D_lp_13 = 0
    D_lp_14 = 0
    D_lp_15 = 0
    D_lp_16 = 0
    D_lp_21 = -(2 / a) * theta_lp
    D_lp_22 = -(s_i**2 / (16 * a**2 * eta**4)) * (1 - 10 * Ttheta * c_i**2) * (2 * (q1 * c_th - q2 * s_th) + eps1 * c_2th)
    D_lp_23 = (s_2i / (16 * a**2 * eta**4)) * (
        5 * q1 * q2 * (11 + 112 * Ttheta * c_i**2 + 520 * Ttheta**2 * c_i**4 + 800 * Ttheta**3 * c_i**6)
        - (2 * q1 * q2 + (2 + eps2) * (q1 * s_th + q2 * c_th))
        * ((1 - 10 * Ttheta * c_i**2) + 10 * Ttheta * s_i**2 * (1 + 5 * Ttheta * c_i**2))
    )
    D_lp_24 = (1 / (16 * a**2 * eta**6)) * (
        (eta**2 + 4 * q1**2)
        * (q2 * (3 - 55 * c_i**2 - 280 * Ttheta * c_i**4 - 400 * Ttheta**2 * c_i**6) - s_i**2 * (1 - 10 * Ttheta * c_i**2) * (3 * q2 + 2 * s_th))
        - 2 * s_i**2 * (1 - 10 * Ttheta * c_i**2) * (4 * q2 + s_th * (1 + eps1)) * q1 * c_th
    )
    D_lp_25 = (1 / (16 * a**2 * eta**6)) * (
        (eta**2 + 4 * q2**2)
        * (q1 * (3 - 55 * c_i**2 - 280 * Ttheta * c_i**4 - 400 * Ttheta**2 * c_i**6) - s_i**2 * (1 - 10 * Ttheta * c_i**2) * (3 * q1 + 2 * c_th))
        - 2 * s_i**2 * (1 - 10 * Ttheta * c_i**2) * (4 * q1 + c_th * (1 + eps1)) * q2 * s_th
    )
    D_lp_26 = 0
    D_lp_31 = -(2 / a) * i_lp
    D_lp_32 = 0
    D_lp_33 = ((q1**2 - q2**2) / (16 * a**2 * eta**4)) * (c_2i * (1 - 10 * Ttheta * c_i**2) + 5 * Ttheta * s_2i**2 * (1 + 5 * Ttheta * c_i**2))
    D_lp_34 = (q1 * s_2i / (16 * a**2 * eta**6)) * (1 - 10 * Ttheta * c_i**2) * (eta**2 + 2 * (q1**2 - q2**2))
    D_lp_35 = -(q2 * s_2i / (16 * a**2 * eta**6)) * (1 - 10 * Ttheta * c_i**2) * (eta**2 - 2 * (q1**2 - q2**2))
    D_lp_36 = 0
    D_lp_41 = -(2 / a) * q1_lp
    D_lp_42 = 0
    D_lp_43 = -(q1 * s_2i / (16 * a**2 * eta**4)) * (
        eta**2 * ((1 - 10 * Ttheta * c_i**2) + 10 * Ttheta * s_i**2 * (1 + 5 * Ttheta * c_i**2))
        + 5 * q2**2 * (11 + 112 * Ttheta * c_i**2 + 520 * Ttheta**2 * c_i**4 + 800 * Ttheta**3 * c_i**6)
    )
    D_lp_44 = -(1 / (16 * a**2 * eta**6)) * (
        eta**2 * s_i**2 * (1 - 10 * Ttheta * c_i**2) * (eta**2 + 2 * q1**2)
        + q2**2 * (eta**2 + 4 * q1**2) * (3 - 55 * c_i**2 - 280 * Ttheta * c_i**4 - 400 * Ttheta**2 * c_i**6)
    )
    D_lp_45 = -(q1 * q2 / (8 * a**2 * eta**6)) * (
        eta**2 * s_i**2 * (1 - 10 * Ttheta * c_i**2)
        + (eta**2 + 2 * q2**2) * (3 - 55 * c_i**2 - 280 * Ttheta * c_i**4 - 400 * Ttheta**2 * c_i**6)
    )
    D_lp_46 = 0
    D_lp_51 = -(2 / a) * q2_lp
    D_lp_52 = 0
    D_lp_53 = (q2 * s_2i / (16 * a**2 * eta**4)) * (
        eta**2 * (1 - 10 * Ttheta * c_i**2)
        + 10 * Ttheta * eta**2 * s_i**2 * (1 + 5 * Ttheta * c_i**2)
        + 5 * q1**2 * (11 + 112 * Ttheta * c_i**2 + 520 * Ttheta**2 * c_i**4 + 800 * Ttheta**3 * c_i**6)
    )
    D_lp_54 = (q1 * q2 / (8 * a**2 * eta**6)) * (
        eta**2 * s_i**2 * (1 - 10 * Ttheta * c_i**2)
        + (3 - 55 * c_i**2 - 280 * Ttheta * c_i**4 - 400 * Ttheta**2 * c_i**6) * (eta**2 + 2 * q1**2)
    )
    D_lp_55 = (1 / (16 * a**2 * eta**6)) * (
        eta**2 * s_i**2 * (1 - 10 * Ttheta * c_i**2) * (eta**2 + 2 * q2**2)
        + q1**2 * (3 - 55 * c_i**2 - 280 * Ttheta * c_i**4 - 400 * Ttheta**2 * c_i**6) * (eta**2 + 4 * q2**2)
    )
    D_lp_56 = 0
    D_lp_61 = -(2 / a) * Omega_lp
    D_lp_62 = 0
    D_lp_63 = -(q1 * q2 * s_i / (8 * a**2 * eta**4)) * ((11 + 80 * Ttheta * c_i**2 + 200 * Ttheta**2 * c_i**4) + 160 * Ttheta * c_i**2 * (1 + 5 * Ttheta * c_i**2) ** 2)
    D_lp_64 = (q2 * c_i / (8 * a**2 * eta**6)) * (eta**2 + 4 * q1**2) * (11 + 80 * Ttheta * c_i**2 + 200 * Ttheta**2 * c_i**4)
    D_lp_65 = (q1 * c_i / (8 * a**2 * eta**6)) * (eta**2 + 4 * q2**2) * (11 + 80 * Ttheta * c_i**2 + 200 * Ttheta**2 * c_i**4)
    D_lp_66 = 0

    D_lp = np.array([
        [D_lp_11, D_lp_12, D_lp_13, D_lp_14, D_lp_15, D_lp_16],
        [D_lp_21, D_lp_22, D_lp_23, D_lp_24, D_lp_25, D_lp_26],
        [D_lp_31, D_lp_32, D_lp_33, D_lp_34, D_lp_35, D_lp_36],
        [D_lp_41, D_lp_42, D_lp_43, D_lp_44, D_lp_45, D_lp_46],
        [D_lp_51, D_lp_52, D_lp_53, D_lp_54, D_lp_55, D_lp_56],
        [D_lp_61, D_lp_62, D_lp_63, D_lp_64, D_lp_65, D_lp_66],
    ], dtype=float)

    lam_sp1 = (eps3 * (1 - 3 * c_i**2) / (4 * a**2 * eta**4 * (1 + eta))) * ((1 + eps2) ** 2 + (1 + eps2) + eta**2) + (3 * (1 - 5 * c_i**2) / (4 * a**2 * eta**4)) * (the_lam + eps3)
    a_sp1 = ((1 - 3 * c_i**2) / (2 * a * eta**6)) * ((1 + eps2) ** 3 - eta**3)
    theta_sp1 = lam_sp1 - (eps3 * (1 - 3 * c_i**2) / (4 * a**2 * eta**4 * (1 + eta))) * ((1 + eps2) ** 2 + eta * (1 + eta))
    i_sp1 = 0
    q1_sp1 = -(3 * q2 * (1 - 5 * c_i**2) / (4 * a**2 * eta**4)) * (the_lam + eps3) + ((1 - 3 * c_i**2) / (4 * a**2 * eta**4 * (1 + eta))) * (((1 + eps2) ** 2 + eta**2) * (q1 + (1 + eta) * c_th) + (1 + eps2) * ((1 + eta) * c_th + q1 * (eta - eps2)))
    q2_sp1 = (3 * q1 * (1 - 5 * c_i**2) / (4 * a**2 * eta**4)) * (the_lam + eps3) + ((1 - 3 * c_i**2) / (4 * a**2 * eta**4 * (1 + eta))) * (((1 + eps2) ** 2 + eta**2) * (q2 + (1 + eta) * s_th) + (1 + eps2) * ((1 + eta) * s_th + q2 * (eta - eps2)))
    Omega_sp1 = (3 * c_i / (2 * a**2 * eta**4)) * (the_lam + eps3)

    D_sp1_11 = -(1 / a) * a_sp1
    D_sp1_12 = -(3 * eps3 / (2 * a * eta**6)) * (1 - 3 * c_i**2) * (1 + eps2) ** 2
    D_sp1_13 = (3 * s_2i / (2 * a * eta**6)) * ((1 + eps2) ** 3 - eta**3)
    D_sp1_14 = (3 * (1 - 3 * c_i**2) / (2 * a * eta**8)) * (2 * q1 * (1 + eps2) ** 3 + eta**2 * (1 + eps2) ** 2 * c_th - eta**3 * q1)
    D_sp1_15 = (3 * (1 - 3 * c_i**2) / (2 * a * eta**8)) * (2 * q2 * (1 + eps2) ** 3 + eta**2 * (1 + eps2) ** 2 * s_th - eta**3 * q2)
    D_sp1_16 = 0
    D_sp1_21 = -(2 / a) * theta_sp1
    D_sp1_22 = ((1 - 3 * c_i**2) / (4 * a**2 * eta**4 * (1 + eta))) * (eps2 * (1 + eps2 - eta) - eps3**2) + (3 * (1 - 5 * c_i**2) / (4 * a**2 * eta**4 * (1 + eps2) ** 2)) * ((1 + eps2) ** 3 - eta**3)
    D_sp1_23 = (3 * eps3 * s_2i / (4 * a**2 * eta**4 * (1 + eta))) * ((1 + eps2) + (5 + 4 * eta)) + (15 * s_2i / (4 * a**2 * eta**4)) * the_lam
    D_sp1_24 = ((1 - 3 * c_i**2) / (4 * a**2 * eta**6 * (1 + eta) ** 2)) * (eta**2 * (eps1 * s_th + (1 + eta) * (eps2 * s_th + eps3 * c_th)) + q1 * eps3 * (4 * (eps1 + eps2) + eta * (2 + 5 * eps2))) + (3 * (1 - 5 * c_i**2) / (4 * a**2 * eta**6)) * (4 * q1 * (the_lam + eps3) + eta**2 * s_th) - (3 * (1 - 5 * c_i**2) / (4 * a**2 * eta**4)) * lam_q1
    D_sp1_25 = -((1 - 3 * c_i**2) / (4 * a**2 * eta**6 * (1 + eta) ** 2)) * (eta**2 * (eps1 * c_th + (1 + eta) * (eps2 * c_th - eps3 * s_th)) - q2 * eps3 * (4 * (eps1 + eps2) + eta * (2 + 5 * eps2))) + (3 * (1 - 5 * c_i**2) / (4 * a**2 * eta**6)) * (4 * q2 * (the_lam + eps3) - eta**2 * c_th) - (3 * (1 - 5 * c_i**2) / (4 * a**2 * eta**4)) * lam_q2
    D_sp1_26 = 0
    D_sp1_31 = -(2 / a) * i_sp1
    D_sp1_32 = 0
    D_sp1_33 = 0
    D_sp1_34 = 0
    D_sp1_35 = 0
    D_sp1_36 = 0
    D_sp1_41 = -(2 / a) * q1_sp1
    D_sp1_42 = -((1 - 3 * c_i**2) / (4 * a**2 * eta**4)) * ((1 + eps2) * (2 * s_th + eps2 * s_th + 2 * eps3 * c_th) + eps3 * (q1 + c_th) + eta**2 * s_th) - (3 * q2 * (1 - 5 * c_i**2) / (4 * a**2 * eta**4 * (1 + eps2) ** 2)) * ((1 + eps2) ** 3 - eta**3)
    D_sp1_43 = (3 * q1 * s_2i / (4 * a**2 * eta**2 * (1 + eta))) + (3 * s_2i / (4 * a**2 * eta**4)) * ((1 + eps2) * (q1 + (2 + eps2) * c_th) - 5 * q2 * eps3 + eta**2 * c_th) - (15 * q2 * s_2i / (4 * a**2 * eta**4)) * the_lam
    D_sp1_44 = ((1 - 3 * c_i**2) / (4 * a**2 * eta**2 * (1 + eta))) + ((1 - 3 * c_i**2) * q1**2 * (4 + 5 * eta) / (4 * a**2 * eta**6 * (1 + eta) ** 2)) + ((1 - 3 * c_i**2) / (8 * a**2 * eta**6)) * (eta**2 * (5 + 2 * (5 * q1 * c_th + 2 * q2 * s_th) + (3 + 2 * eps2) * c_2th) + 2 * q1 * (4 * (1 + eps2) * (2 + eps2) * c_th + (3 * eta + 4 * eps2) * q1)) - (3 * q2 * (1 - 5 * c_i**2) / (4 * a**2 * eta**6)) * (4 * q1 * eps3 + eta**2 * s_th) - (3 * q1 * q2 * (1 - 5 * c_i**2) / (a**2 * eta**6)) * the_lam + (3 * q2 * (1 - 5 * c_i**2) / (4 * a**2 * eta**4)) * lam_q1
    D_sp1_45 = ((1 - 3 * c_i**2) / (8 * a**2 * eta**6)) * (eta**2 * (2 * (q1 * s_th + 2 * q2 * c_th) + (3 + 2 * eps2) * s_2th) + 2 * q2 * (4 * (1 + eps2) * (2 + eps2) * c_th + (3 * eta + 4 * eps2) * q1)) + ((1 - 3 * c_i**2) * q1 * q2 * (4 + 5 * eta) / (4 * a**2 * eta**6 * (1 + eta) ** 2)) - (3 * (1 - 5 * c_i**2) / (4 * a**2 * eta**6)) * (eps3 * (eta**2 + 4 * q2**2) - eta**2 * q2 * c_th) - (3 * (1 - 5 * c_i**2) / (4 * a**2 * eta**6)) * (the_lam * (eta**2 + 4 * q2**2)) + (3 * q2 * (1 - 5 * c_i**2) / (4 * a**2 * eta**4)) * lam_q2
    D_sp1_46 = 0
    D_sp1_51 = -(2 / a) * q2_sp1
    D_sp1_52 = ((1 - 3 * c_i**2) / (4 * a**2 * eta**4)) * ((1 + eps2) * (2 * c_th + eps2 * c_th - 2 * eps3 * s_th) - eps3 * (q2 + s_th) + eta**2 * c_th) + (3 * q1 * (1 - 5 * c_i**2) / (4 * a**2 * eta**4 * (1 + eps2) ** 2)) * ((1 + eps2) ** 3 - eta**3)
    D_sp1_53 = (3 * q2 * s_2i / (4 * a**2 * eta**2 * (1 + eta))) + (3 * s_2i / (4 * a**2 * eta**4)) * ((1 + eps2) * (q2 + (2 + eps2) * s_th) + 5 * q1 * eps3 + eta**2 * s_th) - (15 * q1 * s_2i / (4 * a**2 * eta**4)) * the_lam
    D_sp1_54 = ((1 - 3 * c_i**2) / (8 * a**2 * eta**6)) * (eta**2 * (2 * (2 * q1 * s_th + q2 * c_th) + (3 + 2 * eps2) * s_2th) + 2 * q1 * (4 * (1 + eps2) * (2 + eps2) * s_th + (3 * eta + 4 * eps2) * q2)) + ((1 - 3 * c_i**2) * q1 * q2 * (4 + 5 * eta) / (4 * a**2 * eta**6 * (1 + eta) ** 2)) + (3 * (1 - 5 * c_i**2) / (4 * a**2 * eta**6)) * (eps3 * (eta**2 + 4 * q1**2) + eta**2 * q1 * s_th) + (3 * (1 - 5 * c_i**2) / (4 * a**2 * eta**6)) * (the_lam * (eta**2 + 4 * q1**2)) - (3 * q1 * (1 - 5 * c_i**2) / (4 * a**2 * eta**4)) * lam_q1
    D_sp1_55 = ((1 - 3 * c_i**2) / (4 * a**2 * eta**2 * (1 + eta))) + ((1 - 3 * c_i**2) * q2**2 * (4 + 5 * eta) / (4 * a**2 * eta**6 * (1 + eta) ** 2)) + ((1 - 3 * c_i**2) / (8 * a**2 * eta**6)) * (eta**2 * (5 + 2 * (2 * q1 * c_th + 5 * q2 * s_th) - (3 + 2 * eps2) * c_2th) + 2 * q2 * (4 * (1 + eps2) * (2 + eps2) * s_th + (3 * eta + 4 * eps2) * q2)) + (3 * q1 * (1 - 5 * c_i**2) / (4 * a**2 * eta**6)) * (4 * q2 * eps3 - eta**2 * c_th) + (3 * q1 * q2 * (1 - 5 * c_i**2) / (a**2 * eta**6)) * the_lam - (3 * q1 * (1 - 5 * c_i**2) / (4 * a**2 * eta**4)) * lam_q2
    D_sp1_56 = 0
    D_sp1_61 = -(2 / a) * Omega_sp1
    D_sp1_62 = (3 * c_i / (2 * a**2 * eta**4 * (1 + eps2) ** 2)) * ((1 + eps2) ** 3 - eta**3)
    D_sp1_63 = -(3 * eps3 * s_i / (2 * a**2 * eta**4)) - (3 * s_i / (2 * a**2 * eta**4)) * the_lam
    D_sp1_64 = (3 * c_i / (2 * a**2 * eta**6)) * (4 * q1 * eps3 + eta**2 * s_th) + (6 * q1 * c_i / (a**2 * eta**6)) * the_lam - (3 * c_i / (2 * a**2 * eta**4)) * lam_q1
    D_sp1_65 = (3 * c_i / (2 * a**2 * eta**6)) * (4 * q2 * eps3 - eta**2 * c_th) + (6 * q2 * c_i / (a**2 * eta**6)) * the_lam - (3 * c_i / (2 * a**2 * eta**4)) * lam_q2
    D_sp1_66 = 0

    D_sp1 = np.array([
        [D_sp1_11, D_sp1_12, D_sp1_13, D_sp1_14, D_sp1_15, D_sp1_16],
        [D_sp1_21, D_sp1_22, D_sp1_23, D_sp1_24, D_sp1_25, D_sp1_26],
        [D_sp1_31, D_sp1_32, D_sp1_33, D_sp1_34, D_sp1_35, D_sp1_36],
        [D_sp1_41, D_sp1_42, D_sp1_43, D_sp1_44, D_sp1_45, D_sp1_46],
        [D_sp1_51, D_sp1_52, D_sp1_53, D_sp1_54, D_sp1_55, D_sp1_56],
        [D_sp1_61, D_sp1_62, D_sp1_63, D_sp1_64, D_sp1_65, D_sp1_66],
    ], dtype=float)

    lam_sp2 = -(3 * eps3 * s_i**2 * c_2th / (4 * a**2 * eta**4 * (1 + eta))) * (1 + eps2) * (2 + eps2) - (s_i**2 / (8 * a**2 * eta**2 * (1 + eta))) * (3 * (q1 * s_th + q2 * c_th) + (q1 * s_3th - q2 * c_3th)) - ((3 - 5 * c_i**2) / (8 * a**2 * eta**4)) * (3 * (q1 * s_th + q2 * c_th) + 3 * s_2th + (q1 * s_3th - q2 * c_3th))
    a_sp2 = -(3 * s_i**2 / (2 * a * eta**6)) * (1 + eps2) ** 3 * c_2th
    theta_sp2 = lam_sp2 - (s_i**2 / (32 * a**2 * eta**4 * (1 + eta))) * (36 * q1 * q2 - 4 * (3 * eta**2 + 5 * eta - 1) * (q1 * s_th + q2 * c_th) + 12 * eps2 * q1 * q2 - 32 * (1 + eta) * s_2th - (eta**2 + 12 * eta + 39) * (q1 * s_3th - q2 * c_3th) + 36 * q1 * q2 * c_4th - 18 * (q1**2 - q2**2) * s_4th + 3 * q2 * (3 * q1**2 - q2**2) * c_5th - 3 * q1 * (q1**2 - 3 * q2**2) * s_5th)
    i_sp2 = -(s_2i / (8 * a**2 * eta**4)) * (3 * (q1 * c_th - q2 * s_th) + 3 * c_2th + (q1 * c_3th + q2 * s_3th))
    q1_sp2 = (q2 * (3 - 5 * c_i**2) / (8 * a**2 * eta**4)) * (3 * (q1 * s_th + q2 * c_th) + 3 * s_2th + (q1 * s_3th - q2 * c_3th)) + (s_i**2 / (8 * a**2 * eta**4)) * (3 * (eta**2 - q1**2) * c_th + 3 * q1 * q2 * s_th - (eta**2 + 3 * q1**2) * c_3th - 3 * q1 * q2 * s_3th) - (3 * s_i**2 * c_2th / (16 * a**2 * eta**4)) * (10 * q1 + (8 + 3 * q1**2 + q2**2) * c_th + 2 * q1 * q2 * s_th + 6 * (q1 * c_2th + q2 * s_2th) + (q1**2 - q2**2) * c_3th + 2 * q1 * q2 * s_3th)
    q2_sp2 = -(q1 * (3 - 5 * c_i**2) / (8 * a**2 * eta**4)) * (3 * (q1 * s_th + q2 * c_th) + 3 * s_2th + (q1 * s_3th - q2 * c_3th)) - (s_i**2 / (8 * a**2 * eta**4)) * (3 * (eta**2 - q2**2) * s_th + 3 * q1 * q2 * c_th + (eta**2 + 3 * q2**2) * s_3th + 3 * q1 * q2 * c_3th) - (3 * s_i**2 * c_2th / (16 * a**2 * eta**4)) * (10 * q2 + (8 + q1**2 + 3 * q2**2) * s_th + 2 * q1 * q2 * c_th + 6 * (q1 * s_2th - q2 * c_2th) + (q1**2 - q2**2) * s_3th - 2 * q1 * q2 * c_3th)
    Omega_sp2 = -(c_i / (4 * a**2 * eta**4)) * (3 * (q1 * s_th + q2 * c_th) + 3 * s_2th + (q1 * s_3th - q2 * c_3th))

    D_sp2_11 = -(1 / a) * a_sp2
    D_sp2_12 = (3 * s_i**2 / (2 * a * eta**6)) * (1 + eps2) ** 2 * (3 * eps3 * c_2th + 2 * (1 + eps2) * s_2th)
    D_sp2_13 = -(3 * s_2i * c_2th / (2 * a * eta**6)) * (1 + eps2) ** 3
    D_sp2_14 = -(9 * s_i**2 * c_2th / (2 * a * eta**8)) * (1 + eps2) ** 2 * (2 * q1 * (1 + eps2) + eta**2 * c_th)
    D_sp2_15 = -(9 * s_i**2 * c_2th / (2 * a * eta**8)) * (1 + eps2) ** 2 * (2 * q2 * (1 + eps2) + eta**2 * s_th)
    D_sp2_16 = 0
    D_sp2_21 = -(2 / a) * theta_sp2
    D_sp2_22 = -(1 / (8 * a**2 * eta**4)) * (3 * (3 - 5 * c_i**2) * ((q1 * c_th - q2 * s_th) + 2 * c_2th + (q1 * c_3th + q2 * s_3th)) - s_i**2 * (5 * (q1 * c_th - q2 * s_th) + 16 * c_2th + 9 * (q1 * c_3th + q2 * s_3th)))
    D_sp2_23 = -(s_2i / (8 * a**2 * eta**4)) * (10 * (q1 * s_th + q2 * c_th) + 7 * s_2th + 2 * (q1 * s_3th - q2 * c_3th))
    D_sp2_24 = -((3 - 5 * c_i**2) / (8 * a**2 * eta**6)) * (4 * q1 * (3 * s_2th + q2 * (3 * c_th - c_3th)) + (eta**2 + 4 * q1**2) * (3 * s_th + s_3th)) - (s_i**2 * (3 * s_th + s_3th) / (8 * a**2 * eta**2 * (1 + eta))) - (s_i**2 / (32 * a**2 * eta**4 * (1 + eta))) * (36 * q2 - 4 * (2 + 3 * eta) * s_th - (eta * (12 + eta) + 39) * s_3th + 9 * eps1 * s_5th + 12 * q2 * (2 * q1 * c_th + q2 * s_th) + 9 * q1 * (q1 * s_3th - q2 * c_3th) + 18 * (3 * q1 * s_4th + 2 * q2 * c_4th) - 3 * q1 * (q1 * s_5th - 11 * q2 * c_5th) + 24 * ((1 + eps2) * (2 + eps2) * s_th + eps3 * (3 + 2 * eps2) * c_th) * c_2th) - (3 * s_i**2 / (32 * a**2 * eta**4 * (1 + eta) ** 2)) * (4 * s_th - 6 * q1 * s_4th - q1 * (q1 * s_5th + q2 * c_5th)) + (q1 * s_i**2 / (8 * a**2 * eta**6 * (1 + eta))) * (20 * (1 + eta) * (q1 * s_th + q2 * c_th) + 32 * (1 + eta) * s_2th + 3 * (4 + 3 * eta) * (q1 * s_3th - q2 * c_3th)) - (q1 * s_i**2 * (4 + 5 * eta) / (32 * a**2 * eta**6 * (1 + eta) ** 2)) * (24 * (q1 * s_th + q2 * c_th) + 24 * eps3 * (1 + eps2) * (2 + eps2) * c_2th - (27 + 3 * eta) * (q1 * s_3th - q2 * c_3th) - 18 * s_4th - 3 * (q1 * s_5th + q2 * c_5th) + 12 * q2 * ((3 + eps2) * q1 + 3 * (q1 * c_4th + q2 * s_4th) + q1 * (q1 * c_5th + q2 * s_5th)))
    D_sp2_25 = -((3 - 5 * c_i**2) / (8 * a**2 * eta**6)) * (4 * q2 * (3 * s_2th + q1 * (3 * s_th + s_3th)) + (eta**2 + 4 * q2**2) * (3 * c_th - c_3th)) - (s_i**2 * (3 * c_th - c_3th) / (8 * a**2 * eta**2 * (1 + eta))) - (s_i**2 / (32 * a**2 * eta**4 * (1 + eta))) * (36 * q1 - 4 * (2 + 3 * eta) * c_th + (eta * (12 + eta) + 39) * c_3th + 9 * eps1 * c_5th + 12 * q1 * (q1 * c_th + 2 * q2 * s_th) + 9 * q2 * (q1 * s_3th - q2 * c_3th) + 18 * (2 * q1 * c_4th + 7 * q2 * s_4th) + 3 * q2 * (11 * q1 * s_5th - q2 * c_5th) + 24 * (eps3 * (3 + 2 * eps2) * s_th - (1 + eps2) * (2 + eps2) * c_th) * c_2th) - (3 * s_i**2 / (32 * a**2 * eta**4 * (1 + eta) ** 2)) * (4 * c_th - 6 * q2 * s_4th - q2 * (q1 * s_5th + q2 * c_5th)) + (q2 * s_i**2 / (8 * a**2 * eta**6 * (1 + eta))) * (20 * (1 + eta) * (q1 * s_th + q2 * c_th) + 32 * (1 + eta) * s_2th + 3 * (4 + 3 * eta) * (q1 * s_3th - q2 * c_3th)) - (q2 * s_i**2 * (4 + 5 * eta) / (32 * a**2 * eta**6 * (1 + eta) ** 2)) * (24 * (q1 * s_th + q2 * c_th) + 24 * eps3 * (1 + eps2) * (2 + eps2) * c_2th - (27 + 3 * eta) * (q1 * s_3th - q2 * c_3th) - 18 * s_4th - 3 * (q1 * s_5th + q2 * c_5th) + 12 * q2 * ((3 + eps2) * q1 + 3 * (q1 * c_4th + q2 * s_4th) + q1 * (q1 * c_5th + q2 * s_5th)))
    D_sp2_26 = 0
    D_sp2_31 = -(2 / a) * i_sp2
    D_sp2_32 = (3 * s_2i / (8 * a**2 * eta**4)) * ((q1 * s_th + q2 * c_th) + 2 * s_2th + (q1 * s_3th - q2 * c_3th))
    D_sp2_33 = -(c_2i / (4 * a**2 * eta**4)) * (3 * (q1 * c_th - q2 * s_th) + 3 * c_2th + (q1 * c_3th + q2 * s_3th))
    D_sp2_34 = -(s_2i / (8 * a**2 * eta**6)) * (4 * q1 * (3 * c_2th - q2 * (3 * s_th - s_3th)) + (eta**2 + 4 * q1**2) * (3 * c_th + c_3th))
    D_sp2_35 = -(s_2i / (8 * a**2 * eta**6)) * (4 * q2 * (3 * c_2th + q1 * (3 * c_th + c_3th)) - (eta**2 + 4 * q2**2) * (3 * s_th - s_3th))
    D_sp2_36 = 0
    D_sp2_41 = -(2 / a) * q1_sp2
    D_sp2_42 = (3 * q2 * (3 - 5 * c_i**2) / (8 * a**2 * eta**4)) * ((q1 * c_th - q2 * s_th) + 2 * c_2th + (q1 * c_3th + q2 * s_3th)) + (3 * s_i**2 / (16 * a**2 * eta**4)) * ((2 * eps2 * q2 - 9 * q2 * (q1 * c_3th + q2 * s_3th) + 12 * (q1 * s_4th - q2 * c_4th) - 5 * q2 * (q1 * c_5th + q2 * s_5th)) + 0.5 * (4 * (1 + 3 * q1**2) * s_th + 40 * q1 * s_2th + (28 + 17 * eps1) * s_3th + 5 * eps1 * s_5th))
    D_sp2_43 = -(s_2i / (16 * a**2 * eta**4)) * ((36 * q1 * (q1 * c_th - q2 * s_th) + 30 * (q1 * c_2th - q2 * s_2th) - q2 * (q1 * s_3th - q2 * c_3th) + 9 * (q1 * c_4th + q2 * s_4th) + 3 * q2 * (q1 * s_5th - q2 * c_5th)) + 0.5 * (6 * q1 * (3 + 2 * q1 * c_th) + 12 * (1 - 4 * eps1) * c_th + (28 + 17 * eps1) * c_3th + 3 * eps1 * c_5th))
    D_sp2_44 = (q2 * (3 - 5 * c_i**2) / (8 * a**2 * eta**6)) * (4 * q1 * (3 * s_2th + q2 * (3 * c_th - c_3th)) + (eta**2 + 4 * q1**2) * (3 * s_th + s_3th)) - (s_i**2 / (8 * a**2 * eta**4)) * ((8 * q1 * c_3th - 3 * q2 * (s_th - s_3th)) + 3 * (5 + eps2 + 3 * c_2th + 3 * (q1 * c_3th + q2 * s_3th)) * c_2th) - (3 * q1 * s_i**2 / (4 * a**2 * eta**6)) * (2 * q1 * ((q1 * c_th - q2 * s_th) + (q1 * c_3th + q2 * s_3th)) + (9 * c_th - c_3th + 2 * q1 * (5 + eps2) + 6 * (q1 * c_2th + q2 * s_2th) + 2 * q1 * (q1 * c_3th + q2 * s_3th)) * c_2th)
    D_sp2_45 = ((3 - 5 * c_i**2) / (8 * a**2 * eta**6)) * ((eta**2 + 4 * q2**2) * (3 * s_2th + q1 * (3 * s_th + s_3th)) + 2 * (eta**2 + 2 * q2**2) * q2 * (3 * c_th - c_3th)) + (s_i**2 / (16 * a**2 * eta**4)) * (6 * (q1 * s_th + 2 * q2 * c_th) - (9 * q1 * s_3th + q2 * c_3th) - 9 * s_4th - 3 * (q1 * s_5th + q2 * c_5th)) - (3 * q2 * s_i**2 / (8 * a**2 * eta**6)) * (2 * q1 * (3 + 2 * (2 * q1 * c_th - q2 * s_th) + 10 * c_2th + 3 * (q1 * c_3th + q2 * s_3th) + (q1 * c_5th + q2 * s_5th)) + (8 * c_th + 9 * c_3th + 6 * (q1 * c_4th + q2 * s_4th) - c_5th))
    D_sp2_46 = 0
    D_sp2_51 = -(2 / a) * q2_sp2
    D_sp2_52 = -(3 * q1 * (3 - 5 * c_i**2) / (8 * a**2 * eta**4)) * ((q1 * c_th - q2 * s_th) + 2 * c_2th + (q1 * c_3th + q2 * s_3th)) + (3 * s_i**2 / (16 * a**2 * eta**4)) * ((2 * eps2 * q1 + 9 * q1 * (q1 * c_3th + q2 * s_3th) - 12 * (q1 * c_4th + q2 * s_4th) - 5 * q1 * (q1 * c_5th + q2 * s_5th)) + 0.5 * (4 * (1 + 3 * q2**2) * c_th + 40 * q2 * s_2th - (28 + 17 * eps1) * c_3th + 5 * eps1 * c_5th))
    D_sp2_53 = -(s_2i / (16 * a**2 * eta**4)) * ((36 * q1 * (q1 * s_th + q2 * c_th) + 30 * (q1 * s_2th + q2 * c_2th) + q1 * (q1 * s_3th - q2 * c_3th) + 9 * (q1 * s_4th - q2 * c_4th) + 3 * q1 * (q1 * s_5th - q2 * c_5th)) - 0.5 * (6 * q2 * (3 + 2 * q2 * s_th) + 12 * (1 + 2 * eps1) * s_th - (28 + 17 * eps1) * s_3th + 3 * eps1 * s_5th))
    D_sp2_54 = -((3 - 5 * c_i**2) / (8 * a**2 * eta**6)) * ((eta**2 + 4 * q1**2) * (3 * s_2th + q2 * (3 * c_th - c_3th)) + 2 * (eta**2 + 2 * q1**2) * q1 * (3 * s_th + s_3th)) - (s_i**2 / (16 * a**2 * eta**4)) * (6 * (2 * q1 * s_th + q2 * c_th) + (q1 * s_3th + 9 * q2 * c_3th) + 9 * s_4th - 3 * (q1 * s_5th + q2 * c_5th)) + (3 * q1 * s_i**2 / (8 * a**2 * eta**6)) * (2 * q2 * (3 - 2 * (q1 * c_th - 2 * q2 * s_th) - 10 * c_2th - 3 * (q1 * c_3th + q2 * s_3th) + (q1 * c_5th + q2 * s_5th)) + (8 * s_th - 9 * s_3th - 6 * (q1 * s_4th - q2 * c_4th) - s_5th))
    D_sp2_55 = -(q1 * (3 - 5 * c_i**2) / (8 * a**2 * eta**6)) * ((eta**2 + 4 * q2**2) * (3 * c_th - c_3th) + 4 * q2 * (3 * s_2th + q1 * (3 * s_th + s_3th))) - (s_i**2 / (8 * a**2 * eta**4)) * (8 * q2 * s_3th + 3 * q1 * (c_th + c_3th) + 3 * (5 + eps2 - 3 * c_2th - (q1 * c_3th - q2 * s_3th)) * c_2th) - (3 * s_i**2 * q2 * c_2th / (4 * a**2 * eta**6)) * (9 * s_th - s_3th + 2 * q2 * (5 + eps2) + 6 * (q1 * s_2th - q2 * c_2th) + 2 * q1 * (q1 * s_3th - q2 * c_3th))
    D_sp2_56 = 0
    D_sp2_61 = -(2 / a) * Omega_sp2
    D_sp2_62 = -(3 * c_i / (4 * a**2 * eta**4)) * ((q1 * c_th - q2 * s_th) + 2 * c_2th + (q1 * c_3th + q2 * s_3th))
    D_sp2_63 = (s_i / (4 * a**2 * eta**4)) * (3 * (q1 * s_th + q2 * c_th) + 3 * s_2th + (q1 * s_3th - q2 * c_3th))
    D_sp2_64 = -(c_i / (4 * a**2 * eta**6)) * (4 * q1 * (3 * s_2th + q2 * (3 * c_th - c_3th)) + (eta**2 + 4 * q1**2) * (3 * s_th + s_3th))
    D_sp2_65 = -(c_i / (4 * a**2 * eta**6)) * (4 * q2 * (3 * s_2th + q1 * (3 * s_th + s_3th)) + (eta**2 + 4 * q2**2) * (3 * c_th - c_3th))
    D_sp2_66 = 0

    D_sp2 = np.array([
        [D_sp2_11, D_sp2_12, D_sp2_13, D_sp2_14, D_sp2_15, D_sp2_16],
        [D_sp2_21, D_sp2_22, D_sp2_23, D_sp2_24, D_sp2_25, D_sp2_26],
        [D_sp2_31, D_sp2_32, D_sp2_33, D_sp2_34, D_sp2_35, D_sp2_36],
        [D_sp2_41, D_sp2_42, D_sp2_43, D_sp2_44, D_sp2_45, D_sp2_46],
        [D_sp2_51, D_sp2_52, D_sp2_53, D_sp2_54, D_sp2_55, D_sp2_56],
        [D_sp2_61, D_sp2_62, D_sp2_63, D_sp2_64, D_sp2_65, D_sp2_66],
    ], dtype=float)

    lam_osc = lambda_ + coef * (lam_lp + lam_sp1 + lam_sp2)
    a_osc = a + coef * (a_lp + a_sp1 + a_sp2)
    theta_osc = theta + coef * (theta_lp + theta_sp1 + theta_sp2)
    i_osc = i + coef * (i_lp + i_sp1 + i_sp2)
    q1_osc = q1 + coef * (q1_lp + q1_sp1 + q1_sp2)
    q2_osc = q2 + coef * (q2_lp + q2_sp1 + q2_sp2)
    Omega_osc = Omega + coef * (Omega_lp + Omega_sp1 + Omega_sp2)

    D_J2 = DI + coef * (D_lp + D_sp1 + D_sp2)
    osc_c = np.array([a_osc, theta_osc, i_osc, q1_osc, q2_osc, Omega_osc], dtype=float)

    _ = total_E, n, lam_osc
    return D_J2, osc_c


def mean2osc(oe_mean_in, J2=J2, mu=mu_E, R=R_E):
    oe_mean_arr, scalar_input = _normalize_oe_input(oe_mean_in)
    if J2 == 0:
        return _restore_oe_output(oe_mean_arr.copy(), scalar_input)

    oe_osc_out = np.zeros_like(oe_mean_arr)
    for oe_idx, oe_row in enumerate(oe_mean_arr):
        _ = oe_idx
        oe_mean = _koe_to_nonsingular(oe_row)
        _, oe_osc = meanoscclosed(oe_mean, J2=J2, Re=R, mu=mu)
        oe_osc_out[oe_idx] = _nonsingular_to_koe(oe_osc)
    return _restore_oe_output(oe_osc_out, scalar_input)


def osc2mean(oe_osc_in, J2=J2, tol=1e-8, max_iter=20, method="iterative", mu=mu_E, R=R_E):
    oe_osc_arr, scalar_input = _normalize_oe_input(oe_osc_in)
    if J2 == 0:
        return _restore_oe_output(oe_osc_arr.copy(), scalar_input)

    if method not in {"iterative", "first_order"}:
        raise ValueError("method must be 'iterative' or 'first_order'")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    oe_mean_out = np.zeros_like(oe_osc_arr)
    for idx, oe_row in enumerate(oe_osc_arr):
        oe_osc = _koe_to_nonsingular(oe_row)
        oe_mean = oe_osc.copy()

        if method == "first_order":
            _, oe_mean = meanoscclosed(oe_mean, J2=-J2, Re=R, mu=mu)
        else:
            d_oe = np.full(6, 10.0 * tol, dtype=float)
            for _iter in range(max_iter):
                _ = _iter
                D_J2, oe_osc_calc = meanoscclosed(oe_mean, J2=J2, Re=R, mu=mu)
                residual = oe_osc_calc - oe_osc
                try:
                    d_oe = -np.linalg.solve(D_J2, residual)
                except np.linalg.LinAlgError as exc:
                    raise RuntimeError(f"osc2mean failed for row {idx}: singular Jacobian during Newton solve") from exc
                oe_mean = oe_mean + d_oe
                if np.all(np.abs(d_oe) < tol):
                    break
            else:
                raise RuntimeError(
                    f"osc2mean failed to converge for row {idx} within {max_iter} iterations; final update norm={np.linalg.norm(d_oe):.3e}"
                )

        oe_mean_out[idx] = _nonsingular_to_koe(oe_mean)
    return _restore_oe_output(oe_mean_out, scalar_input)
