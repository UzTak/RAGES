import itertools
import warnings

import numpy as np

import dynamics.dynamics_trans as dyn


def _as_vector(name, value, size):
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return array


def impulsive_control(
    chief_oe,
    initial_aroe,
    desired_aroe,
    dt,
    stm=None,
    cm=None,
    Rp=dyn.R_E,
    mu=dyn.mu_E,
    J2=dyn.J2,
    options=None,
):
    """
    Port of ``impulsive_control.m`` using the repo's Python dynamics helpers.

    Inputs use the repo-native chief orbital-element convention
    ``[a, e, i, RAAN, w, M]`` with mean anomaly in the sixth entry.
    The returned maneuvers are row-major with shape ``(N, 3)``.
    """
    del options

    chief_oe = _as_vector("chief_oe", chief_oe, 6)
    initial_aroe = _as_vector("initial_aroe", initial_aroe, 6)
    desired_aroe = _as_vector("desired_aroe", desired_aroe, 6)
    dt = float(dt)

    stm = dyn.stm_roe if stm is None else stm
    cm = dyn.cim_roe if cm is None else cm

    a_droe = desired_aroe - stm(chief_oe, dt) @ initial_aroe

    a = chief_oe[0]
    e = chief_oe[1]
    inc = chief_oe[2]

    n = np.sqrt(mu / a**3)
    eta = np.sqrt(1.0 - e**2)
    kappa = 0.75 * J2 * Rp**2 * np.sqrt(mu) / (a ** 3.5 * eta**4)
    P = 3.0 * np.cos(inc) ** 2 - 1.0
    Q = 5.0 * np.cos(inc) ** 2 - 1.0

    raan_dot = -2.0 * np.cos(inc) * kappa
    aop_dot = kappa * Q
    M_dot = n + kappa * eta * P

    dvmin_da, dvmin_dlambda, dvmin_de, dvmin_di = _calculate_reachable_delta_v(
        a_droe,
        chief_oe,
        dt,
        n,
        stm,
        cm,
    )

    dv_candidates = np.array([dvmin_da, dvmin_dlambda, dvmin_de], dtype=float)
    dvmin_dom = int(np.argmax(dv_candidates)) + 1
    dvmin = float(dv_candidates[dvmin_dom - 1])

    if dvmin_dom == 2:
        warnings.warn(
            "dlambda dominant case. de change is not addressed; retrigger guidance "
            "when the problem is no longer dlambda dominant.",
            RuntimeWarning,
            stacklevel=2,
        )

    T_opts_ip, m_ip = _compute_optimal_maneuver_times_ip(
        a_droe,
        chief_oe,
        dt,
        n,
        aop_dot,
        M_dot,
    )
    T_opts_oop, m_oop = _compute_optimal_maneuver_times_oop(
        a_droe,
        chief_oe,
        dt,
        n,
        kappa,
        eta,
        P,
        Q,
    )

    Sn_ip, dv_ip = _compute_nested_reachable_sets_and_deltav_ip(
        T_opts_ip,
        m_ip,
        a_droe,
        chief_oe,
        dt,
        dvmin,
        dvmin_dom,
        stm,
        cm,
        raan_dot,
        aop_dot,
        M_dot,
    )
    Sn_oop, dv_oop = _compute_nested_reachable_sets_and_deltav_oop(
        T_opts_oop,
        m_oop,
        chief_oe,
        dt,
        dvmin_di,
        stm,
        cm,
        raan_dot,
        aop_dot,
        M_dot,
    )

    ts_ip, dvs_ip, cost_ip = _determine_best_maneuver_plan_ip(
        Sn_ip,
        dv_ip,
        T_opts_ip,
        m_ip,
        a_droe,
        dvmin,
        dvmin_dom,
        dt,
    )
    ts_oop, dvs_oop, cost_oop = _determine_best_maneuver_plan_oop(
        dv_oop,
        T_opts_oop,
        dvmin_di,
    )

    time_parts = []
    burn_parts = []

    if ts_ip.size:
        time_parts.append(ts_ip)
        burn_parts.append(dvs_ip)

    if ts_oop.size:
        time_parts.append(ts_oop)
        burn_parts.append(dvs_oop)

    if time_parts:
        t_maneuvers = np.concatenate(time_parts)
        maneuvers_internal = np.concatenate(burn_parts, axis=1)
        sort_idx = np.argsort(t_maneuvers)
        t_maneuvers = t_maneuvers[sort_idx]
        maneuvers = maneuvers_internal[:, sort_idx].T
    else:
        t_maneuvers = np.empty(0, dtype=float)
        maneuvers = np.empty((0, 3), dtype=float)

    total_cost = float(cost_ip + cost_oop)
    return t_maneuvers, maneuvers, total_cost


def _calculate_reachable_delta_v(a_droe, chief_oe, dt, n, stm, cm):
    dvmin_da = abs(a_droe[0]) / (2.0 / n)
    dvmin_de = np.linalg.norm(a_droe[2:4]) / (2.0 / n)

    M0 = np.mod(chief_oe[5], 2.0 * np.pi)
    t0 = np.mod(2.0 * np.pi - M0, 2.0 * np.pi) / n
    if t0 < 0.0:
        raise AssertionError("t0 >= 0 must hold")

    chief_oe_peri = chief_oe.copy()
    chief_oe_peri[5] = 0.0
    droe0 = stm(chief_oe_peri, dt - t0) @ cm(chief_oe_peri) @ np.array([0.0, 1.0, 0.0])
    m = -2.0 * abs(droe0[0]) / abs(droe0[1])
    # ``dyn.cim_roe`` already returns dimensionalized ROE, unlike the MATLAB
    # helper this port was based on, so no extra ``a`` scaling belongs here.
    dvmin_dlambda = abs((m * a_droe[1] - a_droe[0]) / droe0[0])

    dvmin_di = np.linalg.norm(a_droe[4:6]) / (1.0 / n)
    return float(dvmin_da), float(dvmin_dlambda), float(dvmin_de), float(dvmin_di)


def _compute_optimal_maneuver_times_ip(a_droe, chief_oe, dt, n, aop_dot, M_dot):
    del n

    phi0 = aop_dot * dt + chief_oe[4]
    x0 = np.arctan2(a_droe[3], a_droe[2]) - phi0
    m_ip = 0
    M0 = np.mod(chief_oe[5], 2.0 * np.pi)

    while x0 + m_ip * np.pi < 0.0:
        m_ip += 1

    x0 = x0 + m_ip * np.pi
    if x0 < M0:
        x0 = x0 + 2.0 * np.pi

    T_opts_ip = [(x0 - M0) / M_dot]
    dTopt = np.pi / M_dot
    while T_opts_ip[-1] + dTopt < dt:
        T_opts_ip.append(T_opts_ip[-1] + dTopt)

    return np.asarray(T_opts_ip, dtype=float), int(m_ip)


def _compute_optimal_maneuver_times_oop(a_droe, chief_oe, dt, n, kappa, eta, P, Q):
    x = np.arctan2(a_droe[5], a_droe[4]) - chief_oe[4]
    denom = n + kappa * (eta * P + Q)

    m_oop = 0
    M0 = np.mod(chief_oe[5], 2.0 * np.pi)
    while x + m_oop * np.pi < 0.0:
        m_oop += 1

    x0 = x + m_oop * np.pi
    if x0 < M0:
        x0 = x0 + 2.0 * np.pi

    T_opts_oop = [(x0 - M0) / denom]
    dTopt = np.pi / denom
    while T_opts_oop[-1] + dTopt < dt:
        T_opts_oop.append(T_opts_oop[-1] + dTopt)

    return np.asarray(T_opts_oop, dtype=float), int(m_oop)


def _alternating_signs(count, phase_offset):
    signs = np.ones(count, dtype=float)
    odd_mask = (np.arange(count) + phase_offset) % 2 == 1
    signs[odd_mask] = -1.0
    return signs


def _compute_nested_reachable_sets_and_deltav_ip(
    T_opts_ip,
    m_ip,
    a_droe,
    chief_oe,
    dt,
    dvmin,
    dvmin_dom,
    stm,
    cm,
    raan_dot,
    aop_dot,
    M_dot,
):
    count = T_opts_ip.size
    if dvmin_dom == 1:
        dv_ip = np.tile(np.array([[0.0], [np.sign(a_droe[0])], [0.0]]), (1, count))
    else:
        signs = _alternating_signs(count, m_ip)
        dv_ip = np.vstack(
            (
                np.zeros(count, dtype=float),
                signs,
                np.zeros(count, dtype=float),
            )
        )

    Sn_ip = np.zeros((6, count), dtype=float)
    M0 = np.mod(chief_oe[5], 2.0 * np.pi)
    for idx, t_opt in enumerate(T_opts_ip):
        chief_oe_k = np.array(
            [
                chief_oe[0],
                chief_oe[1],
                chief_oe[2],
                chief_oe[3] + raan_dot * t_opt,
                chief_oe[4] + aop_dot * t_opt,
                M0 + M_dot * t_opt,
            ],
            dtype=float,
        )
        # The repo's control map is already in dimensionalized ROE units.
        Sn_ip[:, idx] = dvmin * stm(chief_oe_k, dt - t_opt) @ cm(chief_oe_k) @ dv_ip[:, idx]

    return Sn_ip, dv_ip


def _compute_nested_reachable_sets_and_deltav_oop(
    T_opts_oop,
    m_oop,
    chief_oe,
    dt,
    dvmin_di,
    stm,
    cm,
    raan_dot,
    aop_dot,
    M_dot,
):
    count = T_opts_oop.size
    Sn_oop = np.zeros((6, count), dtype=float)

    signs = _alternating_signs(count, m_oop)
    dv_oop = np.tile(np.array([[0.0], [0.0], [1.0]]), (1, count)) * signs.reshape(1, -1)

    M0 = np.mod(chief_oe[5], 2.0 * np.pi)
    for idx, t_opt in enumerate(T_opts_oop):
        chief_oe_k = np.array(
            [
                chief_oe[0],
                chief_oe[1],
                chief_oe[2],
                chief_oe[3] + raan_dot * t_opt,
                chief_oe[4] + aop_dot * t_opt,
                M0 + M_dot * t_opt,
            ],
            dtype=float,
        )
        Sn_oop[:, idx] = dvmin_di * stm(chief_oe_k, dt - t_opt) @ cm(chief_oe_k) @ dv_oop[:, idx]

    return Sn_oop, dv_oop


def _solve_linear_system(M, b):
    cond = np.linalg.cond(M)
    rcond = 0.0 if not np.isfinite(cond) else 1.0 / cond
    if rcond < 1e-6:
        return np.array([-1.0, -1.0, -1.0], dtype=float)
    return np.linalg.solve(M, b)


def _de_sign_for_da_dominant(a_droe, m_ip, inds):
    signs = np.ones(len(inds), dtype=float)
    for j, idx in enumerate(inds):
        is_odd = (m_ip + idx) % 2 != 0
        if (a_droe[0] > 0.0 and is_odd) or (a_droe[0] < 0.0 and not is_odd):
            signs[j] = -1.0
    return signs


def _determine_best_maneuver_plan_ip(Sn_ip, dv_ip, T_opts_ip, m_ip, a_droe, dvmin, dvmin_dom, dt):
    del dt

    dvs_ip = np.empty((3, 0), dtype=float)
    ts_ip = np.empty(0, dtype=float)
    cost_ip = np.inf

    for inds in itertools.combinations(range(T_opts_ip.size), 3):
        viable_Sn = Sn_ip[:, inds]

        if dvmin_dom == 1:
            aDemag_signs = _de_sign_for_da_dominant(a_droe, m_ip, inds)
            M = np.vstack(
                (
                    np.ones(3, dtype=float),
                    viable_Sn[1, :],
                    aDemag_signs * np.sqrt(viable_Sn[2, :] ** 2 + viable_Sn[3, :] ** 2),
                )
            )
            b = np.array([1.0, a_droe[0], np.linalg.norm(a_droe[2:4])], dtype=float)
            cs = _solve_linear_system(M, b)

            if np.any(cs < 0.0) or np.any(cs > 1.0):
                aDemag_signs = _de_sign_for_da_dominant(a_droe, m_ip, inds)
                M = np.vstack(
                    (
                        viable_Sn[0, :],
                        viable_Sn[1, :],
                        aDemag_signs * np.sqrt(viable_Sn[2, :] ** 2 + viable_Sn[3, :] ** 2),
                    )
                )
                b = np.array([a_droe[0], a_droe[1], np.linalg.norm(a_droe[2:4])], dtype=float)
                cs = _solve_linear_system(M, b)

        elif dvmin_dom == 2:
            M = np.vstack(
                (
                    viable_Sn[0, :],
                    viable_Sn[1, :],
                    np.sqrt(viable_Sn[2, :] ** 2 + viable_Sn[3, :] ** 2),
                )
            )
            b = np.array([a_droe[0], a_droe[1], np.linalg.norm(a_droe[2:4])], dtype=float)
            cs = _solve_linear_system(M, b)

        else:
            M = np.vstack(
                (
                    np.ones(3, dtype=float),
                    viable_Sn[1, :],
                    viable_Sn[0, :],
                )
            )
            b = np.array([1.0, a_droe[1], a_droe[0]], dtype=float)
            cs = _solve_linear_system(M, b)

            if np.any(cs < 0.0) or np.any(cs > 1.0):
                M = np.vstack(
                    (
                        viable_Sn[0, :],
                        viable_Sn[1, :],
                        np.sqrt(viable_Sn[2, :] ** 2 + viable_Sn[3, :] ** 2),
                    )
                )
                b = np.array([a_droe[0], a_droe[1], np.linalg.norm(a_droe[2:4])], dtype=float)
                cs = _solve_linear_system(M, b)

        candidate_cost = float(np.sum(dvmin * np.abs(cs)))
        if candidate_cost < cost_ip:
            dvs_ip = dvmin * dv_ip[:, inds] * cs.reshape(1, -1)
            ts_ip = T_opts_ip[np.array(inds, dtype=int)]
            cost_ip = candidate_cost

    return ts_ip, dvs_ip, float(cost_ip)


def _determine_best_maneuver_plan_oop(dv_oop, T_opts_oop, dvmin_di):
    if T_opts_oop.size == 0:
        return np.empty(0, dtype=float), np.empty((3, 0), dtype=float), float(np.inf)

    dvs_oop = dvmin_di * dv_oop[:, [0]]
    ts_oop = T_opts_oop[[0]]
    cost_oop = np.linalg.norm(dvs_oop[:, 0])
    return ts_oop, dvs_oop, float(cost_oop)
