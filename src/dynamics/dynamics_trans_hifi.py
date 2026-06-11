"""
Ephemeris-based Cartesian dynamics in km/s units.

This module currently provides a minimal Earth spherical-harmonics path backed by
an ICGEM-style `.gfc` gravity file. Coefficients are assumed to be fully
normalized and accelerations are evaluated in a body-fixed frame before being
rotated back to the inertial frame.

"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import spiceypy as sp
from scipy.special import lpmn

from dynamics.constants import get_gm

@dataclass(slots=True)
class GravityModel:
    gm: float
    radius: float
    max_degree: int
    norm: str
    tide_system: str
    C: np.ndarray
    S: np.ndarray


@dataclass(slots=True)
class Spacecraft:
    mass: float = 100.0 # kg
    area: float = 5.0 # m^2
    cd: float  = 2.2  # drag coefficient 
    cr: float  = 1.5  # radiation pressure coefficient
    

def load_gfc(path: str | Path, requested_degree: int) -> GravityModel:
    """Load a fully normalized ICGEM gravity field up to the requested degree."""
        
    def _parse_gfc_value(value: str) -> float:
        return float(value.replace("D", "E").replace("d", "e"))

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    header: dict[str, str] = {}
    in_header = False
    header_complete = False
    lines = file_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("begin_of_head"):
            in_header = True
            continue
        if stripped.startswith("end_of_head"):
            header_complete = True
            break
        if not in_header or not stripped or stripped.startswith("key"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            header[parts[0]] = parts[-1]

    if not header_complete:
        raise ValueError(f"Gravity file '{file_path}' is missing an ICGEM header terminator.")
    if header.get("product_type") != "gravity_field":
        raise ValueError(f"Unsupported product_type '{header.get('product_type')}'.")
    if header.get("norm") != "fully_normalized":
        raise ValueError(f"Unsupported normalization '{header.get('norm')}'.")

    file_degree = int(header["max_degree"])
    degree = min(int(requested_degree), file_degree)
    C = np.zeros((degree + 1, degree + 1), dtype=float)
    S = np.zeros((degree + 1, degree + 1), dtype=float)

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("gfc"):
            continue
        parts = stripped.split()
        n = int(parts[1])
        m = int(parts[2])
        if n > degree or m > degree:
            continue
        C[n, m] = _parse_gfc_value(parts[3])
        S[n, m] = _parse_gfc_value(parts[4])

    return GravityModel(
        gm=_parse_gfc_value(header["earth_gravity_constant"]) / 1e9,  # Convert from m^3/s^2 to km^3/s^2
        radius=_parse_gfc_value(header["radius"]) / 1e3,  # Convert from m to km
        max_degree=degree,
        norm=header["norm"],
        tide_system=header.get("tide_system", ""),
        C=C,
        S=S,
    )


"""
Dynamics class 
"""

class EphemerisDynamics:
    def __init__(self, primary, epoch, spacecraft: Spacecraft, perturbations=None, frame="J2000"):
        self.primary = primary
        self.epoch = epoch
        self.sc = spacecraft
        self.frame = frame
        self.perturbations = perturbations or []
        self.mu = get_gm(primary)[0]

    def point_mass_acceleration(self, r):
        r_norm = np.linalg.norm(r)
        return -self.mu * r / r_norm**3

    def acceleration(self, et, state, sc=None):
        a_total = self.point_mass_acceleration(state[:3])

        for pert in self.perturbations:
            a_total += pert.acceleration(et, state, self, sc=self.sc)

        return a_total

    def rhs(self, et, state, sc=None):
        v = state[3:]
        a = self.acceleration(et, state, sc=self.sc)
        return np.hstack((v, a))


class SphericalHarmonicGravity:
    def __init__(
        self,
        body,
        degree=10,
        order=10,
        includes_central=False,
        gravity_file=None,
        body_fixed_frame=None,
    ):
        if body.lower() != "earth":
            raise NotImplementedError("Spherical-harmonic gravity is only implemented for Earth in v1.")
        self.body = body
        self.degree = int(degree)
        self.order = int(order)
        self.includes_central = includes_central
        self.gravity_file = Path(gravity_file) if gravity_file else Path(__file__).with_name("GGM05G.gfc")
        self.body_fixed_frame = body_fixed_frame or "IAU_EARTH"
        self._model: GravityModel | None = None

    @staticmethod
    def _legendre_n(max_degree: int, sin_lat: float) -> tuple[np.ndarray, np.ndarray]:
        """Return fully normalized associated Legendre functions and d/dphi."""

        Pmn, dPmn_dx = lpmn(max_degree, max_degree, sin_lat)
        Pbar = np.zeros((max_degree + 1, max_degree + 1), dtype=float)
        dPbar_dphi = np.zeros_like(Pbar)
        cos_lat = np.sqrt(max(0.0, 1.0 - sin_lat * sin_lat))

        for n in range(max_degree + 1):
            for m in range(n + 1):
                norm = np.sqrt(
                    (2.0 - (1.0 if m == 0 else 0.0))
                    * (2 * n + 1)
                    * math.factorial(n - m)
                    / math.factorial(n + m)
                )
                phase_free = ((-1) ** m) * Pmn[m, n]
                phase_free_dx = ((-1) ** m) * dPmn_dx[m, n]
                Pbar[n, m] = norm * phase_free
                dPbar_dphi[n, m] = norm * phase_free_dx * cos_lat

        return Pbar, dPbar_dphi

    def _acc_body_fixed(self, r_bf: np.ndarray) -> np.ndarray:
        """Evaluate the full gravity acceleration in body-fixed Cartesian coordinates."""

        model = self.model
        x, y, z = r_bf
        r = np.linalg.norm(r_bf)
        if r == 0.0:
            raise ValueError("Gravity acceleration is undefined at the central body's origin.")

        lon = np.arctan2(y, x)
        sin_lat = np.clip(z / r, -1.0, 1.0)
        lat = np.arcsin(sin_lat)
        cos_lat = np.cos(lat)

        Pbar, dPbar_dphi = self._legendre_n(model.max_degree, sin_lat)
        ar = -model.gm / (r * r)
        aphi, alam = 0.0, 0.0

        for n in range(2, model.max_degree + 1):
            rho_n = (model.radius / r) ** n
            radial_sum = 0.0
            lat_sum = 0.0
            lon_sum = 0.0
            for m in range(min(self.order, n) + 1):
                cos_mlon = np.cos(m * lon)
                sin_mlon = np.sin(m * lon)
                trig = model.C[n, m] * cos_mlon + model.S[n, m] * sin_mlon
                radial_sum += Pbar[n, m] * trig
                lat_sum += dPbar_dphi[n, m] * trig
                lon_sum += m * Pbar[n, m] * (model.S[n, m] * cos_mlon - model.C[n, m] * sin_mlon)
            ar += -model.gm / (r * r) * (n + 1) * rho_n * radial_sum
            aphi += model.gm / (r * r) * rho_n * lat_sum
            if abs(cos_lat) > 1.0e-12:
                alam += model.gm / (r * r) * rho_n * lon_sum / cos_lat

        er = np.array([cos_lat * np.cos(lon), cos_lat * np.sin(lon), sin_lat])
        ephi = np.array([-sin_lat * np.cos(lon), -sin_lat * np.sin(lon), cos_lat])
        elam = np.array([-np.sin(lon), np.cos(lon), 0.0])
        return ar * er + aphi * ephi + alam * elam

    @property
    def model(self) -> GravityModel:
        if self._model is None:
            self._model = load_gfc(self.gravity_file, self.degree)
        return self._model

    def acceleration(self, et, state, dyn, sc=None):
        r_inertial = np.asarray(state[:3], dtype=float)
        rot_ib = np.asarray(sp.pxform(dyn.frame, self.body_fixed_frame, et), dtype=float)
        r_body_fixed = rot_ib @ r_inertial
        full_body_fixed = self._acc_body_fixed(r_body_fixed)

        if not self.includes_central:
            r_norm = np.linalg.norm(r_body_fixed)
            central_body_fixed = -self.model.gm * r_body_fixed / r_norm**3
            full_body_fixed -= central_body_fixed

        rot_bi = np.asarray(sp.pxform(self.body_fixed_frame, dyn.frame, et), dtype=float)
        return rot_bi @ full_body_fixed


class ThirdBodyGravity:
    def __init__(self, body):
        self.body = body
        self.mu_body = get_gm(body)[0]

    def acceleration(self, et, state, dyn, sc=None):
        r_sc = state[:3]
        r_body, _ = sp.spkezr(self.body, et, dyn.frame, "NONE", dyn.primary)
        r_body = np.asarray(r_body[:3], dtype=float)

        rho = r_body - r_sc
        return self.mu_body * (rho / np.linalg.norm(rho) ** 3 - r_body / np.linalg.norm(r_body) ** 3)


class SolarRadiationPressure:

    def acceleration(self, et, state, dyn, sc=None):
        r = state[:3]
        r_sun, _ = sp.spkezr("Sun", et, dyn.frame, "NONE", dyn.primary)
        r_sun = np.asarray(r_sun[:3], dtype=float)
        rho = r_sun - r
        rho_norm = np.linalg.norm(rho)
        
        cr, area, mass = sc.cr, sc.area, sc.mass

        p_sr = 4.56e-6   # Solar radiation pressure at 1 AU in N/m^2
        return cr * area / mass * p_sr * (rho / rho_norm) / 1e3  # Convert from m/s^2 to km/s^2
    

class AtmosphericDrag:

    def __init__(self, model="harris-priester", body_fixed_frame="IAU_EARTH"):
        self.model = model
        self.body_fixed_frame = body_fixed_frame

    def _density_harris_priester(self, et, xeci, xecef, inertial_frame):
        """Return atmospheric density in kg/m^3."""

        _HP_H_TABLE = np.array([
            100.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,
            210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,
            320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0,
            520.0, 540.0, 560.0, 580.0, 600.0, 620.0, 640.0, 660.0, 680.0, 700.0,
            720.0, 740.0, 760.0, 780.0, 800.0, 840.0, 880.0, 920.0, 960.0, 1000.0,
        ], dtype=float)
        _HP_RHO_MIN_TABLE = np.array([
            4.974e+05, 2.490e+04, 8.377e+03, 3.899e+03, 2.122e+03, 1.263e+03,
            8.008e+02, 5.283e+02, 3.617e+02, 2.557e+02, 1.839e+02, 1.341e+02,
            9.949e+01, 7.488e+01, 5.709e+01, 4.403e+01, 3.430e+01, 2.697e+01,
            2.139e+01, 1.708e+01, 1.099e+01, 7.214e+00, 4.824e+00, 3.274e+00,
            2.249e+00, 1.558e+00, 1.091e+00, 7.701e-01, 5.474e-01, 3.916e-01,
            2.819e-01, 2.042e-01, 1.488e-01, 1.092e-01, 8.070e-02, 6.012e-02,
            4.519e-02, 3.430e-02, 2.632e-02, 2.043e-02, 1.607e-02, 1.281e-02,
            1.036e-02, 8.496e-03, 7.069e-03, 4.680e-03, 3.200e-03, 2.210e-03,
            1.560e-03, 1.150e-03,
        ], dtype=float)
        _HP_RHO_MAX_TABLE = np.array([
            4.974e+05, 2.490e+04, 8.710e+03, 4.059e+03, 2.215e+03, 1.344e+03,
            8.758e+02, 6.010e+02, 4.297e+02, 3.162e+02, 2.396e+02, 1.853e+02,
            1.455e+02, 1.157e+02, 9.308e+01, 7.555e+01, 6.182e+01, 5.095e+01,
            4.226e+01, 3.526e+01, 2.511e+01, 1.819e+01, 1.337e+01, 9.955e+00,
            7.492e+00, 5.684e+00, 4.355e+00, 3.362e+00, 2.612e+00, 2.042e+00,
            1.605e+00, 1.267e+00, 1.005e+00, 7.997e-01, 6.390e-01, 5.123e-01,
            4.121e-01, 3.325e-01, 2.691e-01, 2.185e-01, 1.779e-01, 1.452e-01,
            1.190e-01, 9.776e-02, 8.059e-02, 5.741e-02, 4.210e-02, 3.130e-02,
            2.360e-02, 1.810e-02,
        ], dtype=float)

        re = 6378.1363
        flattening = 1.0 / 298.257223563
        _, _, alt = sp.recgeo(np.asarray(xecef, dtype=float), re, flattening)
        h = alt
        if h < 100.0 or h >= 1000.0:
            return 0.0

        r_sun, _ = sp.spkezr("10", et, inertial_frame, "NONE", "399")
        r_sun = np.asarray(r_sun[:3], dtype=float)
        reci = np.asarray(xeci, dtype=float)
        er = reci / np.linalg.norm(reci)

        lag_sun = np.deg2rad(30.0)
        a_sun = np.arctan2(r_sun[1], r_sun[0])
        d_sun = np.arctan2(r_sun[2], np.sqrt(r_sun[0] ** 2 + r_sun[1] ** 2))

        eb = np.array([
            np.cos(d_sun) * np.cos(a_sun + lag_sun),
            np.cos(d_sun) * np.sin(a_sun + lag_sun),
            np.sin(d_sun),
        ])
        cospsi = (0.5 * (1.0 + np.dot(er, eb))) ** 1.0

        idx = np.searchsorted(_HP_H_TABLE, h, side="right") - 1
        idx = int(np.clip(idx, 0, len(_HP_H_TABLE) - 2))

        H_min = (_HP_H_TABLE[idx] - _HP_H_TABLE[idx + 1]) / np.log(
            _HP_RHO_MIN_TABLE[idx + 1] / _HP_RHO_MIN_TABLE[idx]
        )
        H_max = (_HP_H_TABLE[idx] - _HP_H_TABLE[idx + 1]) / np.log(
            _HP_RHO_MAX_TABLE[idx + 1] / _HP_RHO_MAX_TABLE[idx]
        )

        rho_min = _HP_RHO_MIN_TABLE[idx] * np.exp((_HP_H_TABLE[idx] - h) / H_min)
        rho_max = _HP_RHO_MAX_TABLE[idx] * np.exp((_HP_H_TABLE[idx] - h) / H_max)
        return (rho_min + (rho_max - rho_min) * cospsi) * 1.0e-12

    def acceleration(self, et, state, dyn, sc=None):
        state_inertial = np.asarray(state, dtype=float)
        state_transform = np.asarray(sp.sxform(dyn.frame, self.body_fixed_frame, et), dtype=float)
        state_body_fixed = state_transform @ state_inertial
        
        cd = sc.cd 
        area = sc.area
        mass = sc.mass

        r_inertial = state_inertial[:3]
        r_body_fixed = state_body_fixed[:3]
        v_body_fixed = state_body_fixed[3:]
        
        if self.model == "harris-priester":
            rho_atm = self._density_harris_priester(et, r_inertial, r_body_fixed, dyn.frame)
        else: 
            raise NotImplementedError(f"Atmospheric model '{self.model}' is not implemented.")
        
        if rho_atm == 0.0:
            return np.zeros(3)

        v_rel_mps = v_body_fixed * 1.0e3
        v_rel_norm_mps = np.linalg.norm(v_rel_mps)
        a_drag_body_fixed = -0.5 * cd * area / mass * rho_atm * v_rel_norm_mps * v_rel_mps
        a_drag_body_fixed /= 1.0e3

        rot_bi = np.asarray(sp.pxform(self.body_fixed_frame, dyn.frame, et), dtype=float)
        return rot_bi @ a_drag_body_fixed
