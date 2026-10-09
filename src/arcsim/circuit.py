"""External circuit: AC source in series with R-L, closed through the arc.

    L dI/dt = v_s(t) - R I - U_arc,     U_arc = I / G_arc + U_fall(I)

The ohmic part of the arc is treated implicitly so the update stays stable when the
arc resistance becomes very large (extinction / post-arc phase).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CircuitConfig:
    V_rms: float = 230.0
    frequency: float = 50.0
    R: float = 0.08
    L: float = 0.9e-3
    # Phase of the *prospective current* at t = 0 (contact separation), in degrees.
    current_phase_deg: float = 20.0


class Circuit:
    def __init__(self, cfg: CircuitConfig):
        self.cfg = cfg
        self.omega = 2 * np.pi * cfg.frequency
        self.Vpk = np.sqrt(2) * cfg.V_rms
        self.Z = np.hypot(cfg.R, self.omega * cfg.L)
        self.theta = np.arctan2(self.omega * cfg.L, cfg.R)
        self.Ipk_prospective = self.Vpk / self.Z
        self.alpha = np.deg2rad(cfg.current_phase_deg) + self.theta
        self.I = self.prospective_current(0.0)

    def source_voltage(self, t):
        return self.Vpk * np.sin(self.omega * t + self.alpha)

    def prospective_current(self, t):
        return self.Ipk_prospective * np.sin(self.omega * t + self.alpha - self.theta)

    def step(self, t_new: float, dt: float, G_arc: float, U_fall: float) -> float:
        """Advance the current to t_new. U_fall carries the sign of the current."""
        R_arc = 1.0 / max(G_arc, 1e-30)
        L, R = self.cfg.L, self.cfg.R
        self.I = (L * self.I + dt * (self.source_voltage(t_new) - U_fall)) / (L + dt * (R + R_arc))
        return self.I


def fall_voltage(I: float, n_series: int, U_fall: float, I_smooth: float) -> float:
    """Electrode (cathode + anode) fall voltage of n_series arc segments, smoothed at I = 0."""
    return n_series * U_fall * np.tanh(I / I_smooth)
