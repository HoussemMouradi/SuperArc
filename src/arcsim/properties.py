"""LTE plasma property tables (temperature dependent, fixed pressure).

The solver uses a "1-atm table + ideal-gas-like EOS" closure that is standard for
low-voltage breaker arcs:

    p = rho * R_eff(T) * T,     R_eff(T) = p0 / (rho_table(T) * T)
    e(T) = h_table(T) - p0 / rho_table(T)

so that dissociation/ionization (falling molar mass, large enthalpy jumps) are
captured through the tables even though the pressure dependence is neglected.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

P0 = 101325.0


@dataclass
class PlasmaProperties:
    T: np.ndarray
    rho: np.ndarray
    h: np.ndarray
    sigma: np.ndarray
    kappa: np.ndarray
    mu: np.ndarray
    nec: np.ndarray
    p0: float = P0

    @classmethod
    def from_csv(cls, path: str | Path | None = None, dT: float = 10.0, p0: float = P0) -> "PlasmaProperties":
        if path is None:
            path = resources.files("arcsim") / "data" / "air_1atm.csv"
        lines = [ln for ln in Path(str(path)).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        header = [c.strip() for c in lines[0].split(",")]
        table = np.array([[float(x) for x in ln.split(",")] for ln in lines[1:]])
        raw = {name: table[:, k] for k, name in enumerate(header)}
        Tg = np.arange(raw["T"][0], raw["T"][-1] + 0.5 * dT, dT)

        def lin(name):
            return np.interp(Tg, raw["T"], raw[name])

        def log(name):
            return np.exp(np.interp(Tg, raw["T"], np.log(np.maximum(raw[name], 1e-300))))

        # Monotone (log-linear) interpolation keeps rho and h smooth enough for an EOS.
        return cls(T=Tg, rho=log("rho"), h=lin("h"), sigma=log("sigma"), kappa=lin("kappa"),
                   mu=lin("mu"), nec=log("nec"), p0=p0)

    def __post_init__(self):
        self.pr = self.p0 / self.rho  # p/rho = R_eff * T, monotone increasing in T
        self.e = self.h - self.pr
        if np.any(np.diff(self.e) <= 0) or np.any(np.diff(self.pr) <= 0):
            raise ValueError("property table gives a non-monotone e(T) or p/rho(T)")
        self.cv = np.gradient(self.e, self.T)
        self.T_min = float(self.T[0])
        self.T_max = float(self.T[-1])

    def _at(self, T, arr):
        return np.interp(T, self.T, arr)

    def sigma_of_T(self, T):
        return self._at(T, self.sigma)

    def kappa_of_T(self, T):
        return self._at(T, self.kappa)

    def nec_of_T(self, T):
        return self._at(T, self.nec)

    def rho_of_T(self, T):
        return self._at(T, self.rho)

    def cv_of_T(self, T):
        return self._at(T, self.cv)

    def e_of_T(self, T):
        return self._at(T, self.e)

    def T_of_e(self, e):
        return np.interp(e, self.e, self.T)

    def T_of_pr(self, pr):
        """Temperature from p/rho."""
        return np.interp(pr, self.pr, self.T)

    def e_of_pr(self, pr):
        return np.interp(pr, self.pr, self.e)

    def pr_of_T(self, T):
        return self._at(T, self.pr)
