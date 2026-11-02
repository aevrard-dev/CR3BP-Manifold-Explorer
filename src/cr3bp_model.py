from typing import Optional
from numpy import ndarray
import numpy as np
import src.physics as phy
import src.numerical as num
from src.data import CelestialBody


class CR3BPModel:
    def __init__(
            self,
            mu: float,
            primary: Optional[CelestialBody] = None,
            secondary: Optional[CelestialBody] = None,
            lstar: float = 1.0  # Defaults to 1.0 (purely dimensionless)
    ):
        self.mu = mu
        self.lstar = lstar

        # If no bodies are provided, create generic dimensionless placeholders
        self.primary = primary or CelestialBody(name="Primary", mass=1.0 - mu, radius=0.0)
        self.secondary = secondary or CelestialBody(name="Secondary", mass=mu, radius=0.0)

    @property
    def r1_normalized(self) -> float:
        return self.primary.radius / self.lstar if self.lstar > 0 else 0.0

    @property
    def r2_normalized(self) -> float:
        return self.secondary.radius / self.lstar if self.lstar > 0 else 0.0

    # --- Presets for Standard Systems ---
    @classmethod
    def sun_jupiter(cls) -> "CR3BPModel":
        """Sun-Jupiter system preset."""
        mu = 9.53707e-4
        lstar = 7.78479e8  # km (approx 5.2 AU)
        sun = CelestialBody(name="Sun", mass=1.989e30, radius=696340.0)
        jupiter = CelestialBody(name="Jupiter", mass=1.898e27, radius=69911.0)
        return cls(mu=mu, primary=sun, secondary=jupiter, lstar=lstar)

    @classmethod
    def sun_earth(cls) -> "CR3BPModel":
        """Sun-Earth (+ Moon barycenter roughly) system preset."""
        mu = 3.040423398e-6
        lstar = 1.495978707e8  # km (1 AU)
        sun = CelestialBody(name="Sun", mass=1.989e30, radius=696340.0)
        earth = CelestialBody(name="Earth", mass=5.972e24, radius=6371.0)
        return cls(mu=mu, primary=sun, secondary=earth, lstar=lstar)

    @classmethod
    def earth_moon(cls) -> "CR3BPModel":
        """Earth-Moon system preset."""
        mu = 0.0121505856
        lstar = 384400.0  # km
        earth = CelestialBody(name="Earth", mass=5.972e24, radius=6371.0)
        moon = CelestialBody(name="Moon", mass=7.342e22, radius=1737.4)
        return cls(mu=mu, primary=earth, secondary=moon, lstar=lstar)

    # --- Core Physics Wrappers ---
    def pot_eff(self, x: float, y: float):
        """Effective potential U(x,y) for the CR3BP system."""
        return phy.pot_eff(x, y, self.mu)

    def pot_eff_x_prime(self, x: float):
        return phy.pot_eff_x_prime(self.mu, x)

    def pot_eff_y_prime(self, y: float) -> float:
        return phy.pot_eff_y_prime(self.mu, y)

    def pot_eff_x_second(self, x: float):
        return phy.pot_eff_x_second(self.mu, x)

    def jacobian(self, x: float, y: float):
        """Jacobian matrix for the CR3BP system (4x4 matrix for [x, y, vx, vy])."""
        return phy.jacobian(x, y, self.mu)

    def dynamics(self, t: float, state):
        """Computes the time derivative of the state for the PCR3BP system."""
        return phy.pcr3bp_rhs(t, state, self.mu)

    def hills_radius(self):
        return (self.mu / 3.0) ** (1.0 / 3.0)

    def compute_libration_points(self, precision: float) -> tuple[ndarray, ndarray, ndarray]:
        r_h = self.hills_radius()

        x_l1_guess = (1.0 - self.mu) - r_h * (1.0 - (1.0 / 3.0) * r_h - (1.0 / 9.0) * (r_h ** 2))
        x_l2_guess = (1.0 - self.mu) + r_h * (1.0 + (1.0 / 3.0) * r_h + (1.0 / 9.0) * (r_h ** 2))
        x_l3_guess = -1.0 - (5.0 / 12.0) * self.mu

        l1 = num.newton_raphson(self.pot_eff_x_prime, self.pot_eff_x_second, x_l1_guess, precision=precision)
        l2 = num.newton_raphson(self.pot_eff_x_prime, self.pot_eff_x_second, x_l2_guess, precision=precision)
        l3 = num.newton_raphson(self.pot_eff_x_prime, self.pot_eff_x_second, x_l3_guess, precision=precision)

        return (
            np.array([l1, 0.0, 0.0, 0.0]),
            np.array([l2, 0.0, 0.0, 0.0]),
            np.array([l3, 0.0, 0.0, 0.0])
        )

    def compute_jacobi(self, state: np.ndarray) -> float:
        return phy.compute_jacobi(mu=self.mu, state=state)

    def initial_condition(self, c, libration_point):
        return phy.initial_condition(self.mu, c, libration_point)

    def nvz_limit_x(self, state, prec):
        x, y, vx, vy = state[:4]
        energy = self.pot_eff(x, y) + 0.5 * (vx ** 2 + vy ** 2)
        l1, l2, l3 = self.compute_libration_points(precision=1e-10)
        l1, l2, l3 = l1[0], l2[0], l3[0]

        def f(pos):
            return self.pot_eff(pos, 0) - energy

        e1 = self.pot_eff(l1, 0)
        e2 = self.pot_eff(l2, 0)
        e3 = self.pot_eff(l3, 0)

        if energy < e1:
            left = num.newton_raphson(f, self.pot_eff_x_prime, 1.1 * l3, prec)
            mid = num.newton_raphson(f, self.pot_eff_x_prime, 0.9 * l3, prec)
            right = num.newton_raphson(f, self.pot_eff_x_prime, 0.9 * l1, prec)
            return [left, mid, right]
        elif e1 < energy < e2:
            left = num.newton_raphson(f, self.pot_eff_x_prime, 1.1 * l3, prec)
            mid = num.newton_raphson(f, self.pot_eff_x_prime, 0.9 * l3, prec)
            return [left, mid, None]
        elif e2 < energy < e3:
            left = num.newton_raphson(f, self.pot_eff_x_prime, 1.1 * l3, prec)
            mid = num.newton_raphson(f, self.pot_eff_x_prime, 0.9 * l3, prec)
            return [left, mid, None]
        else:
            return [l3, l3, None]

    def nvz_limit_y(self, state, prec):
        x, y, vx, vy = state[:4]
        energy = self.pot_eff(x, y) + 0.5 * (vx ** 2 + vy ** 2)

        def f(pos):
            return self.pot_eff(1.0 - self.mu, pos) - energy

        top = num.newton_raphson(f, self.pot_eff_y_prime, 0.01, prec)
        bot = num.newton_raphson(f, self.pot_eff_y_prime, -0.01, prec)

        if top is not None and bot is not None:
            return [top, bot]
        else:
            r_s = self.hills_radius()
            return [r_s, -r_s]

    def define_poincare_maps(self, state):
        x_lim = self.nvz_limit_x(state, 1e-5)
        y_lim = self.nvz_limit_y(state, 1e-5)

        u_1 = [[x_lim[1], 0], [-self.mu, 0]]
        u_2 = [[1.0 - self.mu, 0], [1.0 - self.mu, y_lim[1]]]
        u_3 = [[1.0 - self.mu, y_lim[0]], [1.0 - self.mu, 0]]
        u_4 = [[x_lim[0], 0], [None, None]]
        return [u_1, u_2, u_3, u_4]
