from dataclasses import dataclass
from enum import Enum


class SystemType(str, Enum):
    SUN_JUPITER = "SUN-JUPITER"
    SUN_EARTH = "SUN-EARTH"
    EARTH_MOON = "EARTH-MOON"
    CUSTOM = "CUSTOM"


@dataclass(frozen=True)
class CelestialBody:
    name: str | None = None
    mass: float | None = None  # kg
    radius: float | None = None  # km
    # spice_id: Optional[int] = None


class PoincareSection(str, Enum):
    U1 = "U1"
    U2 = "U2"
    U3 = "U3"
    U4 = "U4"


class PropagationDirection(int, Enum):
    FORWARD = 1
    BACKWARD = -1


class ManifoldBranch(Enum):
    STABLE = "STABLE"
    UNSTABLE = "UNSTABLE"


class OrbitType(Enum):
    lyapunov = 'LYAPUNOV'
    halo = 'HALO'
    lissajous = 'LISSAJOUS'


class LibrationPoint(Enum):
    L1 = 'L1'
    L2 = 'L2'
    L3 = 'L3'


class TransferType(Enum):
    Heteroclinic = 'HETEROCLINIC'
    Homoclinic = 'HOMOCLINIC'


class Realm(Enum):
    P = 'Primary'
    S = 'Secondary'
    E = 'Exterior'
    R1 = 'L1'
    R2 = 'L2'


class Integrator(Enum):
    RK4 = 'RK4'
    RK6 = 'RK6'
    RK8 = 'RK8'
    RKF45 = 'RKF45'


class IntegratorType(Enum):
    FIXED = 'FIXED'
    ADAPTIVE = 'ADAPTIVE'
