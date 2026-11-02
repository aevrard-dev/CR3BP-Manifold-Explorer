from pathlib import Path
from dataclasses import dataclass
import yaml
from src.data import CelestialBody, SystemType, PoincareSection, LibrationPoint, ManifoldBranch


CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_FILE = (CONFIG_DIR / 'page76.yaml').resolve()
DATA_FOLDER = (CONFIG_DIR.parent / 'data').resolve()

DATA_FOLDER.mkdir(parents=True, exist_ok=True)


@dataclass
class NumericalConfig:
    time_step: float
    seed_density: int
    tolerance: float
    max_propagation_time: float

    @classmethod
    def from_dict(cls, numerical_data: dict) -> "NumericalConfig":
        numerical_data = numerical_data or {}

        return NumericalConfig(
            time_step=numerical_data.get('time_step', 0.001),
            seed_density=numerical_data.get('seed_density', 40),
            tolerance=numerical_data.get('tolerance', 1e-5),
            max_propagation_time=numerical_data.get('max_propagation_time', 20.0)
        )


@dataclass
class SystemConfig:
    type: SystemType
    mu: float
    primary: CelestialBody | None = None
    secondary: CelestialBody | None = None
    l_star: float | None = 1.0

    def __post_init__(self):
        if self.mu is not None and not (0.0 < self.mu < 1.0):
            raise ValueError(f"Invalid mass parameter mu = {self.mu}.")
        if self.l_star is not None:
            self.l_star = float(self.l_star)

    @classmethod
    def from_dict(cls, system_data: dict) -> "SystemConfig":
        """Parses raw system dictionary into a typed SystemConfig."""
        system_data = system_data or {}

        # Parse System Type
        raw_type_str = str(system_data.get('type', 'custom')).upper().strip()
        try:
            system_type_data = SystemType(raw_type_str)
        except ValueError:
            system_type_data = SystemType.CUSTOM

        # Parse Primary Body
        primary_data = system_data.get('primary')
        primary_config = CelestialBody(**primary_data) if primary_data else None

        # Parse Secondary Body
        secondary_data = system_data.get('secondary')
        secondary_config = CelestialBody(**secondary_data) if secondary_data else None

        return cls(
            type=system_type_data,
            mu=system_data.get('mu', 0.1),
            primary=primary_config,
            secondary=secondary_config,
            l_star=system_data.get('l_star', 1.0)
        )


@dataclass
class ProjectionConfig:
    name: str
    libration_point: LibrationPoint
    branch: ManifoldBranch
    section: PoincareSection
    intersection: int = 1

    @classmethod
    def from_dict(cls, name: str, proj_data: dict) -> "ProjectionConfig":
        """Parses a single projection block."""
        proj_data = proj_data or {}

        try:
            lib_point = LibrationPoint(str(proj_data.get('libration_point', 'L1')).upper().strip())
        except ValueError:
            raise ValueError(f"Invalid libration_departure in '{name}'. Choose from L1-L2.")

        try:
            branch = ManifoldBranch(str(proj_data.get('branch', 'unstable')).upper().strip())
        except ValueError:
            raise ValueError(f"Invalid branch in '{name}'. Choose 'stable' or 'unstable'.")

        try:
            section = PoincareSection(str(proj_data.get('section')).upper().strip())
        except ValueError:
            raise ValueError(f"Invalid section in '{name}'. Choose from U1-U4")
        return cls(
            name=name,
            libration_point=lib_point,
            branch=branch,
            intersection=int(proj_data.get('intersection', 1)),
            section=section
        )


@dataclass
class ScenarioConfig:
    jacobi: float | str
    jacobi_base: LibrationPoint | None = None

    def __post_init__(self):
        self.jacobi = float(eval(self.jacobi))

    @classmethod
    def from_dict(cls, scenario_data: dict) -> "ScenarioConfig":
        scenario_data = scenario_data or {}
        # Parse System Type
        raw_base_str = str(scenario_data.get('jacobi_base', None)).upper().strip()
        try:
            jacobi_base = LibrationPoint(raw_base_str)
        except ValueError:
            jacobi_base = None
        return cls(
            jacobi=scenario_data.get('jacobi'),
            jacobi_base=jacobi_base
        )


@dataclass
class AppConfig:
    system: SystemConfig
    projections: dict[str, ProjectionConfig]
    scenario: ScenarioConfig
    numerical: NumericalConfig
    data_folder: Path = DATA_FOLDER

    @classmethod
    def from_yaml(cls, filepath: Path = CONFIG_FILE) -> "AppConfig":
        """Loads and coordinates parsing of the master YAML configuration."""
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found at: {filepath}")

        with open(filepath, "r") as file:
            raw_cfg = yaml.safe_load(file) or {}

        # 1. Delegate system parsing
        system_cfg = SystemConfig.from_dict(raw_cfg.get('system', {}))

        # 2. Delegate projections parsing
        projections_raw = raw_cfg.get('projections', {})
        projections_dict = {
            proj_key: ProjectionConfig.from_dict(proj_key, proj_data)
            for proj_key, proj_data in projections_raw.items()
        }

        # 3. Delegate scenario parsing
        scenario_cfg = ScenarioConfig.from_dict(raw_cfg.get('scenario', {}))

        # 4. Delegate numerical parsing
        numerical_cfg = NumericalConfig.from_dict(raw_cfg.get('numerical', {}))
        return cls(
            system=system_cfg,
            projections=projections_dict,
            scenario=scenario_cfg,
            numerical=numerical_cfg
        )
