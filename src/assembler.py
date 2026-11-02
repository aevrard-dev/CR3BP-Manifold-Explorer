import graphic.plot as pl
from src.manifold_projection import ManifoldProjection
from src.cr3bp_model import CR3BPModel
from src.numerical import NumericalSettings
from src.configuration import AppConfig
from src.data import SystemType
from pathlib import Path


class Assembler:
    def __init__(self, config_file_path: Path):
        self.file_path = config_file_path
        self.config = AppConfig.from_yaml(self.file_path)
        self.model = self._create_model()
        self.numerical = self._create_numerical_settings()

    def _create_numerical_settings(self) -> NumericalSettings:
        return NumericalSettings(
            time_step=self.config.numerical.time_step,
            seed_density=self.config.numerical.seed_density,
            tolerance=float(self.config.numerical.tolerance),
            max_propagation_time=self.config.numerical.max_propagation_time,
        )

    def _create_model(self) -> CR3BPModel:
        """Instantiates the physics model using a clean dispatch dictionary."""

        # 1. Map each enum to its corresponding preset factory method
        factory_map = {
            SystemType.SUN_JUPITER: CR3BPModel.sun_jupiter,
            SystemType.SUN_EARTH: CR3BPModel.sun_earth,
            SystemType.EARTH_MOON: CR3BPModel.earth_moon,
        }

        # 2. Check if the type exists in our map
        if self.config.system.type in factory_map:
            return factory_map[self.config.system.type]()

        # 3. Fallback for custom systems
        return CR3BPModel(
            mu=self.config.system.mu,
            primary=self.config.system.primary,
            secondary=self.config.system.secondary,
            lstar=self.config.system.l_star
        )

    def process_projection(self):

        projection_list = []
        for key, value in self.config.projections.items():
            projection = ManifoldProjection(
                model=self.model,
                numerical=self.numerical,
                libration_departure=value.libration_point,
                branch=value.branch,
                jacobi=self.config.scenario.jacobi,
                jacobi_base=self.config.scenario.jacobi_base,
                section=value.section,
                intersection=value.intersection,
                direction=None,
                seed_density=self.config.numerical.seed_density,
                max_propagation_time=self.numerical.max_propagation_time,
            )
            transfer = projection.run()
            for manifold in transfer:
                projection_list.append(manifold)
        # print(projection_list[0].completed[10].numerical_analysis())
        # print(projection_list[0].completed[10].time[-1])
        # results = projection_list[0].completed[10].numerical_analysis()
        # print(results)
        # pl.plot_numerical_convergence(results)
        pl.plot_projection(projection_list)
        pl.draw_plot(projection_list)


