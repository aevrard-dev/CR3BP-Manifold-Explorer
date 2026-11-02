import src.manifold as mf
from src.cr3bp_model import CR3BPModel
from src.manifold import ManifoldBranch
from src.data import LibrationPoint, Realm, PoincareSection, PropagationDirection, Integrator
from src.poincare_map import PoincareMap
from src.Lyapunov import Lyapunov
from src.numerical import NumericalSettings


class ManifoldProjection:
    def __init__(
            self,
            model: CR3BPModel,
            numerical: NumericalSettings,

            libration_departure: LibrationPoint,
            branch: ManifoldBranch,

            jacobi: float,
            section: PoincareSection,

            intersection: int = 1,
            direction: PropagationDirection | None = None,

            jacobi_base: LibrationPoint | None = None,
            libration_precision: float | None = None,
            lyapunov_precision: float | None = None,
            lyapunov_time_step: float | None = None,
            manifold_precision: float | None = None,
            max_propagation_time: float = 10,
            seed_density: int | None = None,
    ):
        self.system = model
        self.numerical = numerical

        self.lib_departure = libration_departure
        self.branch = branch
        self.jacobi = jacobi

        self.jacobi_base = jacobi_base
        self.libration_precision = (1e-10 if libration_precision is None else libration_precision)
        self.lyapunov_precision = (1e-5 if lyapunov_precision is None else lyapunov_precision)
        self.lyapunov_time_step = (1e-3 if lyapunov_time_step is None else lyapunov_time_step)
        self.manifold_precision = (1e-5 if manifold_precision is None else manifold_precision)
        self.density = (10 if seed_density is None else seed_density)

        self.section = section

        self.intersection = intersection
        self.direction = direction

        self.max_propagation_time = max_propagation_time

    def _generate_periodic_orbits(self) -> None:
        lyapunov = Lyapunov(system=self.system,
                            integrator=Integrator.RK4,
                            numerical=self.numerical,
                            libration_point=self.lib_departure,
                            time_step=self.lyapunov_time_step)
        self.periodic_orbit = lyapunov.find_jacobi(jacobi_target=self.jacobi,
                                                   tolerance=self.lyapunov_precision)

    def _generate_poincare_maps(self) -> None:
        self.depart_map = PoincareMap(section=self.section,
                                      max_intersections=self.intersection,
                                      direction=self.direction)

    def _generate_path(self) -> None:
        if self.lib_departure is LibrationPoint.L1:
            self.path = [Realm.R1]
        elif self.lib_departure is LibrationPoint.L2:
            self.path = [Realm.R2]
        else:
            raise RuntimeError(f'Wrong departure libration point: {self.lib_departure}, must be L1 or L2')
        if self.path[0] is Realm.R1:
            if self.section is PoincareSection.U1:
                self.path.append(Realm.P)
            else:
                self.path.append(Realm.S)
        elif self.path[0] is Realm.R2:
            if self.section is PoincareSection.U4:
                self.path.append(Realm.E)
            else:
                self.path.append(Realm.S)
        else:
            raise RuntimeError(f'Wrong departure realm: {self.path[0]} should be R1 or R2')

    def _generate_manifold_config(self) -> None:

        self.depart_config_f = mf.ManifoldConfig(
            system=self.system,
            numerical=self.numerical,
            integrator=Integrator.RKF45,
            periodic_orbit=self.periodic_orbit,
            lib_point=self.lib_departure,
            branch=self.branch,
            direction=mf.PropagationDirection.FORWARD,
            orbit_id='depart_f',
            epsilon=1e-4,
            tol=self.manifold_precision,
            poincare_map=self.depart_map,
            t_max=self.max_propagation_time,
            realm_cond=self.path,
            density=self.density
        )
        self.depart_config_b = mf.ManifoldConfig(
            system=self.system,
            numerical=self.numerical,
            integrator=Integrator.RKF45,
            periodic_orbit=self.periodic_orbit,
            lib_point=self.lib_departure,
            branch=self.branch,
            direction=mf.PropagationDirection.BACKWARD,
            orbit_id='depart_b',
            epsilon=1e-4,
            tol=self.manifold_precision,
            poincare_map=self.depart_map,
            t_max=self.max_propagation_time,
            realm_cond=self.path,
            density=self.density
        )

        self.manifold_configs = [
            self.depart_config_f,
            self.depart_config_b,
        ]

    def _check(self) -> None:
        print(f'Checking parameters')
        if not 0 < self.system.mu < 1:
            print(f'µ parameter must be 0 < µ < 1, µ={self.system.mu}')
        if self.libration_precision == 1e-10:
            print(f'Using default libration precision value: {self.libration_precision}')
        else:
            print(f'Libration precision parameter: {self.libration_precision}, '
                  f'type float: {type(self.libration_precision) is float}')
        if self.lyapunov_precision == 1e-5:
            print(f'Using default lyapunov precision value: {self.lyapunov_precision}')
        else:
            print(f'Libration precision parameter: {self.lyapunov_precision}, '
                  f'type float: {type(self.lyapunov_precision) is float}')
        if self.lyapunov_time_step == 1e-3:
            print(f'Using default lyapunov time step value: {self.lyapunov_time_step}')
        else:
            print(f'Lyapunov time step parameter: {self.lyapunov_time_step}')
        if self.manifold_precision == 1e-5:
            print(f'Using default manifold precision value: {self.manifold_precision}')
        else:
            print(f'Manifold precision parameter: {self.manifold_precision}, '
                  f'type float: {type(self.manifold_precision) is float}')
        if self.density == 5:
            print(f'Using default manifold density value: {self.density}')
        else:
            print(f'Manifold density parameter: {self.density}')

        print(f'Poincare section: {self.section.value}')

        l1, l2, l3 = self.system.compute_libration_points(self.libration_precision)
        l1_jacobi = float(self.system.compute_jacobi(l1))
        l2_jacobi = float(self.system.compute_jacobi(l2))
        l3_jacobi = float(self.system.compute_jacobi(l3))

        if self.jacobi_base is LibrationPoint.L1:
            self.jacobi_base = l1_jacobi
        elif self.jacobi_base is LibrationPoint.L2:
            self.jacobi_base = l2_jacobi
        elif self.jacobi_base is None:
            self.jacobi_base = 0.0
        else:
            raise RuntimeError(f'Jacobi base must be either L1, L2 or None')
        self.jacobi = self.jacobi_base + self.jacobi
        print(f'\nL1 jacobi: {l1_jacobi}\nl2_jacobi: {l2_jacobi}\nl3_jacobi: {l3_jacobi}')
        print(f'Target jacobi: {self.jacobi}')

        if not self.jacobi >= l3_jacobi:
            raise RuntimeError(f'The system jacobi must verify: jacobi >= l3_jacobi')
        else:
            print(f'Jacobi parameter: j={self.jacobi:.3f}, type float: {type(self.jacobi) is float}\n'
                  f'Jacobi base     : j={self.jacobi_base:.3f}, type float:{type(self.jacobi_base) is float}')

        if (self.lib_departure is LibrationPoint.L1 and self.section is PoincareSection.U4) or \
                (self.lib_departure is LibrationPoint.L2 and self.section is PoincareSection.U1):
            raise RuntimeError(f'Wrong intersection, {self.lib_departure.value} cannot intersect with '
                               f'{self.section.value}')

    def _propagate_manifolds(self) -> list[mf.ManifoldResult]:
        self._generate_periodic_orbits()
        self._generate_path()
        self._generate_poincare_maps()
        self._generate_manifold_config()
        manifolds = mf.ManifoldCatalog(configs=self.manifold_configs)
        manifolds_trajectories = manifolds.generate()
        return manifolds_trajectories

    def run(self):
        self._check()
        return self._propagate_manifolds()


#
# system = CR3BPModel(mu=0.1)
#
# jacobi_target = -0.03715 * 2
# poincare_section = PoincareSection.U1
# density = 10
# #
# test_1 = ManifoldProjection(
#     model=system,
#     libration_departure=LibrationPoint.L1,
#     branch=ManifoldBranch.STABLE,
#     jacobi=jacobi_target,
#     # jacobi_base=LibrationPoint.L1,
#     section=poincare_section,
#     intersection=1,
#     # direction=None,
#     seed_density=density,
#     max_propagation_time=20
# )
#
# test_2 = ManifoldProjection(
#     model=system,
#     libration_departure=LibrationPoint.L1,
#     branch=ManifoldBranch.UNSTABLE,
#     jacobi=jacobi_target,
#     jacobi_base=LibrationPoint.L1,
#     section=poincare_section,
#     intersection=2,
#     # direction=None,
#     seed_density=density,
#     max_propagation_time=20
# )
#
# # test_3 = TransferTrajectory(
# #     system=system,
# #     libration_departure=LibrationPoint.L2,
# #     branch=ManifoldBranch.STABLE,
# #     jacobi=jacobi_target,
# #     # jacobi_base=LibrationPoint.L1,
# #     section=pm.PoincareSection.U2,
# #     intersection=1,
# #     direction=None,
# #     density=density,
# #     max_propagation_time=20
# # )
# #
# # test_4 = TransferTrajectory(
# #     system=system,
# #     libration_departure=LibrationPoint.L1,
# #     branch=ManifoldBranch.UNSTABLE,
# #     jacobi=jacobi_target,
# #     # jacobi_base=LibrationPoint.L1,
# #     section=pm.PoincareSection.U3,
# #     intersection=1,
# #     density=density
# # )
#
# transfer_1 = test_1.run()
# transfer_2 = test_2.run()
# # transfer_3 = test_3.run()
# # transfer_4 = test_4.run()
#
# pl.plot_projection([
#     *transfer_1,
#     *transfer_2,
#     # *transfer_3,
#     # *transfer_4
# ])
#
# # for manifold in transfer_1:
# #     print(len(manifold.completed_trajectories), len(manifold.trajectories), len(manifold.excluded_trajectories),
# #           manifold.label, manifold.config.poincare_map.section.value)
# for manifold in transfer_2:
#     print(len(manifold.completed), len(manifold.trajectories), len(manifold.excluded),
#           manifold.label, manifold.config.poincare_map.section.value)
# #
# transfers = [
#     *transfer_1,
#     *transfer_2,
#     # *transfer_3,
#     # *transfer_4
# ]
#
# pl.draw_plot(transfers)
