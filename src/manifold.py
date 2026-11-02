from dataclasses import dataclass, replace
import numpy as np
import src.numerical as num
from src.cr3bp_model import CR3BPModel
import time
from src.Lyapunov import Seed, PeriodicOrbit
from src.physics import compute_jacobi_statistics
from src.propagator import Propagator, PropagationRequest, RealmCondition, PoincareMaps, Condition
from src.poincare_map import PoincareMap
from src.data import ManifoldBranch, PropagationDirection, LibrationPoint, Realm, PoincareSection, Integrator
from src.numerical import NumericalSettings


@dataclass(frozen=True)
class ManifoldConfig:
    system: CR3BPModel
    numerical: NumericalSettings
    integrator: Integrator
    periodic_orbit: PeriodicOrbit
    lib_point: LibrationPoint
    branch: ManifoldBranch
    direction: PropagationDirection
    orbit_id: str
    epsilon: float
    poincare_map: PoincareMap
    tol: float
    realm_cond: list[Realm] | None = None
    t0: float = 0.0
    t_max: float = 10.0
    density: int = 5


@dataclass
class ManifoldStrand:
    id: str  # e.g., "037.04.03" (using zero-padded dot notation)
    config: ManifoldConfig

    initial_state: np.ndarray
    seed: Seed
    max_propagation_time: float

    time = None
    strand = None
    is_completed: bool = False

    def propagate(self):
        if self.config.branch is ManifoldBranch.UNSTABLE:
            time_direction = PropagationDirection.FORWARD
        else:
            time_direction = PropagationDirection.BACKWARD
        request = PropagationRequest(
            system=self.config.system,
            integrator=self.config.integrator,
            numerical=self.config.numerical,
            state=self.initial_state,
            conditions=self._set_conditions(),
            max_propagation_time=self.max_propagation_time,
            time_direction=time_direction
        )
        propagator = Propagator(request=request)

        t, traj = propagator.propagate()
        self.time = t
        self.strand = traj
        self.is_completed = self._set_completed(self.strand)

    def _propagate_symmetrical(self):
        # 1. Determine opposite time direction
        sym_direction = (
            PropagationDirection.FORWARD
            if self.config.branch is ManifoldBranch.STABLE
            else PropagationDirection.BACKWARD
        )

        sym_state = np.copy(self.initial_state)
        sym_state[1] *= -1.0  # Flip y
        sym_state[2] *= -1.0  # Flip vx

        request = PropagationRequest(
            system=self.config.system,
            integrator=self.config.integrator,
            numerical=self.config.numerical,
            state=sym_state,
            conditions=self._set_conditions(symmetry=True),
            max_propagation_time=self.max_propagation_time,
            time_direction=sym_direction
        )

        propagator = Propagator(request=request)
        t_sym, traj_sym = propagator.propagate()
        return t_sym, traj_sym

    def numerical_analysis(self):
        tolerance_array = np.logspace(-3, -14, 12)
        initial_tolerance = self.config.numerical.tolerance
        initial_strand = self.strand
        initial_time = self.time
        results = []
        if self.config.branch is ManifoldBranch.UNSTABLE:
            time_direction = PropagationDirection.FORWARD
        else:
            time_direction = PropagationDirection.BACKWARD
        for tolerance in tolerance_array:
            numerical_config = replace(self.config.numerical, tolerance=tolerance)
            request = PropagationRequest(
                system=self.config.system,
                integrator=self.config.integrator,
                numerical=numerical_config,
                state=self.initial_state,
                conditions=self._set_conditions(symmetry=True),
                max_propagation_time=self.max_propagation_time,
                time_direction=time_direction
            )
            propagator = Propagator(request=request)

            t, traj = propagator.propagate()
            self.time, self.strand = t, traj
            jacobi_mean, jacobi_std = self.compute_jacobi_statistics()
            results.append([tolerance, jacobi_mean, jacobi_std])
            self.config.numerical.tolerance = initial_tolerance
            self.strand = initial_strand
            self.time = initial_time
        return results

    def check_symmetry(self):
        """
        Validates mirror symmetry by propagating the mirrored initial state
        in the opposite time direction using a cloned configuration.
        """

        t_sym, traj_sym = self._propagate_symmetrical()

        symmetry_vector = np.array([-1.0, 1.0, 1.0, -1.0])
        errors = []

        for i in range(min(len(self.strand), len(traj_sym))):
            # Transform symmetry state into the reference frame of the original strand
            transformed_sym = traj_sym[i][:4] * symmetry_vector

            # Compute absolute difference vector
            diff = self.strand[i][:4] + transformed_sym
            error_norm = np.linalg.norm(diff)
            errors.append(error_norm)

        residual = np.sum(errors)
        error_max = max(errors)
        error_mean = np.mean(errors)

        print(f'Residual error: {residual:.15e}')
        # print(f"Max symmetry residual error: {error_max:.15e}")
        # print(f"Mean symmetry residual error: {error_mean:.15e}")
        return residual, error_max, error_mean

    def _set_conditions(self, symmetry=False) -> list[Condition]:

        map_def = self.config.system.define_poincare_maps(self.config.periodic_orbit.trajectory[0])
        if symmetry:
            if self.config.poincare_map.section is PoincareSection.U2:
                poincare_map = replace(self.config.poincare_map, section=PoincareSection.U3)
            elif self.config.poincare_map.section is PoincareSection.U3:
                poincare_map = replace(self.config.poincare_map, section=PoincareSection.U2)
            else:
                poincare_map = self.config.poincare_map
        else:
            poincare_map = self.config.poincare_map
        conditions = [PoincareMaps(system=self.config.system,
                                   maps_def=map_def,
                                   map_list=poincare_map,
                                   direction=poincare_map.direction)]
        if self.config.realm_cond is not None:
            path_condition = RealmCondition(system=self.config.system, path=self.config.realm_cond,
                                            orbit=self.config.periodic_orbit)
            conditions.append(path_condition)
        return conditions

    def compute_jacobi_statistics(self):
        return compute_jacobi_statistics(self.config.system.mu, self.strand)

    def _set_completed(self, trajectory):
        x, y, vx, vy = trajectory[-1][:4]
        if self.config.poincare_map.section in (PoincareSection.U1, PoincareSection.U4):
            return abs(y) <= 1e-16
        elif self.config.poincare_map.section in (PoincareSection.U2, PoincareSection.U3):
            return abs(x - (1 - self.config.system.mu)) <= 1e-16


@dataclass
class ManifoldResult:
    config: ManifoldConfig
    trajectories: list[ManifoldStrand]
    completed: list[ManifoldStrand]
    excluded: list[ManifoldStrand]
    label: str

    def sort_strands(self) -> None:
        """Sorts all internal strand collections numerically by their ID."""

        def numerical_key(strand):
            try:
                # Handles plain numbers ("1400" -> 1400) and dot-notation ("037.04.03" -> [37, 4, 3])
                return [int(part) for part in str(strand.id).split(".")]
            except ValueError:
                # Fallback for pure strings if any exist
                return [str(strand.id)]

        self.trajectories.sort(key=numerical_key)
        self.completed.sort(key=numerical_key)
        self.excluded.sort(key=numerical_key)

    def get_scaled_coordinates(self) -> list[tuple[str, np.ndarray]]:
        """
        Computes scaled coordinates from completed projections and
        returns a list of (id, scaled_4d_state) pairs conserving [x, y, vx, vy].
        """
        points = self.get_projection()
        if not points:
            return []

        states = np.array([p[1] for p in points])

        mean = np.mean(states, axis=0)
        std = np.std(states, axis=0)
        std[std == 0.0] = 1.0

        scaled_states = (states - mean) / std

        scaled_points = []
        for i, (strand_id, _) in enumerate(points):
            scaled_points.append((strand_id, scaled_states[i]))
        return scaled_points

    def get_projection(self) -> list[tuple[str, np.ndarray]]:
        """Returns a list of (id, final_state) pairs for completed strands."""
        projection = []
        for strand in self.completed:
            projection.append((strand.id, strand.strand[-1]))
        return projection

    def get_strand_by_id(self, identification: str) -> ManifoldStrand | None:
        """Searches across all strand collections to find a specific ID."""
        for strand in self.trajectories:
            if strand.id == identification:
                return strand
        return None

    def get_sparse_pairs(self, list_state: list[tuple[str, np.ndarray]], k_sigma: float = 5.0) -> list[
        tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]]
    ]:
        """
        Identifies sparse pairs from a list of (id, state) tuples.
        Compute distances between adjacent neighbors using a circular wrap-around
        for closed phase space loops.
        """
        if not list_state or len(list_state) < 2:
            return []

        n = len(list_state)

        # Extract states and isolate 2D subspace coordinates
        states = np.array([p[1] for p in list_state])
        section = self.config.poincare_map.section

        if section in (PoincareSection.U1, PoincareSection.U4):
            coordinates_2d = states[:, [0, 2]]
        elif section in (PoincareSection.U2, PoincareSection.U3):
            coordinates_2d = states[:, [1, 3]]
        else:
            raise RuntimeError(f"Wrong section, must be U(1,2,3,4), is: {section}")

        # Compute distances between consecutive adjacent pairs with circular wrap-around
        valid_pairs = []
        distances = []

        for i in range(n):
            next_i = (i + 1) % n
            p1, p2 = list_state[i], list_state[next_i]
            dist = np.linalg.norm(coordinates_2d[next_i] - coordinates_2d[i])
            distances.append(dist)
            valid_pairs.append((p1, p2))

        distances = np.array(distances)
        if len(distances) == 0:
            return []

        # Statistical Thresholding
        threshold = np.mean(distances) + (k_sigma * np.std(distances))

        # Return only pairs exceeding the dynamic distance threshold
        return [pair for pair, dist in zip(valid_pairs, distances) if dist > threshold]


def filter_completed(manifold_trajectories):
    curated_list = []
    excluded_list = []
    for manifold in manifold_trajectories:
        if manifold.is_completed:
            curated_list.append(manifold)
        else:
            excluded_list.append(manifold)
    return curated_list, excluded_list


class Manifold:
    def __init__(self, config: ManifoldConfig):
        self.config = config
        self.epsilon = config.epsilon
        self.map = config.poincare_map
        self.map_def = self.config.system.define_poincare_maps(self.config.periodic_orbit.trajectory[0])
        self.t_max = config.t_max
        if self.config.branch is ManifoldBranch.UNSTABLE:
            self.time_direction = PropagationDirection.FORWARD
        else:
            self.time_direction = PropagationDirection.BACKWARD

        # self.name = fr'$W{self.section.sections[0].max_intersections}^{{{self.config.branch.value[0]},\chi}}$'

    def compute_monodromy_vector(self):
        eigen_values, eigen_vectors = np.linalg.eig(self.config.periodic_orbit.monodromy_matrix)
        abs_vals = np.abs(eigen_values)
        idx_unstable = np.argmax(abs_vals)
        idx_stable = np.argmin(abs_vals)
        print(num.check_stm_properties(self.config.periodic_orbit.monodromy_matrix))

        if self.config.branch is ManifoldBranch.STABLE:
            eig_vec = eigen_vectors[:, idx_stable]
        else:
            eig_vec = eigen_vectors[:, idx_unstable]

        eig_vec /= np.linalg.norm(eig_vec)
        return eig_vec

    def compute_perturbed_vector(self, seed, eigen_vector):
        x = seed.state[:4]
        stm = seed.state[4:].reshape(4, 4)
        vec = np.real(stm @ eigen_vector)
        vec /= np.linalg.norm(vec)
        x = x + self.config.epsilon * self.config.direction.value * vec
        return np.concatenate((x, np.eye(4).flatten()))

    def propagate(self):
        manifolds = []
        eig_vec = self.compute_monodromy_vector()
        seeds_number = len(self.config.periodic_orbit.seeds)
        start_time = time.perf_counter()

        for n, seed in enumerate(self.config.periodic_orbit.seeds):
            if n % self.config.density != 0:
                continue
            print(f'Propagating seed {n}/{seeds_number}')

            initial_state = self.compute_perturbed_vector(seed, eig_vec)

            manifold_strand = ManifoldStrand(
                id=seed.identification,
                config=self.config,
                initial_state=initial_state,
                seed=seed,
                max_propagation_time=self.t_max,
            )
            manifold_strand.propagate()

            manifolds.append(manifold_strand)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time:.6f} seconds")
        return manifolds

    # def propagate_prime(self, seeds_list: list[Seed]):
    #     manifolds = []
    #     eig_vec = self.compute_monodromy_vector()
    #     seeds_number = len(self.config.periodic_orbit.seeds)
    #     start_time = time.perf_counter()
    #     for n, seed in enumerate(seeds_list):
    #         if int(seed.identification) % self.iter_density != 0:
    #             continue
    #         print(f'Propagating seed id: {seed.identification}/{seeds_number}')
    #
    #         # Make this condition setup into a method
    #         conditions = [PoincareMaps(system=self.config.system,
    #                                    maps_def=self.map_def,
    #                                    map_list=self.map,
    #                                    direction=self.map.direction)]
    #         if self.config.realm_cond is not None:
    #             path_condition = RealmCondition(system=self.config.system, path=self.config.realm_cond,
    #                                             orbit=self.config.periodic_orbit)
    #             conditions.append(path_condition)
    #         ##################
    #
    #         initial_state = self.compute_perturbed_vector(seed, eig_vec)
    #
    #         request = PropagationRequest(
    #             system=self.config.system,
    #             numerical=self.config.numerical,
    #             integrator=self.config.integrator,
    #             state=initial_state,
    #             conditions=conditions,
    #             max_propagation_time=self.t_max
    #         )
    #
    #         propagator = Propagator(request=request)
    #
    #         t, traj = propagator.propagate()
    #
    #         manifold_strand = ManifoldStrand(
    #             id=seed.identification,
    #             system=self.config.system,
    #             strand=traj,
    #             time=t,
    #             seed=seed,
    #             is_completed=self._set_completed(traj)
    #
    #         )
    #
    #         manifolds.append(manifold_strand)
    #     end_time = time.perf_counter()
    #     elapsed_time = end_time - start_time
    #     print(f"Execution time: {elapsed_time:.6f} seconds")
    #     return manifolds

    def generate_trajectories(self):
        trajectories = self.propagate()
        completed_trajectories, excluded_trajectories = filter_completed(trajectories)

        if self.map.section:
            direction_sign = "+" if self.config.direction.value == 1 else "-"

            label = fr'$\Gamma_{self.map.max_intersections}^{{{self.config.branch.value[0]}{direction_sign}, ' \
                    fr'{self.map.section.value.title()}}}$'
        else:
            label = 'test'

        self.output = ManifoldResult(
            config=self.config,
            trajectories=trajectories,
            completed=completed_trajectories,
            excluded=excluded_trajectories,
            label=label
        )
        # self._propagate_unused_seeds()
        # for strand in self.output.completed:
        # print(strand.id)
        # self._interpolate_seed_gap(str(75), str(76))
        # self._interpolate_seed_gap(str(1094), str(1095))
        return self.output

    # def _interpolate_seed_gap(self, seed_id_1, seed_id_2, steps=100):
    #     # seed_id_list = self._check_projection_density()
    #     # if seed_id_list is None:
    #     #     return None
    #     if not self.output.completed:
    #         return None
    #
    #     # make sure the pair is a true gap in seeds propagation
    #     seed_1 = self.config.periodic_orbit.get_seed_by_id(seed_id_1)
    #     # seed_2 = self.config.periodic_orbit.get_seed_by_id(seed_id_2)
    #
    #     strand_1 = self.output.get_strand_by_id(seed_id_1)
    #     strand_2 = self.output.get_strand_by_id(seed_id_2)
    #
    #     if strand_1.is_completed is not strand_2.is_completed:
    #         print(self.config.periodic_orbit.config.dt)
    #         # Replay the RK step to recover the dense output data
    #         _, _, dense = num.rk45_step(
    #             f=self.config.system.dynamics,
    #             t=seed_1.time,
    #             y=seed_1.state,
    #             dt=self.config.periodic_orbit.config.dt,
    #             tol=0,
    #             replay=True,
    #         )
    #         print(dense)
    #         n = np.linspace(0.1, 1.0, steps)
    #         new_seeds = []
    #         for step in n:
    #             test = num.dense_eval(dense, step)
    #             new_seeds.append(
    #                 Seed(identification=f'{seed_id_1}.{step}', state=test, time=seed_1.time + dense['dt'] * step))
    #         trajectories = self.propagate_prime(new_seeds)
    #         completed_trajectories, excluded_trajectories = filter_completed(trajectories)
    #         for strand in trajectories:
    #             self.output.trajectories.append(strand)
    #         for strand in completed_trajectories:
    #             self.output.completed.append(strand)
    #         for strand in excluded_trajectories:
    #             self.output.excluded.append(strand)
    #         # print(seed_2.state - test)

    # def _propagate_unused_seeds(self):
    #     self.iter_density = np.round(self.config.density / 2)
    #     # while self._check_projection_density() is not None:
    #     seed_id_list = self._check_projection_density()
    #     if seed_id_list is None:
    #         return None
    #     self._propagate_seed_id_list(seed_id_list=seed_id_list)
    #     self.output.sort_strands()
    #     self.iter_density = np.round(self.iter_density / 2)
        # while len(self.output.get_sparse_pairs(self.output.get_scaled_coordinates())) >= 1:
        #     seed_id_list = self._check_projection_density()
        #     if seed_id_list is None:
        #         return None
        #     self._propagate_seed_id_list(seed_id_list=seed_id_list)
        #     self.output.sort_strands()
        #     if self.iter_density >= 2.0:
        #         self.iter_density = np.round(self.iter_density / 2)

    def _check_projection_density(self):
        sparse_pairs = self.output.get_sparse_pairs(self.output.get_scaled_coordinates())
        if not sparse_pairs:
            return None
        pairs_id = []
        if sparse_pairs:
            print('Sparse pairs')
        for pair in sparse_pairs:
            if [pair[0][0], pair[1][0]] not in pairs_id:
                pairs_id.append([pair[0][0], pair[1][0]])

        seeds_id_list = []
        for pair in pairs_id:
            print(pair)
            # print(int(pair[0]) + self.config.density, int(pair[1]) - self.config.density)

            for i in range(int(pair[0]) + 1, int(pair[0]) + self.config.density):
                if i not in seeds_id_list:
                    seeds_id_list.append(i)
            for i in range(int(pair[1]) - self.config.density + 1, int(pair[1])):
                if i not in seeds_id_list:
                    seeds_id_list.append(i)
        print(seeds_id_list)
        return seeds_id_list

    # def _propagate_seed_id_list(self, seed_id_list):
    #     seeds_list = []
    #     for identification in seed_id_list:
    #         if self.output.get_strand_by_id(identification) is None:
    #             seeds_list.append(self.config.periodic_orbit.get_seed_by_id(str(identification)))
    #
    #     trajectories = self.propagate_prime(seeds_list)
    #     completed_trajectories, excluded_trajectories = filter_completed(trajectories)
    #     for strand in trajectories:
    #         self.output.trajectories.append(strand)
    #     for strand in completed_trajectories:
    #         self.output.completed.append(strand)
    #     for strand in excluded_trajectories:
    #         self.output.excluded.append(strand)


@dataclass(frozen=True)
class ManifoldCatalog:
    configs: list[ManifoldConfig]

    def generate(self):
        return [Manifold(config).generate_trajectories() for config in self.configs]
