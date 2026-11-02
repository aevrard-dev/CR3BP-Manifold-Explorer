"""
lyapunov.py

Purpose
-------
Generation and continuation of planar Lyapunov orbit families in the PCR3BP.

This module provides:
    - Initial condition estimation near libration points
    - Differential correction stabilization
    - Orbit family continuation
    - Seed export for downstream analysis

Implemented Methods
-------------------
- Single shooting differential correction
- Numerical continuation
- RK4 propagation
- State transition matrix initialization

Assumptions
-----------
- Planar Circular Restricted Three-Body Problem (PCR3BP)
- Non-dimensional canonical units
- Fixed integration timestep
- Symmetric periodic orbit constraints

Dependencies
------------
- physics.py
- propagator.py
- differential_correction.py
- numerical.py

"""
import src.physics as phy
import src.differential_correction as dif
import numpy as np
import src.configuration as cfg
from src.propagator import Propagator, PropagationRequest, FullOrbit
from src.continuation import continuation_step
from dataclasses import dataclass
from src.data import LibrationPoint, OrbitType, Integrator
from src.cr3bp_model import CR3BPModel
from src.numerical import NumericalSettings


@dataclass
class Seed:
    identification: str
    state: np.ndarray
    time: float


class Lyapunov:
    def __init__(self, system: CR3BPModel,
                 integrator: Integrator,
                 numerical: NumericalSettings,
                 libration_point,
                 max_iter=100,
                 time_step=0.001,
                 lyapunov_precision=1e-10
                 ):
        self.system = system
        self.numerical = numerical
        self.integrator = integrator

        self.lyapunov_precision = lyapunov_precision
        l1, l2, l3 = self.system.compute_libration_points(self.lyapunov_precision)
        self.libration_point = libration_point
        mapping = {
            LibrationPoint.L1: l1[0],
            LibrationPoint.L2: l2[0],
            LibrationPoint.L3: l3[0],
        }
        self.l_i = mapping[self.libration_point]

        self.max_iter = max_iter
        self.dt = self.numerical.time_step
        trajectory_file = f'{self.integrator.value}'.upper()+f'_{self.dt}.dat'
        system_folder_name = f'mu_{self.system.mu:.15g}'
        system_folder_path = (cfg.DATA_FOLDER / system_folder_name).resolve()

        l1_folder = (system_folder_path / 'L1').resolve()
        l2_folder = (system_folder_path / 'L2').resolve()

        system_folder_path.mkdir(parents=True, exist_ok=True)
        l1_folder.mkdir(parents=True, exist_ok=True)
        l2_folder.mkdir(parents=True, exist_ok=True)

        if libration_point is LibrationPoint.L1:
            self.trajectory_file = (l1_folder / trajectory_file).resolve()
        elif libration_point is LibrationPoint.L2:
            self.trajectory_file = (l2_folder / trajectory_file).resolve()
        else:
            raise RuntimeError(f'Wrong libration point, only L1 and L2 are allowed, {libration_point}')

    def initial_condition(self, c):
        return self.system.initial_condition(c, self.l_i)

    def initialize_seeds(self):
        stm = np.eye(4).flatten()
        init_1 = self.system.initial_condition(c=-0.001, libration_point=self.l_i)
        init_1 = self.stabilize(init_1, self.lyapunov_precision)
        init_1 = np.concatenate((init_1, stm))
        init_2 = self.system.initial_condition(c=-0.002, libration_point=self.l_i)
        init_2 = self.stabilize(init_2, self.lyapunov_precision)
        init_2 = np.concatenate((init_2, stm))

        return [init_1, init_2]

    def stabilize(self, initial_condition, tol, full_list=False):
        state = initial_condition.copy()
        dx = np.zeros(4)
        init_conditions = []
        if full_list:
            init_conditions = [state[:4] + dx]
        for iteration in range(self.max_iter):
            dx, r = dif.update_dx(
                system=self.system,
                integrator=self.integrator,
                numerical=self.numerical,
                x=state, dx=dx,
                dt=self.dt
            )
            if full_list:
                init_conditions.append(state[:4] + dx)
            if np.linalg.norm(r) < tol:
                print(f"Converged in {iteration} iterations state: {state[:4] + dx}")
                break
        result = state[:4] + dx
        return init_conditions if full_list else result

    def initialize_family(self):
        orbit_family = []
        orbit_time = []

        # INITIALIZE SEED 1
        init = self.system.initial_condition(c=-0.001, libration_point=self.l_i)
        init = dif.differential_correction(system=self.system, integrator=self.integrator, numerical=self.numerical, x=init, dt=self.dt)[0]
        request = PropagationRequest(
            system=self.system,
            integrator=self.integrator,
            numerical=self.numerical,
            state=init,
            conditions=[FullOrbit(system=self.system)]
        )
        propagator = Propagator(request=request)
        time, traj = propagator.propagate()
        orbit_family.append(traj)
        orbit_time.append(time)

        # INITIALIZE SEED 2
        init = self.system.initial_condition(c=-0.002, libration_point=self.l_i)
        init = dif.differential_correction(system=self.system, integrator=self.integrator, numerical=self.numerical, x=init, dt=self.dt)[0]
        request = PropagationRequest(
            system=self.system,
            integrator=self.integrator,
            numerical=self.numerical,
            state=init,
            conditions=[FullOrbit(system=self.system)]
        )
        propagator = Propagator(request=request)
        time, traj = propagator.propagate()
        orbit_family.append(traj)
        orbit_time.append(time)

        return orbit_family, orbit_time

    def generate_family(self, jacobi_limit, orbit_number=100, scale=1e-2):
        orbit_family, orbit_time = self.initialize_family()
        trigger = False
        # CONTINUATE FAMILY
        while True:
            init, iteration = continuation_step(
                self.system,
                trajectories=[orbit_family[-2][0], orbit_family[-1][0]],
                integrator=self.integrator,
                numerical=self.numerical,
                time_step=self.dt,
                scale=scale)
            if iteration >= 100 or len(orbit_family) >= orbit_number:
                break
            request = PropagationRequest(
                system=self.system,
                integrator=self.integrator,
                numerical=self.numerical,
                state=init,
                conditions=[FullOrbit(system=self.system)]
            )
            propagator = Propagator(request=request)
            p, traj = propagator.propagate()
            jacobi_mean, _ = phy.compute_jacobi_statistics(mu=self.system.mu, trajectory=traj)
            orbit_family.append(traj)
            orbit_time.append(p)
            if trigger:
                break
            if jacobi_mean < jacobi_limit:
                trigger = True
        return orbit_family, orbit_time

    def find_jacobi(self, jacobi_target, tolerance=1e-3):
        time, trajectory = self.find_jacobi_temp(jacobi_target=jacobi_target, tolerance=tolerance)
        seeds = []
        for n, point in enumerate(trajectory):
            seed = Seed(identification=str(n), state=point, time=time[n])
            seeds.append(seed)
        orbit = PeriodicOrbit(
            config=self,
            family=OrbitType.lyapunov,
            trajectory=trajectory,
            seeds=seeds,
            jacobi=jacobi_target
        )
        return orbit

    # FIX THESE TWO METHODS #
    def find_jacobi_temp(self, jacobi_target, tolerance=1e-3):

        # FETCHS THE CLOSEST LOWER SEED
        if self.trajectory_file.exists():
            seeds = []
            print('file found: ', self.trajectory_file)
            with open(self.trajectory_file, 'r') as f:
                data = []
                for line in f:
                    data.append(line.split())
                data = np.array(data, dtype=float)
            print(np.abs(jacobi_target - data[-1][5]), tolerance)
            if jacobi_target > data[0][5] + tolerance:
                raise RuntimeError(f'Target jacobi: {jacobi_target} cannot be greater than L1 jacobi')
            if jacobi_target < data[-1][5] - tolerance:
                print(f'Jacobi value out of range, generating family first')
                self.generate_seeds(jacobi_limit=jacobi_target,
                                    orbit_number=100
                                    )
                return self.find_jacobi_temp(jacobi_target=jacobi_target, tolerance=tolerance)
            else:
                for i in range(len(data)):
                    if jacobi_target > data[i][5]:
                        seeds_buff = data[i-2:i]
                        break
                    elif np.abs(jacobi_target - data[i][5]) <= tolerance:
                        init = np.concatenate((data[i][:4], np.eye(4).flatten()))
                        request = PropagationRequest(
                            system=self.system,
                            integrator=self.integrator,
                            numerical=self.numerical,
                            state=init,
                            conditions=[FullOrbit(system=self.system)]
                        )

                        propagator = Propagator(request=request)
                        print(f'Found jacobi within tolerance: {data[i]}')
                        return propagator.propagate()

                for seed in seeds_buff:
                    init = np.concatenate((seed[:4], np.eye(4).flatten()))
                    request = PropagationRequest(
                        system=self.system,
                        integrator=self.integrator,
                        numerical=self.numerical,
                        state=init,
                        conditions=[FullOrbit(system=self.system)]
                    )
                    propagator = Propagator(request=request)
                    time, traj = propagator.propagate()
                    jacobi_mean, jacobi_std = phy.compute_jacobi_statistics(mu=self.system.mu, trajectory=traj)
                    seeds.append(np.array([*traj[0][:4], time[-1], jacobi_mean, jacobi_std]))
            orbit_family = seeds
        else:
            orbit_family = []
            seeds_traj, seeds_time = self.initialize_family()
            for traj, time in zip(seeds_traj, seeds_time):
                jacobi_mean, jacobi_std = phy.compute_jacobi_statistics(self.system.mu, traj)
                orbit_family.append(np.array([*traj[0][:4], time[-1], jacobi_mean, jacobi_std]))
            data = np.array(orbit_family)
        scale = 1e-2

        while True:
            jacobi_1 = orbit_family[-2][5]
            jacobi_2 = orbit_family[-1][5]
            # print(jacobi_1)
            # print(" ".join(f"{v:.16e}" for v in orbit_family[-1][:4]))

            seed = orbit_family[-1][:4]
            exists_data = np.any(
                np.all(
                    np.isclose(data[:, :4], seed, rtol=1e-12, atol=1e-12),
                    axis=1
                )
            )
            exists_family = any(
                np.allclose(orbit[:4], seed,
                            rtol=1e-12, atol=1e-12)
                for orbit in orbit_family
            )

            # print(exists_data, exists_family)
            if np.abs(jacobi_1 - jacobi_target) < tolerance:
                print('Jacobi found -2: ', orbit_family[-2][5])
                # append here
                # Load existing data
                # print(len(orbit_family), orbit_family)

                return orbit_family[-2]
            elif np.abs(jacobi_2 - jacobi_target) < tolerance:
                print('Jacobi found -1: ', orbit_family[-1][5])
                # orbit_family is assumed to be a NumPy array with the same number of columns
                combined = np.vstack((data, orbit_family))
                # Sort by column 5 (6th column) in descending order
                combined = combined[np.argsort(combined[:, 5])[::-1]]
                # Save back to file
                with open(self.trajectory_file, "w") as f:
                    for row in combined:
                        f.write("".join(f"{v:25.16e}" for v in row))
                        f.write("\n")
                # print(len(orbit_family), orbit_family)

                init = np.concatenate((orbit_family[-1][:4], np.eye(4).flatten()))

                request = PropagationRequest(
                    system=self.system,
                    integrator=self.integrator,
                    numerical=self.numerical,
                    state=init,
                    conditions=[FullOrbit(self.system)],
                )

                propagator = Propagator(request=request)
                return propagator.propagate()
            else:
                init, iteration = continuation_step(
                    system=self.system,
                    integrator=self.integrator,
                    numerical=self.numerical,
                    trajectories=orbit_family,
                    time_step=self.dt,
                    scale=scale
                )
                # print('iteration = ', iteration)
                if iteration >= 100:
                    break
                request = PropagationRequest(
                    system=self.system,
                    integrator=self.integrator,
                    numerical=self.numerical,
                    state=init,
                    conditions=[FullOrbit(self.system)],
                )

                propagator = Propagator(request=request)
                p, traj = propagator.propagate()
                jacobi_mean_temp, jacobi_sdt_temp = phy.compute_jacobi_statistics(self.system.mu, traj)
                print('jacobi tried: ', jacobi_mean_temp)
                if jacobi_mean_temp < jacobi_target < jacobi_2 or jacobi_mean_temp > jacobi_target > jacobi_2:
                    print('scaled down')
                    scale /= 2
                else:
                    orbit_family.append(np.array([*traj[0][:4], p[-1], jacobi_mean_temp, jacobi_sdt_temp]))

    def generate_seeds(self, jacobi_limit, orbit_number=100, scale=1e-2):
        print(self.trajectory_file)
        seeds, seeds_time = self.generate_family(jacobi_limit=jacobi_limit, orbit_number=orbit_number, scale=scale)
        print(f'writing {len(seeds)} seeds')
        with open(self.trajectory_file, 'w') as f:
            for time, trajectory in zip(seeds_time, seeds[:]):
                init_state = trajectory[0][:4].copy()
                period = time[-1]
                jacobi_mean, jacobi_sigma = phy.compute_jacobi_statistics(self.system.mu, trajectory)
                values = np.append(init_state, [period, jacobi_mean, jacobi_sigma])
                formatted = "".join(f"{v:25.16e}" for v in values)
                f.write(formatted + "\n")
        return seeds, seeds_time


class PeriodicOrbit:
    def __init__(self, config: Lyapunov,
                 family: OrbitType,
                 trajectory: list[np.ndarray],
                 seeds: list[Seed],
                 jacobi: float
                 ):
        self.config = config
        self.family = family
        self.trajectory = trajectory
        self.seeds = seeds
        self.jacobi = jacobi
        self.monodromy_matrix = self.trajectory[-1][4:].reshape(4, 4)
        self.x_min = min(np.array(self.trajectory).T[0])
        self.x_max = max(np.array(self.trajectory).T[0])
        self.y_min = min(np.array(self.trajectory).T[1])
        self.y_max = max(np.array(self.trajectory).T[1])

    def get_seed_by_id(self, identification: str) -> Seed | None:
        """Searches across all strand collections to find a specific ID."""
        for seed in self.seeds:
            if seed.identification == identification:
                return seed
        return None
