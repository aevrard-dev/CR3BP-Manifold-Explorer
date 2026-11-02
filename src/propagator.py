import src.numerical as num
import numpy as np
from src.cr3bp_model import CR3BPModel
from src.data import Realm, PoincareSection, Integrator, PropagationDirection
from src.numerical import NumericalSettings, rk4_step, rk45_step, rk6_step
from dataclasses import dataclass, field
from typing import Callable


class Condition:
    def __init__(self, system):
        self.system = system

    def evaluate(self, trajectory, t):
        raise NotImplementedError

    def refine(self, prev_state, curr_state, t, time_step):
        return curr_state


@dataclass
class PropagationRequest:
    """Task-specific execution instructions for a single propagation run."""
    integrator: Integrator
    system: CR3BPModel
    state: np.ndarray
    numerical: NumericalSettings
    conditions: list[Condition]
    max_propagation_time: float = 20
    time_direction: PropagationDirection = PropagationDirection.FORWARD

    # Internal cache for zero-overhead execution
    _integrator_func: Callable = field(init=False, repr=False)

    def __post_init__(self):
        integrator_map = {
            Integrator.RK4: rk4_step,
            Integrator.RK6: rk6_step,
            Integrator.RKF45: rk45_step,
        }
        if self.integrator not in integrator_map:
            raise ValueError(f"Unsupported integrator type: {self.integrator}")

        self._integrator_func = integrator_map[self.integrator]

    def execute_step(self, f, t, y, dt, tol, replay=False):
        return self._integrator_func(f, t, y, dt, tol, replay=replay)


class Propagator:
    """
        Time-marching propagator for dynamical systems.

        This class advances the system state forward in time using a specified
        numerical integrator and dynamical model.

        The system state vector is expected to be formed as:

        [x, y, vx, vy, STM] where STM is the flattened 4x4 State Transition
        Matrix.

        Parameters
        ----------
        request: PropagationRequest object

        Attributes
        ----------
        state : array-like
            Current state vector.
    """

    def __init__(self, request: PropagationRequest):

        # self.propagation_request: propagation_request
        self.system = request.system
        self.dynamics = self.system.dynamics

        self.numerical = request.numerical
        self.dt = self.numerical.time_step * float(request.time_direction.value)

        self.state = request.state
        self.integrator = request.execute_step
        self.conditions = request.conditions
        self.t = 0
        self.t_max = request.max_propagation_time
        self.duration = 0
        self.tol = self.numerical.tolerance

    def propagate(self):
        self._ensure_default_condition()

        # Initialize trajectory array
        time = [self.t]
        trajectory = [self.state.copy()]

        while True:
            # Advance one step
            dt_try = min(self.dt, self.t_max - self.t)
            # dt_try = self.dt
            while True:
                y_new, t_new, dt_new, accept = self.integrator(
                    self.dynamics, self.t, self.state, dt_try, self.tol
                )

                if accept:
                    dt_used = dt_try
                    self.state = y_new
                    self.t = t_new
                    self.dt = dt_new
                    self.duration += np.abs(dt_used)

                    break
                else:
                    dt_try = dt_new  # retry with smaller step
            time.append(self.t)
            trajectory.append(self.state.copy())

            # Check propagation conditions
            trigger = None
            for cond in self.conditions:
                if not cond.evaluate(trajectory, self.duration):
                    trigger = cond
                    break
            if trigger:
                trajectory[-1] = trigger.refine(*trajectory[-2:], self.t-dt_used*2, dt_used)
                break

        return np.array(time), np.array(trajectory)

    def _ensure_default_condition(self):
        existing_condition = {type(c) for c in self.conditions}
        if TimeCondition not in existing_condition:
            self.conditions.append(TimeCondition(self.system, self.t_max))
        if ImpactCondition not in existing_condition:
            self.conditions.append(ImpactCondition(self.system))


class RealmCondition(Condition):
    def __init__(self, system, path: list[Realm], orbit):
        super().__init__(system)
        self.path = path
        self.orbit = orbit
        self._valid = True
        self._triggered = False
        self.tolerance = 0.2

    def _define_realm(self, state) -> Realm:
        x, y, vx, vy = state[:4]
        x_min = self.orbit.x_min
        x_max = self.orbit.x_max
        x_amp = x_max - x_min

        if self.path[0] is Realm.R1:
            if x <= x_min - x_amp * self.tolerance:
                return Realm.P
            elif x >= x_max + x_amp * self.tolerance:
                return Realm.S
            else:
                return Realm.R1

        elif self.path[0] is Realm.R2:
            if x <= x_min - x_amp * self.tolerance:
                return Realm.S
            elif x >= x_max + x_amp * self.tolerance:
                return Realm.E
            else:
                return Realm.R2
        else:
            raise RuntimeError(f'Path error, first value must be R1 or R2, here path: {self.path[0]}')

    def evaluate(self, trajectory, t) -> bool:
        if self._triggered:
            return self._valid
        prev_state, curr_state = trajectory[-2:][:4]
        prev_realm = self._define_realm(prev_state)
        curr_realm = self._define_realm(curr_state)
        if curr_realm is not prev_realm and (prev_realm is self.path[0] and curr_realm is not self.path[-1]):
            print(f'Wrong realm crossing {prev_realm} to {curr_realm}')
            self._valid = False
        elif curr_realm is not prev_realm and (prev_realm is self.path[0] and curr_realm is self.path[1]):
            self._triggered = True
        return self._valid

    def refine(self, prev_state, curr_state, t, time_step):
        # Replay the RK step to recover the dense output data
        _, _, dense = num.rk45_step(
            f=self.system.dynamics,
            t=t,
            y=prev_state,
            dt=time_step,
            tol=0,
            replay=True,
        )

        prev_realm = self._define_realm(prev_state)
        curr_realm = self._define_realm(curr_state)

        x_min, x_max = self.orbit.x_min, self.orbit.x_max
        x_amp = x_max - x_min
        target = 0.0
        if prev_realm is Realm.R1:
            if curr_realm is Realm.P:
                target = x_min - x_amp * self.tolerance
            elif curr_realm is Realm.S:
                target = x_max + x_amp * self.tolerance
        elif prev_realm is Realm.R2:
            if curr_realm is Realm.S:
                target = x_min - x_amp * self.tolerance
            elif curr_realm is Realm.E:
                target = x_max + x_amp * self.tolerance
        else:
            raise RuntimeError(f'TARGET ERROR')
        # Initial guess from linear interpolation
        x1 = prev_state[0] - target
        x2 = curr_state[0] - target
        theta = -x1 / (x2 - x1)

        max_iter = 20

        for _ in range(max_iter):
            state_theta = num.dense_eval(dense, theta)

            # Event function: y = 0
            g = state_theta[0] - target

            if abs(g) < 1e-16:
                break

            # Analytical derivative:
            # d(y)/dθ = dt * vy
            dg = time_step * state_theta[2]

            # Avoid division by a nearly-zero derivative
            if abs(dg) < 1e-14:
                break

            theta -= g / dg

            # Keep the solution within the current integration step
            theta = np.clip(theta, 0.0, 1.0)
        return num.dense_eval(dense, theta)


class TimeCondition(Condition):
    def __init__(self, system, t_max):
        super().__init__(system)
        self.t_max = t_max

    def evaluate(self, trajectory, time):
        return time < self.t_max


class ImpactCondition(Condition):
    def __init__(self, system: CR3BPModel):
        super().__init__(system)

    def evaluate(self, trajectory, t):
        x, y, vx, vy = trajectory[-1][:4]
        r1 = np.sqrt((x + self.system.mu) ** 2 + y ** 2)
        r2 = np.sqrt((x + self.system.mu - 1) ** 2 + y ** 2)

        r_m1 = self.system.r1_normalized
        r_m2 = self.system.r2_normalized

        if r1 <= r_m1:
            print(f'Impact on {self.system.primary.name}')
            return False
        elif r2 <= r_m2:
            print(f'Impact on {self.system.secondary.name}')
            return False
        else:
            return True


class PlaneCrossing(Condition):
    def __init__(self, system):
        super().__init__(system)
        pass

    def evaluate(self, trajectory, time):
        if len(trajectory) < 2:
            return True
        y_prev, y_curr = trajectory[-2][1], trajectory[-1][1]
        if y_prev > 0 > y_curr or y_curr > 0 > y_prev:
            return False
        else:
            return True

    def refine(self, prev_state, curr_state, t, time_step):
        # Replay the RK step to recover the dense output data
        _, _, dense = num.rk45_step(
            f=self.system.dynamics,
            t=t,
            y=prev_state,
            dt=time_step,
            tol=0,
            replay=True,
        )

        target = 0.0

        # Initial guess from linear interpolation
        y1 = prev_state[1] - target
        y2 = curr_state[1] - target
        theta = -y1 / (y2 - y1)

        max_iter = 20

        for _ in range(max_iter):
            state_theta = num.dense_eval(dense, theta)

            # Event function: y = 0
            g = state_theta[1] - target

            if abs(g) < 1e-16:
                break

            # Analytical derivative:
            # d(y)/dθ = dt * vy
            dg = time_step * state_theta[3]

            # Avoid division by a nearly-zero derivative
            if abs(dg) < 1e-14:
                break

            theta -= g / dg

            # Keep the solution within the current integration step
            theta = np.clip(theta, 0.0, 1.0)

        return num.dense_eval(dense, theta)


class PoincareMaps(Condition):
    def __init__(self, system, maps_def, map_list, direction):
        super().__init__(system)
        self.direction = direction

        self.map = map_list
        self.count = np.zeros(4)
        if maps_def is None:
            maps_def = []
        self.maps_def = maps_def
        self.trigger = None
        self.direction = (0 if direction is None else direction.value)

    def evaluate(self, trajectory, t):
        # print(self.count)
        x_prev, y_prev, vx_prev, vy_prev = trajectory[-2][:4]
        x_curr, y_curr, vx_curr, vy_curr = trajectory[-1][:4]

        if self.map.section is PoincareSection.U1:
            x_1, x_2 = self.maps_def[0]
            if np.sign(y_prev * y_curr) < 0 and x_1[0] < x_curr < x_2[0]:

                if self.direction == 0:
                    self.count[0] += 1
                else:
                    if np.sign(y_curr) == np.sign(self.direction):
                        self.count[0] += 1

                if self.count[0] >= self.map.max_intersections:
                    self.trigger = 0
                    return False
        elif self.map.section is PoincareSection.U2:
            x_1, x_2 = self.maps_def[1]
            if np.sign((1 - self.system.mu - x_prev) * (1 - self.system.mu - x_curr)) < 0 and x_1[1] > y_curr > x_2[1]:

                if self.direction == 0:
                    self.count[1] += 1
                else:
                    if np.sign(x_curr - (1 - self.system.mu)) == np.sign(self.direction):
                        self.count[1] += 1

                if self.count[1] >= self.map.max_intersections:
                    self.trigger = 1
                    return False
        elif self.map.section is PoincareSection.U3:
            x_1, x_2 = self.maps_def[2]
            if np.sign((1 - self.system.mu - x_prev) * (1 - self.system.mu - x_curr)) < 0 and x_1[1] > y_curr > x_2[1]:

                if self.direction == 0:
                    self.count[2] += 1
                else:
                    if np.sign(x_curr - (1 - self.system.mu)) == np.sign(self.direction):
                        self.count[2] += 1

                if self.count[2] >= self.map.max_intersections:
                    self.trigger = 2
                    return False
        elif self.map.section is PoincareSection.U4:
            x_1, x_2 = self.maps_def[3]
            if x_curr < x_1[0] and np.sign(y_curr * y_prev) < 0:

                if self.direction == 0:
                    self.count[3] += 1
                else:
                    if np.sign(y_curr) == np.sign(self.direction):
                        self.count[3] += 1

                if self.count[3] >= self.map.max_intersections:
                    self.trigger = 3
                    return False
        return True

    def refine(self, prev_state, curr_state, t, time_step):
        if self.trigger == 0 or self.trigger == 3:
            _, _, dense = num.rk45_step(
                f=self.system.dynamics,
                t=t,
                y=prev_state,
                dt=time_step,
                tol=0,
                replay=True,
            )
            target = 0.0
            y1 = prev_state[1] - target
            y2 = curr_state[1] - target
            theta = -y1 / (y2 - y1)
            max_iter = 20
            for _ in range(max_iter):
                state_theta = num.dense_eval(dense, theta)
                g = state_theta[1] - target
                if abs(g) < 1e-16:
                    break
                dg = time_step * state_theta[3]
                if abs(dg) < 1e-14:
                    break
                theta -= g / dg
                theta = np.clip(theta, 0.0, 1.0)
            state = num.dense_eval(dense, theta)

            if abs(state[1]) <= 1e-16:
                state[1] = 0.0
            return state

        elif 0 < self.trigger < 3:
            curr_state_2, _, dense = num.rk45_step(
                f=self.system.dynamics,
                t=t,
                y=prev_state,
                dt=time_step,
                tol=0,
                replay=True
            )

            target = 1 - self.system.mu

            # initial guess (linear interpolation)
            x1 = prev_state[0] - target
            x2 = curr_state[0] - target
            theta = -x1 / (x2 - x1)

            max_iter = 20
            eps = 1e-8

            g = float("inf")
            it = 0

            while abs(g) > 1e-16 and it < max_iter:

                y_theta = num.dense_eval(dense, theta)
                g = y_theta[0] - target

                # numerical derivative
                yp = num.dense_eval(dense, theta + eps)
                dg = (yp[0] - y_theta[0]) / eps

                # safeguard against bad derivative
                if abs(dg) < 1e-14:
                    break

                theta -= g / dg

                # keep theta in bounds
                theta = min(1.0, max(0.0, theta))
                # print(abs(g))
                it += 1
            state = num.dense_eval(dense, theta)

            if abs(state[0] - (1 - self.system.mu)) <= 1e-16:
                state[0] = 1 - self.system.mu
            return state


class FullOrbit(Condition):
    def __init__(self, system):
        super().__init__(system)
        pass

    def evaluate(self, trajectory, time):
        if len(trajectory) < 2:
            return True
        y_prev, y_curr, vy_curr, vy_init = trajectory[-2][1], trajectory[-1][1], trajectory[-1][3], trajectory[0][3]

        if np.sign(y_prev * y_curr) < 0 < np.sign(vy_init * vy_curr):
            # print('crossing detected')
            return False
        else:
            return True

    def refine(self, prev_state, curr_state, t, time_step):
        # Replay the RK step to recover the dense output data
        _, _, dense = num.rk45_step(
            f=self.system.dynamics,
            t=t,
            y=prev_state,
            dt=time_step,
            tol=0,
            replay=True,
        )

        target = 0.0

        # Initial guess from linear interpolation
        y1 = prev_state[1] - target
        y2 = curr_state[1] - target
        theta = -y1 / (y2 - y1)

        max_iter = 20

        for _ in range(max_iter):
            state_theta = num.dense_eval(dense, theta)

            # Event function: y = 0
            g = state_theta[1] - target

            if abs(g) < 1e-16:
                break

            # Analytical derivative:
            # d(y)/dθ = dt * vy
            dg = time_step * state_theta[3]

            # Avoid division by a nearly-zero derivative
            if abs(dg) < 1e-14:
                break

            theta -= g / dg

            # Keep the solution within the current integration step
            theta = np.clip(theta, 0.0, 1.0)

        return num.dense_eval(dense, theta)
