import numpy as np
from dataclasses import dataclass


@dataclass
class NumericalSettings:
    time_step: float = 0.001
    seed_density: int = 20
    tolerance: float = 1e-5
    lyapunov_precision: float = 1e-10
    max_propagation_time: float = 20.0


def newton_raphson(f, f_prime, x_start, precision):
    x = x_start
    iteration = 0
    while np.abs(f(x)) >= precision:
        iteration += 1
        x = x - f(x) / f_prime(x)
        if iteration >= 100:
            return None
    return x


def rk4_step(f, t, y, dt, tol, replay=False):
    """
    Perform a single RK4 step for dy/dt = f(t, y)

    Parameters:
        f  : function f(t, y)
        t  : current time
        y  : current value (scalar or numpy array)
        dt : time step size
        tol: placeholder
        replay: placeholder

    Returns:
        y_next : value at t + dt
    """
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt*k1/2)
    k3 = f(t + dt/2, y + dt*k2/2)
    k4 = f(t + dt, y + dt*k3)

    y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    t_new = t + dt

    return y_new, t_new, dt, True


def rk45_step(f, t, y, dt, tol, replay=False):
    """
    Perform a single adaptive RK45 (Dormand–Prince) step.

    Parameters:
        f      : function f(t, y)
        t      : current time
        y      : current state (numpy array)
        dt     : current step size
        tol    : desired error tolerance
        replay : used for last step interpolation, return the dense output with intermediate k values

    Returns:
        y_next : next state (accepted step)
        t_next : next time
        dt_new : suggested next step size
        accept : whether the step was accepted
    """

    # Coefficients (Dormand–Prince)
    k1 = f(t, y)
    k2 = f(t + dt*(1/5), y + dt*(1/5)*k1)
    k3 = f(t + dt*(3/10), y + dt*(3/40)*k1 + dt*(9/40)*k2)
    k4 = f(t + dt*(4/5), y + dt*(44/45)*k1 - dt*(56/15)*k2 + dt*(32/9)*k3)
    k5 = f(t + dt*(8/9), y + dt*(19372/6561)*k1 - dt*(25360/2187)*k2 + dt*(64448/6561)*k3 - dt*(212/729)*k4)
    k6 = f(t + dt, y + dt*(9017/3168)*k1 - dt*(355/33)*k2 + dt*(46732/5247)*k3 + dt*(49/176)*k4 - dt*(5103/18656)*k5)

    # 5th order solution
    y5 = y + dt * (
        (35/384)*k1 +
        (500/1113)*k3 +
        (125/192)*k4 -
        (2187/6784)*k5 +
        (11/84)*k6
    )
    if replay:
        k7 = f(t + dt, y5)
        return y5, t + dt, {
            "y0": y,
            "dt": dt,
            "k": [k1, k2, k3, k4, k5, k6, k7]
        }

    # 4th order solution (for error estimate)
    k7 = f(t + dt, y5)
    y4 = y + dt * (
        (5179/57600)*k1 +
        (7571/16695)*k3 +
        (393/640)*k4 -
        (92097/339200)*k5 +
        (187/2100)*k6 +
        (1/40)*k7
    )

    # Error estimate
    err = np.linalg.norm(y5 - y4, ord=np.inf)

    # Step size control
    safety = 0.9
    if err == 0:
        dt_new = dt * 2
        accept = True
    else:
        dt_new = dt * safety * (tol / err)**0.2  # 0.2 = 1/(order+1)
        accept = err < tol
    if accept:
        return y5, t + dt, dt_new, True
    else:
        return y, t, dt_new, False


def rk6_step(f, t, y, dt, tol, replay=False):
    """
    Perform a single RK6 step for dy/dt = f(t, y).

    Parameters:
        f      : function f(t, y)
        t      : current time
        y      : current value (scalar or numpy array)
        dt     : time step size
        tol    : error tolerance (unused in this fixed-step method)
        replay : replay flag (unused)

    Returns:
        y_next : value at t + dt
        t_next : next time
        dt     : time step size
        success: True
    """
    k1 = f(t, y)

    k2 = f(t + dt/3,
           y + dt * k1/3)

    k3 = f(t + dt/3,
           y + dt * (k1 + k2)/6)

    k4 = f(t + dt/2,
           y + dt * (k1 + 3*k3)/8)

    k5 = f(t + 2*dt/3,
           y + dt * (k1 - 3*k3 + 4*k4)/2)

    k6 = f(t + dt,
           y + dt * (k1 + 4*k4 + k5)/6)

    k7 = f(t + dt,
           y + dt * (k1 + 4*k4 + k5 + k6)/6)

    y_new = y + (dt/6) * (k1 + 4*k4 + k5 + k6)
    t_new = t + dt

    return y_new, t_new, dt, True


def linear_interpolate(state1, state2):
    """
    This targets for y=0, assumes state = [x, y, vx, vy]
    :param state1:
    :param state2:
    :return:
    """
    y1 = state1[1]
    y2 = state2[1]

    alpha = -y1/(y2-y1)

    target = state1 + alpha * (state2 - state1)

    return target


def check_stm_properties(stm: np.ndarray, tol: float = 1e-4) -> dict:
    """
    Validates canonical symplecticity and eigenvalue invariants for a 4x4 Planar CR3BP STM.
    """
    if stm.shape != (4, 4):
        raise ValueError(f"Expected 4x4 STM, got shape {stm.shape}")

    # 1. Canonical Coordinate Transformation Matrix T ([x, y, vx, vy] -> [x, y, px, py])
    T = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 1.0, 0.0],  # px = vx - y
        [1.0, 0.0, 0.0, 1.0]  # py = vy + x
    ])

    T_inv = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],  # vx = px + y
        [-1.0, 0.0, 0.0, 1.0]  # vy = py - x
    ])

    # Transform STM to canonical phase space
    stm_canonical = T @ stm @ T_inv

    # Standard Canonical Symplectic Matrix J
    J = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0]
    ])

    # Symplecticity check: Phi_can^T * J * Phi_can = J
    symp_diff = stm_canonical.T @ J @ stm_canonical - J
    symplectic_error = np.linalg.norm(symp_diff) / np.linalg.norm(J)

    # 2. Determinant check
    det_error = np.abs(np.linalg.det(stm_canonical) - 1.0)

    # 3. Eigenvalues & Reciprocal Pair Check (Monodromy)
    eigenvalues, eigenvectors = np.linalg.eig(stm)
    algebraic_error = np.linalg.norm(stm @ eigenvectors - eigenvectors @ np.diag(eigenvalues)) / np.linalg.norm(stm)

    # Sort eigenvalues by magnitude to check unstable/stable reciprocal pair
    abs_eigs = np.abs(eigenvalues)
    idx = np.argsort(abs_eigs)
    sorted_eigs = eigenvalues[idx]  # lambda_min at [0], lambda_max at [-1]

    reciprocal_pair_error = np.abs(sorted_eigs[0] * sorted_eigs[-1] - 1.0)

    checks = {
        "valid": bool(
            symplectic_error < tol and
            det_error < tol and
            algebraic_error < tol and
            reciprocal_pair_error < tol
        ),
        "symplectic_error": float(symplectic_error),
        "det_error": float(det_error),
        "eigen_error": float(algebraic_error),
        "reciprocal_pair_error": float(reciprocal_pair_error)
    }

    return checks


def dense_eval(data, theta):

    y0 = data["y0"]
    dt = data["dt"]

    k1, k2, k3, k4, k5, k6, k7 = data["k"]

    # Dormand–Prince dense output polynomials
    b1 = theta + theta**2*(-183/64 + theta*(37/12 + theta*(-145/128)))
    b3 = theta**2*(1500/371 + theta*(-1000/159 + theta*(1000/371)))
    b4 = theta**2*(-125/32 + theta*(125/12 + theta*(-375/64)))
    b5 = theta**2*(9477/3392 + theta*(-729/106 + theta*(25515/6784)))
    b6 = theta**2*(-11/7 + theta*(11/3 + theta*(-55/28)))
    b7 = theta**2*(3/2 + theta*(-4 + theta*(5/2)))

    return y0 + dt * (
        b1*k1 +
        b3*k3 +
        b4*k4 +
        b5*k5 +
        b6*k6 +
        b7*k7
    )


def find_manifold_intersections(sample_1, sample_2, closed=False, tol=1e-12):
    """
    Find intersections between two manifold traces projected onto (x, vx).

    Parameters
    ----------
    sample_1 : (N,4) array_like
        States [x, y, vx, vy].
    sample_2 : (M,4) array_like
        States [x, y, vx, vy].
    closed : bool
        If True, include segment (last -> first).
    tol : float
        Numerical tolerance.

    Returns
    -------
    intersections : list of dict
    """

    sample_1 = [x[-1][:4] for x in sample_1]
    sample_2 = [x[-1][:4] for x in sample_2]

    sample_1 = np.asarray(sample_1, dtype=float)
    sample_2 = np.asarray(sample_2, dtype=float)

    intersections = []

    n1 = len(sample_1)
    n2 = len(sample_2)

    if n1 < 2 or n2 < 2:
        return intersections

    nseg1 = n1 if closed else n1 - 1
    nseg2 = n2 if closed else n2 - 1

    for i in range(nseg1):

        i_next = (i + 1) % n1

        A_state = sample_1[i]
        B_state = sample_1[i_next]

        # projection: (x, vx)
        A = np.array([A_state[1], A_state[3]])
        B = np.array([B_state[1], B_state[3]])

        r = B - A

        for j in range(nseg2):

            j_next = (j + 1) % n2

            C_state = sample_2[j]
            D_state = sample_2[j_next]

            C = np.array([C_state[1], C_state[3]])
            D = np.array([D_state[1], D_state[3]])

            s = D - C

            # Solve:
            # A + u*r = C + v*s
            M = np.column_stack((r, -s))
            rhs = C - A

            det = np.linalg.det(M)

            if abs(det) < tol:
                continue

            try:
                u, v = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                continue

            if (
                -tol <= u <= 1.0 + tol
                and -tol <= v <= 1.0 + tol
            ):

                P = A + u * r
                Q = C + v * s

                if np.linalg.norm(P - Q) < 1e-8:

                    intersections.append(
                        {
                            "seg1": (i, i_next),
                            "seg2": (j, j_next),
                            "u": float(u),
                            "v": float(v),
                            "intersection": P,
                            "states": (
                                A_state,
                                B_state,
                                C_state,
                                D_state,
                            ),
                        }
                    )

    return intersections
