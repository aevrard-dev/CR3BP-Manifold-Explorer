import numpy as np
from src.propagator import Propagator, PropagationRequest, PlaneCrossing
from src.cr3bp_model import CR3BPModel
from src.numerical import NumericalSettings
from src.data import Integrator


def update_dx(system: CR3BPModel,  integrator: Integrator, numerical: NumericalSettings, x, dx, dt, alpha=0.5):
    """
    Compute the Newton correction for a Lyapunov orbit half-period.
    x      : current initial state [x0, y0, vx0, vy0]
    dx     : current perturbation (correction vector)
    dt     : integration timestep
    alpha  : damping factor for robustness
    Returns:
        dx_new : updated perturbation vector
        r      : residual vector at half-period
    """

    """
    Compute the Newton correction for a Planar Lyapunov orbit half-period.
    Bound to CR3BPSystem instance (self).
    """
    # 1. Compute perturbed initial condition
    init_state = x[:4] + dx

    # 2. Append STM identity
    state0 = np.concatenate((init_state, np.eye(4).flatten()))

    # 3. Propagate to half-period using system dynamics and system-aware plane crossing

    request = PropagationRequest(
        system=system,
        numerical=numerical,
        integrator=integrator,
        state=state0,
        conditions=[PlaneCrossing(system=system)]
    )

    propagation = Propagator(request=request)

    time, state_half = propagation.propagate()

    if len(state_half) < 2:
        raise RuntimeError("Half-period event not detected; trajectory invalid.")

    # 4. Extract state & STM at half-period
    final_record = state_half[-1]
    yh, vxh = final_record[1], final_record[2]
    phi = final_record[4:].reshape((4, 4))

    # 5. Residual vector r = [y_half, vx_half]^T
    r = np.array([yh, vxh])

    # 6. Jacobian J = [d(y)/d(vy0), d(vx)/d(vy0)]^T
    J = np.array([[phi[1, 3]],
                  [phi[2, 3]]])  # Shape (2, 1)

    # 7. Least-squares / Pseudo-inverse solve
    delta_vy0, _, _, _ = np.linalg.lstsq(J, -r, rcond=None)

    # delta_vy0 = (J_pinv @ (-r))[0]

    # 8. Apply update
    dx_new = dx.copy()
    dx_new[3] += alpha * delta_vy0[0]

    return dx_new, r


def differential_correction(system: CR3BPModel, integrator: Integrator, numerical: NumericalSettings, x, dt, tol=1e-10, max_iter=100):
    """
    Differential correction routine to compute Lyapunov orbits.
    x        : initial guess [x0, y0, vx0, vy0]
    dt       : timestep for propagation
    tol      : convergence tolerance for residual
    max_iter : maximum number of Newton iterations
    Returns:
        corrected_state : [x0, y0, vx0, vy0, STM flattened]
    """

    dx = np.zeros(4)  # perturbation vector
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        dx, r = update_dx(system=system, integrator=integrator, numerical=numerical, x=x, dx=dx, dt=dt)
        if np.linalg.norm(r) < tol:
            line = " ".join(f"{v:.16e}" for v in x[:4] + dx)
            # print(line)
            print(f"Converged in {iteration} iterations state: {line}")
            break

    if iteration >= max_iter:
        print("Warning: differential correction did not converge within max_iter")

    # return corrected initial state + STM identity
    stm = np.eye(4).flatten()
    corrected_state = np.concatenate((x[:4] + dx, stm))
    return corrected_state, iteration
