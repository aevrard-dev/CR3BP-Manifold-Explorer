import numpy as np


def pot_eff(x: float, y: float, mu: float) -> float:
    """Effective potential U(x, y) for CR3BP."""
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x + mu - 1) ** 2 + y ** 2)
    return - (1-mu)/r1 - mu/r2 - 0.5 * (x**2 + y**2) - 0.5 * mu*(1-mu)


def grad_pot_x(x: float, y: float, mu: float) -> float:
    """Effective potential gradiant U(x,y) in x dimension for CR3BP."""
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x + mu - 1)**2 + y**2)
    grad_x = (1 - mu) * (x + mu) / r1**3 + mu * (x + mu - 1) / r2**3 - x
    return grad_x


def grad_pot_y(x: float, y: float, mu: float) -> float:
    """Effective potential gradiant U(x,y) in y dimension for CR3BP."""
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x + mu - 1)**2 + y**2)
    grad_y = (1 - mu) * y / r1**3 + mu * y / r2**3 - y
    return grad_y


def pcr3bp_rhs(t, state, mu):
    """
    Computes the time derivative of the state for the planar PCR3BP.

    Parameters:
        mu : reduced mass parameter
        t : time array, unused for now
        state : array_like [x, y, vx, vy]

    Returns:
        dstate_dt : ndarray [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """
    x, y, vx, vy = state[:4]
    stm = jacobian(x, y, mu) @ state[4:].reshape(4, 4)
    # Compute gradients of the effective potential
    ux = grad_pot_x(x, y, mu)
    uy = grad_pot_y(x, y, mu)

    # Equations of motion in rotating frame
    dxdt = vx
    dydt = vy
    dvxdt = 2 * vy - ux
    dvydt = -2 * vx - uy

    return np.array([dxdt, dydt, dvxdt, dvydt, *stm.flatten()])


def pot_eff_x(mu, x):
    return - 0.5*x**2 - (1-mu) / np.abs(x+mu) - mu / np.abs(x-1+mu)


def pot_eff_x_prime(mu, x):
    return - x + (1-mu)*(x+mu) / np.abs(x+mu)**3 + mu * (x-1+mu) / np.abs(x-1+mu)**3


def pot_eff_y(mu, y):
    return -0.5*((1-mu)**2 + y**2) - (1-mu)/np.sqrt(1 + y**2) - mu/np.abs(y) - 0.5*mu*(1-mu)


def pot_eff_y_prime(mu, y):
    return -y + (1-mu)*y / (1 + y**2)**(3/2) + mu * y / np.abs(y)**3


def pot_eff_x_second(mu, x):
    first = (1-mu) * (1/np.abs(x+mu)**3 - 3*(x+mu)**2/np.abs(x+mu)**5)
    second = mu * (1/np.abs(x-1+mu)**3 - 3*(x-1+mu)**2/np.abs(x-1+mu)**5)

    return - 1 + first + second


def compute_hill_radius(mu):
    return ((1-mu)/3)**(1/3), (mu/3)**(1/3)


def compute_jacobi(mu, state):
    x, y, vx, vy = state
    c = 2 * (- pot_eff(x, y, mu)) - (vx**2 + vy**2)
    return c


def rot_inert_to_pcr3bp(t, theta0=0.0, omega=1.0):
    """
    Inertial -> PCR3BP rotating frame
    """
    theta = omega * t + theta0
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c,  s],
        [-s,  c]]
    )


def rot_pcr3bp_to_inert(t, theta0=0.0, omega=1.0):
    """
    PCR3BP rotating frame -> inertial
    """
    theta = omega * t + theta0
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s],
        [s,  c]]
    )


def skew_matrix():
    """2D equivalent of cross product with +z"""
    return np.array([[0.0, -1.0],
                     [1.0,  0.0]])


def state_rotating_to_inertial(state_R, t, omega=1.0, theta0=0.0):
    """
    Convert [x, y, vx, vy] from rotating (PCR3BP) to inertial frame
    """
    x, y, vx, vy = state_R[:4]
    r_R = np.array([x, y])
    v_R = np.array([vx, vy])

    theta = omega * t + theta0
    R = rot_pcr3bp_to_inert(theta)
    J = skew_matrix()

    r_I = R @ r_R
    v_I = R @ (v_R + omega * J @ r_R)

    return np.hstack((r_I, v_I))


def state_inertial_to_rotating(state_I, t, omega=1.0, theta0=0.0):
    """
    Convert [x, y, vx, vy] from inertial to rotating (PCR3BP) frame
    """
    x, y, vx, vy = state_I
    r_I = np.array([x, y])
    v_I = np.array([vx, vy])

    theta = omega * t + theta0
    R = rot_pcr3bp_to_inert(theta)
    RT = R.T
    J = skew_matrix()

    r_R = RT @ r_I
    v_R = RT @ v_I - omega * J @ r_R

    return np.hstack((r_R, v_R))


def jacobian(x: float, y: float, mu: float):
    """
    Jacobian matrix A of the planar PCR3BP
    evaluated at any [x, y] point of space.

    State ordering: [x, y, vx, vy]
    ! reformulate this function using the previously defined ones !
    """

    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x + mu - 1)**2 + y**2)

    r1_3 = r1**3
    r1_5 = r1**5
    r2_3 = r2**3
    r2_5 = r2**5

    u_xx = (
        (1 - mu) * (3*(x + mu)**2 / r1_5 - 1/r1_3)
        + mu * (3*(x + mu - 1)**2 / r2_5 - 1/r2_3)
        - 1
    )

    u_yy = (
        (1 - mu) * (3*y**2 / r1_5 - 1/r1_3)
        + mu * (3*y**2 / r2_5 - 1/r2_3)
        - 1
    )

    u_xy = (
        3*(1 - mu)*(x + mu)*y / r1_5
        + 3*mu*(x + mu - 1)*y / r2_5
    )

    a = np.array([
        [0,     0,     1,   0],
        [0,     0,     0,   1],
        [u_xx,  u_xy,  0,   2],
        [u_xy,  u_yy, -2,   0]
    ])

    return a


def confirm_stm_eigen_values(stm):
    eigen_values = np.linalg.eigvals(stm)
    if np.abs(np.prod(eigen_values) - 1) < 1e-6:
        check = True
    else:
        check = False
    print(f'Eigen values check: {check}')


def initial_condition(mu, c, libration_point):
    """
    Computes the initial conditions from the linearized equation
    of motion at the colinear libration point L_i from:\n
    X(t0) = c ( a cos(omega t0) - b sin(omega t0))
    :param
    mu: Reduced mass parameter
    c: Amplitude factor
    libration_point: Libration point of the initial condition
    """
    c *= np.sign(libration_point)
    a = jacobian(libration_point, 0, mu)
    eigen_values, eigen_vectors = np.linalg.eig(a)

    """Selects the purely imaginary eigen value and its vector"""
    idx_center = None
    for i, lam in enumerate(eigen_values):
        if abs(np.real(lam)) < 1e-10 < abs(np.imag(lam)):
            idx_center = i
            break
    vector = eigen_vectors[:, idx_center]
    omega = abs(np.imag(eigen_values[idx_center]))

    t = np.linspace(0, 5, 1000)
    a = np.real(vector)
    b = np.imag(vector)

    xi = []
    for i in range(4):
        xi.append(c * (a[i] * np.cos(omega * t) - b[i] * np.sin(omega * t)))
        xi[i] = np.array(xi[i])
        if i == 0:
            xi[i] += libration_point
    # xi[-1] = np.abs(xi[-1])
    stm = np.eye(4).flatten()
    x0 = np.concatenate((np.array(xi).T[0], stm))
    for i in range(len(x0)):
        if np.abs(x0[i]) <= 1e-16:
            x0[i] = 0
    return np.array(x0)


def compute_jacobi_statistics(mu, trajectory):

    jacobi_values = []
    mean = 0
    count = 0

    # Compute Jacobi values and mean
    for point in trajectory:
        jacobi = compute_jacobi(mu, point[:4])
        jacobi_values.append(jacobi)
        mean += jacobi
        count += 1
    mean /= count

    # Compute standard deviation
    sigma = 0
    for jacobi in jacobi_values:
        sigma += (jacobi - mean) ** 2
    sigma = (sigma / count) ** 0.5

    return mean, sigma
