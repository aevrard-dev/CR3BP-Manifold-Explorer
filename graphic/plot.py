import src.physics as phy
import src.configuration as cfg
from src.data import PoincareSection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from src.manifold import ManifoldResult, ManifoldBranch
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt


def plot_numerical_convergence(results):
    data = np.array(results)

    # Sort data by tolerance ascending to ensure proper curve fitting and plotting
    data = data[data[:, 0].argsort()]

    tolerances = data[:, 0]
    sigmas = data[:, 2]

    # --- plotting ---
    plt.figure(figsize=(8, 6))

    # --- log-log regression ---
    log_tol = np.log10(tolerances)
    log_sigma = np.log10(sigmas)

    plt.loglog(
        tolerances,
        sigmas,
        'o',
        label='Measured'
    )

    # Dynamically find the turning point (the local minimum / optimal sweet spot)
    # where both preceding and succeeding points have higher error values.
    turning_idx = np.argmin(sigmas)

    # Split the data around the dynamic turning point for dual-regime fitting
    slope_1, intercept_1 = np.polyfit(log_tol[:turning_idx + 1], log_sigma[:turning_idx + 1], 1)
    slope_2, intercept_2 = np.polyfit(log_tol[turning_idx:], log_sigma[turning_idx:], 1)

    # fitted curves
    fit_1 = 10 ** intercept_1 * tolerances ** slope_1
    fit_2 = 10 ** intercept_2 * tolerances ** slope_2

    plt.loglog(
        tolerances,
        fit_1,
        '--',
        label=f'Fit: σ ~ tol^{slope_1:.2f}'
    )
    plt.loglog(
        tolerances,
        fit_2,
        '--',
        label=f'Fit: σ ~ tol^{slope_2:.2f}'
    )
    print(f"Dynamic turning point index: {turning_idx} (Tolerance: {tolerances[turning_idx]:.1e})")
    print(f"Power law exponent = {slope_1:.6f}")

    plt.xlabel('Integrator tolerance')
    plt.ylabel('Standard deviation')
    plt.title("Jacobi drift vs tolerance using adaptive DOP45")
    plt.grid(True, which='both')
    plt.legend()
    plt.show()


def plot_projection(manifold_results: list[ManifoldResult]) -> None:
    fig, ax1 = plt.subplots()

    for result in manifold_results:
        if not result.completed:
            continue

        section = result.config.poincare_map.section
        branch = result.config.branch
        label = result.label
        color = 'red' if branch is ManifoldBranch.UNSTABLE else 'green'

        raw_projection = result.get_projection()
        raw_labeled = False

        for identification, state in raw_projection:
            if section in (PoincareSection.U1, PoincareSection.U4):
                x, y = state[0], state[2]
            elif section in (PoincareSection.U2, PoincareSection.U3):
                x, y = state[1], state[3]
            else:
                raise RuntimeError(f'Wrong section')
            ax1.scatter(x, y, color=color, label=label if not raw_labeled else "")
            raw_labeled = True

        ax1.set_title(f"Phase space projection onto {section.value}")
        if section in (PoincareSection.U1, PoincareSection.U4):
            ax1.set_xlabel("x [nondimensionnalized]")
            ax1.set_ylabel("vx [nondimensionnalized]")
        else:
            ax1.set_xlabel("y [nondimensionnalized]")
            ax1.set_ylabel("vy [nondimensionnalized]")

        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def draw_plot(manifolds_config_list, intersections=None, section=''):
    # ----------------------------
    # FIRST FIGURE
    # ----------------------------
    fig, ax = plt.subplots()

    legend_handles = []

    trajectory = manifolds_config_list[0].trajectories[0].strand
    # print(trajectory)
    mu = manifolds_config_list[0].config.system.mu
    primary_label = manifolds_config_list[0].config.system.primary.name
    secondary_label = manifolds_config_list[0].config.system.secondary.name
    x_lim = manifolds_config_list[0].config.system.nvz_limit_x(trajectory[0][:4], 1e-5)
    y_lim = manifolds_config_list[0].config.system.nvz_limit_y(trajectory[0][:4], 1e-5)

    E = phy.pot_eff(trajectory[0][0], trajectory[0][1], mu) + 0.5 * (trajectory[0][2] ** 2 + trajectory[0][3] ** 2)
    print(E)
    xmin = -1.5
    xmax = 1.2
    xlim = (-1.5, 1.5)
    ylim = (-1.5, 1.5)
    res = 5000
    # ----------------------------
    # Grid
    # ----------------------------
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(*ylim, res)
    X, Y = np.meshgrid(x, y)

    Z = phy.pot_eff(X, Y, mu) - E
    allowed = Z >= 0
    ax.contourf(
        X, Y,
        allowed.astype(float),
        levels=[0, 0.5, 1],
        colors=["white", "lightgray"],
        alpha=1.0
    )

    ax.contour(
        X, Y,
        Z,
        levels=[0],
        colors='black',
        linewidths=1.5,
        alpha=0.7,
        zorder=10
    )

    for manifold in manifolds_config_list:
        if not manifold.completed:
            continue
        color = 'green' if manifold.config.branch is ManifoldBranch.STABLE else 'red'

        for i, traj in enumerate(manifold.completed):

            x, y = traj.strand.T[:2]

            line, = ax.plot(
                x,
                y,
                color=color
            )

            # store actual matplotlib color
            if i == 0:
                color = line.get_color()

        handle = Line2D(
            [0],
            [0],
            color=color,
            lw=2,
            label=manifold.label
        )

        legend_handles.append(handle)

    # initial orbit
    for manifold in manifolds_config_list:
        init_orbit = manifold.config.periodic_orbit.trajectory
        x, y = init_orbit.T[:2]
        ax.plot(x, y, linestyle='--', linewidth=2.5, color='black')

    init_handle = Line2D(
        [0],
        [0],
        color='black',
        lw=2,
        linestyle='--',
        label='init'
    )

    legend_handles.append(init_handle)

    l1_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markeredgecolor='black', markersize=10, label='L1')
    l2_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markeredgecolor='black', markersize=10, label='L2')
    l3_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markeredgecolor='black', markersize=10, label='L3')
    legend_handles.extend([l1_handle, l2_handle, l3_handle])

    # libration points
    l1, l2, l3 = manifolds_config_list[0].config.system.compute_libration_points(1e-10)
    l1 = l1[0]
    l2 = l2[0]
    l3 = l3[0]
    ax.scatter(l1, 0, s=60, color='gold', edgecolor='black', zorder=10, label='L1')
    ax.scatter(l2, 0, s=60, color='lightblue', edgecolor='black', zorder=10, label='L2')
    ax.scatter(l3, 0, s=60, color='purple', edgecolor='black', zorder=10, label='L3')

    # Earth
    r_p = manifolds_config_list[0].config.system.r1_normalized
    primary_center = (-mu, 0)
    primary = Circle(primary_center, r_p, facecolor='deepskyblue', edgecolor='black', lw=1, zorder=10)
    ax.add_patch(primary)
    primary_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='deepskyblue', markeredgecolor='black', markersize=12, label=primary_label)
    legend_handles.append(primary_handle)

    # moon
    r_s = manifolds_config_list[0].config.system.r2_normalized
    secondary_center = (1 - mu, 0)
    secondary = Circle(secondary_center, r_s, facecolor='darkgray', edgecolor='black', lw=1, zorder=10)
    ax.add_patch(secondary)
    secondary_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgray', markeredgecolor='black', markersize=12, label=secondary_label)
    legend_handles.append(secondary_handle)

    # Poincare Maps
    # U1
    ax.plot([-mu, x_lim[1]], [0, 0], color='black', lw=2)
    plt.text((-mu + x_lim[1]) / 2, 0.02, f"U1")
    # U2
    ax.plot([1 - mu, 1 - mu], [0, y_lim[1]], color='black', lw=2)
    plt.text(1 - mu + 0.01, y_lim[1] / 2, f"U2")
    # U3
    ax.plot([1 - mu, 1 - mu], [0, y_lim[0]], color='black', lw=2)
    plt.text(1 - mu + 0.01, y_lim[0] / 2, f"U3")
    # U4
    ax.plot([xmin, x_lim[0]], [0, 0], color='black', lw=2)
    plt.text((xmin + x_lim[0]) / 2, 0.02, f"U4")

    ax.axis('equal')

    ax.set_xlabel("x [non-dimensional]")
    ax.set_ylabel("y [non-dimensional]")

    ax.legend(handles=legend_handles, fontsize=14)
    legend_handles.extend([secondary])

    plt.show()





    # # ----------------------------
    # # SECOND FIGURE
    # # ----------------------------
    # fig2, ax2 = plt.subplots()
    #
    # for manifold in manifolds_config_list:
    #
    #     color = 'green' if manifold.config.branch.value == 'STABLE' else 'red'
    #
    #     # if manifold.projections:
    #     #     continue
    #
    #     test = identify_gaps(manifold.config, manifold.completed)
    #
    #     for i in range(len(test)):
    #         # print(test[i])
    #         state_1, state_2 = test[i]
    #         x_1, y_1, vx_1, vy_1 = state_1
    #         x_2, y_2, vx_2, vy_2 = state_2
    #         if manifold.config.poincare_map.section is PoincareSection.U1 or manifold.config.poincare_map.section is PoincareSection.U4:
    #
    #             ax2.scatter(x_1, vx_1, color='blue')
    #             ax2.scatter(x_2, vx_2, color='blue')
    #         else:
    #             ax2.scatter(y_1, vy_1, color='blue')
    #             ax2.scatter(y_2, vy_2, color='blue')
    #
    #
    #     for i, proj in enumerate(manifold.completed):
    #
    #         # if proj.size != 0:
    #         #     continue
    #         x, y, vx, vy = proj.strand[-1][:4]
    #
    #         if manifold.config.poincare_map.section is PoincareSection.U1 or manifold.config.poincare_map.section is PoincareSection.U4:
    #             x, y = x, vx
    #         else:
    #             x, y = y, vy
    #         if i == 0:
    #             ax2.text(x - 0.13, vx, manifold.label, fontsize=16)
    #             ax2.scatter(
    #                 x,
    #                 y,
    #                 color=color,
    #                 label=manifold.label,
    #             )
    #         else:
    #             ax2.scatter(
    #                 x,
    #                 y,
    #                 color=color,
    #                 s=15
    #             )
    # if intersections is not None:
    #     for intersection in intersections:
    #         x, y, vx, vy = intersection
    #         ax2.scatter(y, vy, color='blue')
    # if manifolds_config_list[0].config.poincare_map.section is PoincareSection.U1 or manifolds_config_list[0].config.poincare_map.section is PoincareSection.U4:
    #     label_x = rf'$x [non-dimensional]$'
    #     label_y = rf'$v_x [non-dimensional]$'
    # else:
    #     label_x = rf'$y [non-dimensional]$'
    #     label_y = rf'$v_y [non-dimensional]$'
    # ax2.set_xlabel(label_x)
    # ax2.set_ylabel(label_y)
    #
    # ax2.legend()
    # ax2.grid(
    #     True,
    #     linestyle=(0, (1, 10)),
    #     linewidth=1,
    #     color='black',
    #     alpha=1.0
    # )
    # plt.show()


from src.manifold import ManifoldStrand, ManifoldConfig


def identify_gaps(
        manifold_config: ManifoldConfig,
        manifold_list: list[ManifoldStrand],
        factor: float = 1.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Identifies pairs of successive Poincaré intersection points in their original
    list order (treated as a circular sequence) whose separation distance
    exceeds the median spacing.

    Parameters:
        manifold_config: tteteet
        manifold_list: eokefokfeok
        points : list of state arrays [x, y, vx, vy, ...]
        factor : multiplier applied to the median threshold (default 1.0)

    Returns:
        sparse_pairs : list of tuples containing the two adjacent points (p_i, p_{i+1})
                       that require refinement.
    """
    points = []
    for manifold in manifold_list:
        points.append(np.array(manifold.strand[-1][:4]))

    if len(points) < 2:
        return []

    pts_array = np.array(points)

    # Extract relevant 2D coordinates (x and vx, indices 0 and 2)

    if manifold_config.poincare_map.section is PoincareSection.U1 or manifold_config.poincare_map.section is PoincareSection.U4:
        coords_2d = pts_array[:, [0, 2]]
    else:
        coords_2d = pts_array[:, [1, 3]]


    n = len(pts_array)
    # --- Z-SCORE STANDARDIZATION ---
    mean = np.mean(coords_2d, axis=0)
    std = np.std(coords_2d, axis=0)
    # Safeguard against zero variance (constant coordinate)
    std[std == 0.0] = 1.0

    scaled_coords = (coords_2d - mean) / std

    distances = np.zeros(n)
    for i in range(n):
        next_i = (i + 1) % n  # Wraps last point back to the first point
        distances[i] = np.linalg.norm(scaled_coords[next_i] - scaled_coords[i])

    # Compute the median distance across the sequence
    median_dist = np.median(distances)
    threshold = median_dist * factor

    # Find and collect pairs exceeding the threshold
    sparse_pairs = []
    for i in range(n):
        next_i = (i + 1) % n
        if distances[i] > threshold:
            sparse_pairs.append((scaled_coords[i], scaled_coords[next_i]))



    for i in range(len(scaled_coords)):
        plt.scatter(scaled_coords[i][0], scaled_coords[i][1], color='red')
    for j in range(len(sparse_pairs)):
        state_1, state_2 = sparse_pairs[j]
        x_1, y_1 = state_1
        x_2, y_2 = state_2
        plt.scatter(x_1, y_1, color='blue', s=20)
        plt.scatter(x_2, y_2, color='blue', s=20)
    plt.show()


    # Compute distances between direct neighbors in the list (circular wrap-around)
    distances = np.zeros(n)
    for i in range(n):
        next_i = (i + 1) % n  # Wraps last point back to the first point
        distances[i] = np.linalg.norm(coords_2d[next_i] - coords_2d[i])

    # Compute the median distance across the sequence
    median_dist = np.median(distances)
    threshold = median_dist * factor

    # Find and collect pairs exceeding the threshold
    sparse_pairs = []
    for i in range(n):
        next_i = (i + 1) % n
        if distances[i] > threshold:
            sparse_pairs.append((pts_array[i], pts_array[next_i]))

    return sparse_pairs


def plot_manifolds(
        trajectory,
        unstable_manifolds,
        stable_manifolds,
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
        res=5000,
        hill_sphere=False
):

    l1, l2, l3 = phy.compute_libration_points(mu, 1e-10)
    unstable_plus, unstable_minus = unstable_manifolds
    stable_plus, stable_minus = stable_manifolds

    # xlim=(l1-0.05, l1+0.05)
    # ylim=(-0.05, 0.05)
    # if stable_manifolds:
    #     print('yes')
    #     xmin = 1.1 * min(min(np.min(traj.T[0]) for traj in stable) for stable in stable_manifolds)
    #     xmax = 1.1 * max(max(np.max(traj.T[0]) for traj in stable) for stable in stable_manifolds)
    #
    #     xmin = min(xmin, -1.5)
    #     xmax = max(xmax, 1.5)
    # else:
    #     xmin = 1.1 * min(min(np.min(traj.T[0]) for traj in stable) for stable in unstable_manifolds)
    #     xmax = 1.1 * max(max(np.max(traj.T[0]) for traj in stable) for stable in unstable_manifolds)
    #
    #     xmin = min(xmin, -1.5)
    #     xmax = max(xmax, 1.5)
    xmin = -1.5
    xmax = 1.2
    # ----------------------------
    # Grid
    # ----------------------------
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(*ylim, res)
    X, Y = np.meshgrid(x, y)

    # ----------------------------
    # Potential (Omega)
    # ----------------------------
    U = phy.pot_eff(X, Y)  # should be Omega, not your negative version

    # ----------------------------
    # Jacobi constant from orbit
    # ----------------------------
    x0, y0, vx0, vy0 = trajectory[0][0][:4]

    E = max(np.max(phy.pot_eff(state[0][0], state[0][1]) + 0.5 * (state[0][2] ** 2 + state[0][3] ** 2)) for state in
            trajectory)
    # ----------------------------
    # Plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    Z = phy.pot_eff(X, Y) - E
    allowed = Z >= 0
    ax.contourf(
        X, Y,
        allowed.astype(float),
        levels=[0, 0.5, 1],
        colors=["white", "lightgray"],
        alpha=1.0
    )

    ax.contour(
        X, Y,
        Z,
        levels=[0],
        colors='black',
        linewidths=1.5,
        alpha=0.7,
        zorder=10
    )

    # ----------------------------
    # Overlay boundary (zero velocity curve)
    # ----------------------------
    if unstable_plus:
        for w_u in unstable_plus:
            x, y = w_u.T[0:2]
            ax.plot(x, y, lw=1.5, color='red')
    if unstable_minus:
        for w_u in unstable_minus:
            x, y = w_u.T[0:2]
            ax.plot(x, y, lw=1.5, color='orange')
    if stable_plus:
        for w_s in stable_plus:
            x, y = w_s.T[0:2]
            ax.plot(x, y, lw=1.5, color='green')
    if stable_minus:
        for w_s in stable_minus:
            x, y = w_s.T[0:2]
            ax.plot(x, y, lw=1.5, color='yellow')

    for orbit in trajectory:
        x, y = orbit.T[0:2]
        ax.plot(x, y, linestyle='--', linewidth=2.5, color='black')

    theta = np.linspace(0, 2 * np.pi, 300)
    if hill_sphere:
        r_p, r_s = phy.compute_hill_radius()

        # Hill sphere around primary 1
        x_circle1 = -mu + r_p * np.cos(theta)
        y_circle1 = 0 + r_p * np.sin(theta)
        ax.plot(x_circle1, y_circle1, 'r--', label='Hill sphere (secondary)')

        # Hill sphere around primary 2
        x_circle2 = (1 - mu) + r_s * np.cos(theta)
        y_circle2 = 0 + r_s * np.sin(theta)
        ax.plot(x_circle2, y_circle2, 'r--', label='Hill sphere (secondary)')

    # ax.annotate(
    #     '',
    #     xy=(0.8, 0.2),  # arrow head
    #     xytext=(0.7, 0.2),  # arrow tail
    #     arrowprops=dict(
    #         arrowstyle='-|>',
    #         color='blue',
    #         lw=2,
    #         mutation_scale=18
    #     )
    # )

    # ----------------------------
    # Primaries
    # ----------------------------
    # ax.scatter(-mu, 0, c='blue', s=80, label='Primary')
    # ax.scatter(1-mu, 0, c='gray', s=40, label='Secondary')
    l1_scatter = ax.scatter(l1, 0, facecolors='cyan', edgecolors='black', s=50, zorder=10, label='L1')
    l2_scatter = ax.scatter(l2, 0, facecolors='green', edgecolors='black', s=50, zorder=10, label='L2')
    l3_scatter = ax.scatter(l3, 0, facecolors='red', edgecolors='black', s=50, zorder=10, label='L3')

    # primary_scatter = ax.scatter(-mu, 0, facecolors='lightgray', edgecolors='black', s=50, zorder=10, label='Primary')
    # secondary_scatter = ax.scatter(1-mu, 0, facecolors='gray', edgecolors='black', s=50, zorder=10, label='Secondary')

    r_m = 0.00452
    moon_center = (1 - mu, 0)

    moon = Circle(
        moon_center,
        r_m,
        facecolor='lightgray',
        edgecolor='black',
        lw=1,
        zorder=10
    )

    ax.add_patch(moon)
    custom_circle = Line2D(
        [0], [0],
        marker='o',
        color='black',
        markerfacecolor='lightgray',
        markersize=10,
        linestyle='None'
    )
    x_lim = phy.nvz_limit_x(trajectory[0][0][:4], 1e-5)
    y_lim = phy.nvz_limit_y(trajectory[0][0][:4], 1e-5)

    if x_lim:
        for elem in x_lim:
            ax.scatter(elem, 0, facecolors='purple', edgecolors='black', s=50, zorder=10, label='L1')
    if y_lim:
        for elem in y_lim:
            ax.scatter(1 - mu, elem, facecolors='purple', edgecolors='black', s=50, zorder=10, label='L1')

    # Poincare Maps
    # U1
    ax.plot([-mu, x_lim[1]], [0, 0], color='black', lw=2)
    plt.text((-mu + x_lim[1]) / 2, 0.02, f"U1")

    # U2
    ax.plot([1 - mu, 1 - mu], [0, y_lim[1]], color='black', lw=2)
    plt.text(1 - mu + 0.01, y_lim[1] / 2, f"U2")

    # U3
    ax.plot([1 - mu, 1 - mu], [0, y_lim[0]], color='black', lw=2)
    plt.text(1 - mu + 0.01, y_lim[0] / 2, f"U3")

    # U4
    ax.plot([xmin, x_lim[0]], [0, 0], color='black', lw=2)
    plt.text((xmin + x_lim[0]) / 2, 0.02, f"U4")

    custom_line1 = Line2D([0], [0], color='red', lw=2)
    custom_line2 = Line2D([0], [0], color='orange', lw=2)
    custom_line3 = Line2D([0], [0], color='green', lw=2)
    custom_line4 = Line2D([0], [0], color='yellow', lw=2)
    custom_line5 = Line2D([0], [0], color='lightgray', lw=3)
    custom_line6 = Line2D([0], [0], color='black', lw=3)
    custom_line7 = Line2D([0], [0], color='black', linestyle=':', lw=3)

    plt.legend([custom_line7, custom_line1, custom_line2, custom_line3, custom_line4, custom_line5, custom_line6,
                l1_scatter, l2_scatter, l3_scatter, custom_circle],
               ['Lyapunov orbit', 'Unstable +', 'Unstable -', 'Stable +', 'Stable -', 'Forbidden Realm', 'zvc', 'L1',
                'L2', 'L3',
                'secondary'])

    ax.set_aspect('equal')
    ax.set_xlabel("x[nondimensionalized]")
    ax.set_ylabel("y[nondimensionalized]")
    ax.set_title("Invariant manifold of Lyapunov orbits")

    plt.show()

    x_set = []
    y_set = []
    vx_set = []
    vy_set = []
    for w_s in stable_plus:
        x, y, vx, vy = w_s[-1].T[:4]
        if x == 1 - cfg.mu:
            x_set.append(x)
            y_set.append(y)
            vx_set.append(vx)
            vy_set.append(vy)
    plt.scatter(y_set, vy_set, color='red', label=r'$\Gamma_1^{s,\chi}$')
    x_set = []
    y_set = []
    vx_set = []
    vy_set = []
    for w_u in unstable_minus:
        x, y, vx, vy = w_u[-1].T[:4]
        if x == 1 - cfg.mu:
            x_set.append(x)
            y_set.append(y)
            vx_set.append(vx)
            vy_set.append(vy)

    plt.scatter(y_set, vy_set, color='green', label=r'$\Gamma_1^{u,\chi}$')
    plt.axis('equal')
    plt.legend(fontsize=14)
    plt.show()
