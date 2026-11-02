import matplotlib.pyplot as plt
import src.configuration as cfg


def plot_family(lib_points, orbit_family):
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'font.family': 'serif',
        'mathtext.fontset': 'cm'
    })

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    colors = ['tab:blue', 'tab:red', 'tab:green']

    for i, lya_group in enumerate(orbit_family):

        color = colors[i % len(colors)]

        for j, traj in enumerate(lya_group):
            label = f'L{i+1} Lyapunov Family' if j == 0 else None

            x, y = traj[:, 0], traj[:, 1]
            ax.plot(
                x,
                y,
                color=color,
                lw=1.2,
                alpha=0.85,
                label=label
            )





    # Libration points
    lp_colors = ['gold', 'lightgray', 'violet']

    for i, l_i in enumerate(lib_points):
        ax.text(l_i, 0.01, rf'$L_{i + 1}$')
        # ax.scatter(
        #     l_i,
        #     0,
        #     s=80,
        #     color=lp_colors[i % len(lp_colors)],
        #     edgecolor='black',
        #     zorder=10,
        #     label=f'L{i+1}'
        # )

    # Moon
    ax.scatter(
        1 - cfg.mu,
        0,
        s=120,
        color='black',
        zorder=12,
        label='Secondary'
    )

    ax.set_aspect('equal')

    ax.set_xlabel(r'$x$ (non-dimensional)')
    ax.set_ylabel(r'$y$ (non-dimensional)')

    ax.legend(frameon=True)
    ax.grid(alpha=0.3)

    # plt.tight_layout()
    plt.show()


def plot_corrections(trajectories):
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'font.family': 'serif',
        'mathtext.fontset': 'cm'
    })

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, traj in enumerate(trajectories):
        x, y = traj.T[:2]
        ax.plot(x, y, lw=1.2, alpha=0.85)
        ax.text(x[-1], -0.0001, rf'${i + 1}$')
    ax.set_aspect('equal')

    ax.set_xlabel(r'$x$ (non-dimensional)')
    ax.set_ylabel(r'$y$ (non-dimensional)')

    # ax.legend(frameon=True)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    pass
