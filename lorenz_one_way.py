import numpy as np
import matplotlib.pyplot as plt

# Double the default font size (10 -> 30); labels, titles, legends and ticks all scale with it
plt.rcParams["font.size"] = 30

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0


def lorenz(X):
    x, y, z = X
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


def one_way_lorenz(U, use_sender_x):
    X1 = U[:3]
    x1 = X1[0]
    x2, y2, z2 = U[3:]
    if use_sender_x:
        dx2 = -sigma * x1 + sigma * y2
    else:
        dx2 = -sigma * x2 + sigma * y2
    dy2 = -x1 * z2 + rho * x1 - y2
    dz2 = x1 * y2 - beta * z2
    dX2 = np.array([dx2, dy2, dz2])
    return np.concatenate([lorenz(X1), dX2])


def rk4_step(U, h, use_sender_x):
    k1 = one_way_lorenz(U, use_sender_x)
    k2 = one_way_lorenz(U + h / 2 * k1, use_sender_x)
    k3 = one_way_lorenz(U + h / 2 * k2, use_sender_x)
    k4 = one_way_lorenz(U + h * k3, use_sender_x)
    return U + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


t_end = 60.0
h = 0.02
n_steps = int(t_end / h)
t = np.linspace(0.0, t_end, n_steps + 1)
X1_0 = [1.0, 1.0, 1.0]
X2_0 = [2.0, 3.0, 4.0]
sync_tolerance = 1e-5
late_window = t >= 40.0
plot_window = (t >= 50.0) & (t <= 60.0)


def simulate(use_sender_x):
    U = np.array(X1_0 + X2_0, dtype=float)
    history = np.empty((n_steps + 1, 6))
    history[0] = U
    for n in range(n_steps):
        U = rk4_step(U, h, use_sender_x)
        history[n + 1] = U
    return history


for use_sender_x, name in [(True, "book"), (False, "x2_damping")]:
    U = simulate(use_sender_x)
    difference = U[:, 3:] - U[:, :3]
    error = np.linalg.norm(difference, axis=1)
    yz_error = np.linalg.norm(difference[:, 1:], axis=1)
    max_error = np.max(error[late_window])
    print(name)
    print(f"Initial error: {error[0]:.6e}")
    print(f"Final x2-x1: {difference[-1, 0]:.6e}")
    print(f"Maximum y,z error for 40 <= t <= 60: {np.max(yz_error[late_window]):.6e}")
    print(f"Maximum full-state error for 40 <= t <= 60: {max_error:.6e}")
    print(f"Full-state synchronisation: {max_error < sync_tolerance}\n")

    fig, ax = plt.subplots(2, 2, figsize=(22, 13))
    for coordinate, label in enumerate(["x", "y", "z"]):
        ax[0, 0].semilogy(t, np.maximum(np.abs(difference[:, coordinate]), 1e-15), label=label)
    ax[0, 0].semilogy(t, np.maximum(error, 1e-15), color="black", ls="--", lw=0.9, label="Full state")
    ax[0, 0].axhline(sync_tolerance, color="gray", ls="--", label="Criterion")
    ax[0, 0].set_xlabel("Time t")
    ax[0, 0].set_ylabel("Coordinate error")
    ax[0, 0].set_title("Plot floor: 1e-15", pad=150)
    ax[0, 0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, columnspacing=0.8, handlelength=1.2)

    for row, coordinate, label in [(0, 0, "x"), (1, 1, "y")]:
        ax[row, 1].plot(t[plot_window], U[plot_window, coordinate], color="black", lw=0.9, label="Sender")
        ax[row, 1].plot(t[plot_window], U[plot_window, coordinate + 3], color="tab:orange", ls="--", lw=0.9, label="Receiver")
        ax[row, 1].set_xlabel("Time t")
        ax[row, 1].set_ylabel(f"{label} coordinate")
        ax[row, 1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

    ax[1, 0].plot(U[plot_window, 0], U[plot_window, 3], color="black", lw=0.7)
    ax[1, 0].plot([-25, 25], [-25, 25], color="gray", ls="--", lw=0.8)
    ax[1, 0].set_xlim(-25, 25)
    ax[1, 0].set_ylim(-25, 25)
    ax[1, 0].set_aspect("equal", adjustable="box")
    ax[1, 0].set_xticks([-20, 0, 20])
    ax[1, 0].set_yticks([-20, 0, 20])
    ax[1, 0].set_xlabel("Sender x1")
    ax[1, 0].set_ylabel("Receiver x2")
    for axis in ax.flat:
        axis.grid(alpha=0.25)
    fig.suptitle("Equation 9.36 as printed" if use_sender_x else "Receiver with -sigma*x2 damping")
    fig.tight_layout()
    fig.savefig(f"lorenz_one_way_{name}.png", dpi=180)

plt.show()
