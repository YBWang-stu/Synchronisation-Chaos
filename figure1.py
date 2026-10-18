"""Figure 1: two-way Lorenz synchronisation, using Times New Roman."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0


def lorenz(X):
    x, y, z = X
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


def coupled_lorenz(U, c):
    X1 = U[:3]
    X2 = U[3:]
    dX1 = lorenz(X1) + c * (X2 - X1)
    dX2 = lorenz(X2) + c * (X1 - X2)
    return np.concatenate([dX1, dX2])


def rk4_step(U, h, c):
    k1 = coupled_lorenz(U, c)
    k2 = coupled_lorenz(U + h / 2 * k1, c)
    k3 = coupled_lorenz(U + h / 2 * k2, c)
    k4 = coupled_lorenz(U + h * k3, c)
    return U + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

from matplotlib import font_manager


def configure_times_new_roman():
    # Register installed Windows fonts; other systems use their font registry.
    for name in ("times.ttf", "timesi.ttf", "timesbd.ttf", "timesbi.ttf"):
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "text.usetex": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.sf": "Times New Roman",
        "mathtext.tt": "Times New Roman",
        "mathtext.cal": "Times New Roman:italic",
        "mathtext.fallback": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "savefig.facecolor": "white",
    })
    if "mathtext.bfit" in plt.rcParams:
        plt.rcParams["mathtext.bfit"] = "Times New Roman:italic:bold"

# Time settings and initial conditions
t_end = 1000.0
h = 0.02
n_steps = int(t_end / h)
t = np.linspace(0.0, t_end, n_steps + 1)
start_step = int(800.0 / h)

X1_0 = [1.0, 1.0, 1.0]
X2_0 = [1.01, 1.0, 1.0]
sync_tolerance = 1e-5


def simulate(c, save=False):
    # Keep the whole trajectory only when a figure needs it.
    U = np.array(X1_0 + X2_0, dtype=float)
    history = np.empty((n_steps + 1, 6)) if save else None
    if save:
        history[0] = U

    max_error = 0.0
    for n in range(n_steps):
        U = rk4_step(U, h, c)
        if save:
            history[n + 1] = U
        if n + 1 >= start_step:
            error = np.linalg.norm(U[:3] - U[3:])
            max_error = max(max_error, error)

    return max_error, history



SHOW_PLOTS = True

def plot_figure1():
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.04, hspace=0.04)
    window = (t >= 900.0) & (t <= 930.0)

    for column, c in enumerate((0.30, 0.50)):
        maximum, history = simulate(c, save=True)
        x1, x2 = history[window, 0], history[window, 3]
        time_ax, state_ax = axes[:, column]

        time_ax.plot(t[window], x1, color="black", lw=1.4, label=r"$x_1$")
        time_ax.plot(t[window], x2, color="tab:orange", ls="--", lw=1.4,
                     label=r"$x_2$")
        time_ax.set(title=rf"$c={c:.2f}$", xlabel=r"Time $t$",
                    ylabel=r"$x$ coordinate", xlim=(900, 930), ylim=(-25, 25),
                    xticks=[900, 910, 920, 930], yticks=[-20, 0, 20])
        time_ax.legend(loc="upper right", ncol=2, framealpha=0.95)

        state_ax.plot([-25, 25], [-25, 25], color="0.55", ls="--", lw=1.2)
        state_ax.plot(x1, x2, color="black", lw=1.0)
        state_ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$",
                     xlim=(-25, 25), ylim=(-25, 25),
                     xticks=[-20, 0, 20], yticks=[-20, 0, 20])
        state_ax.set_aspect("equal", adjustable="box")
        for axis in (time_ax, state_ax):
            axis.grid(alpha=0.2, lw=0.6)
            axis.tick_params(direction="out", pad=3)
        print(f"c={c:.2f}, maximum error over the final window={maximum:.6e}")
    return fig


def main():
    configure_times_new_roman()
    fig = plot_figure1()
    path = Path(__file__).resolve().parent / "figure_914_rk4.pdf"
    fig.savefig(path, format="pdf", dpi=800, facecolor="white")
    print(f"Saved: {path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
