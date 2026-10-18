"""Figure 2: combined Lorenz receivers, using Times New Roman."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator


# Model and simulation settings.
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
T_END = 60.0
H = 0.02
SENDER_INITIAL = [1.0, 1.0, 1.0]
RECEIVER_INITIAL = [2.0, 3.0, 4.0]
SYNC_TOLERANCE = 1e-5
PLOT_FLOOR = 1e-15
LATE_START = 40.0
PLOT_START = 50.0

# Files are saved beside this script, independently of the working directory.
OUTPUT_DIR = Path(__file__).resolve().parent
SHOW_PLOTS = True

PLOT_STYLE = {
    "font.size": 15,
    "axes.titlesize": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.titlesize": 19,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "savefig.facecolor": "white",
}



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


def lorenz(state):
    """Return the three derivatives of the independent sender."""
    x, y, z = state
    return np.array([
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z,
    ])


def one_way_lorenz(state, use_sender_x):
    """Return all six derivatives using the selected receiver equation."""
    sender = state[:3]
    x1 = sender[0]
    x2, y2, z2 = state[3:]

    # True: the printed equation; False: receiver-dependent x damping.
    dx2 = -SIGMA * x1 + SIGMA * y2 if use_sender_x else -SIGMA * x2 + SIGMA * y2
    dy2 = -x1 * z2 + RHO * x1 - y2
    dz2 = x1 * y2 - BETA * z2
    return np.concatenate([lorenz(sender), np.array([dx2, dy2, dz2])])


def rk4_step(state, step, use_sender_x):
    """Advance both systems together by one classical RK4 step."""
    k1 = one_way_lorenz(state, use_sender_x)
    k2 = one_way_lorenz(state + step / 2 * k1, use_sender_x)
    k3 = one_way_lorenz(state + step / 2 * k2, use_sender_x)
    k4 = one_way_lorenz(state + step * k3, use_sender_x)
    return state + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(use_sender_x):
    """Return the time array and a history with shape (steps + 1, 6)."""
    steps = int(T_END / H)
    times = np.linspace(0.0, T_END, steps + 1)
    state = np.array(SENDER_INITIAL + RECEIVER_INITIAL, dtype=float)
    history = np.empty((steps + 1, 6))
    history[0] = state

    for n in range(steps):
        state = rk4_step(state, H, use_sender_x)
        history[n + 1] = state
    return times, history


def measure_errors(history):
    """Measure receiver minus sender errors without changing stored states."""
    difference = history[:, 3:] - history[:, :3]
    full_error = np.linalg.norm(difference, axis=1)
    yz_error = np.linalg.norm(difference[:, 1:], axis=1)
    return difference, full_error, yz_error


def print_summary(name, times, difference, full_error, yz_error):
    """Print the same numerical checks as the original script."""
    late = times >= LATE_START
    maximum = np.max(full_error[late])
    print(name)
    print(f"Initial error: {full_error[0]:.6e}")
    print(f"Final x2-x1: {difference[-1, 0]:.6e}")
    print(f"Maximum y,z error for 40 <= t <= 60: {np.max(yz_error[late]):.6e}")
    print(f"Maximum full-state error for 40 <= t <= 60: {maximum:.6e}")
    print(f"Full-state synchronisation: {maximum < SYNC_TOLERANCE}\n")


def draw_block(subfig, times, history, label):
    difference, full_error, _ = measure_errors(history)
    window = (times >= PLOT_START) & (times <= T_END)

    axes = subfig.subplots(
        2, 2, gridspec_kw={"width_ratios": [1, 2.1]}
    )
    error_ax, x_ax = axes[0]
    state_ax, y_ax = axes[1]
    error_ax.set_box_aspect(1)
    state_ax.set_box_aspect(1)

    for coordinate, name in enumerate([r"$x$", r"$y$", r"$z$"]):
        error_ax.semilogy(
            times, np.maximum(np.abs(difference[:, coordinate]), PLOT_FLOOR),
            lw=1.3, label=name,
        )
    error_ax.semilogy(
        times, np.maximum(full_error, PLOT_FLOOR),
        color="black", ls="--", lw=1.4, label="Full state",
    )
    error_ax.axhline(
        SYNC_TOLERANCE, color="0.45", ls=":", lw=1.4, label="Criterion",
    )
    error_ax.set(
        title="(1) Synchronisation error",
        xlabel=r"Time $t$", ylabel="Absolute error",
        xlim=(0, T_END), ylim=(PLOT_FLOOR / 3, 1e2),
        xticks=[0, 20, 40, 60],
    )
    error_ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    error_ax.yaxis.set_minor_locator(NullLocator())
    error_ax.legend(
        loc="center right", ncol=2, columnspacing=0.7,
        handlelength=1.4, borderpad=0.35, framealpha=0.95,
    )
    error_ax.text(
        0.98, 0.98, r"Display floor: $10^{-15}$",
        transform=error_ax.transAxes, ha="right", va="top", fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1},
    )

    y_ax.sharex(x_ax)
    for axis, coordinate, name, number in [
        (x_ax, 0, "x", 2),
        (y_ax, 1, "y", 4),
    ]:
        axis.plot(
            times[window], history[window, coordinate],
            color="black", lw=1.4, label="Sender",
        )
        axis.plot(
            times[window], history[window, coordinate + 3],
            color="tab:orange", ls="--", lw=1.4, label="Receiver",
        )
        axis.set_title(f"({number}) ${name}$ time series", loc="left", pad=5)
        axis.set_ylabel(f"${name}$ coordinate")
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    x_ax.set_xlim(PLOT_START, T_END)
    x_ax.tick_params(labelbottom=False)
    x_ax.legend(loc="upper right", ncol=2, framealpha=0.95)
    y_ax.set_xlabel(r"Time $t$")

    state_ax.plot(
        [-25, 25], [-25, 25], color="0.55", ls="--", lw=1.2,
        label=r"$x_2=x_1$",
    )
    state_ax.plot(history[window, 0], history[window, 3], color="tab:blue", lw=1.3)
    state_ax.set(
        title="(3) State comparison",
        xlabel=r"Sender $x_1$", ylabel=r"Receiver $x_2$",
        xlim=(-25, 25), ylim=(-25, 25),
        xticks=[-20, 0, 20], yticks=[-20, 0, 20],
    )
    state_ax.legend(loc="lower right", framealpha=0.95)

    for axis in axes.flat:
        axis.grid(alpha=0.2, lw=0.6)
        axis.tick_params(direction="out", pad=3)
    # Place the part label level with the xlabel at the right edge of panel (4).
    y_ax.annotate(
        label, xy=(1.0, 0.5),
        xycoords=("axes fraction", y_ax.xaxis.label),
        ha="right", va="center", fontsize=17, fontweight="bold",
        annotation_clip=False,
    )


def main():
    configure_times_new_roman()
    with plt.rc_context(PLOT_STYLE):
        fig = plt.figure(figsize=(12, 14.6), layout="constrained")
        fig.get_layout_engine().set(
            w_pad=0.035, h_pad=0.035, wspace=0.035, hspace=0.025
        )
        blocks = fig.subfigures(2, 1, hspace=0.025)

        for subfig, use_sender_x, name, label in [
            (blocks[0], True, "book", "(a)"),
            (blocks[1], False, "x2_damping", "(b)"),
        ]:
            times, history = simulate(use_sender_x)
            difference, full_error, yz_error = measure_errors(history)
            print_summary(name, times, difference, full_error, yz_error)
            draw_block(subfig, times, history, label)

        path = OUTPUT_DIR / "lorenz_one_way_combined.pdf"
        fig.savefig(path, dpi=800, facecolor="white")
        print(f"Saved: {path}")
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
