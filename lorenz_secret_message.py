from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Triple the default font size (10 -> 30); labels, titles, legends and ticks all scale with it
plt.rcParams["font.size"] = 30

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Send no message until the receiver has had time to synchronise.
t_end = 60.0
h = 0.01
message_start = 40.0
X1_0 = [1.0, 1.0, 1.0]
X2_0 = [2.0, 3.0, 4.0]

# These are the two main settings to change for your own message.
amplitude = 0.1
frequency = 5.0  # Cycles per Lorenz time unit, not physical Hz.

# Save the results beside this Python file.
output_folder = Path(__file__).resolve().parent / "output" / "lorenz_message"
output_folder.mkdir(parents=True, exist_ok=True)


def lorenz(X):
    x, y, z = X
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


def message(time, A, f):
    if time < message_start:
        return 0.0
    return A * np.sin(2.0 * np.pi * f * (time - message_start))


def one_way_lorenz(time, U, A, f):
    X1 = U[:3]
    x1 = X1[0]
    x2, y2, z2 = U[3:]

    # The receiver only receives s, not the clean x1 or the message separately.
    s = x1 + message(time, A, f)

    # Extend the original x2_damping receiver using the transmitted signal s.
    dx2 = sigma * (y2 - x2)
    dy2 = s * (rho - z2) - y2
    dz2 = s * y2 - beta * z2
    dX2 = np.array([dx2, dy2, dz2])
    return np.concatenate([lorenz(X1), dX2])


def rk4_step(time, U, step, A, f):
    # Evaluate the time-dependent message at all four RK4 stage times.
    k1 = one_way_lorenz(time, U, A, f)
    k2 = one_way_lorenz(time + step / 2, U + step / 2 * k1, A, f)
    k3 = one_way_lorenz(time + step / 2, U + step / 2 * k2, A, f)
    k4 = one_way_lorenz(time + step, U + step * k3, A, f)
    return U + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(A, f, step=h):
    n_steps = int(round(t_end / step))
    t = np.arange(n_steps + 1) * step
    U = np.array(X1_0 + X2_0, dtype=float)
    history = np.empty((n_steps + 1, 6))
    history[0] = U
    for n in range(n_steps):
        U = rk4_step(t[n], U, step, A, f)
        history[n + 1] = U
    return t, history


# First check synchronisation with no message.
t, U = simulate(0.0, frequency)
late_window = t >= 40.0
error = np.linalg.norm(U[:, :3] - U[:, 3:], axis=1)
max_error = np.max(error[late_window])
print(f"No-message maximum full-state error: {max_error:.6e}")
print(f"No-message synchronisation: {max_error < 1e-5}\n")

# Also test a larger amplitude and a slower message.
cases = [
    ("small_message", amplitude, frequency),
    ("larger_message", 2.0, frequency),
    ("slow_message", amplitude, 0.1),
]
results = []

for name, A, f in cases:
    t, U = simulate(A, f)
    original = np.array([message(time, A, f) for time in t])
    transmitted = U[:, 0] + original
    recovered = transmitted - U[:, 3]

    # Use the final ten time units, after the message-onset transient.
    late_window = t >= 50.0
    rmse = np.sqrt(np.mean((recovered[late_window] - original[late_window]) ** 2))
    message_rms = np.sqrt(np.mean(original[late_window] ** 2))
    relative_error = rmse / message_rms
    correlation = np.corrcoef(original[late_window], recovered[late_window])[0, 1]

    # Repeat with half the step size to check the overall recovery quality.
    t_fine, U_fine = simulate(A, f, h / 2)
    original_fine = np.array([message(time, A, f) for time in t_fine])
    recovered_fine = U_fine[:, 0] + original_fine - U_fine[:, 3]
    fine_window = t_fine >= 50.0
    fine_rmse = np.sqrt(np.mean((recovered_fine[fine_window] - original_fine[fine_window]) ** 2))
    fine_rms = np.sqrt(np.mean(original_fine[fine_window] ** 2))
    fine_relative_error = fine_rmse / fine_rms

    print(f"{name}: A = {A}, f = {f}")
    print(f"RMSE: {rmse:.6e}")
    print(f"RMSE / message RMS: {relative_error:.2%}")
    print(f"Correlation: {correlation:.4f}")
    print(f"Half-step RMSE / message RMS: {fine_relative_error:.2%}\n")
    results.append([A, f, rmse, relative_error, correlation, fine_relative_error])

    # Show ten fast cycles, or one full slow cycle.
    plot_window = t >= (58.0 if f >= 1.0 else 50.0)
    fig, ax = plt.subplots(3, 1, figsize=(22, 16), sharex=True)
    ax[0].plot(t[plot_window], original[plot_window], color="black")
    ax[0].set_title("Original message")
    ax[0].set_ylabel("m(t)")

    ax[1].plot(t[plot_window], transmitted[plot_window], color="tab:blue")
    ax[1].set_title("Transmitted signal: s(t) = x1(t) + m(t)")
    ax[1].set_ylabel("s(t)")

    ax[2].plot(t[plot_window], original[plot_window], color="black", label="Original")
    ax[2].plot(t[plot_window], recovered[plot_window], color="tab:orange", ls="--", label="Recovered")
    ax[2].set_title("Message recovery: recovered(t) = s(t) - x2(t)", loc="left")
    ax[2].set_ylabel("Message")
    ax[2].set_xlabel("Time t")
    ax[2].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, handlelength=1.5, columnspacing=0.8)
    for axis in ax:
        axis.grid(alpha=0.25)
    fig.suptitle(f"{name.replace('_', ' ').title()}: A = {A}, f = {f}")
    fig.tight_layout()
    fig.savefig(output_folder / f"{name}.png", dpi=180)

np.savetxt(
    output_folder / "recovery_results.csv",
    results,
    delimiter=",",
    header="amplitude,frequency,rmse,rmse_over_message_rms,correlation,half_step_rmse_over_message_rms",
    comments="",
)
print(f"Results saved to: {output_folder}")
plt.show()
