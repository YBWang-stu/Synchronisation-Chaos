"""Reproduce Figure 9.13(b): strong coupling, a=0.1, c=0.07.

The two trajectories of system (9.27) start from x0=5 and y0=-4.5.
Because a - 2c < 0, their separation decays exponentially and the
trajectories approach synchronisation.
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled_scalar_solver import solve_coupled_scalar


A = 0.1
C = 0.07  # strong coupling
X0, Y0 = 5.0, -4.5
T_SPAN = (0.0, 45.0)


t, x, y = solve_coupled_scalar(
    a=A,
    c=C,
    initial_state=(X0, Y0),
    t_span=T_SPAN,
    num_points=2001,
)

fig, ax = plt.subplots(figsize=(5, 4))
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
ax.plot(t, x, color="black", linewidth=1.6)
ax.plot(t, y, color="black", linewidth=1.6)
ax.annotate(
    "x(t)",
    xy=(t[-1], x[-1]),
    xytext=(4, 4),
    textcoords="offset points",
    fontsize=10,
)
ax.annotate(
    "y(t)",
    xy=(t[-1], y[-1]),
    xytext=(4, -10),
    textcoords="offset points",
    fontsize=10,
)
ax.set_xlabel("t")
ax.set_yticks([])
ax.set_title(f"(b) strong coupling: a={A}, c={C}")
ax.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig_9_13b.png", dpi=150)

initial_separation = abs(X0 - Y0)
final_separation = abs(x[-1] - y[-1])
predicted_factor = math.exp((A - 2 * C) * T_SPAN[1])

print(
    f"|x(T)-y(T)| = {final_separation:.3f} "
    f"(started at {initial_separation:.1f})"
)
print(
    "predicted decay factor of |x-y|: "
    f"exp((a-2c)*T) = {predicted_factor:.3f}"
)
