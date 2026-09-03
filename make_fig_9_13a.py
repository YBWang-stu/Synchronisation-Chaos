"""Reproduce Figure 9.13(a): weak coupling, a=0.1, c=0.03.

Two solutions of the coupled scalar system (9.27), started from
x0=5, y0=-4.5. With weak coupling the (x-y) mode decays slower than
it is "pulled together", so the two trajectories drift apart instead
of synchronizing.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coupled_scalar_solver import solve_coupled_scalar

A = 0.1
C = 0.03  # weak coupling
X0, Y0 = 5.0, -4.5
T_SPAN = (0.0, 45.0)

t, x, y = solve_coupled_scalar(
    a=A, c=C, initial_state=(X0, Y0), t_span=T_SPAN, num_points=2001
)

fig, ax = plt.subplots(figsize=(5, 4))
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
ax.plot(t, x, color="black", linewidth=1.6)
ax.plot(t, y, color="black", linewidth=1.6)
ax.annotate("x(t)", xy=(t[-1], x[-1]), xytext=(4, 4),
            textcoords="offset points", fontsize=10)
ax.annotate("y(t)", xy=(t[-1], y[-1]), xytext=(4, -10),
            textcoords="offset points", fontsize=10)
ax.set_xlabel("t")
ax.set_yticks([])
ax.set_title(f"(a) weak coupling: a={A}, c={C}")
ax.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig_9_13a.png", dpi=150)

print(f"|x(T)-y(T)| = {abs(x[-1]-y[-1]):.3f} (started at {abs(X0-Y0)})")
print(f"predicted growth of |x-y|: exp((a-2c)*T) = "
      f"{__import__('math').exp((A-2*C)*T_SPAN[1]):.3f}")
