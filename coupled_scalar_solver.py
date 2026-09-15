
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def coupled_scalar_rhs(
    _t: float, state: np.ndarray, a: float, c: float
) -> np.ndarray:
    """Evaluate the right-hand side of system (9.27)."""
    x, y = state
    return np.array(
        [a * x + c * (y - x), a * y + c * (x - y)], dtype=float
    )


def solve_coupled_scalar(
    *,
    a: float,
    c: float,
    initial_state: tuple[float, float],
    t_span: tuple[float, float],
    num_points: int = 1001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate system (9.27) and return t, x(t), and y(t)."""
    t_start, t_end = t_span
    if t_end <= t_start:
        raise ValueError("t_span must satisfy t_end > t_start")
    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    solution = solve_ivp(
        coupled_scalar_rhs,
        t_span=(t_start, t_end),
        y0=np.asarray(initial_state, dtype=float),
        t_eval=np.linspace(t_start, t_end, num_points),
        args=(a, c),
        method="DOP853",
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")

    return solution.t, solution.y[0], solution.y[1]
