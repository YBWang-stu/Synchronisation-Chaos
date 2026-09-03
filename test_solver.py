"""Sanity check: compare the numerical solution to the closed-form
solution x(t)+y(t) = (x0+y0)e^{at}, x(t)-y(t) = (x0-y0)e^{(a-2c)t}
derived in Step 2 (eigen-decomposition of A in eq. 9.28)."""
import numpy as np
from coupled_scalar_solver import solve_coupled_scalar

a, c = 0.1, 0.03
x0, y0 = 5.0, -4.5
t, x, y = solve_coupled_scalar(
    a=a, c=c, initial_state=(x0, y0), t_span=(0.0, 45.0), num_points=500
)

sum_exact = (x0 + y0) * np.exp(a * t)
diff_exact = (x0 - y0) * np.exp((a - 2 * c) * t)

assert np.allclose(x + y, sum_exact, rtol=1e-6)
assert np.allclose(x - y, diff_exact, rtol=1e-6)
print("OK: numerical solution matches closed-form eigenmode solution.")
