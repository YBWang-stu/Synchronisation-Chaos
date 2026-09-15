# Coupled Scalar System Solver

Python/SciPy solver for system (9.27):

```text
dx/dt = a*x + c*(y - x)
dy/dt = a*y + c*(x - y)
```

## Requirements

- Python 3.10+
- NumPy and SciPy

```bash
python -m pip install numpy scipy
```

## Usage

```python
from coupled_scalar_solver import solve_coupled_scalar

t, x, y = solve_coupled_scalar(
    a=0.1,
    c=0.03,
    initial_state=(5.0, -4.5),
    t_span=(0.0, 50.0),
    num_points=1001,
)
```

The function returns NumPy arrays `t`, `x`, and `y`. Use `c=0.03` for weak
coupling or `c=0.07` for strong coupling.

## Method

`solve_ivp` with DOP853, `rtol=1e-10`, and `atol=1e-12`.
