# Figure 9.13(b) - strong coupling

`make_fig_9_13b.py` qualitatively reproduces Figure 9.13(b) for the
two-way coupled scalar system (9.27). It uses
`a=0.1`, `c=0.07`, `x0=5`, `y0=-4.5`, and `t` in `[0, 45]`.

## Requirements

- Python 3.10+
- NumPy, SciPy, and Matplotlib
- `coupled_scalar_solver.py` in the same directory

Install the dependencies with:

```bash
python -m pip install numpy scipy matplotlib
```

## Run

```bash
python make_fig_9_13b.py
```

The script saves `fig_9_13b.png`. Since
`a - 2c = -0.04 < 0`, the separation satisfies
`|x(t)-y(t)| = |x0-y0| exp((a-2c)t)` and therefore decays with time.
Over the plotted interval, the numerical separation decreases from `9.5`
to approximately `1.570`, so the trajectories approach synchronisation.
