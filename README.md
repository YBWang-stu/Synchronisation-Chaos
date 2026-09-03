### Figure 9.13(a) — weak coupling

## python make_fig_9_13a.py
`make_fig_9_13a.py` reproduces Figure 9.13(a): `a=0.1`, `c=0.03`,
`x0=5`, `y0=-4.5`, `t in [0, 45]`. 

Saves `fig_9_13a.png`. Since `a-2c = 0.04 > 0`, `|x(t)-y(t)|` grows
as `exp((a-2c)t)` — the trajectories diverge instead of syncing.

# Verification

`test_solver.py` checks the numerical result against the closed-form
solution (`x±y = (x0±y0) e^{...}`, from Step 2):

## python test_solver.py
Expected: `OK: numerical solution matches closed-form eigenmode solution.`
