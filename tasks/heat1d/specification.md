# One-dimensional heat task

## Input parameters

`nx` is the number of spatial points, `nt` the number of timesteps, `alpha` the positive diffusivity, `length` the positive domain length, and `t_final` the positive final time.

## Numerical method

Integrate $u_t=\alpha u_{xx}$ with FTCS on the uniform physical grid. The initial field is the fundamental sine mode and both endpoints are pinned to zero Dirichlet boundary values. Inputs with FTCS ratio greater than `0.5` are rejected.

## Output archive

`result.npz` contains float64 vectors `x` and `state`, each of shape `(nx,)`. `x` is the grid over `[0, length]`; `state` is the final temperature field at `t_final`.
