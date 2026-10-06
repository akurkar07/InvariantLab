# One-dimensional wave task

## Input parameters

`nx` is the number of spatial points, `nt` the number of timesteps, `c` the positive wave speed, `length` the positive domain length, and `t_final` the positive final time. Optional `courant`, when supplied, must equal the Courant value derived from these physical parameters.

## Numerical method

Integrate $u_{tt}=c^2u_{xx}$ with the corrected zero-initial-velocity leapfrog method. The fundamental sine displacement is initialized with the half-acceleration first step, and endpoints are pinned to zero Dirichlet values. The physical CFL value is derived as $C=c(t_{final}/nt)/(length/(nx-1))$ and must satisfy $|C|\leq1$; `courant` is only a consistency check.

## Output archive

`result.npz` contains float64 vectors `x` and `state`, each of shape `(nx,)`. `x` is the grid over `[0, length]`; `state` is the final displacement field at `t_final`.
