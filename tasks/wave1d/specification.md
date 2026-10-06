# One-dimensional wave task

## Input parameters

`nx` is the number of spatial points, `nt` the number of timesteps, `c` the positive wave speed, `length` the positive domain length, and `t_final` the positive final time. Optional `courant`, when supplied, must equal the Courant value derived from these physical parameters.

## Numerical method

Integrate $u_{tt}=c^2u_{xx}$ with the corrected zero-initial-velocity leapfrog method. The fundamental sine displacement is initialized with the half-acceleration first step, and endpoints are pinned to zero Dirichlet values. The physical CFL value is derived as $C=c(t_{final}/nt)/(length/(nx-1))$ and must satisfy $|C|\leq1$; `courant` is only a consistency check.

## Output archive

`result.npz` contains float64 vectors `x` and `state`, each of shape `(nx,)`. `x` is the grid over `[0, length]`; `state` is the final displacement field at `t_final`.

## Scientific thresholds

The contract tolerance `state_relative_l2` bounds the final displacement error. The hidden tests also bound the fundamental-mode amplitude error by `5.0e-6` absolute, because the reference leapfrog solver reaches about `1e-6` on the non-special cases and a phase or amplitude defect would exceed it. The single-step startup case on a 5-point grid is bounded by a state relative L2 error of `5.0e-3`, because its error is dominated by spatial truncation (about `3e-3`) rather than by the startup formula.
