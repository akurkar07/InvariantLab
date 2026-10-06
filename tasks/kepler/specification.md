# Planar Kepler task

## Input parameters

`rx`, `ry`, `vx`, and `vy` define the initial planar position and velocity. `mu` is the positive gravitational parameter, `dt` the positive timestep, and `n_steps` the positive number of updates.

## Numerical method

Integrate $r''=-\mu r/|r|^3$ using planar velocity Verlet, including the initial state and one row per completed timestep.

## Output archive

`result.npz` contains float64 arrays `time` and `state`. `time` has shape `(n_steps + 1,)`; `state` has shape `(n_steps + 1, 4)` and each row is `[rx, ry, vx, vy]`.
