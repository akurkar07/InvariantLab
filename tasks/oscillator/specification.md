# Harmonic oscillator task

## Input parameters

`x0` and `v0` are the initial displacement and velocity. `omega` is the positive angular frequency, `dt` the positive timestep, and `n_steps` the positive number of updates.

## Numerical method

Integrate $x''=-\omega^2x$ with velocity Verlet, including the initial state and one row per completed timestep.

## Output archive

`result.npz` contains float64 arrays `time` and `state`. `time` has shape `(n_steps + 1,)`; `state` has shape `(n_steps + 1, 2)`, with `state[:, 0] = x` and `state[:, 1] = v`.
