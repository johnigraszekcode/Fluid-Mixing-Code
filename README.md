# Incompressible Fluid Mixing as Constrained Optimal Transport

Code for the numerical results in Sections 6.1 and 7.1 of the paper.

Two systems are solved, both on the flat torus `Omega = [0, 2*pi)^2` with
periodic boundary conditions.

**Geodesic equations (Section 6.1)** — the enstrophy-penalized case, in which
the density and the costate are transported together:

```
d_t rho    = -grad(rho) . v
d_t lambda = -grad(lambda) . v
v          = Pi[ Lap^-1 ( rho grad(lambda) ) ]
```

**Gradient flow (Section 7.1)** — steepest descent of the homogeneous `H^-1`
energy `E(rho) = 1/2 ||rho - rho*||^2`:

```
d_t rho = -div(rho v)
v       = Pi[ Lap^-1 ( rho grad(dE(rho)) ) ]
dE(rho) = -Lap^-1 (rho - rho*),     rho* = mean(rho)
```

`Pi` is the Leray projection and `Lap^-1` the inverse Laplacian, both applied
spectrally. The two runs share the same velocity pipeline; the only difference
is where the scalar potential comes from — the transported costate `lambda`, or
the first variation of the energy recomputed from `rho` at every step.

## Requirements

Python 3.10 or newer, with `numpy` and `matplotlib`. The viewer also needs
`tkinter`.

## Running

From the command line, which writes PNGs to `results/` and prints the
diagnostics:

```
python mixing.py geodesic      --grid 128
python mixing.py gradient-flow --grid 128
```

Useful flags: `--time` (final time), `--cfl`, `--snapshots`, `--rho` and `--lam`
for the initial conditions, `--out` for the output directory. `--rho "sin(x+y)"`
and similar formulas in `x` and `y` are accepted.

From Python:

```python
from mixing import Settings, run_gradient_flow, diagnostics_report, save_run_figures

data = run_gradient_flow(Settings(N=128, final_time=75.0))
print(diagnostics_report(data))
save_run_figures(data, "results")
```

For an interactive viewer with a slider over the recorded snapshots:

```
python mixing_gui.py
```

`mixing_notebook.ipynb` walks through both runs and reproduces the figures
inline; it imports `mixing.py`, so keep them in the same directory.

## Files

| file | contents |
| --- | --- |
| `mixing.py` | grid, spectral operators, upwind schemes, both solvers, diagnostics, figures, command line |
| `mixing_gui.py` | Tk viewer; all numerics come from `mixing.py` |
| `mixing_notebook.ipynb` | worked example of both runs |
| `test_mixing.py` | checks on the operators, conservation and energy dissipation |
| `build_notebook.py` | regenerates the notebook |

## Numerical scheme

Both systems use the conservative flux form of first-order upwind differences,
selected pointwise from the sign of the local velocity, so mass is preserved to
round-off for every transported field (`rho` and `lambda` in the geodesic
system, `rho` in the gradient flow); face velocities are averaged from
neighbouring cells and the upwind side decides which cell value is carried.
Under incompressibility the advective and conservative forms of the transport
equation agree analytically, so this is a discretization choice, not a change
to the continuous dynamics.

The inverse Laplacian and the Leray projection are applied in Fourier space.
Each component of `rho grad(potential)` is zero-mean centred before the solve,
so the constant mode never appears in a denominator. In the time loop the two
multipliers are composed in spectral space and applied in a single transform
pair; `test_mixing.py` checks this against applying them separately.

Time steps are chosen adaptively from the CFL condition,
`dt = min(dt_max, CFL / max(|u|/dx + |v|/dy))`, with `dt_max` set to the
snapshot interval so that recorded frames stay evenly spaced and the first step
is not enormous when the run starts from a nearly stationary state.

## Diagnostics

Every run reports:

- **mass and `L^2` norm of `rho`.** Both are conserved by the exact dynamics —
  under incompressibility the level sets of `rho` are only rearranged, so the
  signature `rho # L` and hence every `L^p` norm is invariant. Mass is
  conserved by both solvers to round-off, by construction of the conservative
  upwind scheme; `L^2` still drifts under discretization, and that drift
  measures the numerical dissipation of the first-order scheme.
- **the `H^-1` energy**, recorded at every step rather than only at the
  snapshots. For the gradient flow the energy dissipation equality says it can
  never increase, which is the sharpest available check that the code really is
  integrating a gradient flow of `E`.
- **`div(v)`**, measured two independent ways. The spectral check should sit at
  round-off; the finite-difference check is larger because centred differences
  are only second-order accurate on a spectrally defined field, and the gap
  between the two is itself informative.

