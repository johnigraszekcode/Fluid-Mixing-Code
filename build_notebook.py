"""Regenerate mixing_notebook.ipynb.  Run this after changing the text below."""

import json

CELLS = [
    (
        "markdown",
        """# Incompressible Fluid Mixing as Constrained Optimal Transport

Numerical companion to the paper.  Everything here calls into `mixing.py`,
which must sit next to this notebook.

Two systems are solved, both on the flat torus $\\Omega = [0,2\\pi)^2$:

**Geodesic equations (Section 6.1)** --- the enstrophy-penalized case, where
$\\rho$ and the costate $\\lambda$ are transported together,

$$\\partial_t \\rho = -\\nabla\\rho \\cdot v, \\qquad
  \\partial_t \\lambda = -\\nabla\\lambda \\cdot v, \\qquad
  v = \\Pi\\big[\\Delta^{-1}(\\rho\\,\\nabla\\lambda)\\big].$$

**Gradient flow (Section 7.1)** --- steepest descent of the homogeneous
$\\dot H^{-1}$ energy $E(\\rho) = \\tfrac12\\|\\rho - \\rho^*\\|^2_{\\dot H^{-1}}$,

$$\\partial_t \\rho = -\\nabla\\cdot(\\rho v), \\qquad
  v = \\Pi\\big[\\Delta^{-1}(\\rho\\,\\nabla\\,\\delta E(\\rho))\\big], \\qquad
  \\delta E(\\rho) = -\\Delta^{-1}\\tilde\\rho .$$

The two share a velocity pipeline: the only difference is whether the scalar
potential is the transported costate or the first variation of the energy.
$\\Pi$ is the Leray projection and $\\Delta^{-1}$ the inverse Laplacian, both
applied spectrally.""",
    ),
    (
        "code",
        """import matplotlib.pyplot as plt
import numpy as np

import mixing as mx

%matplotlib inline""",
    ),
    ("markdown", "## Geodesic equations (Section 6.1)"),
    (
        "code",
        """geodesic = mx.run_geodesic_equations(
    mx.Settings(
        N=128,
        final_time=mx.DEFAULT_FINAL_TIME["geodesic"],
        cfl=0.45,
        rho_expr="sin(x)*cos(y)",
        lambda_expr="cos(x)*sin(y)",
    )
)

print(mx.diagnostics_report(geodesic))""",
    ),
    (
        "markdown",
        """Every stage of the velocity pipeline at the final time.  Under
incompressibility the level sets of $\\rho$ are only rearranged, never created
or destroyed, so the range of $\\rho$ and all of its $L^p$ norms are conserved
by the exact dynamics --- the drift reported above is the numerical
dissipation of the first-order upwind scheme.""",
    ),
    ("code", "mx.figure_overview(geodesic)"),
    ("markdown", "## Gradient flow (Section 7.1)"),
    (
        "code",
        """gradient_flow = mx.run_gradient_flow(
    mx.Settings(
        N=128,
        final_time=mx.DEFAULT_FINAL_TIME["gradient-flow"],
        cfl=0.45,
    )
)

print(mx.diagnostics_report(gradient_flow))""",
    ),
    ("code", "mx.figure_overview(gradient_flow)"),
    (
        "markdown",
        """The energy dissipation equality says $E$ can only decrease along the flow.
That is the sharpest check that the code is really implementing a gradient
flow of $E$, so the energy is recorded at every step rather than only at the
display snapshots.""",
    ),
    ("code", "mx.figure_energy_history(gradient_flow)"),
    ("code", "mx.figure_diagnostics(gradient_flow)"),
    (
        "markdown",
        """## Looking at a single snapshot

`data["snapshots"]` is a list of dictionaries, one per recorded frame, holding
the state, every stage of the velocity pipeline, and the diagnostics for that
instant.  The keys are listed in `mx.quantities_for(mode)`.""",
    ),
    (
        "code",
        """table = mx.quantities_for(gradient_flow["mode"])
snapshot = gradient_flow["snapshots"][len(gradient_flow["snapshots"]) // 2]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
mx.FieldPanel(fig, axes[0], gradient_flow["grid"], snapshot, table["rho"], colorbar=True)
mx.FieldPanel(fig, axes[1], gradient_flow["grid"], snapshot, table["velocity v"], colorbar=True)
fig.suptitle(f"t = {snapshot['time']:.2f}")
fig.tight_layout()""",
    ),
    (
        "markdown",
        """## Writing the figures out

`save_run_figures` writes the standard set of PNGs used in the paper.  The same
thing is available from the command line:

```
python mixing.py geodesic      --grid 128
python mixing.py gradient-flow --grid 128
```

and `python mixing_gui.py` opens an interactive viewer where you can scrub
through the snapshots.""",
    ),
    (
        "code",
        """for path in mx.save_run_figures(gradient_flow, "results"):
    print(path)""",
    ),
]


def build():
    cells = []

    for kind, source in CELLS:
        lines = source.split("\n")
        source_lines = [line + "\n" for line in lines[:-1]] + [lines[-1]]

        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source_lines})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source_lines,
                }
            )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open("mixing_notebook.ipynb", "w") as handle:
        json.dump(notebook, handle, indent=1)
        handle.write("\n")

    print(f"wrote mixing_notebook.ipynb ({len(cells)} cells)")


if __name__ == "__main__":
    build()
