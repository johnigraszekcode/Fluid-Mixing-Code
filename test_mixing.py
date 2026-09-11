"""
Checks on the numerics in mixing.py.

Run with `python test_mixing.py` (no pytest needed).  These are the properties
the paper actually leans on: the spectral operators do what they claim, the
fused velocity path agrees with the literal one, mass and the L^2 norm are
conserved by the appropriate scheme, div(v) vanishes, and the gradient flow
dissipates the H^-1 energy.
"""

from __future__ import annotations

import numpy as np

import mixing as mx


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}{'  ' + detail if detail else ''}")
    if not condition:
        raise AssertionError(name)


def test_inverse_laplacian():
    """Lap^-1 of a known eigenfunction: Lap sin(2x)cos(y) = -5 sin(2x)cos(y)."""
    grid = mx.Grid(64)
    field = np.sin(2.0 * grid.X) * np.cos(grid.Y)

    got = mx.inverse_laplacian(-5.0 * field, grid)
    error = np.max(np.abs(got - field))

    check("inverse Laplacian on an eigenfunction", error < 1.0e-12, f"error {error:.2e}")


def test_leray_projection():
    """Pi should annihilate a pure gradient and leave a divergence-free field alone."""
    grid = mx.Grid(64)

    # Gradient of cos(x)sin(y): entirely curl free, so Pi kills it.
    gx, gy = -np.sin(grid.X) * np.sin(grid.Y), np.cos(grid.X) * np.cos(grid.Y)
    px, py = mx.leray_projection(gx, gy, grid)

    check("Leray kills a pure gradient", max(np.max(np.abs(px)), np.max(np.abs(py))) < 1.0e-10)

    # Cellular flow: already divergence free, so Pi is the identity on it.
    U = np.sin(grid.X) * np.cos(grid.Y)
    V = -np.cos(grid.X) * np.sin(grid.Y)
    pu, pv = mx.leray_projection(U, V, grid)

    error = max(np.max(np.abs(pu - U)), np.max(np.abs(pv - V)))
    check("Leray fixes a divergence-free field", error < 1.0e-12, f"error {error:.2e}")


def test_fused_velocity_matches_stages():
    """The single-transform velocity must equal the stage-by-stage version."""
    grid = mx.Grid(48)
    rho = 2.0 + 0.5 * np.sin(grid.X) * np.cos(grid.Y)
    lam = np.cos(grid.X) * np.sin(2.0 * grid.Y)

    U, V = mx.mixing_velocity(rho, lam, grid)
    stages = mx.mixing_velocity_stages(rho, lam, grid)

    error = max(
        np.max(np.abs(U - stages["velocity_x"])), np.max(np.abs(V - stages["velocity_y"]))
    )
    check("fused velocity matches staged velocity", error < 1.0e-12, f"error {error:.2e}")


def test_velocity_is_divergence_free():
    grid = mx.Grid(64)
    rho = 2.0 + 0.4 * np.cos(2.0 * grid.X + grid.Y)
    lam = np.sin(grid.X) * np.cos(grid.Y)

    U, V = mx.mixing_velocity(rho, lam, grid)
    spectral = np.max(np.abs(mx.spectral_divergence(U, V, grid)))
    scale = np.max(np.hypot(U, V))

    check("spectral div v is zero", spectral / scale < 1.0e-12, f"relative {spectral / scale:.2e}")


def test_conservative_step_preserves_mass():
    grid = mx.Grid(64)
    rho = 2.0 + 0.5 * np.sin(grid.X) * np.cos(grid.Y)

    U = np.sin(grid.X) * np.cos(grid.Y)
    V = -np.cos(grid.X) * np.sin(grid.Y)

    before = mx.total_mass(rho, grid)
    for _ in range(20):
        rho = mx.conservative_upwind_step(rho, U, V, grid, 0.01)
    after = mx.total_mass(rho, grid)

    error = abs(after - before) / abs(before)
    check("conservative step preserves mass", error < 1.0e-13, f"relative {error:.2e}")


def test_geodesic_run():
    """A short geodesic run: incompressible, and rho stays in its initial range."""
    data = mx.run_geodesic_equations(mx.Settings(N=48, final_time=0.5, max_snapshots=8))
    d = data["diagnostics"]

    check("geodesic run finishes", data["steps"] > 0, f"{data['steps']} steps")
    check(
        "geodesic velocity is incompressible",
        d["divergence_spectral_relative"] < 1.0e-10,
        f"relative {d['divergence_spectral_relative']:.2e}",
    )
    check(
        "geodesic rho stays bounded by its initial range",
        d["rho_min"] >= np.min(data["rho_start"]) - 1.0e-9
        and d["rho_max"] <= np.max(data["rho_start"]) + 1.0e-9,
    )
    check(
        "geodesic state actually moves",
        d["rho_change_linf"] > 1.0e-6,
        f"max change {d['rho_change_linf']:.2e}",
    )


def test_gradient_flow_dissipates_energy():
    """The energy dissipation equality: E must never increase."""
    data = mx.run_gradient_flow(mx.Settings(N=48, final_time=2.0, max_snapshots=8))
    d = data["diagnostics"]

    check("gradient flow run finishes", data["steps"] > 0, f"{data['steps']} steps")
    check(
        "gradient flow conserves mass",
        abs(d["mass_relative_error"]) < 1.0e-12,
        f"relative {d['mass_relative_error']:.2e}",
    )
    check(
        "H^-1 energy never increases",
        d["energy_increase_count"] == 0,
        f"{d['energy_increase_count']} increases",
    )
    check(
        "H^-1 energy actually drops",
        d["energy_relative_drop"] > 0.01,
        f"relative drop {d['energy_relative_drop']:.3f}",
    )
    check(
        "gradient flow velocity is incompressible",
        d["divergence_spectral_relative"] < 1.0e-10,
        f"relative {d['divergence_spectral_relative']:.2e}",
    )


def test_expression_parser():
    grid = mx.Grid(16)

    got = mx.evaluate_expression("2sin(x)^2", grid, "rho")
    want = 2.0 * np.sin(grid.X) ** 2
    check("expression parser handles 2sin(x)^2", np.allclose(got, want))

    got = mx.evaluate_expression("1.5", grid, "rho")
    check("scalar expressions broadcast", got.shape == grid.X.shape and np.allclose(got, 1.5))


def test_figures_build():
    data = mx.run_gradient_flow(mx.Settings(N=32, final_time=0.5, max_snapshots=4))

    for name, builder in mx.FIGURE_BUILDERS.items():
        fig = builder(data)
        check(f"figure '{name}' builds", fig is not None)


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()

    print("\nAll checks passed.")
