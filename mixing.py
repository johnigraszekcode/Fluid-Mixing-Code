"""
Numerical solvers for "Incompressible Fluid Mixing as Constrained Optimal Transport".

This module reproduces the simulations reported in Sections 6.1 and 7.1 of the
paper.  Both solvers live on the flat torus Omega = [0, 2*pi)^2 and share the
same velocity pipeline, so there is really only one nontrivial routine here and
two ways of feeding it a scalar potential.

    Geodesic equations (Section 6.1, enstrophy-penalized case)

        d_t rho    = -grad(rho) . v
        d_t lambda = -grad(lambda) . v
        v          = Pi[ Lap^-1 ( rho grad(lambda) ) ]

    Gradient flow (Section 7.1, homogeneous H^-1 energy)

        d_t rho    = -div(rho v)
        v          = Pi[ Lap^-1 ( rho grad(dE(rho)) ) ]
        dE(rho)    = -Lap^-1 (rho - rho*),     rho* = mean(rho)

Here Pi is the Leray projection and Lap^-1 the inverse Laplacian, both applied
spectrally.  The only structural difference between the two runs is where the
scalar potential comes from: the costate lambda, transported alongside rho, or
the first variation of the energy, recomputed from rho at every step.  Note
that the gradient flow velocity is v = -grad_m E(rho), so it is a genuine
steepest descent -- the sign lives in dE = -Lap^-1 rho_tilde.

Everything is written against numpy only; matplotlib is used for figures but
never for interactive state, so this module is safe to import headless.  The
Tk front end lives in mixing_gui.py.

Usage:

    python mixing.py geodesic       --grid 128 --time 2.0
    python mixing.py gradient-flow  --grid 128 --time 4.0

or, from Python,

    from mixing import Settings, run_gradient_flow, save_run_figures
    data = run_gradient_flow(Settings(N=128, final_time=4.0))
    save_run_figures(data, "results")
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from matplotlib.figure import Figure

# Figures are grayscale throughout so they drop straight into the paper.
COLORMAP = "gray"

# Default initial data.  Smooth on purpose: first-order upwind is diffusive and
# sharp interfaces smear out fast, which makes the pictures harder to read.
DEFAULT_RHO_EXPR = "sin(x)*cos(y)"
DEFAULT_LAMBDA_EXPR = "cos(x)*sin(y)"

# The gradient flow needs something with structure to flatten out, and it is
# tidier to start from a strictly positive density.
DEFAULT_MIXING_RHO_EXPR = "2.0 + 0.5*sin(x)*cos(y) + 0.4*cos(2*x + y) + 0.3*sin(x + 2*y)"

# Sensible horizons for each mode.  The gradient flow runs on a much longer
# clock: at the natural gradient flow rate the velocity is small, so it takes a
# while for anything visible to happen.  Rescaling v by a constant is just a
# reparameterization of time and leaves the trajectory in the reachable set
# unchanged, so running longer is exactly equivalent to speeding the flow up --
# and it costs nothing, because the CFL condition lets dt grow in proportion.
DEFAULT_FINAL_TIME = {"geodesic": 2.0, "gradient-flow": 75.0}

# Guards against a run that will never finish because the CFL condition keeps
# shrinking dt.
DEFAULT_MAX_STEPS = 50_000

# Slack used when comparing accumulated times against the final time.
TIME_EPS = 1.0e-14


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


@dataclass
class Grid:
    """Uniform N x N periodic grid on [0, Lx) x [0, Ly).

    The Fourier wave numbers are built once here rather than on every spectral
    call -- they never change, and rebuilding them was a surprisingly large
    share of the runtime in the original version of this code.
    """

    N: int
    Lx: float = 2.0 * np.pi
    Ly: float = 2.0 * np.pi

    def __post_init__(self):
        self.dx = self.Lx / self.N
        self.dy = self.Ly / self.N

        # X, Y use "ij" indexing, so the first axis is x and the second is y.
        self.x = np.arange(self.N) * self.dx
        self.y = np.arange(self.N) * self.dy
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # xi = (xi_1, xi_2) in the notation of Section 6.1.
        kx = 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")
        self.K2 = self.KX * self.KX + self.KY * self.KY

        # 1/|xi|^2 with the constant mode set to zero.  Multiplying by this is
        # both faster and less error-prone than masking on every call.
        self.inv_K2 = np.zeros_like(self.K2)
        np.divide(1.0, self.K2, out=self.inv_K2, where=self.K2 > 0.0)

        # Wave numbers for *first* derivatives.  For even N, fftfreq reports the
        # Nyquist mode as -N/2 in both halves of the spectrum, so an odd-order
        # multiplier built from it is not odd across the mode pairs and the
        # result of the transform is no longer the transform of a real field.
        # Taking the real part then leaves a small inconsistency, which shows up
        # as div(v) ~ 1e-8 relative instead of round-off, entirely on the
        # Nyquist row and column.  Dropping that mode is the usual remedy and
        # makes the Leray projection an exact projection on real fields.
        self.KX_d = self.KX.copy()
        self.KY_d = self.KY.copy()

        if self.N % 2 == 0:
            self.KX_d[self.N // 2, :] = 0.0
            self.KY_d[:, self.N // 2] = 0.0

        K2_d = self.KX_d * self.KX_d + self.KY_d * self.KY_d
        self.inv_K2_d = np.zeros_like(K2_d)
        np.divide(1.0, K2_d, out=self.inv_K2_d, where=K2_d > 0.0)

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def tidy_expression(expr: str) -> str:
    """Normalize a hand-typed formula: ^ -> **, implicit multiplication, etc.

    Not a real parser, just enough forgiveness that "2sin(x)^2" and
    "(x)(y)" do what the user obviously meant.
    """
    expr = (expr or "").strip()
    expr = expr.replace("^", "**")
    expr = expr.replace("\u03c0", "pi")
    expr = expr.replace("E", "e")
    expr = expr.replace(")(", ")*(")

    expr = re.sub(r"(?<=\))(?=[A-Za-z_])", "*", expr)  # )x   -> )*x
    expr = re.sub(r"(?<=[0-9])(?=[A-Za-z_])", "*", expr)  # 2x   -> 2*x
    expr = re.sub(r"(?<=[A-Za-z_])(?=\()", "", expr)  # sin*( -> sin(

    return expr


# Names visible to a user-supplied initial condition.  Builtins are stripped out
# below so a stray expression cannot reach the filesystem.
_EXPRESSION_NAMESPACE = {
    "pi": np.pi,
    "e": np.e,
    "np": np,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "where": np.where,
}


def evaluate_expression(expr: str, grid: Grid, label: str) -> np.ndarray:
    """Evaluate a formula in x and y on the grid and sanity-check the result."""
    cleaned = tidy_expression(expr)

    if not cleaned:
        raise ValueError(f"The {label} initial condition cannot be blank.")

    namespace = dict(_EXPRESSION_NAMESPACE)
    namespace.update({"x": grid.X, "y": grid.Y, "X": grid.X, "Y": grid.Y})

    try:
        value = eval(cleaned, {"__builtins__": {}}, namespace)  # noqa: S307
    except Exception as exc:
        raise ValueError(
            f"Could not read the {label} initial condition:\n\n{cleaned}\n\n"
            "Examples that work:\n"
            "  sin(x)*cos(y)\n"
            "  sin(x + y)\n"
            "  exp(cos(x))\n"
            "  2 + 0.5*cos(2*x + y)"
        ) from exc

    field = np.asarray(value, dtype=float)

    if field.shape == ():
        field = np.full(grid.X.shape, float(field))
    else:
        field = np.broadcast_to(field, grid.X.shape).astype(float).copy()

    if not np.all(np.isfinite(field)):
        raise ValueError(f"The {label} initial condition produced NaN or infinity.")

    return field


def periodic_bilinear_sample(field: np.ndarray, xq, yq, grid: Grid) -> np.ndarray:
    """Sample `field` at arbitrary points, wrapping periodically (Section 6.1).

    Query points are wrapped with modular arithmetic and then interpolated from
    the four surrounding nodes.  Not needed by either solver -- both are purely
    grid-based -- but it is the sampling rule quoted in the paper and is handy
    for tracer particles or for comparing two runs on different grids.
    """
    N = grid.N

    gx = np.mod(xq, grid.Lx) / grid.dx
    gy = np.mod(yq, grid.Ly) / grid.dy

    i0 = np.floor(gx).astype(int) % N
    j0 = np.floor(gy).astype(int) % N
    i1 = (i0 + 1) % N
    j1 = (j0 + 1) % N

    wx = gx - np.floor(gx)
    wy = gy - np.floor(gy)

    return (
        (1.0 - wx) * (1.0 - wy) * field[i0, j0]
        + wx * (1.0 - wy) * field[i1, j0]
        + (1.0 - wx) * wy * field[i0, j1]
        + wx * wy * field[i1, j1]
    )


# ---------------------------------------------------------------------------
# Norms and finite differences
# ---------------------------------------------------------------------------


def total_mass(field: np.ndarray, grid: Grid) -> float:
    return float(np.sum(field) * grid.cell_area)


def l1_norm(field: np.ndarray, grid: Grid) -> float:
    return float(np.sum(np.abs(field)) * grid.cell_area)


def l2_norm(field: np.ndarray, grid: Grid) -> float:
    return float(np.sqrt(np.sum(field * field) * grid.cell_area))


def max_abs(field: np.ndarray) -> float:
    return float(np.max(np.abs(field)))


def zero_mean(field: np.ndarray) -> np.ndarray:
    """Subtract the spatial average, killing the xi = 0 mode before Lap^-1."""
    return field - np.mean(field)


def centered_gradient(field: np.ndarray, grid: Grid):
    """Centered differences.

    Upwinding needs to know the flow direction, but here the gradient *is* what
    builds the velocity, so there is nothing to upwind against yet.  Centered
    differences keep the symmetry of the operator.
    """
    field_x = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2.0 * grid.dx)
    field_y = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2.0 * grid.dy)
    return field_x, field_y


def centered_divergence(U: np.ndarray, V: np.ndarray, grid: Grid) -> np.ndarray:
    U_x = (np.roll(U, -1, axis=0) - np.roll(U, 1, axis=0)) / (2.0 * grid.dx)
    V_y = (np.roll(V, -1, axis=1) - np.roll(V, 1, axis=1)) / (2.0 * grid.dy)
    return U_x + V_y


def spectral_divergence(U: np.ndarray, V: np.ndarray, grid: Grid) -> np.ndarray:
    """Divergence via FFT (d_x -> i*xi_1, d_y -> i*xi_2).

    Kept as an independent second opinion on div(v) = 0: the finite-difference
    check and the spectral check fail in different ways, so agreement between
    them is worth more than either on its own.
    """
    div_hat = 1j * (grid.KX_d * np.fft.fft2(U) + grid.KY_d * np.fft.fft2(V))
    return np.real(np.fft.ifft2(div_hat))


# ---------------------------------------------------------------------------
# Spectral operators
# ---------------------------------------------------------------------------


def inverse_laplacian(field: np.ndarray, grid: Grid) -> np.ndarray:
    """Lap^-1 on the torus, zero mean.

    In Fourier the Laplacian is multiplication by -|xi|^2, so the inverse is a
    division by -|xi|^2 with the constant mode dropped.
    """
    field_hat = np.fft.fft2(zero_mean(field))
    return np.real(np.fft.ifft2(-field_hat * grid.inv_K2))


def inverse_vector_laplacian(W1: np.ndarray, W2: np.ndarray, grid: Grid):
    """The vector Laplacian acts component-wise, so just do it twice."""
    return inverse_laplacian(W1, grid), inverse_laplacian(W2, grid)


def leray_projection(U: np.ndarray, V: np.ndarray, grid: Grid):
    """Pi = I - grad Lap^-1 div: strip the curl-free part, leaving div(v) = 0.

    In Fourier this is u_hat - xi (xi . u_hat) / |xi|^2.  The constant mode has
    inv_K2 = 0, so any mean flow passes through untouched (it is already
    divergence free).
    """
    U_hat = np.fft.fft2(U)
    V_hat = np.fft.fft2(V)

    dot_hat = grid.KX_d * U_hat + grid.KY_d * V_hat

    U_hat = U_hat - grid.KX_d * dot_hat * grid.inv_K2_d
    V_hat = V_hat - grid.KY_d * dot_hat * grid.inv_K2_d

    return np.real(np.fft.ifft2(U_hat)), np.real(np.fft.ifft2(V_hat))


def _leray_inverse_laplacian(W1: np.ndarray, W2: np.ndarray, grid: Grid):
    """Pi[Lap^-1 W] using a single transform pair per component.

    Lap^-1 and Pi are both Fourier multipliers, so composing them in spectral
    space gives exactly the same answer as calling inverse_vector_laplacian and
    then leray_projection -- it just halves the number of FFTs, which is the
    dominant cost of a time step.  test_mixing.py checks the two agree.
    """
    A_hat = -np.fft.fft2(W1) * grid.inv_K2
    B_hat = -np.fft.fft2(W2) * grid.inv_K2

    dot_hat = grid.KX_d * A_hat + grid.KY_d * B_hat

    A_hat = A_hat - grid.KX_d * dot_hat * grid.inv_K2_d
    B_hat = B_hat - grid.KY_d * dot_hat * grid.inv_K2_d

    return np.real(np.fft.ifft2(A_hat)), np.real(np.fft.ifft2(B_hat))


# ---------------------------------------------------------------------------
# The velocity field
# ---------------------------------------------------------------------------


def mixing_velocity(rho: np.ndarray, potential: np.ndarray, grid: Grid):
    """v = Pi[ Lap^-1 ( rho grad(potential) ) ].

    `potential` is the costate lambda for the geodesic equations and the first
    variation dE(rho) for the gradient flow; nothing else changes between the
    two.  Each component of rho grad(potential) is zero-meaned before the
    solve, as in Section 6.1.
    """
    potential_x, potential_y = centered_gradient(potential, grid)

    W1 = zero_mean(rho * potential_x)
    W2 = zero_mean(rho * potential_y)

    return _leray_inverse_laplacian(W1, W2, grid)


def mixing_velocity_stages(rho: np.ndarray, potential: np.ndarray, grid: Grid) -> dict:
    """Same computation as `mixing_velocity`, keeping every intermediate.

    This is the version the plotting code uses, since the whole point of the
    overview figures is to show the pipeline stage by stage.  It runs the
    operators separately and so costs roughly twice as much as the fused path;
    it is only ever called when a snapshot is actually recorded.
    """
    potential_x, potential_y = centered_gradient(potential, grid)

    W1 = zero_mean(rho * potential_x)
    W2 = zero_mean(rho * potential_y)

    laplace_x, laplace_y = inverse_vector_laplacian(W1, W2, grid)
    U, V = leray_projection(laplace_x, laplace_y, grid)

    return {
        "potential": potential,
        "potential_x": potential_x,
        "potential_y": potential_y,
        "rho_grad_potential_x": W1,
        "rho_grad_potential_y": W2,
        "laplacian_inverse_x": laplace_x,
        "laplacian_inverse_y": laplace_y,
        "velocity_x": U,
        "velocity_y": V,
    }


def velocity_diagnostics(U: np.ndarray, V: np.ndarray, grid: Grid) -> dict:
    """Speed field, both divergence checks, and the CFL rate max|u|/dx + |v|/dy."""
    finite_div = centered_divergence(U, V, grid)
    spectral_div = spectral_divergence(U, V, grid)
    speed = np.hypot(U, V)

    rate = np.abs(U) / grid.dx + np.abs(V) / grid.dy
    max_rate = float(np.max(rate))

    # A dead velocity field would make the relative divergences blow up.
    scale = max_rate if max_rate > 1.0e-300 else 1.0

    return {
        "speed": speed,
        "divergence_finite": finite_div,
        "divergence_spectral": spectral_div,
        "max_speed": float(np.max(speed)),
        "mean_speed": float(np.mean(speed)),
        "divergence_finite_linf": max_abs(finite_div),
        "divergence_finite_l2": l2_norm(finite_div, grid),
        "divergence_spectral_linf": max_abs(spectral_div),
        "divergence_spectral_l2": l2_norm(spectral_div, grid),
        "divergence_finite_relative": max_abs(finite_div) / scale,
        "divergence_spectral_relative": max_abs(spectral_div) / scale,
        "cfl_rate": max_rate,
    }


# ---------------------------------------------------------------------------
# The H^-1 energy
# ---------------------------------------------------------------------------


def hminus1_potential(rho: np.ndarray, grid: Grid) -> np.ndarray:
    """psi = Lap^-1 rho_tilde, where rho_tilde = rho - mean(rho).

    Both the energy and its first variation are built from this one solve, so
    the solvers compute it once per step and reuse it.
    """
    return inverse_laplacian(zero_mean(rho), grid)


def hminus1_energy_from_potential(psi: np.ndarray, grid: Grid) -> float:
    """E = 1/2 ||grad psi||^2_{L^2} = 1/2 ||rho - rho*||^2_{H^-1}."""
    psi_x, psi_y = centered_gradient(psi, grid)
    return 0.5 * float(np.sum(psi_x * psi_x + psi_y * psi_y) * grid.cell_area)


def hminus1_energy(rho: np.ndarray, grid: Grid) -> float:
    return hminus1_energy_from_potential(hminus1_potential(rho, grid), grid)


def hminus1_first_variation(rho: np.ndarray, grid: Grid) -> np.ndarray:
    """dE(rho) = -Lap^-1 rho_tilde."""
    return -hminus1_potential(rho, grid)


# ---------------------------------------------------------------------------
# Time stepping
# ---------------------------------------------------------------------------


def upwind_advection_step(field, U, V, grid: Grid, dt: float) -> np.ndarray:
    """One explicit step of d_t u = -grad(u) . v (Section 6.1).

    Upwind means differencing on the side the flow is coming *from*.  The other
    choice is unstable, so the difference operator has to be selected pointwise
    from the sign of the local velocity -- hence np.where rather than a single
    stencil.
    """
    backward_x = (field - np.roll(field, 1, axis=0)) / grid.dx
    forward_x = (np.roll(field, -1, axis=0) - field) / grid.dx

    backward_y = (field - np.roll(field, 1, axis=1)) / grid.dy
    forward_y = (np.roll(field, -1, axis=1) - field) / grid.dy

    field_x = np.where(U >= 0.0, backward_x, forward_x)
    field_y = np.where(V >= 0.0, backward_y, forward_y)

    return field - dt * (U * field_x + V * field_y)


def conservative_upwind_step(field, U, V, grid: Grid, dt: float) -> np.ndarray:
    """One explicit step of d_t rho = -div(rho v) in flux form (Section 6.1).

    The gradient flow has to conserve mass exactly, which the advective form
    above does not.  Face velocities are the average of the two neighbouring
    cells and the upwind side decides which cell value gets carried across.
    """
    U_right = 0.5 * (U + np.roll(U, -1, axis=0))
    field_right = np.where(U_right >= 0.0, field, np.roll(field, -1, axis=0))
    flux_x_right = U_right * field_right
    flux_x_left = np.roll(flux_x_right, 1, axis=0)  # left face = right face of the previous cell

    V_top = 0.5 * (V + np.roll(V, -1, axis=1))
    field_top = np.where(V_top >= 0.0, field, np.roll(field, -1, axis=1))
    flux_y_top = V_top * field_top
    flux_y_bottom = np.roll(flux_y_top, 1, axis=1)

    return field - dt * (
        (flux_x_right - flux_x_left) / grid.dx + (flux_y_top - flux_y_bottom) / grid.dy
    )


def cfl_timestep(U, V, grid: Grid, cfl: float, max_dt: float) -> float:
    """dt = min(max_dt, cfl / max(|u|/dx + |v|/dy)).

    Both solvers regenerate v from the state at every step, so dt is chosen
    adaptively rather than fixed up front.  max_dt keeps us from stepping past
    the final time.
    """
    max_rate = float(np.max(np.abs(U) / grid.dx + np.abs(V) / grid.dy))

    if max_rate == 0.0:
        return max_dt

    return min(max_dt, cfl / max_rate)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    """Everything a run needs.  `rho_expr` falls back to a per-solver default."""

    N: int = 128
    final_time: float = 2.0
    cfl: float = 0.45
    max_snapshots: int = 180
    max_steps: int = DEFAULT_MAX_STEPS
    rho_expr: str | None = None
    lambda_expr: str = DEFAULT_LAMBDA_EXPR

    def validate(self):
        if self.N < 16:
            raise ValueError("Grid size N must be at least 16.")
        if self.final_time <= 0.0:
            raise ValueError("Final time must be positive.")
        if self.cfl <= 0.0:
            raise ValueError("CFL number must be positive.")
        if self.max_snapshots < 2:
            raise ValueError("Need at least 2 display snapshots.")
        return self


class SnapshotClock:
    """Decides when to record a display snapshot.

    Snapshots are for looking at, not for accuracy, so they are spaced evenly
    in time and capped -- otherwise a fine grid produces thousands of them and
    the viewer bogs down.
    """

    def __init__(self, final_time: float, max_snapshots: int):
        self.final_time = final_time
        self.interval = final_time / float(max(2, max_snapshots) - 1)
        self.next_time = self.interval

    def due(self, t: float) -> bool:
        if t >= self.next_time - TIME_EPS or abs(t - self.final_time) < TIME_EPS:
            while self.next_time <= t + TIME_EPS:
                self.next_time += self.interval
            return True
        return False


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def _record(step, t, rho, potential, grid, extra=None) -> dict:
    """Build one display snapshot: state, the velocity pipeline, diagnostics."""
    snapshot = {"step": step, "time": t, "rho": rho.copy()}
    snapshot.update(mixing_velocity_stages(rho, potential, grid))
    snapshot.update(
        velocity_diagnostics(snapshot["velocity_x"], snapshot["velocity_y"], grid)
    )

    if extra:
        snapshot.update(extra)

    return snapshot


def run_geodesic_equations(settings: Settings) -> dict:
    """Section 6.1: the coupled (rho, lambda) geodesic system.

    rho and lambda are transported together by v = Pi[Lap^-1(rho grad lambda)],
    which is rebuilt from the state at every step.  Both use the conservative
    flux-form upwind scheme (same as Section 7.1); under incompressibility the
    advective and conservative forms agree analytically, and using the
    conservative one keeps mass exact and the two equations symmetric.
    """
    settings.validate()

    grid = Grid(settings.N)

    rho = evaluate_expression(settings.rho_expr or DEFAULT_RHO_EXPR, grid, "rho")
    lam = evaluate_expression(settings.lambda_expr, grid, "lambda")

    rho_start = rho.copy()
    lam_start = lam.copy()

    clock = SnapshotClock(settings.final_time, settings.max_snapshots)

    U, V = mixing_velocity(rho, lam, grid)
    snapshots = [_record(0, 0.0, rho, lam, grid)]
    energy_history = [(0.0, hminus1_energy(rho, grid))]

    t = 0.0
    step = 0
    dt = settings.final_time

    while t < settings.final_time - TIME_EPS:
        # The snapshot interval plays the role of dt_max in the CFL formula of
        # Section 6.1.  Without it the first step is enormous whenever the run
        # starts from a nearly stationary state, which is exactly where the
        # explicit scheme is least trustworthy.
        max_dt = min(settings.final_time - t, clock.interval)
        dt = cfl_timestep(U, V, grid, settings.cfl, max_dt)

        # Both fields ride the same velocity, so step them off the same v.
        rho_next = conservative_upwind_step(rho, U, V, grid, dt)
        lam = conservative_upwind_step(lam, U, V, grid, dt)
        rho = rho_next

        t += dt
        step += 1

        U, V = mixing_velocity(rho, lam, grid)
        energy_history.append((t, hminus1_energy(rho, grid)))

        if clock.due(t):
            snapshots.append(_record(step, t, rho, lam, grid))

        if step > settings.max_steps:
            raise RuntimeError(
                "Too many time steps. Shorten the final time, lower the CFL number, "
                "or coarsen the grid."
            )

    if snapshots[-1]["step"] != step:
        snapshots.append(_record(step, t, rho, lam, grid))

    data = {
        "mode": "geodesic",
        "grid": grid,
        "settings": settings,
        "rho_start": rho_start,
        "rho_final": rho,
        "potential_start": lam_start,
        "potential_final": lam,
        "velocity_x": U,
        "velocity_y": V,
        "dt": dt,
        "steps": step,
        "snapshots": snapshots,
        "energy_history": energy_history,
    }

    return annotate_diagnostics(data)


def run_gradient_flow(settings: Settings) -> dict:
    """Section 7.1: the H^-1 gradient flow.

    Identical machinery to the geodesic solver with the costate replaced by
    dE(rho), recomputed from rho at every step, plus the conservative flux step
    so that mass is preserved to round-off.  The energy is tracked every step
    as a check on the energy dissipation equality: it should be monotonically
    non-increasing.
    """
    settings.validate()

    grid = Grid(settings.N)

    rho = evaluate_expression(settings.rho_expr or DEFAULT_MIXING_RHO_EXPR, grid, "rho")
    rho_start = rho.copy()

    clock = SnapshotClock(settings.final_time, settings.max_snapshots)

    psi = hminus1_potential(rho, grid)
    energy = hminus1_energy_from_potential(psi, grid)
    delta_E = -psi

    U, V = mixing_velocity(rho, delta_E, grid)

    snapshots = [_record(0, 0.0, rho, delta_E, grid, _gradient_flow_extras(rho, energy))]
    energy_history = [(0.0, energy)]

    t = 0.0
    step = 0
    dt = settings.final_time

    while t < settings.final_time - TIME_EPS:
        # Same dt_max convention as the geodesic solver: never step past a
        # snapshot time, so the recorded frames stay evenly spaced.
        max_dt = min(settings.final_time - t, clock.interval)
        dt = cfl_timestep(U, V, grid, settings.cfl, max_dt)

        if dt <= 0.0 or not np.isfinite(dt):
            raise RuntimeError("Gradient flow time step collapsed; check the initial condition.")

        rho = conservative_upwind_step(rho, U, V, grid, dt)

        t += dt
        step += 1

        psi = hminus1_potential(rho, grid)
        energy = hminus1_energy_from_potential(psi, grid)
        delta_E = -psi

        U, V = mixing_velocity(rho, delta_E, grid)
        energy_history.append((t, energy))

        if clock.due(t):
            snapshots.append(
                _record(step, t, rho, delta_E, grid, _gradient_flow_extras(rho, energy))
            )

        if step > settings.max_steps:
            raise RuntimeError(
                "Too many time steps. Shorten the final time, lower the CFL number, "
                "or coarsen the grid."
            )

    if snapshots[-1]["step"] != step:
        snapshots.append(
            _record(step, t, rho, delta_E, grid, _gradient_flow_extras(rho, energy))
        )

    data = {
        "mode": "gradient-flow",
        "grid": grid,
        "settings": settings,
        "rho_start": rho_start,
        "rho_final": rho,
        "potential_start": snapshots[0]["potential"],
        "potential_final": delta_E,
        "velocity_x": U,
        "velocity_y": V,
        "dt": dt,
        "steps": step,
        "snapshots": snapshots,
        "energy_history": energy_history,
    }

    return annotate_diagnostics(data)


def _gradient_flow_extras(rho: np.ndarray, energy: float) -> dict:
    """rho*, rho_tilde and the energy: the pieces specific to the gradient flow."""
    target = np.full_like(rho, float(np.mean(rho)))
    return {"target_rho": target, "rho_tilde": rho - target, "energy": energy}


SOLVERS: dict[str, Callable[[Settings], dict]] = {
    "geodesic": run_geodesic_equations,
    "gradient-flow": run_gradient_flow,
}

MODE_LABELS = {
    "geodesic": "geodesic equations (Section 6.1)",
    "gradient-flow": "H^-1 gradient flow (Section 7.1)",
}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def annotate_diagnostics(data: dict) -> dict:
    """Attach per-snapshot and run-level diagnostics in place.

    The quantities worth watching are: mass and the L^2 norm of rho (both are
    conserved by the exact dynamics -- see the conserved-signature result in
    Section 3.1 -- so drift measures the scheme's dissipation), the H^-1 energy,
    and how far div(v) strays from zero.
    """
    grid = data["grid"]
    snapshots = data["snapshots"]
    rho_start = data["rho_start"]

    mass_start = total_mass(rho_start, grid)
    l2_start = l2_norm(rho_start, grid)
    energy_start = data["energy_history"][0][1]

    # The geodesic default has zero total mass, so dividing the mass drift by
    # the mass itself is meaningless.  Measure it against ||rho||_L1 instead,
    # which is a sensible scale for the density either way.
    mass_scale = max(abs(mass_start), l1_norm(rho_start, grid), 1.0e-300)

    for snapshot in snapshots:
        rho = snapshot["rho"]
        change = rho - rho_start

        snapshot["rho_change"] = change
        snapshot["abs_rho_change"] = np.abs(change)
        snapshot["mass"] = total_mass(rho, grid)
        snapshot["mass_error"] = snapshot["mass"] - mass_start
        snapshot["l2_norm"] = l2_norm(rho, grid)
        snapshot["l2_error"] = snapshot["l2_norm"] - l2_start
        snapshot["rho_min"] = float(np.min(rho))
        snapshot["rho_max"] = float(np.max(rho))
        snapshot["rho_change_linf"] = max_abs(change)
        snapshot["rho_change_l2"] = l2_norm(change, grid)
        snapshot.setdefault("energy", hminus1_energy(rho, grid))
        snapshot["energy_drop"] = energy_start - snapshot["energy"]

    # The energy dissipation equality says E should never go up.  Round-off and
    # the explicit time step allow a little slack, so count only real increases.
    energies = [value for _, value in data["energy_history"]]
    rises = [b - a for a, b in zip(energies, energies[1:]) if b - a > 1.0e-10]

    final = snapshots[-1]

    data["diagnostics"] = {
        "mode": data["mode"],
        "steps": data["steps"],
        "snapshots": len(snapshots),
        "mass_start": mass_start,
        "mass_final": final["mass"],
        "mass_error": final["mass_error"],
        "mass_relative_error": final["mass_error"] / mass_scale,
        "l2_start": l2_start,
        "l2_final": final["l2_norm"],
        "l2_relative_error": final["l2_error"] / max(abs(l2_start), 1.0e-300),
        "energy_start": energy_start,
        "energy_final": final["energy"],
        "energy_drop": final["energy_drop"],
        "energy_relative_drop": final["energy_drop"] / max(abs(energy_start), 1.0e-300),
        "energy_increase_count": len(rises),
        "max_energy_increase": max(rises) if rises else 0.0,
        "divergence_finite_linf": final["divergence_finite_linf"],
        "divergence_spectral_linf": final["divergence_spectral_linf"],
        "divergence_finite_relative": final["divergence_finite_relative"],
        "divergence_spectral_relative": final["divergence_spectral_relative"],
        "rho_change_linf": final["rho_change_linf"],
        "rho_change_l2": final["rho_change_l2"],
        "rho_min": final["rho_min"],
        "rho_max": final["rho_max"],
    }

    return data


def diagnostics_report(data: dict) -> str:
    """Human-readable version of the diagnostics dictionary."""
    d = data["diagnostics"]

    lines = [
        MODE_LABELS[d["mode"]],
        "",
        f"time steps taken      {d['steps']}",
        f"display snapshots     {d['snapshots']}",
        "",
        "Conserved quantities",
        f"  initial mass                 {d['mass_start']:.12e}",
        f"  final mass                   {d['mass_final']:.12e}",
        f"  mass drift                   {d['mass_error']:.6e}",
        f"  drift / ||rho(0)||_L1        {d['mass_relative_error']:.6e}",
        f"  initial ||rho||_L2           {d['l2_start']:.12e}",
        f"  final   ||rho||_L2           {d['l2_final']:.12e}",
        f"  relative L2 error            {d['l2_relative_error']:.6e}",
        "",
        "H^-1 energy",
        f"  initial energy               {d['energy_start']:.12e}",
        f"  final energy                 {d['energy_final']:.12e}",
        f"  energy drop                  {d['energy_drop']:.12e}",
        f"  relative energy drop         {d['energy_relative_drop']:.6e}",
        f"  steps where energy rose      {d['energy_increase_count']}",
        f"  largest single-step rise     {d['max_energy_increase']:.6e}",
        "",
        "Incompressibility of v",
        f"  finite-difference max|div v| {d['divergence_finite_linf']:.6e}",
        f"  spectral max|div v|          {d['divergence_spectral_linf']:.6e}",
        f"  relative finite-difference   {d['divergence_finite_relative']:.6e}",
        f"  relative spectral            {d['divergence_spectral_relative']:.6e}",
        "",
        "How far the state moved",
        f"  final max|rho(t) - rho(0)|   {d['rho_change_linf']:.6e}",
        f"  final L2 |rho(t) - rho(0)|   {d['rho_change_l2']:.6e}",
        f"  final min rho                {d['rho_min']:.6e}",
        f"  final max rho                {d['rho_max']:.6e}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plottable quantities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Quantity:
    """One plottable field.

    `label` is what appears in the viewer menus and `title` is the LaTeX form
    used on the figures, so the plots read in the notation of the paper without
    the drop-down boxes turning into a wall of dollar signs.
    """

    label: str
    title: str
    kind: str  # "scalar" or "vector"
    keys: tuple  # (field,) for scalars, (x_field, y_field) for vectors


def _as_table(quantities: Sequence[Quantity]) -> dict:
    return {q.label: q for q in quantities}


GEODESIC_QUANTITIES = _as_table(
    [
        Quantity("rho", r"$\rho$", "scalar", ("rho",)),
        Quantity("lambda", r"$\lambda$", "scalar", ("potential",)),
        Quantity("grad lambda", r"$\nabla\lambda$", "vector", ("potential_x", "potential_y")),
        Quantity(
            "rho grad lambda",
            r"$\rho\,\nabla\lambda$",
            "vector",
            ("rho_grad_potential_x", "rho_grad_potential_y"),
        ),
        Quantity(
            "inverse Laplacian",
            r"$\Delta^{-1}(\rho\,\nabla\lambda)$",
            "vector",
            ("laplacian_inverse_x", "laplacian_inverse_y"),
        ),
        Quantity(
            "velocity v",
            r"$v=\Pi[\Delta^{-1}(\rho\,\nabla\lambda)]$",
            "vector",
            ("velocity_x", "velocity_y"),
        ),
        Quantity("speed", r"$|v|$", "scalar", ("speed",)),
        Quantity(
            "divergence (finite difference)",
            r"$\nabla\cdot v$ (finite difference)",
            "scalar",
            ("divergence_finite",),
        ),
        Quantity(
            "divergence (spectral)", r"$\nabla\cdot v$ (spectral)", "scalar", ("divergence_spectral",)
        ),
    ]
)

GRADIENT_FLOW_QUANTITIES = _as_table(
    [
        Quantity("rho", r"$\rho$", "scalar", ("rho",)),
        Quantity("rho star", r"$\rho^{*}$", "scalar", ("target_rho",)),
        Quantity("rho - rho star", r"$\tilde\rho=\rho-\rho^{*}$", "scalar", ("rho_tilde",)),
        Quantity("change from start", r"$\rho(t)-\rho(0)$", "scalar", ("rho_change",)),
        Quantity(
            "first variation", r"$\delta E(\rho)=-\Delta^{-1}\tilde\rho$", "scalar", ("potential",)
        ),
        Quantity(
            "grad delta E", r"$\nabla\,\delta E(\rho)$", "vector", ("potential_x", "potential_y")
        ),
        Quantity(
            "rho grad delta E",
            r"$\rho\,\nabla\,\delta E(\rho)$",
            "vector",
            ("rho_grad_potential_x", "rho_grad_potential_y"),
        ),
        Quantity(
            "inverse Laplacian",
            r"$\Delta^{-1}(\rho\,\nabla\,\delta E)$",
            "vector",
            ("laplacian_inverse_x", "laplacian_inverse_y"),
        ),
        Quantity(
            "velocity v",
            r"$v=\Pi[\Delta^{-1}(\rho\,\nabla\,\delta E)]$",
            "vector",
            ("velocity_x", "velocity_y"),
        ),
        Quantity("speed", r"$|v|$", "scalar", ("speed",)),
        Quantity(
            "divergence (finite difference)",
            r"$\nabla\cdot v$ (finite difference)",
            "scalar",
            ("divergence_finite",),
        ),
        Quantity(
            "divergence (spectral)", r"$\nabla\cdot v$ (spectral)", "scalar", ("divergence_spectral",)
        ),
    ]
)

# Layout of the overview figure for each mode: (quantity labels, rows, columns).
OVERVIEW_LAYOUT = {
    "geodesic": (list(GEODESIC_QUANTITIES), 3, 3),
    "gradient-flow": (list(GRADIENT_FLOW_QUANTITIES), 3, 4),
}


def quantities_for(mode: str) -> dict:
    return GEODESIC_QUANTITIES if mode == "geodesic" else GRADIENT_FLOW_QUANTITIES



# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def arrow_components(U, V, grid: Grid, skip: int):
    """Downsample and normalize a vector field for quiver.

    Arrows are drawn at a fixed fraction of the cell size so the plot looks the
    same at any N; without this, a fine grid turns into a solid black square.
    """
    Uq = U[::skip, ::skip]
    Vq = V[::skip, ::skip]

    max_speed = float(np.max(np.hypot(Uq, Vq)))

    if max_speed < 1.0e-14:
        return np.zeros_like(Uq), np.zeros_like(Vq)

    target_length = 0.70 * min(grid.dx, grid.dy) * max(1, skip)

    return (Uq / max_speed) * target_length, (Vq / max_speed) * target_length


class FieldPanel:
    """A 2D image of one quantity that can be re-pointed at a new snapshot.

    Vector quantities are drawn as a speed image with arrows on top.  Updating
    reuses the same artists, which is what makes scrubbing through snapshots in
    the viewer feel instant instead of redrawing from scratch.
    """

    def __init__(self, fig, ax, grid: Grid, snapshot: dict, quantity: Quantity, colorbar=False):
        self.fig = fig
        self.ax = ax
        self.grid = grid
        self.quantity = quantity
        self.skip = max(1, grid.N // 20)

        ax.set_aspect("equal")
        ax.set_title(quantity.title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        background, U, V = self._fields(snapshot)

        # imshow wants row = y, so transpose out of the "ij" convention.
        self.image = ax.imshow(
            background.T,
            origin="lower",
            extent=(0.0, grid.Lx, 0.0, grid.Ly),
            cmap=COLORMAP,
            aspect="equal",
        )
        self._set_limits(background)

        if U is None:
            self.quiver = None
        else:
            Uq, Vq = arrow_components(U, V, grid, self.skip)
            self.quiver = ax.quiver(
                grid.X[:: self.skip, :: self.skip],
                grid.Y[:: self.skip, :: self.skip],
                Uq,
                Vq,
                color="black",
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.003,
            )

        if colorbar:
            self.fig.colorbar(self.image, ax=ax, fraction=0.046, pad=0.04)

    def _fields(self, snapshot):
        if self.quantity.kind == "scalar":
            return snapshot[self.quantity.keys[0]], None, None

        U = snapshot[self.quantity.keys[0]]
        V = snapshot[self.quantity.keys[1]]
        return np.hypot(U, V), U, V

    def _set_limits(self, field):
        finite = field[np.isfinite(field)]

        if finite.size == 0:
            self.image.set_clim(-1.0, 1.0)
            return

        vmin = float(np.min(finite))
        vmax = float(np.max(finite))

        # A constant field would give a degenerate colour range.
        if abs(vmax - vmin) < 1.0e-14:
            center = 0.5 * (vmax + vmin)
            pad = max(1.0e-6, abs(center) * 1.0e-3)
            vmin, vmax = center - pad, center + pad

        self.image.set_clim(vmin, vmax)

    def update(self, snapshot):
        background, U, V = self._fields(snapshot)

        self.image.set_data(background.T)
        self._set_limits(background)

        if self.quiver is not None:
            self.quiver.set_UVC(*arrow_components(U, V, self.grid, self.skip))


class SurfacePanel:
    """A 3D view of one quantity.

    Scalars are drawn as a surface; vectors as a low relief of |v| with the
    arrows laid on top.  Matplotlib's 3D axes cannot be updated in place, so
    this one really does redraw each time.
    """

    def __init__(self, fig, ax, grid: Grid, snapshot: dict, quantity: Quantity):
        self.fig = fig
        self.ax = ax
        self.grid = grid
        self.quantity = quantity
        self.skip = max(1, grid.N // 16)

        self.update(snapshot)

    def update(self, snapshot):
        ax = self.ax
        ax.clear()

        if self.quantity.kind == "scalar":
            field = snapshot[self.quantity.keys[0]]
            stride = max(1, self.grid.N // 60)

            ax.plot_surface(
                self.grid.X[::stride, ::stride],
                self.grid.Y[::stride, ::stride],
                field[::stride, ::stride],
                cmap=COLORMAP,
                linewidth=0,
                antialiased=True,
            )
            ax.set_zlabel(self.quantity.label)
        else:
            U = snapshot[self.quantity.keys[0]]
            V = snapshot[self.quantity.keys[1]]

            speed = np.hypot(U, V)
            max_speed = float(np.max(speed))

            # Flatten the relief to a fixed height so the arrows stay readable.
            relief = 0.28 * (speed / max_speed if max_speed > 1.0e-14 else np.zeros_like(speed))

            Xs = self.grid.X[:: self.skip, :: self.skip]
            Ys = self.grid.Y[:: self.skip, :: self.skip]
            Zs = relief[:: self.skip, :: self.skip]

            Uq, Vq = arrow_components(U, V, self.grid, self.skip)

            ax.plot_surface(Xs, Ys, Zs, cmap=COLORMAP, linewidth=0, antialiased=True, alpha=0.72)
            ax.quiver(
                Xs,
                Ys,
                Zs + 0.02,
                Uq,
                Vq,
                np.zeros_like(Uq),
                length=1.0,
                normalize=False,
                color="black",
            )
            ax.set_zlabel("scaled speed")
            ax.set_zlim(0.0, 0.36)

        ax.set_title(self.quantity.title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0.0, self.grid.Lx)
        ax.set_ylim(0.0, self.grid.Ly)
        ax.view_init(elev=30, azim=-60)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_overview(data: dict, index: int = -1) -> Figure:
    """Every stage of the velocity pipeline at one snapshot."""
    grid = data["grid"]
    snapshot = data["snapshots"][index]
    table = quantities_for(data["mode"])
    labels, rows, cols = OVERVIEW_LAYOUT[data["mode"]]

    fig = Figure(figsize=(3.4 * cols, 3.3 * rows), dpi=110)

    for i, label in enumerate(labels):
        ax = fig.add_subplot(rows, cols, i + 1)
        FieldPanel(fig, ax, grid, snapshot, table[label], colorbar=False)

    fig.suptitle(
        f"{MODE_LABELS[data['mode']]}   t = {snapshot['time']:.4f}   N = {grid.N}", fontsize=12
    )
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.90, wspace=0.36, hspace=0.45)

    return fig


def figure_energy_history(data: dict) -> Figure:
    """E(rho_t) against time, recorded every step rather than every snapshot."""
    times = [t for t, _ in data["energy_history"]]
    energies = [e for _, e in data["energy_history"]]

    fig = Figure(figsize=(8.0, 5.0), dpi=110)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(times, energies, color="black", linewidth=1.4)
    ax.set_title(r"$E(\rho_t)=\frac{1}{2}\|\rho_t-\rho^{*}\|^{2}_{\dot{H}^{-1}}$")
    ax.set_xlabel("t")
    ax.set_ylabel("energy")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig


def figure_diagnostics(data: dict) -> Figure:
    """Conservation, dissipation and incompressibility checks in one sheet."""
    snapshots = data["snapshots"]
    times = [s["time"] for s in snapshots]

    fig = Figure(figsize=(12.0, 7.5), dpi=110)
    axes = [fig.add_subplot(2, 3, i + 1) for i in range(6)]

    def line(ax, values, title, ylabel, label=None):
        ax.plot(times, values, marker="o", markersize=2.5, linewidth=1.2, label=label)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    line(axes[0], [s["energy"] for s in snapshots], "H^-1 energy", "E")
    line(axes[1], [s["mass_error"] for s in snapshots], "mass drift", "mass(t) - mass(0)")
    line(axes[2], [s["l2_error"] for s in snapshots], "L2 norm drift", "||rho(t)|| - ||rho(0)||")

    line(
        axes[3],
        [s["divergence_finite_linf"] for s in snapshots],
        "max |div v|",
        "max |div v|",
        label="finite difference",
    )
    line(axes[3], [s["divergence_spectral_linf"] for s in snapshots], "max |div v|", "max |div v|", label="spectral")
    axes[3].legend(fontsize=8)

    line(axes[4], [s["rho_change_linf"] for s in snapshots], "change from start", "change", label="max")
    line(axes[4], [s["rho_change_l2"] for s in snapshots], "change from start", "change", label="L2")
    axes[4].legend(fontsize=8)

    line(axes[5], [s["rho_min"] for s in snapshots], "range of rho", "rho", label="min")
    line(axes[5], [s["rho_max"] for s in snapshots], "range of rho", "rho", label="max")
    axes[5].legend(fontsize=8)

    fig.suptitle(f"{MODE_LABELS[data['mode']]} diagnostics", fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.90, wspace=0.34, hspace=0.40)

    return fig


def figure_initial_and_final(data: dict) -> Figure:
    """Before and after, side by side -- the figure that goes in the paper."""
    grid = data["grid"]
    first, last = data["snapshots"][0], data["snapshots"][-1]

    fig = Figure(figsize=(11.0, 4.6), dpi=110)

    table = quantities_for(data["mode"])
    rho = table["rho"]
    velocity = table["velocity v"]

    ax = fig.add_subplot(1, 3, 1)
    FieldPanel(fig, ax, grid, first, rho, colorbar=True)
    ax.set_title(rf"{rho.title}, $t=0$", fontsize=10)

    ax = fig.add_subplot(1, 3, 2)
    FieldPanel(fig, ax, grid, last, rho, colorbar=True)
    ax.set_title(rf"{rho.title}, $t={last['time']:.2f}$", fontsize=10)

    ax = fig.add_subplot(1, 3, 3)
    FieldPanel(fig, ax, grid, last, velocity, colorbar=True)
    ax.set_title(rf"$v$, $t={last['time']:.2f}$", fontsize=10)

    fig.tight_layout()

    return fig


FIGURE_BUILDERS = {
    "initial_and_final": figure_initial_and_final,
    "overview_final": figure_overview,
    "energy_history": figure_energy_history,
    "diagnostics": figure_diagnostics,
}


def save_run_figures(data: dict, output_dir: str = "results", prefix: str | None = None) -> list:
    """Write the standard set of PNGs for a run and return the paths."""
    os.makedirs(output_dir, exist_ok=True)

    prefix = prefix or data["mode"].replace("-", "_")
    paths = []

    for name, builder in FIGURE_BUILDERS.items():
        fig = builder(data)
        path = os.path.join(output_dir, f"{prefix}_{name}.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulations for 'Incompressible Fluid Mixing as Constrained Optimal Transport'."
    )
    parser.add_argument(
        "mode",
        choices=sorted(SOLVERS),
        help="geodesic (Section 6.1) or gradient-flow (Section 7.1)",
    )
    parser.add_argument("--grid", type=int, default=128, dest="N", help="grid size N (default 128)")
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="final time (default 2 for geodesic, 75 for gradient-flow)",
    )
    parser.add_argument("--cfl", type=float, default=0.45, help="CFL number (default 0.45)")
    parser.add_argument("--snapshots", type=int, default=180, help="display snapshots (default 180)")
    parser.add_argument("--rho", type=str, default=None, help="initial rho(x, y)")
    parser.add_argument(
        "--lam", type=str, default=DEFAULT_LAMBDA_EXPR, help="initial lambda(x, y), geodesic mode only"
    )
    parser.add_argument("--out", type=str, default="results", help="output directory")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    settings = Settings(
        N=args.N,
        final_time=args.time if args.time is not None else DEFAULT_FINAL_TIME[args.mode],
        cfl=args.cfl,
        max_snapshots=args.snapshots,
        rho_expr=args.rho,
        lambda_expr=args.lam,
    )

    print(f"Running {MODE_LABELS[args.mode]} ...")
    data = SOLVERS[args.mode](settings)

    print()
    print(diagnostics_report(data))
    print()

    for path in save_run_figures(data, args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
