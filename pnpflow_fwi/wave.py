"""Acoustic wave solver and multi-shot data misfit (JAX).

The domain is a rectangle on an ny x nx grid, periodic in x, with u_y = 0 on the bottom row and
u_y = q(x, t) on the top row, where q is the source flux. Receivers are the bottom-row grid points.
"""
import jax
import jax.numpy as jnp
from jax import lax


def laplacian(u, c, dx, dy, q_top):
    """c^2 * Laplacian(u), with ghost cells for the two Neumann rows."""
    ny = u.shape[0]
    d2x = (jnp.roll(u, -1, axis=1) - 2.0 * u + jnp.roll(u, 1, axis=1)) / dx**2
    d2y = (jnp.roll(u, -1, axis=0) - 2.0 * u + jnp.roll(u, 1, axis=0)) / dy**2
    d2y = d2y.at[0, :].set(2.0 * (u[1, :] - u[0, :]) / dy**2)
    d2y = d2y.at[ny - 1, :].set(2.0 * (u[ny - 2, :] - u[ny - 1, :]) / dy**2 + 2.0 * q_top / dy)
    return (c**2) * (d2x + d2y)


def forward(c, q_top, dx, dy, dt):
    """Wavefield (nt, ny, nx) of u_tt = c^2 Laplacian(u) from rest, second order in time; q_top is (nt, nx)."""
    u0 = jnp.zeros_like(c)
    u1 = u0 + 0.5 * dt**2 * laplacian(u0, c, dx, dy, q_top[0])

    def step(state, inputs):
        c, q, dx, dy, dt = inputs
        u_prev, u_curr = state
        u_next = 2.0 * u_curr - u_prev + dt**2 * laplacian(u_curr, c, dx, dy, q)
        return (u_curr, u_next), u_next

    # The speed field and the spacings are handed to every time step as per-step inputs. The reported runs
    # were computed this way, and the rounding of the gradient depends on it, so the arrangement is kept.
    steps = q_top.shape[0] - 2
    per_step = (jnp.broadcast_to(c, (steps,) + c.shape), q_top[1:-1]) + tuple(jnp.full((steps,), v) for v in (dx, dy, dt))
    _, u_rest = lax.scan(step, (u0, u1), per_step)
    return jnp.concatenate([u0[None], u1[None], u_rest], axis=0)


def receivers(u):
    """Bottom-row field at every second time level: (nt // 2, nx)."""
    return u[::2, 0, :]


def misfit_and_grad(sources, data, dx, dy, dt):
    """Return c -> (J(c), grad J(c)) with J(c) = 0.5 * sum over shots ||receivers(u(c)) - data||^2.

    sources: (S, nt, nx) top-boundary flux of each shot; data: (S, nt // 2, nx) observed receiver records.
    """
    def shot_misfit(c, q, d):
        return 0.5 * jnp.sum((receivers(forward(c, q, dx, dy, dt)) - d) ** 2)

    def loss(c):
        return jnp.sum(jax.vmap(shot_misfit, in_axes=(None, 0, 0))(c, sources, data))

    return jax.jit(jax.value_and_grad(loss))
