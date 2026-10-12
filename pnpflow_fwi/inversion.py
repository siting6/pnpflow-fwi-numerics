"""The two inversions compared in the tables: PnP-Flow and TV-FWI."""
import jax
import jax.numpy as jnp

CLIP, BETA1, BETA2, EPS = 10.0, 0.9, 0.999, 1.0e-8
TV_EPS = 1.0e-4
DIVERGENCE_MAX_ABS, DIVERGENCE_GROWTH = 50.0, 100.0


def clipped_adam(x, m, v, grad, lr):
    """One Adam update without bias correction, after clipping the gradient to L2 norm CLIP."""
    norm = jnp.sqrt(jnp.sum(grad * grad))
    grad = grad * jnp.minimum(1.0, CLIP / (norm + 1e-12))
    m = BETA1 * m + (1.0 - BETA1) * grad
    v = BETA2 * v + (1.0 - BETA2) * grad * grad
    return x - lr * m / (jnp.sqrt(v) + EPS), m, v


def pnp_flow(c_init, loss_and_grad, prior, *, blocks, block_size, step, t_start, proposals, log=print):
    """PnP-Flow reconstruction.

    Each of the `blocks` blocks runs `block_size` data updates with step size step * (1 - t) and Adam
    moments reset to zero, while the flow time t advances linearly from t_start to 1, then replaces
    the iterate by the mean of `proposals` prior updates prior(x, t). A run that diverges is aborted.
    """
    dt = (1.0 - t_start) / blocks / block_size

    @jax.jit
    def data_block(x, t):
        def body(_, state):
            x, t, m, v = state
            _, grad = loss_and_grad(x)
            x, m, v = clipped_adam(x, m, v, grad, step * (1.0 - t))
            return x, jnp.minimum(t + dt, 1.0), m, v

        return jax.lax.fori_loop(0, block_size, body, (x, t, jnp.zeros_like(x), jnp.zeros_like(x)))[:2]

    x, t = jnp.asarray(c_init), t_start
    for n in range(blocks):
        x, t = data_block(x, t)
        x = sum(prior(x, float(t)) for _ in range(proposals)) / proposals
        loss = float(loss_and_grad(x)[0])
        if n == 0:
            first_loss = loss
        if not loss < DIVERGENCE_GROWTH * first_loss or float(jnp.max(jnp.abs(x))) > DIVERGENCE_MAX_ABS:
            raise RuntimeError(f'PnP-Flow diverged in block {n}: loss {loss:.3e}')
        log(f'block {n + 1}/{blocks}  t={float(t):.4f}  data misfit={loss:.4e}')
    return x


def tv_fwi(c_init, loss_and_grad, *, updates, weight, step, log=print):
    """TV-FWI reconstruction: `updates` clipped Adam steps, moments carried, on J(c) + weight * TV(c).

    TV is the smoothed isotropic total variation sum sqrt(gx^2 + gy^2 + TV_EPS) with forward differences
    and a replicated far boundary.
    """
    def tv(c):
        gx = jnp.diff(c, axis=1, append=c[:, -1:])
        gy = jnp.diff(c, axis=0, append=c[-1:, :])
        return jnp.sum(jnp.sqrt(gx * gx + gy * gy + TV_EPS))

    @jax.jit
    def update(c, m, v):
        loss, grad = loss_and_grad(c)
        tv_value, tv_grad = jax.value_and_grad(tv)(c)
        c, m, v = clipped_adam(c, m, v, grad + weight * tv_grad, step)
        return c, m, v, loss, tv_value

    c = jnp.asarray(c_init)
    m, v = jnp.zeros_like(c), jnp.zeros_like(c)
    for i in range(1, updates + 1):
        c, m, v, loss, tv_value = update(c, m, v)
        if i % 100 == 0 or i == updates:
            log(f'update {i}/{updates}  data misfit={float(loss):.4e}  TV={float(tv_value):.4e}')
    return c
