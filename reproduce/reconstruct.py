"""Reconstruct one target with PnP-Flow or TV-FWI and save the result (needs a GPU for the reported runtimes)."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import START, checkpoint_path, settings, target  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pnpflow_fwi.inversion import pnp_flow, tv_fwi  # noqa: E402
from pnpflow_fwi.prior import FlowPrior, load_prior  # noqa: E402
from pnpflow_fwi.wave import forward, misfit_and_grad, receivers  # noqa: E402

NT = 1000  # time levels
SHOTS, F0 = 12, 3.0  # plane-wave shots with Ricker peak frequencies F0, F0 + 1, ..., F0 + SHOTS - 1


def ricker(t, f0):
    t0 = 1.5 / f0
    x = jnp.pi * f0 * (t - t0)
    return (1.0 - 2.0 * x**2) * jnp.exp(-x**2)


def acquisition(c_true):
    """Noiseless data of the plane-wave experiment on the unit square; returns the misfit-and-gradient function.

    Each shot is a Ricker flux applied uniformly along the top boundary; the 63 bottom points record every
    second of NT time levels. The time step is 0.4 / sqrt(2) times the CFL step of the target's maximum speed.
    """
    ny, nx = c_true.shape
    dx, dy = 1.0 / nx, 1.0 / (ny - 1)
    dt = 0.4 * min(dx, dy) / (jnp.sqrt(2.0) * float(jnp.max(c_true)))
    t = jnp.arange(NT) * dt
    sources, data = [], []
    for k in range(SHOTS):
        q = jnp.tile(ricker(t, F0 + k)[:, None], (1, nx))
        sources.append(q)
        data.append(receivers(forward(c_true, q, dx, dy, float(dt))))
    return misfit_and_grad(jnp.stack(sources), jnp.stack(data), dx, dy, float(dt))


def reconstruct(cohort, method, image, seed, output):
    s = settings(cohort, method, image, seed)
    print(json.dumps(s), flush=True)
    c_true = jnp.asarray(target(cohort, image))
    loss_and_grad = acquisition(c_true)
    c_init = jnp.full_like(c_true, START)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    started = time.perf_counter()
    if method == 'pnp':
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        prior = FlowPrior(load_prior(checkpoint_path(s['prior']), device), device)
        c_est = pnp_flow(c_init, loss_and_grad, prior, blocks=s['blocks'], block_size=s['block_size'],
                         step=s['step'], t_start=s['t_start'], proposals=s['proposals'])
    else:
        c_est = tv_fwi(c_init, loss_and_grad, updates=s['updates'], weight=s['weight'], step=s['step'])
    c_est = np.asarray(jnp.asarray(c_est).block_until_ready())
    seconds = time.perf_counter() - started
    print(f'done in {seconds:.1f} s; relative L2 error {np.linalg.norm(c_est - c_true) / np.linalg.norm(c_true):.4f}')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, c_true=np.asarray(c_true), c_est=c_est, settings=json.dumps(s, sort_keys=True),
             device=f'{jax.devices()[0].device_kind}; {torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}',
             seconds=seconds)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort', required=True, choices=['flat', 'gauss', 'breast'])
    ap.add_argument('--method', required=True, choices=['pnp', 'tv'])
    ap.add_argument('--image', required=True, type=int)
    ap.add_argument('--seed', type=int, help='PnP-Flow seed (torch random state; ignored for TV-FWI)')
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if args.method == 'pnp' and args.seed is None:
        ap.error('--seed is required for PnP-Flow')
    reconstruct(args.cohort, args.method, args.image, args.seed if args.method == 'pnp' else None, args.output)


if __name__ == '__main__':
    main()
