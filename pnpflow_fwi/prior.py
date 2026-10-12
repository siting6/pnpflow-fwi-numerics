"""Flow-matching prior: checkpoint loading and the one-step PnP-Flow denoiser."""
import jax
import jax.numpy as jnp
import numpy as np
import torch

from .unet import UNet2D

C_MIN, C_MAX = 4.0, 12.0  # the network works on fields mapped linearly from [C_MIN, C_MAX] to [-1, 1]


def load_prior(path, device):
    """Build the U-Net described in the checkpoint and load its EMA weights."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    arch = checkpoint['config']['model']
    unet = UNet2D(base_channels=arch['base_channels'], channel_mults=arch['channel_mults'],
                  num_res_blocks=arch['num_res_blocks'], time_embed_dim=arch['time_embed_dim'],
                  num_heads=arch['num_heads'])
    unet.load_state_dict({key[len('model.'):]: value for key, value in checkpoint['ema']['shadow'].items()})
    return unet.to(device).eval()


class FlowPrior:
    """One-step denoiser D_t(x) = z + (1 - t) v(z, t), z = (1 - t) eps + t x, eps ~ N(0, I), on the training grid.

    The network was trained on ny x (nx + 1) fields whose last column repeats the first (periodic x);
    the solver works on the ny x nx grid, so the column is appended before and removed after denoising.
    """

    def __init__(self, unet, device):
        self.unet, self.device = unet, device

    def __call__(self, c, t):
        c = np.array(jax.device_get(jnp.concatenate([c, c[:, :1]], axis=1)), dtype=np.float32)
        x = torch.from_numpy(2.0 * (c - C_MIN) / (C_MAX - C_MIN) - 1.0).float().to(self.device)
        eps = torch.randn(x.shape, device=self.device)
        z = ((1.0 - t) * eps + t * x).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            flow_time = torch.ones(z.shape[0], device=z.device) * float(t)
            out = z + (1.0 - t) * self.unet(z, flow_time)
        out = 0.5 * (out[0, 0].cpu().numpy() + 1.0) * (C_MAX - C_MIN) + C_MIN
        return jnp.array(out.astype(np.float32))[:, :-1]
