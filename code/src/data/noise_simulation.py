"""
Foi (2008) Poisson-Gaussian noise model for CXR degradation simulation.

Var(z|y) = a*y + b

The 'a' term (Poisson) models photon shot noise, which scales with
signal intensity. The 'b' term (Gaussian) models read-out and electronic
noise, which is signal-independent. Both are present in clinical X-ray
acquisition. Pure Gaussian-only noise models (used by DnCNN, SwinIR, etc.)
omit the 'a' term -- this is physically incorrect for X-ray and is one
motivation for using the Foi model.

Reference
---------
Foi, A. et al. (2008). Practical Poissonian-Gaussian noise modelling and
    fitting for single-image raw-data. IEEE TIP 17(10), 1737-1754.
"""

from __future__ import annotations
import torch
import numpy as np


PRESETS: dict[str, dict[str, float]] = {
    "low":  {"a": 0.01, "b": 0.002},  # ~sigma_15 equivalent
    "mid":  {"a": 0.03, "b": 0.005},  # ~sigma_25 equivalent
    "high": {"a": 0.08, "b": 0.010},  # ~sigma_50 equivalent
}


def add_foi_noise(
    x: torch.Tensor,
    a: float,
    b: float,
    clip: bool = True,
) -> torch.Tensor:
    """
    Apply Foi Poisson-Gaussian noise to a clean image tensor.

    x must be in [0, 1]. The Poisson term is simulated via the
    Bartlett-Anscombe transform approximation for efficiency:
    instead of sampling true Poisson values, we draw Gaussian noise
    with signal-dependent variance a*x + b, which is exact in
    expectation and accurate for the intensity ranges typical of CXR.

    Parameters
    ----------
    x   : Tensor [..., H, W], values in [0, 1]
    a   : Poisson variance coefficient
    b   : Gaussian variance floor
    clip: clip output to [0, 1]

    Returns
    -------
    z : noisy image, same shape as x
    """
    variance = a * x.clamp(min=0.0) + b
    noise = torch.randn_like(x) * variance.sqrt()
    z = x + noise
    if clip:
        z = z.clamp(0.0, 1.0)
    return z


def add_foi_noise_preset(
    x: torch.Tensor,
    level: str = "mid",
    clip: bool = True,
) -> torch.Tensor:
    """Convenience wrapper using named presets."""
    params = PRESETS[level]
    return add_foi_noise(x, a=params["a"], b=params["b"], clip=clip)


def effective_sigma(x_mean: float, a: float, b: float) -> float:
    """Effective noise std at a given mean intensity (for reporting)."""
    return float(np.sqrt(a * x_mean + b))
