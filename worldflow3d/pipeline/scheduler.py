"""Flow-matching Euler scheduler, diffusers-native.

Ports ``src/models/schedulers/flow_euler_sampler.py`` to a
``SchedulerMixin``/``ConfigMixin`` with the standard ``set_timesteps`` +
``step`` contract. Classifier-free guidance is **not** done here — the pipeline
combines the conditional/unconditional velocities and passes the already-combined
``model_output`` to :meth:`step`.

Conventions (must match the research sampler exactly):
  * Internal time is the **flow time** ``t in [0, 1]`` (NOT the ``1000*t`` model
    scaling — the pipeline applies that at the model-call site).
  * Velocity prediction; integrate ``t: 1 -> 0`` with explicit Euler:
    ``x_{prev} = x - (t - t_prev) * v``.
"""
from dataclasses import dataclass

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

try:  # location differs slightly across diffusers versions
    from diffusers.schedulers.scheduling_utils import SchedulerOutput
except Exception:  # pragma: no cover
    from diffusers.utils import BaseOutput

    @dataclass
    class SchedulerOutput(BaseOutput):
        prev_sample: torch.Tensor


class WorldFlow3DFlowScheduler(SchedulerMixin, ConfigMixin):
    """Velocity-prediction flow-matching scheduler with explicit Euler steps."""

    order = 1

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 1e-4,
        num_train_timesteps: int = 1000,
        prediction_type: str = "flow_velocity",
    ):
        self.sigma_min = sigma_min
        self.num_inference_steps = None
        self._t_seq = None          # length steps+1, descending 1 -> 0
        self.timesteps = None       # length steps, flow-times t_i (NOT *1000)
        self._step_index = None

    # -- flow-time helpers ------------------------------------------------
    def sigma(self, t):
        """Noise scale at flow time ``t`` (used by add_noise / inpaint blend)."""
        return self.sigma_min + (1.0 - self.sigma_min) * t

    def _v_to_xstart_eps(self, x_t, t, v):
        """Recover (x_0, eps) from a velocity prediction (trajectory export)."""
        x_0 = (1.0 - self.sigma_min) * x_t - self.sigma(t) * v
        eps = (1.0 - t) * v + x_t
        return x_0, eps

    # -- diffusers scheduler contract -------------------------------------
    def set_timesteps(self, num_inference_steps, device=None, rescale_t: float = 1.0):
        t_seq = np.linspace(1.0, 0.0, num_inference_steps + 1)
        if rescale_t != 1.0:
            # reparameterize the whole sequence (matches research `rescale_t`)
            t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
        self._t_seq = torch.tensor(t_seq, dtype=torch.float32, device=device)
        self.timesteps = self._t_seq[:-1].clone()
        self.num_inference_steps = num_inference_steps
        self._step_index = 0

    def scale_model_input(self, sample, timestep=None):
        # flow matching: the network consumes x_t directly, no input scaling.
        return sample

    def step(self, model_output, timestep, sample, return_dict: bool = True):
        """Explicit-Euler velocity step. ``model_output`` is the post-CFG velocity."""
        if self._step_index is None:
            raise RuntimeError("call set_timesteps(...) before step(...)")
        i = self._step_index
        t = self._t_seq[i]
        t_prev = self._t_seq[i + 1]
        dt = (t - t_prev)  # > 0 while integrating 1 -> 0
        prev_sample = sample - dt * model_output
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(self, original_samples, noise, timestep):
        """Forward flow corruption ``x_t = (1-t) x_0 + sigma(t) eps``."""
        t = timestep
        return (1.0 - t) * original_samples + self.sigma(t) * noise

    def __len__(self):
        return self.config.num_train_timesteps
