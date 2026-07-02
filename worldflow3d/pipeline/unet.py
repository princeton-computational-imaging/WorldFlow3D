"""Diffusers-native wrapper around the vendored UNet3D torso.

``WorldFlow3DUNet`` is a ``ModelMixin``/``ConfigMixin`` so it round-trips through
``save_pretrained``/``from_pretrained`` (safetensors) without the research config
system. It composes the vendored :class:`worldflow3d.models.unet3d.UNetModel` as
``self.model`` — so all weight keys are prefixed ``model.`` — and delegates the
forward, which owns all conditioning concatenation (layout/tags/coarse_geometry/
outpaint/inpaint/micro) and the ``geometry``/``map`` FiLM embeds.

``in_channels`` is the **sample (H) channel count**; the conditioning channel
counts (``geometry_channels``, ``map_channels``, ...) are added *internally* by the
torso to form the first-conv input.
"""
from dataclasses import dataclass

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput

from ..models.unet3d import UNetModel


@dataclass
class WorldFlow3DUNetOutput(BaseOutput):
    sample: torch.Tensor


class WorldFlow3DUNet(ModelMixin, ConfigMixin):
    """Velocity-prediction 3D UNet for hierarchical voxel flow matching."""

    @register_to_config
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        # conditioning channel counts (resolved at convert time)
        geometry_channels: int = 0,
        map_channels: int = 0,
        outpainting_channels: int = 0,
        inpainting_channels: int = 0,
        micro_channels: int = 0,
        tag_channels: int = 0,
        # architecture
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions=(16, 8, 4),
        channel_mult=(1, 2, 4, 8),
        dropout: float = 0.0,
        num_heads: int = 8,
        num_head_channels: int = -1,
        use_scale_shift_norm: bool = False,
        use_fp16: bool = False,
        use_separate_color_head: bool = False,
        # serialized LevelSpec + per-level inference metadata (carried for the
        # pipeline; not used in forward).
        level_spec: dict = None,
        level_meta: dict = None,
    ):
        super().__init__()
        self.model = UNetModel(
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=list(attention_resolutions),
            channel_mult=list(channel_mult),
            dropout=dropout,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            use_fp16=use_fp16,
            geometry_channels=geometry_channels,
            map_channels=map_channels,
            outpainting_channels=outpainting_channels,
            micro_channels=micro_channels,
            tag_channels=tag_channels,
            use_separate_color_head=use_separate_color_head,
            level_spec=None,
            inpainting_channels=inpainting_channels,  # consumed via UNetModel **kwargs
        )

    def forward(self, sample, timestep, cond=None, coarse_geometry=None,
                return_dict: bool = True):
        """``timestep`` is the model-space time (= 1000*t); the pipeline scales it."""
        out = self.model(sample, timesteps=timestep, cond=cond,
                         coarse_geometry=coarse_geometry, context=None)
        if not return_dict:
            return (out,)
        return WorldFlow3DUNetOutput(sample=out)
