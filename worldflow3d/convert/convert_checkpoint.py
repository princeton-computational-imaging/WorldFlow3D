"""Convert a research WorldFlow3D checkpoint into the diffusers layout.

Output:
    <out>/model_index.json
    <out>/unet/{config.json, diffusion_pytorch_model.safetensors}
    <out>/scheduler/scheduler_config.json

The generator weights are EMA-baked (when an ``ema_state_dict`` is present) the
same way research inference applies EMA. Architecture is taken from the research
config via ``build_generator`` / ``calculate_generator_channels`` so the converted
``WorldFlow3DUNet`` is bit-for-bit equivalent.

Run inside the research env (needs ``src`` importable). Example:
    python -m worldflow3d.convert.convert_checkpoint \
        --checkpoint-dir model_ckpts/preeccv/flow_direct_0p4_v4_big \
        --level tudf_0p4_latent --output-dir models_diffusers/streets_coarse
"""
import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_src_importable():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


def _json_safe(obj):
    """Coerce a (possibly dataclass / tuple) value into JSON-serializable form."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _levelspec_to_dict(level_spec):
    if dataclasses.is_dataclass(level_spec) and not isinstance(level_spec, type):
        return _json_safe(dataclasses.asdict(level_spec))
    # frozen dataclass / object fallback
    try:
        return _json_safe(vars(level_spec))
    except TypeError:
        return {k: _json_safe(getattr(level_spec, k))
                for k in dir(level_spec)
                if not k.startswith("_") and not callable(getattr(level_spec, k))}


def _find_generator_ckpt(checkpoint_dir, level_name, prefer="last"):
    base = os.path.join(checkpoint_dir, "checkpoints", "generator", level_name)
    order = [prefer, "best" if prefer == "last" else "last"]
    for tag in order:
        p = os.path.join(base, f"generator_{level_name}_{tag}.pt")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no generator checkpoint under {base} (tried {order})")


def _safe(getter, default=None):
    try:
        return getter()
    except Exception:
        return default


def convert_level(
    checkpoint_dir: str,
    output_dir: str,
    level_name: str = None,
    use_ema: bool = True,
    prefer: str = "last",
    sigma_min: float = 1e-4,
):
    """Convert one level. Returns (research_generator, worldflow3d_unet) for testing."""
    _ensure_src_importable()
    from src.configs.schema import HierarchicalConfig
    from src.configs.builders import (
        build_generator, calculate_generator_channels, create_level_spec,
    )
    from src.models.modules.ema_model import EMAModel
    import diffusers
    from worldflow3d.pipeline.unet import WorldFlow3DUNet
    from worldflow3d.pipeline.scheduler import WorldFlow3DFlowScheduler

    cfg = HierarchicalConfig.from_json(os.path.join(checkpoint_dir, "config.json"))
    level_names = [l.name for l in cfg.levels]
    if level_name is None:
        level_name = level_names[0]
    if level_name not in level_names:
        raise ValueError(f"level '{level_name}' not in {level_names}")
    level_config = cfg.levels[level_names.index(level_name)]
    dataset_config = cfg.dataset

    gconf = level_config.diffusion.generator
    if gconf.type != "UNet3D":
        raise NotImplementedError(
            f"converter currently supports UNet3D; got '{gconf.type}'. "
            f"(DiT/Sparse variants are future work.)"
        )

    # --- channel counts (exact, from research logic) ---
    (in_ch, out_ch, geo_ch, map_ch, outp_ch, inp_ch, micro_ch, tag_ch) = \
        calculate_generator_channels(level_config, dataset_config)

    # --- architecture-faithful research generator + (EMA-baked) weights ---
    gen, _, _ = build_generator(level_config, dataset_config, device="cpu")
    gen.eval()
    gpath = _find_generator_ckpt(checkpoint_dir, level_name, prefer=prefer)
    ckpt = torch.load(gpath, map_location="cpu")
    gen.load_state_dict(ckpt["generator_state_dict"])
    baked_ema = False
    if use_ema and ckpt.get("ema_state_dict") is not None:
        ema = EMAModel(gen.parameters())
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.copy_to(gen.parameters())
        baked_ema = True

    # --- metadata for the pipeline ---
    level_spec = create_level_spec(level_config, dataset_config)
    direct = bool(level_config.training["diffusion"].direct_diffusion)
    has_color = bool(level_config.data.get("color", False))

    # Capture the EXACT layout config the rasterizers need (channel count + height
    # handling must match training — e.g. this street model uses include_subtypes=True).
    layout_cfg_meta = None
    if map_ch > 0:
        lc = _safe(lambda: dataset_config.ensure_layout_config())
        if lc is not None:
            layout_cfg_meta = {
                "dataset_type": getattr(lc, "dataset_type", getattr(dataset_config, "type", None)),
                "types": list(getattr(lc, "types", [])),
                "include_vehicles": bool(getattr(lc, "include_vehicles", True)),
                "include_subtypes": bool(getattr(lc, "include_subtypes", False)),
                "preserve_height": bool(getattr(lc, "preserve_height", True)),
            }
    level_meta = {
        "level_name": level_name,
        "generator_type": gconf.type,
        "direct_diffusion": direct,
        "use_separate_color_head": direct and has_color,
        "initial_distribution": getattr(level_config, "initial_distribution", "noise"),
        "truncation": float(getattr(dataset_config, "truncation", 1.0)),
        "dataset_type": getattr(dataset_config, "type", None),
        "is_coarsest": bool(getattr(level_config, "is_coarsest", level_name == level_names[0])),
        "voxel_size": float(level_spec.output_voxel_size),
        "factor": int(level_spec.factor),
        "output_chunk_size": [int(v) for v in level_spec.output_chunk_size],
        "H_channels": int(level_spec.H_channels),
        "L_channels": int(level_spec.L_channels),
        "use_color": bool(_safe(lambda: level_spec.use_color, False)),
        "source_noise_std": float(getattr(level_config, "source_noise_std", 0.0) or 0.0),
        "has_layout": bool(getattr(level_config, "has_layout_conditioning", map_ch > 0)),
        "layout_types": (layout_cfg_meta or {}).get("types"),
        "layout_config": layout_cfg_meta,
        "has_tags": tag_ch > 0,
        "tag_types": _safe(lambda: list(dataset_config.tag_config.types)),
        "channels": {
            "in": in_ch, "out": out_ch, "geometry": geo_ch, "map": map_ch,
            "outpainting": outp_ch, "inpainting": inp_ch, "micro": micro_ch, "tag": tag_ch,
        },
    }

    # --- build the diffusers-native UNet with identical kwargs ---
    unet = WorldFlow3DUNet(
        in_channels=in_ch, out_channels=out_ch,
        geometry_channels=geo_ch, map_channels=map_ch,
        outpainting_channels=outp_ch, inpainting_channels=inp_ch,
        micro_channels=micro_ch, tag_channels=tag_ch,
        model_channels=gconf.model_channels,
        num_res_blocks=gconf.num_res_blocks,
        attention_resolutions=list(gconf.attention_resolutions),
        channel_mult=list(gconf.channel_mult),
        dropout=getattr(gconf, "dropout", 0.0),
        num_heads=gconf.num_heads,
        num_head_channels=-1,
        use_scale_shift_norm=getattr(gconf, "use_scale_shift_norm", False),
        use_fp16=getattr(gconf, "use_fp16", False),
        use_separate_color_head=direct and has_color,
        level_spec=_levelspec_to_dict(level_spec),
        level_meta=level_meta,
    )
    # Architecture-match guarantee: weights from the research generator load strictly.
    missing, unexpected = unet.model.load_state_dict(gen.state_dict(), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch converting {level_name}:\n  missing={missing}\n  unexpected={unexpected}"
        )

    # --- write diffusers layout ---
    os.makedirs(output_dir, exist_ok=True)
    unet.save_pretrained(os.path.join(output_dir, "unet"))
    WorldFlow3DFlowScheduler(sigma_min=sigma_min).save_pretrained(
        os.path.join(output_dir, "scheduler")
    )
    # diffusers 0.29.x resolves a CUSTOM pipeline class only via the LIST form of
    # _class_name (["<module_file>", "<ClassName>"]) + the remote .py placed in the
    # repo. The list form + remote mirrors makes BOTH WorldFlow3DPipeline.from_pretrained
    # and DiffusionPipeline.from_pretrained(trust_remote_code=True) work.
    model_index = {
        "_class_name": ["worldflow3d_pipeline", "WorldFlow3DPipeline"],
        "_diffusers_version": diffusers.__version__,
        "unet": ["worldflow3d_unet", "WorldFlow3DUNet"],
        "scheduler": ["worldflow3d_scheduler", "WorldFlow3DFlowScheduler"],
        "vae": [None, None],
        # NOTE: level_meta is NOT placed here (diffusers would warn "unexpected
        # kwarg"); the authoritative copy lives in unet/config.json (unet.config.level_meta).
    }
    with open(os.path.join(output_dir, "model_index.json"), "w") as f:
        json.dump(_json_safe(model_index), f, indent=2)

    # Stage the trust_remote_code mirrors (root pipeline + per-component .py) so the
    # converted dir loads via from_pretrained without needing from_converted().
    import shutil as _shutil
    from worldflow3d._remote import REMOTE_FILES, remote_file_source
    for _src_name, _rel_dest in REMOTE_FILES.items():
        _dst = os.path.join(output_dir, _rel_dest)
        os.makedirs(os.path.dirname(_dst) or output_dir, exist_ok=True)
        _shutil.copy2(remote_file_source(_src_name), _dst)

    print(f"[convert] {level_name}: in={in_ch} out={out_ch} map={map_ch} geo={geo_ch} "
          f"ema_baked={baked_ema} -> {output_dir}")
    return gen, unet


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--level", default=None, help="level name (default: coarsest)")
    ap.add_argument("--no-ema", action="store_true", help="do not bake EMA weights")
    ap.add_argument("--prefer", default="last", choices=["last", "best"])
    args = ap.parse_args()
    convert_level(args.checkpoint_dir, args.output_dir, level_name=args.level,
                  use_ema=not args.no_ema, prefer=args.prefer)


if __name__ == "__main__":
    main()
