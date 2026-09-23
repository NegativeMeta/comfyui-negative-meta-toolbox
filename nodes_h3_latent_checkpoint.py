"""Persistent checkpoints for MiniMax H3 nested audio/video latents.

This module is intentionally independent from ComfyUI-DaSiWa-Nodes. It only
provides the checkpoint nodes used by the NegativeMeta workflow.
"""

from __future__ import annotations

import os
import time

import comfy.nested_tensor
import folder_paths
from safetensors import safe_open
from safetensors.torch import load_file, save_file


_CHECKPOINT_FILENAME = "negativemeta_h3_last_latent.safetensors"


def _checkpoint_path() -> str:
    directory = os.path.join(folder_paths.get_output_directory(), "latents")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, _CHECKPOINT_FILENAME)


def _streams(value, label: str):
    if value is None:
        return None
    if hasattr(value, "unbind"):
        streams = value.unbind()
    elif isinstance(value, (list, tuple)):
        streams = value
    else:
        raise ValueError(f"NegativeMeta H3 checkpoint: {label} is not an AV latent.")
    if len(streams) != 2:
        raise ValueError(f"NegativeMeta H3 checkpoint: {label} must contain video and audio streams.")
    return streams


class NegativeMeta_H3LatentCheckpointSave:
    """Persist the latest H3 AV latent without evaluating it during resume mode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resume_last_latent": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, keep the existing checkpoint and skip the fresh latent branch.",
                }),
                "lbh_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Kept for workflow compatibility; the latest base latent is saved regardless of this value.",
                }),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "latent": ("LATENT", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("checkpoint_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "NegativeMeta/Toolbox"
    DESCRIPTION = "Overwrites the latest NegativeMeta H3 latent checkpoint after every fresh base sampling pass."

    @classmethod
    def check_lazy_status(cls, resume_last_latent, lbh_enabled, noise_seed=0, latent=None):
        if resume_last_latent:
            return []
        if latent is None:
            return ["latent"]
        return []

    def save(self, resume_last_latent, lbh_enabled, noise_seed, latent=None):
        path = _checkpoint_path()
        if resume_last_latent:
            print(f"[NegativeMeta] Resume active; checkpoint preserved: {path}")
            return (path,)
        if latent is None:
            raise ValueError("NegativeMeta H3 checkpoint: fresh latent was not provided.")

        samples = _streams(latent.get("samples"), "samples")
        payload = {
            "video": samples[0].detach().cpu().contiguous(),
            "audio": samples[1].detach().cpu().contiguous(),
        }
        masks = _streams(latent.get("noise_mask"), "noise_mask")
        if masks is not None:
            payload["video_noise_mask"] = masks[0].detach().cpu().contiguous()
            payload["audio_noise_mask"] = masks[1].detach().cpu().contiguous()

        temporary_path = f"{path}.tmp"
        save_file(
            payload,
            temporary_path,
            metadata={
                "format": "negativemeta_h3_av_checkpoint_v1",
                "noise_seed": str(int(noise_seed)),
                "saved_at_ns": str(time.time_ns()),
            },
        )
        os.replace(temporary_path, path)
        print(f"[NegativeMeta] Saved latest H3 latent: {path} (seed={int(noise_seed)})")
        return (path,)


class NegativeMeta_H3LatentCheckpointLoad:
    """Load the latest NegativeMeta H3 AV checkpoint as a nested LATENT."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "noise_seed")
    FUNCTION = "load"
    CATEGORY = "NegativeMeta/Toolbox"
    DESCRIPTION = "Loads the most recently saved NegativeMeta H3 AV latent checkpoint."

    @classmethod
    def IS_CHANGED(cls):
        try:
            return os.stat(_checkpoint_path()).st_mtime_ns
        except FileNotFoundError:
            return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls):
        # This node sits behind a lazy switch. A missing checkpoint should only
        # fail when resume mode actually selects it.
        return True

    def load(self):
        path = _checkpoint_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "NegativeMeta H3 checkpoint: no saved latent found. "
                "Run once with Resume Last Latent disabled."
            )
        data = load_file(path, device="cpu")
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        if "video" not in data or "audio" not in data:
            raise ValueError("NegativeMeta H3 checkpoint: invalid checkpoint file.")

        latent = {
            "samples": comfy.nested_tensor.NestedTensor((data["video"], data["audio"])),
        }
        if "video_noise_mask" in data and "audio_noise_mask" in data:
            latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (data["video_noise_mask"], data["audio_noise_mask"])
            )
        try:
            noise_seed = int(metadata.get("noise_seed", "0"))
        except ValueError:
            noise_seed = 0
        return (latent, noise_seed)
