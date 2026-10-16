"""Portable MiniMax H3 AV archives; never pickle or apply latent scaling."""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
import re
import tempfile
import time

import comfy.nested_tensor
import folder_paths
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .nodes_h3_latent_checkpoint import _checkpoint_path, _h3_av_streams, _streams


def _file_path(file_path):
    text = str(file_path).strip().strip('"')
    if text.endswith(("[input]", "[temp]", "[output]")):
        return Path(folder_paths.get_annotated_filepath(text))
    return Path(text).expanduser().resolve() if text else Path(_checkpoint_path())


def _validate(latent):
    video, audio = _h3_av_streams(latent, "portable AV latent")
    if video.shape[1] != 24 or audio.shape[2] != 2:
        raise ValueError("H3 latent file: expected 24 video channels and stereo audio latent.")
    if any(size < 1 for t in (video, audio) for size in t.shape):
        raise ValueError("H3 latent file: empty tensors are not supported.")
    if not all(t.is_floating_point() for t in (video, audio)):
        raise ValueError("H3 latent file: AV samples must be floating-point tensors.")
    if not all(torch.isfinite(t).all().item() for t in (video, audio)):
        raise ValueError("H3 latent file: AV samples contain NaN or infinity.")
    return video, audio


class NegativeMeta_H3LatentFileSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT", {"lazy": True, "tooltip": "Raw base H3 AV latent, before RGB color correction/upscale."}),
            "skip_save": ("BOOLEAN", {"default": False, "forceInput": True, "tooltip": "Effective Upscale Last Job / resume flag; preserves the temporary file on resume."}),
            "seed_value": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "forceInput": True, "tooltip": "Actual seed of the saved generation."}),
            "filename_prefix": ("STRING", {"default": "h3_clip", "tooltip": "Download file label. Each fresh clip replaces this node's temporary latent; click Download Latent to keep it."}),
        }, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "NegativeMeta/Toolbox"
    DESCRIPTION = "Always saves fresh native H3 AV samples temporarily. Download Latent keeps a copy; no automatic per-clip archives or VAE round trip."

    @classmethod
    def check_lazy_status(cls, skip_save=False, latent=None, **kwargs):
        return ["latent"] if not skip_save and latent is None else []

    def save(self, skip_save=False, seed_value=0, filename_prefix="h3_clip", latent=None, unique_id=None):
        if skip_save:
            return ("",)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", filename_prefix):
            raise ValueError("H3 latent file: filename_prefix must use 1–100 letters, numbers, underscores or hyphens (no paths).")
        video, audio = _validate(latent)
        payload = {"video": video.detach().cpu().contiguous(), "audio": audio.detach().cpu().contiguous()}
        masks = _streams(latent.get("noise_mask"), "noise_mask")
        if masks is not None:
            for name, mask, sample in zip(("video_noise_mask", "audio_noise_mask"), masks, (video, audio)):
                if mask.ndim != sample.ndim or any(m not in (1, s) for m, s in zip(mask.shape, sample.shape)):
                    raise ValueError("H3 latent file: noise mask is not broadcast-compatible with samples.")
                payload[name] = mask.detach().cpu().contiguous()
        subfolder = "negative_meta_h3_latents"
        directory = Path(folder_paths.get_temp_directory()) / subfolder
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        key = hashlib.sha256(str(unique_id or "default").encode()).hexdigest()[:12]
        path = directory / f"h3_{key}_latest.safetensors"
        # Only replace the previous temporary after a complete successful write.
        fd, temporary = tempfile.mkstemp(prefix=".h3_temporary_", suffix=".tmp", dir=directory)
        os.close(fd)
        try:
            save_file(payload, temporary, metadata={
                "format": "negativemeta_h3_av_checkpoint_v1", "noise_seed": str(int(seed_value)),
                "saved_at_ns": str(time.time_ns()), "fps": "24", "audio_latent_hz": "40",
                "video_shape": str(list(video.shape)), "audio_shape": str(list(audio.shape)),
                "stage": "raw_base_av", "archive_kind": "sampled_window_not_cumulative",
            })
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(f"[NegativeMeta] Saved temporary H3 AV latent: {path} (seed={int(seed_value)})")
        descriptor = {"filename": path.name, "subfolder": subfolder, "type": "temp",
                      "download_name": f"{filename_prefix}_{stamp}_{int(seed_value)}.safetensors"}
        return {"ui": {"text": [str(path)], "h3_latent_file": [descriptor]}, "result": (str(path),)}


class NegativeMeta_H3LatentFileLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"file_path": ("STRING", {"default": "", "multiline": False,
            "tooltip": "Use Select Latent File to browse/upload a saved AV .safetensors, or enter an absolute path. Empty uses the latest checkpoint."})}}

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("latent", "noise_seed", "loaded_path")
    FUNCTION = "load"
    CATEGORY = "NegativeMeta/Toolbox"
    DESCRIPTION = "Loads a chosen native H3 AV latent file for Hard Latent continuation. Empty path falls back to the existing last checkpoint."

    @classmethod
    def IS_CHANGED(cls, file_path=""):
        path = _file_path(file_path)
        try:
            stat = path.stat()
            return (str(path), stat.st_size, stat.st_mtime_ns)
        except FileNotFoundError:
            return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, file_path=""):
        # Only open/check the file when the lazy continuation branch is selected.
        return True

    def load(self, file_path=""):
        path = _file_path(file_path)
        if path.suffix.lower() != ".safetensors":
            raise ValueError("H3 latent file: only AV .safetensors files are supported, not pickle/.pt or generic image .latent files.")
        if not path.is_file():
            raise FileNotFoundError(f"H3 latent file not found: {path}. Choose an archive or run one fresh clip first.")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            keys = set(handle.keys())
        if metadata.get("format") != "negativemeta_h3_av_checkpoint_v1" or not {"video", "audio"} <= keys:
            raise ValueError("H3 latent file: incompatible format; expected a NegativeMeta H3 video+audio checkpoint.")
        if ("video_noise_mask" in keys) != ("audio_noise_mask" in keys):
            raise ValueError("H3 latent file: incomplete AV noise masks.")
        data = load_file(str(path), device="cpu")
        latent = {"samples": comfy.nested_tensor.NestedTensor((data["video"], data["audio"]))}
        _validate(latent)
        if "video_noise_mask" in data:
            masks = (data["video_noise_mask"], data["audio_noise_mask"])
            for mask, sample in zip(masks, (data["video"], data["audio"])):
                if mask.ndim != sample.ndim or any(m not in (1, s) for m, s in zip(mask.shape, sample.shape)):
                    raise ValueError("H3 latent file: incompatible noise mask shape.")
            latent["noise_mask"] = comfy.nested_tensor.NestedTensor(masks)
        seed = int(metadata.get("noise_seed", "0"))
        print(f"[NegativeMeta] Loaded H3 AV latent file: {path}")
        return (latent, seed, str(path))
