"""Persistent checkpoints for MiniMax H3 nested audio/video latents.

This module is intentionally independent from ComfyUI-DaSiWa-Nodes. It only
provides the checkpoint nodes used by the NegativeMeta workflow.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import time

import comfy.nested_tensor
import comfy.utils
import folder_paths
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


_CHECKPOINT_FILENAME = "negativemeta_h3_last_latent.safetensors"
_CLIP_PREFIX_CACHE_FORMAT = "negativemeta_h3_clip_prefix_v1"
_H3_FPS = 24.0
_H3_AUDIO_HZ = 40.0


def _checkpoint_path() -> str:
    directory = os.path.join(folder_paths.get_output_directory(), "latents")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, _CHECKPOINT_FILENAME)


def _clip_prefix_cache_directory() -> str:
    directory = os.path.join(
        folder_paths.get_output_directory(),
        "latents",
        "negativemeta_h3_clip_prefix_cache",
    )
    os.makedirs(directory, exist_ok=True)
    return directory


def _ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
        if executable and os.path.isfile(executable):
            return executable
    except Exception:
        pass
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    raise RuntimeError(
        "NegativeMeta H3 clip tail: ffmpeg was not found. Install FFmpeg or imageio-ffmpeg."
    )


def _video_path(video) -> str | None:
    """Resolve a core VideoFromFile path without asking it to decode components."""
    try:
        source = video.get_stream_source()
    except Exception:
        source = getattr(video, "_VideoFromFile__file", None)
    if isinstance(source, os.PathLike):
        source = os.fspath(source)
    if isinstance(source, str) and os.path.isfile(source):
        return os.path.abspath(source)
    return None


def _h3_context_boundary(requested: int, available: int) -> int:
    """Largest shared 24 fps video / 40 Hz audio boundary within ``available``."""
    cap = min(int(requested), int(available))
    if cap < 39:
        raise ValueError(
            "NegativeMeta H3 clip tail: at least 39 source frames are required for "
            "an exact audio/video continuation boundary."
        )
    return 39 + 51 * ((cap - 39) // 51)


def _tail_duration(context_frames: int) -> float:
    return float(int(context_frames)) / _H3_FPS


def _h3_audio_steps(context_frames: int) -> int:
    exact = int(context_frames) * 5
    if exact % 3:
        raise ValueError(
            "NegativeMeta H3 clip cache: context must be an exact 24 fps / 40 Hz boundary."
        )
    return exact // 3


def _load_tail_with_ffmpeg(video, context_frames: int):
    """Decode only a file-backed video's final H3 context frames and audio."""
    path = _video_path(video)
    if not path:
        raise ValueError(
            "NegativeMeta H3 clip tail needs a file-backed core LoadVideo input. "
            "Use the workflow's OPTIONAL Load continuation clip node."
        )
    try:
        width, height = video.get_dimensions()
    except Exception as exc:
        raise RuntimeError(
            f"NegativeMeta H3 clip tail could not read source dimensions: {exc}"
        ) from exc
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError("NegativeMeta H3 clip tail: source video has invalid dimensions.")

    duration = _tail_duration(context_frames)
    executable = _ffmpeg_executable()
    video_args = [
        executable,
        "-v", "error",
        "-sseof", f"-{duration:.9f}",
        "-i", path,
        "-map", "0:v:0",
        "-vf", "fps=24",
        "-frames:v", str(int(context_frames)),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    video_result = subprocess.run(video_args, capture_output=True, check=False)
    if video_result.returncode:
        detail = video_result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"NegativeMeta H3 clip tail: ffmpeg video extraction failed: {detail}")
    pixels_per_frame = width * height * 3
    raw_video = np.frombuffer(video_result.stdout, dtype=np.uint8)
    frame_count = raw_video.size // pixels_per_frame
    if frame_count < int(context_frames):
        raise ValueError(
            "NegativeMeta H3 clip tail: the source is shorter than the required "
            f"{int(context_frames)}-frame continuation context."
        )
    raw_video = raw_video[: frame_count * pixels_per_frame].reshape(
        frame_count, height, width, 3
    )
    images = torch.from_numpy(raw_video[-int(context_frames) :].copy()).to(torch.float32)
    images.div_(255.0)

    audio_rate = 32000
    audio_args = [
        executable,
        "-v", "error",
        "-sseof", f"-{duration:.9f}",
        "-i", path,
        "-t", f"{duration:.9f}",
        "-map", "0:a:0",
        "-vn",
        "-ac", "2",
        "-ar", str(audio_rate),
        "-f", "f32le",
        "-",
    ]
    audio_result = subprocess.run(audio_args, capture_output=True, check=False)
    if audio_result.returncode:
        detail = audio_result.stderr.decode("utf-8", errors="replace").lower()
        missing_audio = (
            "matches no streams",
            "does not contain any stream",
            "output file does not contain any stream",
            "stream map '0:a:0' matches no streams",
            "stream specifier ':a:0'",
        )
        if not any(marker in detail for marker in missing_audio):
            raise RuntimeError(
                "NegativeMeta H3 clip tail: ffmpeg audio extraction failed: "
                f"{audio_result.stderr.decode('utf-8', errors='replace').strip()}"
            )
        waveform = torch.zeros(
            (1, 2, int(round(duration * audio_rate))), dtype=torch.float32
        )
    else:
        raw_audio = np.frombuffer(audio_result.stdout, dtype=np.float32)
        raw_audio = raw_audio[: (raw_audio.size // 2) * 2]
        waveform = torch.from_numpy(raw_audio.copy().reshape(-1, 2).T).unsqueeze(0)
        wanted = int(round(duration * audio_rate))
        if waveform.shape[-1] < wanted:
            waveform = torch.nn.functional.pad(waveform, (0, wanted - waveform.shape[-1]))
        elif waveform.shape[-1] > wanted:
            waveform = waveform[..., -wanted:]
    return images.contiguous(), {"waveform": waveform.contiguous(), "sample_rate": audio_rate}


def _resize_images(images, width: int, height: int):
    if int(images.shape[0]) < 1:
        raise ValueError("NegativeMeta H3 clip cache: source tail is empty.")
    source = images[..., :3].movedim(-1, 1)
    resized = comfy.utils.common_upscale(source, int(width), int(height), "lanczos", "disabled")
    return resized.movedim(1, -1)


def _stereo_waveform(audio):
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("NegativeMeta H3 clip cache: source audio is invalid.")
    waveform = audio["waveform"]
    if getattr(waveform, "ndim", 0) != 3:
        raise ValueError("NegativeMeta H3 clip cache: source audio must be [B,C,S].")
    waveform = waveform[:1]
    channels = int(waveform.shape[1])
    if channels == 1:
        waveform = waveform.repeat(1, 2, 1)
    elif channels != 2:
        raise ValueError("NegativeMeta H3 clip cache: source audio must be mono or stereo.")
    return waveform.to(dtype=torch.float32), int(audio["sample_rate"])


def _resample_exact(waveform, source_rate: int, target_rate: int, target_samples: int):
    if int(source_rate) == int(target_rate) and int(waveform.shape[-1]) == int(target_samples):
        return waveform
    try:
        import torchaudio

        resampled = torchaudio.functional.resample(
            waveform, int(source_rate), int(target_rate)
        )
    except Exception:
        resampled = torch.nn.functional.interpolate(
            waveform,
            size=max(1, int(round(waveform.shape[-1] * target_rate / source_rate))),
            mode="linear",
            align_corners=False,
        )
    if int(resampled.shape[-1]) != int(target_samples):
        resampled = torch.nn.functional.interpolate(
            resampled, size=int(target_samples), mode="linear", align_corners=False
        )
    return resampled


def _audio_grid_geometry(audio_vae, audio_steps: int):
    sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    first_stage = getattr(audio_vae, "first_stage_model", None)
    samples_per_latent = getattr(first_stage, "samples_per_latent", None)
    if samples_per_latent is None:
        samples_per_latent = getattr(first_stage, "hop_length", None)
    if samples_per_latent is None:
        samples_per_latent = int(round(sample_rate / _H3_AUDIO_HZ))
    samples_per_latent = int(samples_per_latent)
    if samples_per_latent < 1 or abs(sample_rate / samples_per_latent - _H3_AUDIO_HZ) > 1e-6:
        raise RuntimeError(
            "NegativeMeta H3 clip cache: audio VAE is not on H3's 40 Hz latent grid."
        )
    return sample_rate, samples_per_latent, int(audio_steps) * samples_per_latent


def _apply_audio_feather(audio_mask, audio_steps: int, feather_ticks: int):
    feather = max(0, min(int(feather_ticks), int(audio_steps)))
    hard_steps = int(audio_steps) - feather
    if hard_steps:
        audio_mask[..., :hard_steps] = 0.0
    if feather:
        index = torch.arange(1, feather + 1, device=audio_mask.device, dtype=audio_mask.dtype)
        ramp = 0.5 - 0.5 * torch.cos(torch.pi * index / float(feather))
        audio_mask[..., hard_steps:int(audio_steps)] = ramp.view(1, 1, 1, feather)
    return audio_mask


def _ensure_h3_mask_support():
    """Invoke the installed Motion Context pack's compatibility guard once."""
    try:
        import nodes

        context_node = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3ExistingVideoMaskedContext")
        guard = getattr(inspect.getmodule(context_node), "_require_h3_mask_support", None)
        if callable(guard):
            guard()
            return
    except Exception as exc:
        raise RuntimeError(
            "NegativeMeta H3 clip cache requires the installed "
            "ComfyUI-H3-Motion-Context-MultiRef AV-mask support."
        ) from exc
    raise RuntimeError(
        "NegativeMeta H3 clip cache could not find MiniMaxH3ExistingVideoMaskedContext. "
        "Install ComfyUI-H3-Motion-Context-MultiRef."
    )


def _cache_identity(video, target_video, target_audio, vae, audio_vae, context_frames: int):
    path = _video_path(video)
    if not path:
        return None, None
    info = os.stat(path)

    def identity(value):
        names = ("model_path", "filename", "file", "name")
        declared = {name: str(getattr(value, name)) for name in names if getattr(value, name, None)}
        return {"class": f"{type(value).__module__}.{type(value).__qualname__}", **declared}

    payload = {
        "format": _CLIP_PREFIX_CACHE_FORMAT,
        "source": path,
        "source_size": int(info.st_size),
        "source_mtime_ns": int(info.st_mtime_ns),
        "context_frames": int(context_frames),
        "video_geometry": [int(target_video.shape[1]), int(target_video.shape[3]), int(target_video.shape[4])],
        "audio_geometry": [int(target_audio.shape[1]), int(target_audio.shape[2])],
        "video_vae": identity(vae),
        "audio_vae": identity(audio_vae),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _cache_path(cache_key: str) -> str:
    return os.path.join(_clip_prefix_cache_directory(), f"{cache_key}.safetensors")


def _load_cached_prefix(cache_key: str, expected_metadata):
    path = _cache_path(cache_key)
    if not os.path.isfile(path):
        return None
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        if metadata.get("format") != _CLIP_PREFIX_CACHE_FORMAT:
            return None
        if metadata.get("identity") != json.dumps(expected_metadata, sort_keys=True, separators=(",", ":")):
            return None
        data = load_file(path, device="cpu")
        if "video_prefix" not in data or "audio_prefix" not in data:
            return None
        return data["video_prefix"], data["audio_prefix"]
    except Exception as exc:
        print(f"[NegativeMeta] Ignoring unreadable H3 clip prefix cache {path}: {exc}")
        return None


def _save_cached_prefix(cache_key: str, metadata, video_prefix, audio_prefix):
    path = _cache_path(cache_key)
    temporary_path = f"{path}.tmp"
    save_file(
        {
            "video_prefix": video_prefix.detach().cpu().contiguous(),
            "audio_prefix": audio_prefix.detach().cpu().contiguous(),
        },
        temporary_path,
        metadata={
            "format": _CLIP_PREFIX_CACHE_FORMAT,
            "identity": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            "saved_at_ns": str(time.time_ns()),
        },
    )
    os.replace(temporary_path, path)


def _masked_h3_target(latent, video_prefix, audio_prefix, context_frames: int, audio_feather_ticks: int):
    target_video, target_audio = _h3_av_streams(latent, "target latent")
    frames = int(context_frames)
    video_steps = int(video_prefix.shape[2])
    audio_steps = int(audio_prefix.shape[-1])
    expected_video_steps = 2 + 5 * ((frames - 5) // 17)
    expected_audio_steps = _h3_audio_steps(frames)
    if video_steps != expected_video_steps or audio_steps != expected_audio_steps:
        raise RuntimeError("NegativeMeta H3 clip cache: cached prefix does not match the H3 AV timeline.")
    if video_steps >= int(target_video.shape[2]) or audio_steps >= int(target_audio.shape[-1]):
        raise ValueError("NegativeMeta H3 clip cache: context consumes the whole target latent.")
    if tuple(video_prefix.shape[1:2] + video_prefix.shape[3:]) != tuple(
        target_video.shape[1:2] + target_video.shape[3:]
    ):
        raise ValueError("NegativeMeta H3 clip cache: cached video prefix geometry differs from the target.")
    if tuple(audio_prefix.shape[1:3]) != tuple(target_audio.shape[1:3]):
        raise ValueError("NegativeMeta H3 clip cache: cached audio prefix geometry differs from the target.")

    out_video = target_video.clone()
    out_audio = target_audio.clone()
    out_video[:, :, :video_steps] = video_prefix[:1].to(device=out_video.device, dtype=out_video.dtype)
    out_audio[..., :audio_steps] = audio_prefix[:1].to(device=out_audio.device, dtype=out_audio.dtype)
    video_mask = torch.ones(
        (1, 1, out_video.shape[2], out_video.shape[3], out_video.shape[4]),
        device=out_video.device,
        dtype=torch.float32,
    )
    audio_mask = torch.ones(
        (1, 1, out_audio.shape[2], out_audio.shape[3]),
        device=out_audio.device,
        dtype=torch.float32,
    )
    video_mask[:, :, :video_steps] = 0.0
    _apply_audio_feather(audio_mask, audio_steps, audio_feather_ticks)
    output = latent.copy()
    output["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
    output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return output


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


def _h3_av_streams(latent, label: str):
    """Return validated MiniMax H3 video and audio tensors from a LATENT."""
    if not isinstance(latent, dict):
        raise ValueError(f"NegativeMeta H3 continuation: {label} is not a LATENT.")
    streams = _streams(latent.get("samples"), f"{label} samples")
    if streams is None:
        raise ValueError(f"NegativeMeta H3 continuation: {label} has no samples.")

    video, audio = streams
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError(
            "NegativeMeta H3 continuation: expected video [B,C,T,H,W] and "
            f"audio [B,C,2,T], got {tuple(video.shape)} and {tuple(audio.shape)}."
        )
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("NegativeMeta H3 continuation currently requires batch size 1.")
    return video, audio


def _h3_pixel_frames(video_steps: int) -> int:
    # H3 temporal tokens decode as 1, 4, 4, 4, 4 pixel frames repeatedly.
    return sum((1, 4, 4, 4, 4)[index % 5] for index in range(int(video_steps)))


def _exact_av_context_frames(requested: int, source_frames: int, target_frames: int) -> int:
    """Choose the largest exact 24 fps / 40 Hz H3 AV boundary within the inputs."""
    cap = min(int(requested), int(source_frames), int(target_frames))
    # Valid H3 video runs are 5 + 17n. A shared audio boundary additionally
    # needs frames * 40 / 24 to be an integer, giving 39 + 51n frames.
    if cap < 39:
        raise ValueError(
            "NegativeMeta H3 continuation: at least 39 compatible frames are "
            "required for exact audio/video continuation."
        )
    return 39 + 51 * ((cap - 39) // 51)


def _apply_audio_feather(audio_mask, audio_steps: int, feather_ticks: int):
    """Release the final protected audio ticks gradually into fresh generation."""
    feather = max(0, min(int(feather_ticks), int(audio_steps)))
    hard_steps = int(audio_steps) - feather
    if hard_steps:
        audio_mask[..., :hard_steps] = 0.0
    if feather:
        ramp_index = torch.arange(
            1,
            feather + 1,
            device=audio_mask.device,
            dtype=audio_mask.dtype,
        )
        ramp = 0.5 - 0.5 * torch.cos(torch.pi * ramp_index / float(feather))
        audio_mask[..., hard_steps:int(audio_steps)] = ramp.view(1, 1, 1, feather)
    return audio_mask


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


class NegativeMeta_H3LatentTailContinuation:
    """Seed a fresh H3 target with the tail of a saved generated AV latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Fresh MiniMax H3 AV target latent for the next clip.",
                }),
                "source_latent": ("LATENT", {
                    "tooltip": "Previous generated MiniMax H3 AV latent loaded from NegativeMeta Toolbox.",
                }),
                "context_length": ("INT", {
                    "default": 39,
                    "min": 39,
                    "max": 9999,
                    "tooltip": "Requested protected AV prefix. Uses exact shared AV boundaries: 39, 90, 141, ... frames.",
                }),
                "audio_feather_ticks": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 256,
                    "tooltip": "Audio-mask release at the seam. 0 keeps a hard mask; 8 equals 0.2 seconds on H3's 40 Hz audio grid.",
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "trim_frames")
    FUNCTION = "prepare"
    CATEGORY = "NegativeMeta/Toolbox"
    DESCRIPTION = (
        "Continue a generated MiniMax H3 clip directly in latent space. The saved "
        "video/audio tail is copied into a fresh target and protected with native "
        "H3 AV noise masks; no video or audio VAE round trip is used."
    )

    def prepare(self, latent, source_latent, context_length=39, audio_feather_ticks=8):
        target_video, target_audio = _h3_av_streams(latent, "target latent")
        source_video, source_audio = _h3_av_streams(source_latent, "saved source latent")

        target_frames = _h3_pixel_frames(target_video.shape[2])
        source_frames = _h3_pixel_frames(source_video.shape[2])
        frames = _exact_av_context_frames(context_length, source_frames, target_frames)
        video_steps = 2 + 5 * ((frames - 5) // 17)
        audio_steps = frames * 5 // 3

        if _h3_pixel_frames(video_steps) != frames:
            raise RuntimeError("NegativeMeta H3 continuation: invalid internal video-grid mapping.")
        if video_steps >= target_video.shape[2] or audio_steps >= target_audio.shape[-1]:
            raise ValueError(
                "NegativeMeta H3 continuation: context must be shorter than the new target clip."
            )
        if source_video.shape[2] < video_steps or source_audio.shape[-1] < audio_steps:
            raise ValueError("NegativeMeta H3 continuation: saved latent is shorter than the requested context.")
        if tuple(source_video.shape[1:2] + source_video.shape[3:]) != tuple(
            target_video.shape[1:2] + target_video.shape[3:]
        ):
            raise ValueError(
                "NegativeMeta H3 continuation: source and target video latent geometry differs. "
                "Use the same Director resolution, model family, and video VAE, or continue from a clip instead."
            )
        if tuple(source_audio.shape[1:3]) != tuple(target_audio.shape[1:3]):
            raise ValueError(
                "NegativeMeta H3 continuation: source and target audio latent geometry differs. "
                "Use the same compatible H3 audio VAE."
            )

        out_video = target_video.clone()
        out_audio = target_audio.clone()
        out_video[:, :, :video_steps] = source_video[:, :, -video_steps:].to(
            device=out_video.device,
            dtype=out_video.dtype,
        )
        out_audio[..., :audio_steps] = source_audio[..., -audio_steps:].to(
            device=out_audio.device,
            dtype=out_audio.dtype,
        )

        video_mask = torch.ones(
            (1, 1, out_video.shape[2], out_video.shape[3], out_video.shape[4]),
            device=out_video.device,
            dtype=torch.float32,
        )
        audio_mask = torch.ones(
            (1, 1, out_audio.shape[2], out_audio.shape[3]),
            device=out_audio.device,
            dtype=torch.float32,
        )
        video_mask[:, :, :video_steps] = 0.0
        _apply_audio_feather(audio_mask, audio_steps, audio_feather_ticks)

        output = latent.copy()
        output["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
        output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        print(
            "[NegativeMeta] H3 latent continuation: protected "
            f"{frames} frames ({video_steps} video steps / {audio_steps} audio steps)."
        )
        return (output, frames)


class NegativeMeta_H3VideoTail:
    """Read only the final H3 continuation window from a file-backed core VIDEO."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {
                    "tooltip": "Core LoadVideo output. The node seeks directly to the file tail instead of decoding the entire clip.",
                }),
                "context_length": ("INT", {
                    "default": 39,
                    "min": 39,
                    "max": 9999,
                    "tooltip": "Tail length at 24 fps. H3 AV context uses shared boundaries: 39, 90, 141, ... frames.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "audio", "fps")
    FUNCTION = "load_tail"
    CATEGORY = "NegativeMeta/Toolbox"
    DESCRIPTION = (
        "Decodes only the final exact H3 audio/video context from a core file-backed VIDEO. "
        "It avoids materializing the full source clip in RAM."
    )

    def load_tail(self, video, context_length=39):
        frames = _h3_context_boundary(context_length, context_length)
        images, audio = _load_tail_with_ffmpeg(video, frames)
        print(
            "[NegativeMeta] H3 clip tail: decoded only the final "
            f"{frames} frames ({_tail_duration(frames):.3f}s)."
        )
        return (images, audio, float(_H3_FPS))


class NegativeMeta_H3CachedClipContext:
    """Prepare an external clip tail once and reuse its H3 VAE prefix on later runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Fresh H3 AV target latent for the next generated segment.",
                }),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE."}),
                "source_video": ("VIDEO", {
                    "tooltip": "The same file-backed core VIDEO used by H3 Video Tail. It provides the stable cache identity without decoding the clip.",
                }),
                "context_length": ("INT", {
                    "default": 39,
                    "min": 39,
                    "max": 9999,
                    "tooltip": "Protected H3 AV prefix. Uses exact shared boundaries: 39, 90, 141, ... frames.",
                }),
                "audio_feather_ticks": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 256,
                    "tooltip": "Half-cosine release across the final protected audio ticks. 0 keeps a hard audio mask.",
                }),
            },
            "optional": {
                "source_frames": ("IMAGE", {
                    "lazy": True,
                    "tooltip": "Tail images from NegativeMeta H3 Video Tail. Requested only when no matching prefix cache exists.",
                }),
                "source_audio": ("AUDIO", {
                    "lazy": True,
                    "tooltip": "Tail audio from NegativeMeta H3 Video Tail. Requested only when no matching prefix cache exists.",
                }),
                "source_fps": ("FLOAT", {
                    "lazy": True,
                    "tooltip": "Tail frame rate. NegativeMeta H3 Video Tail returns the exact required 24 fps.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "trim_frames")
    FUNCTION = "prepare"
    CATEGORY = "NegativeMeta/Toolbox"
    DESCRIPTION = (
        "External MiniMax H3 continuation with two optimizations: only the source tail is "
        "decoded, and its VAE-encoded AV prefix is cached persistently. Cache hits skip both "
        "H3 VAEs while preserving the same protected H3 AV prefix."
    )

    @classmethod
    def check_lazy_status(
        cls,
        latent,
        vae,
        audio_vae,
        source_video,
        context_length=39,
        audio_feather_ticks=8,
        source_frames=None,
        source_audio=None,
        source_fps=None,
    ):
        try:
            target_video, target_audio = _h3_av_streams(latent, "target latent")
            frames = _h3_context_boundary(context_length, _h3_pixel_frames(target_video.shape[2]))
            cache_key, metadata = _cache_identity(
                source_video, target_video, target_audio, vae, audio_vae, frames
            )
            if cache_key and _load_cached_prefix(cache_key, metadata) is not None:
                return []
        except Exception:
            # Let execute report the full actionable error once the selected path runs.
            pass
        required = []
        if source_frames is None:
            required.append("source_frames")
        if source_audio is None:
            required.append("source_audio")
        if source_fps is None:
            required.append("source_fps")
        return required

    def prepare(
        self,
        latent,
        vae,
        audio_vae,
        source_video,
        context_length=39,
        audio_feather_ticks=8,
        source_frames=None,
        source_audio=None,
        source_fps=None,
    ):
        _ensure_h3_mask_support()
        target_video, target_audio = _h3_av_streams(latent, "target latent")
        target_frames = _h3_pixel_frames(target_video.shape[2])
        frames = _h3_context_boundary(context_length, target_frames)
        cache_key, metadata = _cache_identity(
            source_video, target_video, target_audio, vae, audio_vae, frames
        )
        if cache_key:
            cached = _load_cached_prefix(cache_key, metadata)
            if cached is not None:
                print(f"[NegativeMeta] H3 clip prefix cache hit: {cache_key[:12]}")
                output = _masked_h3_target(
                    latent, cached[0], cached[1], frames, audio_feather_ticks
                )
                return (output, frames)

        if source_frames is None or source_audio is None or source_fps is None:
            raise ValueError(
                "NegativeMeta H3 clip cache: no valid prefix cache was found and the "
                "tail image/audio inputs were not provided."
            )
        if abs(float(source_fps) - _H3_FPS) > 1e-4:
            raise ValueError(
                "NegativeMeta H3 clip cache expects the 24 fps output of NegativeMeta H3 Video Tail."
            )
        if getattr(source_frames, "ndim", 0) != 4 or int(source_frames.shape[0]) < frames:
            raise ValueError(
                "NegativeMeta H3 clip cache: source tail has fewer frames than the selected context."
            )

        width = int(target_video.shape[4]) * 16
        height = int(target_video.shape[3]) * 16
        video_tail = _resize_images(source_frames[-frames:], width, height)
        video_prefix = vae.encode(video_tail)
        if getattr(video_prefix, "ndim", 0) != 5:
            raise ValueError(
                "NegativeMeta H3 clip cache: video VAE must return [B,C,T,H,W]."
            )
        expected_video_steps = 2 + 5 * ((frames - 5) // 17)
        if int(video_prefix.shape[2]) != expected_video_steps:
            raise RuntimeError(
                "NegativeMeta H3 clip cache: video VAE returned a temporal grid that does not "
                "match the exact H3 context boundary."
            )

        audio_steps = _h3_audio_steps(frames)
        sample_rate, _samples_per_latent, grid_samples = _audio_grid_geometry(
            audio_vae, audio_steps
        )
        waveform, source_rate = _stereo_waveform(source_audio)
        waveform = _resample_exact(waveform, source_rate, sample_rate, grid_samples)
        audio_prefix = audio_vae.encode(waveform.movedim(1, -1))
        if getattr(audio_prefix, "ndim", 0) != 4 or int(audio_prefix.shape[-1]) != audio_steps:
            raise RuntimeError(
                "NegativeMeta H3 clip cache: audio VAE did not return the exact H3 audio grid."
            )

        if cache_key:
            try:
                _save_cached_prefix(cache_key, metadata, video_prefix, audio_prefix)
                print(f"[NegativeMeta] H3 clip prefix cached: {cache_key[:12]}")
            except Exception as exc:
                print(f"[NegativeMeta] H3 clip prefix cache write skipped: {exc}")
        else:
            print("[NegativeMeta] H3 clip prefix cache skipped: source VIDEO is not file-backed.")
        output = _masked_h3_target(
            latent, video_prefix, audio_prefix, frames, audio_feather_ticks
        )
        return (output, frames)
