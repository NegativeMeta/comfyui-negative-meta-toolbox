"""Temporal reference color correction for decoded IMAGE batches."""
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
import kornia
import comfy.model_management
import comfy.utils
from comfy_api.latest import io
from .nodes_streaming_color_transfer import (
    _Stats, _reference_stats, _transform, _protect_lightness, _luminance_safe_rgb, _interrupt,
)
from .nodes_adaptive_color_transfer import _tone_curve, _chroma_curve


def _image_lab(image, device, max_edge=0):
    rgb = image[..., :3].to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    h, w = rgb.shape[-2:]
    if max_edge and max(h, w) > max_edge:
        scale = max_edge / max(h, w)
        rgb = F.interpolate(rgb, size=(max(1, round(h * scale)), max(1, round(w * scale))),
                            mode="bilinear", align_corners=False, antialias=True)
    return kornia.color.rgb_to_lab(rgb)


def _image_profile(images, device, fps, max_edge, detect_cuts):
    stats, rows, cuts, previous = _Stats(), [], [0], None
    progress = comfy.utils.ProgressBar(len(images))
    for index, image in enumerate(images):
        _interrupt()
        lab = _image_lab(image, device, max_edge)
        flat = lab.reshape(3, -1)
        mean = flat.mean(1, keepdim=True)
        centered = flat - mean
        cov = centered @ centered.T / flat.shape[1]
        rows.append(np.append(torch.cat([mean.flatten(), cov.flatten()]).cpu().numpy(), index / fps))
        stats.add(lab)
        if detect_cuts:
            luma = F.interpolate(lab[:, :1], size=(18, 32), mode="area").flatten()
            luma = (luma - luma.mean()) / luma.std(unbiased=False).clamp_min(5)
            if previous is not None:
                energy = (luma.square().mean() * previous.square().mean()).sqrt().clamp_min(.01)
                correlation = (luma * previous).mean() / energy
                if correlation.item() < .2 and (luma - previous).abs().mean().item() > .8:
                    cuts.append(index)
            previous = luma
        del lab, flat, mean, centered, cov
        progress.update(1)
    return stats, np.asarray(rows, dtype=np.float64), cuts


class NegativeMeta_AdaptiveImageColorTransfer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NegativeMeta_AdaptiveImageColorTransfer",
            display_name="NegativeMeta · Adaptive Image Color Transfer",
            category="NegativeMeta/Video",
            description="Reference a/b color plus bounded temporal color/lightness stabilization. Processes each frame separately; source and output IMAGE batches remain resident.",
            inputs=[
                io.Image.Input("image_target"), io.Image.Input("image_ref"),
                io.Float.Input("strength", default=.55, min=0, max=1, step=.01, display_name="🎨 Reference Color Strength"),
                io.Float.Input("fps", default=24., min=.001, max=1000, display_name="🕒 Source FPS"),
                io.Combo.Input("method", options=["mkl_lab", "reinhard_lab"], default="mkl_lab"),
                io.Float.Input("exposure_strength", default=1., min=0, max=1, step=.01, display_name="☀️ Exposure Stabilization"),
                io.Float.Input("temporal_window", default=2., min=0, max=10, step=.1, display_name="⏱️ Lightness Window (seconds)"),
                io.Float.Input("max_luma_shift", default=12., min=0, max=100, step=1, display_name="↕️ Max Lightness Correction"),
                io.Float.Input("highlight_protection", default=1., min=0, max=1, step=.01, display_name="🛡️ Highlight Protection"),
                io.Float.Input("temporal_color_strength", default=.8, min=0, max=1, step=.01, display_name="🎨 Temporal Color Stabilization"),
                io.Float.Input("color_temporal_window", default=3., min=0, max=10, step=.1, display_name="⏱️ Color Window (seconds)"),
                io.Float.Input("max_chroma_shift", default=8., min=0, max=40, step=.5, display_name="↔️ Max Tint Correction"),
                io.Boolean.Input("detect_scene_cuts", default=False, display_name="🎬 Detect Color Scene Cuts"),
                io.Int.Input("analysis_max_edge", default=256, min=0, max=8192, step=32),
                io.Combo.Input("device", options=["auto", "cuda", "cpu"], default="auto"),
            ], outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image_target, image_ref, strength=.55, fps=24., method="mkl_lab",
                exposure_strength=1., temporal_window=2., max_luma_shift=12.,
                highlight_protection=1., temporal_color_strength=.8,
                color_temporal_window=3., max_chroma_shift=8.,
                detect_scene_cuts=False, analysis_max_edge=256, device="auto"):
        if image_ref is None or not len(image_target):
            return io.NodeOutput(image_target)
        if image_target.ndim != 4 or image_target.shape[-1] < 3:
            raise ValueError("Expected an RGB IMAGE batch.")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Source FPS must be positive and finite.")
        if not len(image_ref) or image_ref.shape[-1] < 3:
            raise ValueError("Expected an RGB reference image.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; select cpu or auto.")
        if strength == exposure_strength == temporal_color_strength == 0:
            return io.NodeOutput(image_target)
        target = comfy.model_management.get_torch_device() if device == "auto" else torch.device(device)
        with torch.inference_mode():
            refs = _reference_stats(image_ref, target, analysis_max_edge)
            stats, profile, cuts = _image_profile(image_target, target, fps, analysis_max_edge, detect_scene_cuts)
            matrix, offset = _transform(stats, refs, method, chroma_only=True)
            tone = _tone_curve(profile, cuts, matrix, offset, refs, strength,
                               exposure_strength, 0., 0., temporal_window, max_luma_shift,
                               luminance_mode="preserve_video")
            chroma = _chroma_curve(profile, cuts, matrix, offset, strength, tone[:, 2],
                                   temporal_color_strength, color_temporal_window, max_chroma_shift)
            del stats, profile
            output = torch.empty(image_target.shape, dtype=image_target.dtype,
                                 device=comfy.model_management.intermediate_device())
            progress = comfy.utils.ProgressBar(len(image_target))
            for index, image in enumerate(image_target):
                _interrupt()
                lab = _image_lab(image, target)
                matched = (matrix @ lab.reshape(3, -1) + offset).reshape_as(lab)
                gain, shift, activity = map(float, tone[index])
                mixed = torch.lerp(lab, matched, strength * activity)
                mixed[:, 0].mul_(gain).add_(shift)
                if chroma is not None:
                    cg, ca, cb, da, db = map(float, chroma[index])
                    center = mixed.new_tensor([ca, cb]).view(1, 2, 1, 1)
                    delta = mixed.new_tensor([da, db]).view(1, 2, 1, 1)
                    mixed[:, 1:] = (mixed[:, 1:] - center) * cg + center + delta
                    del center, delta
                if highlight_protection > 0:
                    mixed[:, 0] = _protect_lightness(mixed[:, 0], lab[:, 0], highlight_protection)
                rgb = _luminance_safe_rgb(mixed)[0].permute(1, 2, 0)
                output[index, ..., :3].copy_(rgb)
                if image_target.shape[-1] > 3:
                    output[index, ..., 3:].copy_(image[..., 3:])
                del lab, matched, mixed, rgb
                progress.update(1)
        logging.info("[NegativeMeta] Adaptive IMAGE color: %s frames at %.3f fps; temporal color %.2f; cuts %s",
                     len(image_target), fps, temporal_color_strength, len(cuts) - 1)
        return io.NodeOutput(output)
