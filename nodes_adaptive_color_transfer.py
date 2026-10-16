"""Streaming global color grade plus time-varying exposure stabilization."""

import logging
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import av
import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl, io, ui
from .nodes_streaming_color_transfer import (
    _Stats, _frame_lab, _reference_stats, _transform, _encode, _mux_audio, _ffmpeg, _interrupt,
)


def _profile(path, device, frame_count, samples, max_edge, detect_cuts):
    selected = None
    if samples and frame_count > samples:
        selected = ({frame_count // 2} if samples == 1 else
                    {round(i * (frame_count - 1) / (samples - 1)) for i in range(samples)})
    stats, rows, cuts, previous = _Stats(), [], [0], None
    progress = comfy.utils.ProgressBar(max(1, frame_count))
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "SLICE"
        stream.codec_context.thread_count = 2
        for index, frame in enumerate(container.decode(stream)):
            _interrupt()
            if frame.time is None:
                raise ValueError("Adaptive color requires video frames with timestamps.")
            lab = _frame_lab(frame, device, max_edge)
            flat = lab.reshape(3, -1)
            mean = flat.mean(1, keepdim=True)
            centered = flat - mean
            cov = centered @ centered.T / flat.shape[1]
            moments = torch.cat([mean.flatten(), cov.flatten()]).cpu().numpy()
            rows.append(np.append(moments, float(frame.time)))
            if selected is None or index in selected:
                stats.add(lab)
            if detect_cuts:
                luma = F.interpolate(lab[:, :1], size=(18, 32), mode="area").flatten()
                luma = (luma - luma.mean()) / luma.std(unbiased=False).clamp_min(5)
                if previous is not None:
                    energy = (luma.square().mean() * previous.square().mean()).sqrt().clamp_min(.01)
                    correlation = (luma * previous).mean() / energy
                    # Brightness offsets/scales are normalized out: an exposure jump
                    # alone must not become a scene boundary and bypass smoothing.
                    if correlation.item() < .2 and (luma - previous).abs().mean().item() > .8:
                        cuts.append(index)
                previous = luma
            progress.update(1)
    if not rows:
        raise ValueError("No frames were available for adaptive color analysis.")
    profile = np.asarray(rows, dtype=np.float64)
    if np.any(np.diff(profile[:, 12]) <= 0):
        raise ValueError("The source has duplicate or non-increasing video timestamps.")
    logging.info("[NegativeMeta] Adaptive analysis: %s frames; %s scene boundaries; %.2f MiB of scalar statistics",
                 len(profile), len(cuts) - 1, profile.nbytes / 1024**2)
    return stats, profile, cuts


def _smooth(values, times, cuts, seconds):
    """Centered Gaussian in real time; no wraparound and no mixing across cuts."""
    if seconds <= 0:
        return values.copy()
    result = np.empty_like(values)
    sigma, radius = seconds / 4, seconds / 2
    boundaries = list(cuts) + [len(values)]
    for start, end in zip(boundaries, boundaries[1:]):
        t, v = times[start:end], values[start:end]
        left = np.searchsorted(t, t - radius, side="left")
        right = np.searchsorted(t, t + radius, side="right")
        for i in range(end - start):
            _interrupt()
            lo, hi = left[i], right[i]
            weights = np.exp(-.5 * ((t[lo:hi] - t[i]) / sigma)**2)
            result[start + i] = np.dot(weights, v[lo:hi]) / weights.sum()
    return result


def _tone_curve(profile, cuts, matrix, offset, reference, color_strength,
                exposure_strength, contrast_strength, preserve_lighting,
                temporal_window, max_luma_shift, brightness_offset=0.,
                luminance_mode="reference"):
    means = profile[:, :3]
    covs = profile[:, 3:12].reshape(-1, 3, 3)
    original_std = np.sqrt(np.maximum(covs[:, 0, 0], 0))
    # Fade/blank-frame protection gradually reduces correction near flat black/white.
    activity = np.clip(np.minimum(means[:, 0], 100 - means[:, 0]) / 8, 0, 1)
    activity *= np.clip(original_std / 2, 0, 1)
    m = matrix.detach().cpu().numpy().astype(np.float64)
    b = offset.detach().cpu().numpy().reshape(3).astype(np.float64)
    affine = np.eye(3)[None] + color_strength * activity[:, None, None] * (m - np.eye(3))[None]
    graded_mean = (np.einsum('nij,nj->ni', affine, means)
                   + color_strength * activity[:, None] * b)[..., 0]
    row = affine[:, 0, :]
    graded_std = np.sqrt(np.maximum(np.einsum('ni,nij,nj->n', row, covs, row), 0))
    smooth_mean = _smooth(graded_mean, profile[:, 12], cuts, temporal_window)
    smooth_std = _smooth(graded_std, profile[:, 12], cuts, temporal_window)
    if luminance_mode == "preserve_video":
        # The reference may have different subjects/composition: do not use its L.
        target_mean, target_std = smooth_mean, smooth_std
    else:
        ref_mean, ref_cov = reference.result()
        desired_mean = float(np.clip(float(ref_mean[0, 0]) + brightness_offset, 0, 100))
        desired_std = float(ref_cov[0, 0].clamp_min(0).sqrt())
        target_mean = desired_mean + preserve_lighting * (smooth_mean - desired_mean)
        target_std = desired_std + preserve_lighting * (smooth_std - desired_std)
    gain = 1 + contrast_strength * activity * (np.clip(target_std / np.maximum(graded_std, 1), .7, 1.4) - 1)
    exposure = exposure_strength * activity * np.clip(target_mean - graded_mean, -max_luma_shift, max_luma_shift)
    # Compensate raw-frame changes immediately. Only the desired appearance is
    # smoothed: smoothing a correction matrix instead would lag abrupt exposure jumps.
    shift = graded_mean * (1 - gain) + exposure
    curve = np.stack([gain, shift, activity], axis=1).astype(np.float32)
    if not np.isfinite(curve).all():
        raise ValueError("Adaptive exposure produced non-finite parameters.")
    return curve


def _chroma_curve(profile, cuts, matrix, offset, color_strength, activity,
                  strength, temporal_window, max_shift):
    """Bounded a/b mean and saturation correction; no reference L or frame batch."""
    if strength <= 0:
        return None
    means = profile[:, :3]
    covs = profile[:, 3:12].reshape(-1, 3, 3)
    m = matrix.detach().cpu().numpy().astype(np.float64)
    b = offset.detach().cpu().numpy().reshape(3).astype(np.float64)
    affine = np.eye(3)[None] + color_strength * activity[:, None, None] * (m - np.eye(3))[None]
    graded = (np.einsum('nij,nj->ni', affine, means)
              + color_strength * activity[:, None] * b)[:, 1:]
    rows = affine[:, 1:, :]
    spread = np.sqrt(np.maximum(np.einsum('nki,nij,nkj->n', rows, covs, rows), 0))
    times = profile[:, 12]
    target = np.column_stack([_smooth(graded[:, i], times, cuts, temporal_window)
                              for i in range(2)])
    target_spread = _smooth(spread, times, cuts, temporal_window)
    delta = target - graded
    size = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1., max_shift / np.maximum(size, 1e-9))[:, None]
    delta *= (strength * activity)[:, None]
    # Isotropic a/b gain retains hue angles about the frame's color center.
    gain = 1 + strength * activity * (np.clip(target_spread / np.maximum(spread, 1.), .85, 1.15) - 1)
    curve = np.column_stack([gain, graded, delta]).astype(np.float32)
    if not np.isfinite(curve).all():
        raise ValueError("Temporal color stabilization produced non-finite parameters.")
    return curve


class NegativeMeta_AdaptiveStreamingColorTransfer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NegativeMeta_AdaptiveStreamingColorTransfer",
            display_name="NegativeMeta · Adaptive Streaming Color Transfer",
            category="NegativeMeta/Video", is_output_node=True,
            description="Global reference color transfer plus per-frame exposure compensation with a centered temporal appearance curve. Streams frames; retains only small statistics per frame.",
            inputs=[
                io.Video.Input("video"), io.Image.Input("image_ref"),
                io.Combo.Input("method", options=["mkl_lab", "reinhard_lab"], display_name="🎨 Color Method"),
                io.Float.Input("color_strength", default=.55, min=0, max=1, step=.01, display_name="🎨 Color Strength"),
                io.Float.Input("exposure_strength", default=1., min=0, max=1, step=.01, display_name="☀️ Exposure Stabilization",
                               tooltip="0 preserves the global grade's brightness changes; 1 compensates frame brightness toward the reference/temporal target."),
                io.Float.Input("contrast_strength", default=0., min=0, max=1, step=.01, display_name="◐ Contrast Stabilization",
                               tooltip="Optional limited contrast normalization. Start at 0; try 0.15 if contrast also jumps."),
                io.Float.Input("preserve_lighting", default=.2, min=0, max=1, step=.01, display_name="💡 Preserve Lighting",
                               tooltip="Reference mode only: 0 targets fixed reference brightness; 1 preserves the smooth source trajectory. Ignored in preserve_video mode."),
                io.Float.Input("temporal_window", default=1., min=0, max=10, step=.1, display_name="⏱️ Temporal Window (seconds)",
                               tooltip="Centered smoothing of desired brightness, using frame timestamps; excludes detected cuts."),
                io.Float.Input("max_luma_shift", default=25., min=0, max=100, step=1, display_name="↕️ Max Lightness Correction",
                               tooltip="Limit on exposure correction in Lab L units (0–100). Protects against extreme shifts."),
                io.Boolean.Input("detect_scene_cuts", default=True, display_name="🎬 Detect Scene Cuts",
                                 tooltip="Heuristic on exposure-normalized image structure; brightness jumps alone are not treated as cuts."),
                io.Combo.Input("device", options=["auto", "cuda", "cpu"], display_name="⚙️ Processing Device"),
                io.Int.Input("analysis_frames", default=96, min=0, max=10000, display_name="🔎 Global Color Samples",
                             tooltip="Sample count for the base color grade only. Exposure statistics are always measured for every frame. 0 uses all frames."),
                io.Int.Input("analysis_max_edge", default=256, min=0, max=8192, step=32, display_name="🔎 Analysis Max Edge",
                             tooltip="Downscales analysis only. Output resolution is unchanged; 0 uses full resolution."),
                io.Int.Input("crf", default=18, min=0, max=51, display_name="🎞️ H.264 CRF"),
                io.String.Input("filename_prefix", default="video/color_recovery_adaptive/color_recovery", display_name="💾 Output Prefix"),
                io.Float.Input("brightness_offset", default=-8., min=-50, max=50, step=1, display_name="☀️ Target Brightness Offset",
                               tooltip="Reference mode only: Lab L offset from reference brightness. Ignored in preserve_video mode."),
                io.Float.Input("highlight_protection", default=.8, min=0, max=1, step=.01, display_name="🛡️ Highlight Protection",
                               tooltip="Reduces positive brightness lifts in source highlights. Does not undo darkening or recover already clipped detail."),
                io.Combo.Input("luminance_mode", options=["preserve_video", "reference"], default="preserve_video",
                               display_name="💡 Luminance Mode",
                               tooltip="preserve_video transfers a/b color only and stabilizes L from the video. reference also matches reference brightness. With exposure/contrast 0, preserve_video retains source Lab L."),
                io.Float.Input("temporal_color_strength", default=.8, min=0, max=1, step=.01,
                               display_name="🎨 Temporal Color Stabilization",
                               tooltip="0 disables it. Smooths rapid a/b tint and saturation changes from the video, independently of lightness. Content/lighting changes can also affect estimates."),
                io.Float.Input("color_temporal_window", default=3., min=0, max=10, step=.1,
                               display_name="⏱️ Color Temporal Window (seconds)",
                               tooltip="Centered target-color smoothing within each detected scene. Larger windows soften transitions more but can suppress intentional lighting changes."),
                io.Float.Input("max_chroma_shift", default=8., min=0, max=40, step=.5,
                               display_name="↔️ Max Tint Correction",
                               tooltip="Maximum a/b mean-offset vector length in Lab units before intensity scaling. Saturation gain is limited separately to ±15% at full intensity."),
            ], outputs=[io.Video.Output(), io.String.Output(display_name="saved_path")],
        )

    @classmethod
    def execute(cls, video, image_ref, method, color_strength, exposure_strength,
                contrast_strength, preserve_lighting, temporal_window, max_luma_shift,
                detect_scene_cuts, device, analysis_frames, analysis_max_edge, crf, filename_prefix,
                brightness_offset=0., highlight_protection=0., luminance_mode="reference",
                temporal_color_strength=0., color_temporal_window=3., max_chroma_shift=8.):
        if luminance_mode not in ("preserve_video", "reference"):
            raise ValueError("Unknown luminance mode.")
        source = video.get_stream_source() if hasattr(video, "get_stream_source") else None
        if not isinstance(source, (str, os.PathLike)) or not Path(source).is_file():
            raise ValueError("Use core Load Video with a file-backed video upload.")
        source = str(Path(source).resolve())
        if video.get_active_trim_window() != (0., 0.):
            raise ValueError("Use a full untrimmed video upload.")
        if not image_ref.shape[0] or image_ref.shape[-1] < 3:
            raise ValueError("Upload an RGB reference image.")
        executable = _ffmpeg()
        with av.open(source) as container:
            if not container.streams.video:
                raise ValueError("The source contains no video stream.")
            stream = container.streams.video[0]
            w, h = stream.width, stream.height
            rate = stream.average_rate or stream.guessed_rate
            if not rate or w % 2 or h % 2:
                raise ValueError("Use a video with a valid frame rate and even dimensions.")
            if tuple(video.get_dimensions()) != (w, h):
                raise ValueError("Use an uncropped video upload.")
            if stream.codec_context.color_trc in (16, 18):
                raise ValueError("Convert HDR PQ/HLG to SDR before using this workflow.")
            count = int(stream.frames or 0)
            if not count and stream.duration:
                count = max(1, round(float(stream.duration * stream.time_base * rate)))
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; select cpu or auto.")
        target_device = (comfy.model_management.get_torch_device() if device == "auto"
                         else torch.device(device))
        with torch.inference_mode():
            logging.info("[NegativeMeta] Adaptive streaming pass 1/2: global color and per-frame exposure analysis")
            refs = _reference_stats(image_ref, target_device, analysis_max_edge)
            stats, profile, cuts = _profile(source, target_device, count, analysis_frames,
                                           analysis_max_edge, detect_scene_cuts)
            preserve_luminance = luminance_mode == "preserve_video"
            logging.info("[NegativeMeta] Luminance mode: %s", luminance_mode)
            matrix, offset = _transform(stats, refs, method, chroma_only=preserve_luminance)
            curve = _tone_curve(profile, cuts, matrix, offset, refs, color_strength,
                                exposure_strength, contrast_strength, preserve_lighting,
                                temporal_window, max_luma_shift, brightness_offset, luminance_mode)
            chroma_curve = _chroma_curve(profile, cuts, matrix, offset, color_strength,
                                         curve[:, 2], temporal_color_strength,
                                         color_temporal_window, max_chroma_shift)
            del profile, stats
            folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory(), w, h)
            filename = f"{name}_{counter:05}_{uuid.uuid4().hex[:8]}.mp4"
            final = Path(folder) / filename
            with tempfile.TemporaryDirectory(prefix="negativemeta_adaptive_", dir=folder_paths.get_temp_directory()) as temp:
                silent, muxed = Path(temp) / "video.mp4", Path(temp) / "result.mp4"
                logging.info("[NegativeMeta] Adaptive streaming pass 2/2: frame-wise compensation and encoding")
                written = _encode(source, silent, target_device, matrix, offset,
                                  color_strength, crf, len(curve), tone_curve=curve,
                                  highlight_protection=highlight_protection,
                                  preserve_luminance=preserve_luminance,
                                  chroma_curve=chroma_curve)
                _interrupt()
                try:
                    _mux_audio(executable, silent, source, muxed)
                except RuntimeError as exc:
                    logging.warning("[NegativeMeta] Retrying audio mux as AAC: %s", exc)
                    _mux_audio(executable, silent, source, muxed, "aac")
                _interrupt()
                shutil.move(str(muxed), str(final))
        logging.info("[NegativeMeta] Adaptive color output: %s frames -> %s", written, final)
        return io.NodeOutput(InputImpl.VideoFromFile(str(final)), str(final),
                             ui=ui.PreviewVideo([ui.SavedResult(filename, subfolder, io.FolderType.output)]))
