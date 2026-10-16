"""File-backed video color transfer with bounded frame memory."""

import logging
import os
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

import av
import kornia
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl, io, ui


def _ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Streaming Color Transfer requires FFmpeg or imageio-ffmpeg.") from exc


def _interrupt():
    comfy.model_management.throw_exception_if_processing_interrupted()


def _frame_lab(frame, device, max_edge=0):
    if max_edge and max(frame.width, frame.height) > max_edge:
        scale = max_edge / max(frame.width, frame.height)
        frame = frame.reformat(width=max(1, round(frame.width * scale)),
                               height=max(1, round(frame.height * scale)), format="rgb24")
    array = frame.to_ndarray(format="rgb24")
    rgb = torch.from_numpy(array).to(device=device, dtype=torch.float32)
    rgb = rgb.permute(2, 0, 1).unsqueeze(0).div_(255)
    return kornia.color.rgb_to_lab(rgb)


class _Stats:
    """Merge pixel means/covariances; retain only a mean and a 3x3 matrix."""
    def __init__(self):
        self.count = 0
        self.mean = None
        self.scatter = None

    def add(self, lab):
        pixels = lab.reshape(3, -1)
        size = pixels.shape[1]
        mean = pixels.mean(dim=1, keepdim=True)
        centered = pixels - mean
        scatter = (centered @ centered.T).double()
        mean = mean.double()
        if self.count == 0:
            self.mean, self.scatter, self.count = mean, scatter, size
            return
        total = self.count + size
        delta = mean - self.mean
        self.scatter += scatter + (delta @ delta.T) * (self.count * size / total)
        self.mean += delta * (size / total)
        self.count = total

    def result(self):
        if not self.count:
            raise ValueError("No video frames were available for color analysis.")
        return self.mean.float(), (self.scatter / self.count).float()


def _transform(source, reference, method, chroma_only=False):
    source_mean, source_cov = source.result()
    ref_mean, ref_cov = reference.result()
    if chroma_only:
        # Fit only the independent a/b subspace, never correlations with L.
        source_mean, ref_mean = source_mean[1:], ref_mean[1:]
        source_cov, ref_cov = source_cov[1:, 1:], ref_cov[1:, 1:]
    if method == "reinhard_lab":
        scale = (ref_cov.diag().clamp_min(0).sqrt().clamp_min(1e-6)
                 / source_cov.diag().clamp_min(0).sqrt().clamp_min(1e-6))
        matrix = torch.diag(scale)
    else:
        from comfy_extras.nodes_post_processing import ColorTransfer
        matrix = ColorTransfer._mkl_matrix(source_cov, ref_cov, 1e-6)
    offset = ref_mean - matrix @ source_mean
    if chroma_only:
        full_matrix = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
        full_matrix[1:, 1:] = matrix
        full_offset = torch.zeros((3, 1), device=offset.device, dtype=offset.dtype)
        full_offset[1:] = offset
        matrix, offset = full_matrix, full_offset
    if not torch.isfinite(matrix).all() or not torch.isfinite(offset).all():
        raise ValueError("The reference/source color statistics produced an invalid transform.")
    return matrix, offset


def _reference_stats(images, device, max_edge):
    result = _Stats()
    for image in images:
        _interrupt()
        rgb = image[..., :3].to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        h, w = rgb.shape[-2:]
        if max_edge and max(h, w) > max_edge:
            scale = max_edge / max(h, w)
            rgb = F.interpolate(rgb, size=(max(1, round(h * scale)), max(1, round(w * scale))),
                                mode="bilinear", align_corners=False, antialias=True)
        result.add(kornia.color.rgb_to_lab(rgb))
    return result


def _analyse(path, device, frame_count, samples, max_edge):
    selected = None
    if samples and frame_count > samples:
        selected = ({frame_count // 2} if samples == 1 else
                    {round(i * (frame_count - 1) / (samples - 1)) for i in range(samples)})
    stats = _Stats()
    used = 0
    progress = comfy.utils.ProgressBar(max(1, frame_count))
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "SLICE"
        stream.codec_context.thread_count = 2
        for index, frame in enumerate(container.decode(stream)):
            _interrupt()
            if selected is None or index in selected:
                stats.add(_frame_lab(frame, device, max_edge))
                used += 1
            progress.update(1)
    logging.info("[NegativeMeta] Streaming color analysis: %s frames, max edge %s, device=%s",
                 used, max_edge or "original", device)
    return stats


def _protect_lightness(candidate, original, strength):
    """Reduce positive lightness lifts in source highlights; retain darkening."""
    ramp = ((original - 60) / 30).clamp(0, 1)
    weight = 1 - strength * ramp.square() * (3 - 2 * ramp)
    delta = candidate - original
    return original + torch.where(delta > 0, delta * weight, delta)


def _luminance_safe_rgb(lab):
    """Compress out-of-gamut chroma toward neutral at fixed Lab lightness."""
    lab = lab.clone()
    lab[:, :1].clamp_(0, 100)
    rgb = kornia.color.lab_to_rgb(lab, clip=False)
    invalid = ((rgb < 0) | (rgb > 1)).any(dim=1, keepdim=True)
    if invalid.any():
        low = torch.zeros_like(lab[:, :1])
        high = torch.ones_like(low)
        for _ in range(8):
            middle = (low + high) * .5
            trial = torch.cat([lab[:, :1], lab[:, 1:] * middle], dim=1)
            trial_rgb = kornia.color.lab_to_rgb(trial, clip=False)
            valid = ((trial_rgb >= 0) & (trial_rgb <= 1)).all(dim=1, keepdim=True)
            low = torch.where(valid, middle, low)
            high = torch.where(valid, high, middle)
        lab[:, 1:] *= torch.where(invalid, low, torch.ones_like(low))
        rgb = kornia.color.lab_to_rgb(lab, clip=False)
    return rgb.clamp(0, 1)


def _encode(path, output, device, matrix, offset, strength, crf, frame_count,
            tone_curve=None, highlight_protection=0., preserve_luminance=False,
            chroma_curve=None):
    written = 0
    progress = comfy.utils.ProgressBar(max(1, frame_count))
    with av.open(path) as source, av.open(str(output), "w") as destination:
        input_stream = source.streams.video[0]
        rate = input_stream.average_rate or input_stream.guessed_rate
        if not rate:
            raise ValueError("The input video has no usable frame rate.")
        input_stream.thread_type = "SLICE"
        input_stream.codec_context.thread_count = 2
        stream = destination.add_stream("libx264", rate=rate,
                                        options={"crf": str(crf), "preset": "fast"})
        stream.width, stream.height = input_stream.width, input_stream.height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 2
        stream.codec_context.time_base = input_stream.time_base
        stream.codec_context.sample_aspect_ratio = input_stream.sample_aspect_ratio or Fraction(1, 1)
        # This workflow targets standard SDR H3 videos; write explicit BT.709 tags.
        stream.codec_context.color_primaries = 1
        stream.codec_context.color_trc = 1
        stream.codec_context.colorspace = 1
        stream.codec_context.color_range = 1
        reformatter = av.video.reformatter.VideoReformatter()
        first_pts = None
        for frame in source.decode(input_stream):
            _interrupt()
            if frame.width != stream.width or frame.height != stream.height:
                raise ValueError("Videos that change resolution within the stream are unsupported.")
            lab = _frame_lab(frame, device)
            matched = (matrix @ lab.reshape(3, -1) + offset).reshape_as(lab)
            if tone_curve is None:
                mixed = torch.lerp(lab, matched, strength)
            else:
                if written >= len(tone_curve):
                    raise ValueError("The source video changed after the analysis pass.")
                gain, shift, color_activity = map(float, tone_curve[written])
                mixed = torch.lerp(lab, matched, strength * color_activity)
                mixed[:, 0].mul_(gain).add_(shift)
            if chroma_curve is not None:
                if written >= len(chroma_curve):
                    raise ValueError("The source video changed after color analysis.")
                saturation_gain, center_a, center_b, shift_a, shift_b = map(float, chroma_curve[written])
                center = mixed.new_tensor([center_a, center_b]).view(1, 2, 1, 1)
                shift_ab = mixed.new_tensor([shift_a, shift_b]).view(1, 2, 1, 1)
                mixed[:, 1:] = (mixed[:, 1:] - center) * saturation_gain + center + shift_ab
                del center, shift_ab
            if highlight_protection > 0:
                mixed[:, 0] = _protect_lightness(mixed[:, 0], lab[:, 0], highlight_protection)
            rgb = (_luminance_safe_rgb(mixed) if preserve_luminance
                   else kornia.color.lab_to_rgb(mixed).clamp_(0, 1))
            array = rgb[0].permute(1, 2, 0).mul_(255).round_().to(device="cpu", dtype=torch.uint8).numpy()
            corrected = av.VideoFrame.from_ndarray(array, format="rgb24")
            corrected = reformatter.reformat(
                corrected, format="yuv420p", src_colorspace="ITU709", dst_colorspace="ITU709",
                src_color_range="JPEG", dst_color_range="MPEG",
                dst_color_trc=1, dst_color_primaries=1, threads=2)
            if frame.pts is not None:
                if first_pts is None:
                    first_pts = frame.pts
                corrected.pts = frame.pts - first_pts
                corrected.time_base = frame.time_base
            else:
                raise ValueError("Video frames without presentation timestamps are unsupported.")
            for packet in stream.encode(corrected):
                destination.mux(packet)
            written += 1
            progress.update(1)
            del lab, matched, mixed, rgb, array, corrected
        for packet in stream.encode():
            destination.mux(packet)
    if not written:
        raise ValueError("The input video contains no decodable frames.")
    if tone_curve is not None and written != len(tone_curve):
        raise ValueError("The source frame count changed after the analysis pass.")
    return written


def _mux_audio(executable, video, source, output, audio_codec="copy"):
    command = [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-i", str(video), "-i", source, "-map", "0:v:0", "-map", "1:a?",
               "-map_metadata", "1", "-map_chapters", "1", "-c:v", "copy", "-c:a", audio_codec]
    if audio_codec == "aac":
        command += ["-b:a", "192k"]
    command += ["-movflags", "+faststart", str(output)]
    # Disk-backed stderr and polling keep both memory and cancellation bounded.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            while True:
                _interrupt()
                try:
                    code = process.wait(timeout=.25)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if code:
            errors.seek(0, os.SEEK_END)
            errors.seek(max(0, errors.tell() - 4000))
            raise RuntimeError(errors.read().decode("utf-8", errors="replace"))


class NegativeMeta_StreamingColorTransfer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NegativeMeta_StreamingColorTransfer",
            display_name="NegativeMeta · Streaming Video Color Transfer",
            category="NegativeMeta/Video", is_output_node=True,
            description="Analyze a file-backed SDR video, then apply one global color transform and encode one frame at a time. Memory does not grow with video length.",
            inputs=[
                io.Video.Input("video"), io.Image.Input("image_ref"),
                io.Combo.Input("method", options=["mkl_lab", "reinhard_lab"]),
                io.Float.Input("strength", default=.55, min=0, max=1, step=.01),
                io.Combo.Input("device", options=["auto", "cuda", "cpu"]),
                io.Int.Input("analysis_frames", default=96, min=0, max=10000,
                             tooltip="Evenly distributed source frames. 0 analyzes all frames; 1 uses the middle frame."),
                io.Int.Input("analysis_max_edge", default=512, min=0, max=8192, step=32,
                             tooltip="Only analysis is downscaled. 0 analyzes original resolution. Output keeps source resolution."),
                io.Int.Input("crf", default=18, min=0, max=51,
                             tooltip="H.264 output quality. Lower means larger file and higher quality."),
                io.String.Input("filename_prefix", default="video/color_recovery_stream/color_recovery"),
            ],
            outputs=[io.Video.Output(), io.String.Output(display_name="saved_path")],
        )

    @classmethod
    def execute(cls, video, image_ref, method, strength, device, analysis_frames,
                analysis_max_edge, crf, filename_prefix):
        source = video.get_stream_source() if hasattr(video, "get_stream_source") else None
        if not isinstance(source, (str, os.PathLike)) or not Path(source).is_file():
            raise ValueError("Use the core Load Video upload node with a video file on disk.")
        source = str(Path(source).resolve())
        if hasattr(video, "get_active_trim_window") and video.get_active_trim_window() != (0., 0.):
            raise ValueError("Use a full untrimmed Load Video input; trim-aware audio is not supported here.")
        if not image_ref.shape[0] or image_ref.shape[-1] < 3:
            raise ValueError("Upload an RGB color reference image.")
        executable = _ffmpeg()
        with av.open(source) as container:
            if not container.streams.video:
                raise ValueError("The uploaded file contains no video stream.")
            stream = container.streams.video[0]
            w, h = stream.width, stream.height
            if w % 2 or h % 2:
                raise ValueError("H.264 4:2:0 output requires even video dimensions.")
            if tuple(video.get_dimensions()) != (w, h):
                raise ValueError("Use a full uncropped video input.")
            if stream.codec_context.color_trc in (16, 18):
                raise ValueError("Convert PQ/HLG HDR video to SDR before using this color transfer workflow.")
            rate = stream.average_rate or stream.guessed_rate
            if not rate:
                raise ValueError("The uploaded video has no frame rate.")
            frame_count = int(stream.frames or 0)
            if not frame_count and stream.duration:
                frame_count = max(1, round(float(stream.duration * stream.time_base * rate)))
        if device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable. Select auto or cpu.")
            target_device = torch.device("cuda")
        else:
            target_device = (comfy.model_management.get_torch_device() if device == "auto"
                             else torch.device("cpu"))
        # Inference tensors are released each iteration; encoder/decoder queues are small.
        with torch.inference_mode():
            logging.info("[NegativeMeta] Streaming color pass 1/2: reference and global source analysis")
            refs = _reference_stats(image_ref, target_device, analysis_max_edge)
            stats = _analyse(source, target_device, frame_count, analysis_frames, analysis_max_edge)
            matrix, offset = _transform(stats, refs, method)
            full_folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory(), w, h)
            filename = f"{name}_{counter:05}_{uuid.uuid4().hex[:8]}.mp4"
            final_path = Path(full_folder) / filename
            with tempfile.TemporaryDirectory(prefix="negativemeta_color_", dir=folder_paths.get_temp_directory()) as temp:
                silent, muxed = Path(temp) / "video.mp4", Path(temp) / "result.mp4"
                logging.info("[NegativeMeta] Streaming color pass 2/2: correct and encode %sx%s on %s", w, h, target_device)
                written = _encode(source, silent, target_device, matrix, offset, strength, crf, frame_count)
                _interrupt()
                try:
                    _mux_audio(executable, silent, source, muxed)
                except RuntimeError as exc:
                    logging.warning("[NegativeMeta] Audio stream copy failed; retrying AAC: %s", exc)
                    _mux_audio(executable, silent, source, muxed, "aac")
                _interrupt()
                shutil.move(str(muxed), str(final_path))
        logging.info("[NegativeMeta] Streaming color output: %s frames, %sx%s, %s fps -> %s",
                     written, w, h, rate, final_path)
        return io.NodeOutput(InputImpl.VideoFromFile(str(final_path)), str(final_path),
                             ui=ui.PreviewVideo([ui.SavedResult(filename, subfolder, io.FolderType.output)]))
