"""Use a continuation clip's aspect when the Director has no visual reference."""

import json
import math


# Same pixel-area presets as the DaSiWa Director (MP means 1024 * 1024).
RESOLUTION_PRESETS = {
    "144p": .0352, "240p": .0977, "360p": .22, "480p": .391,
    "540p": .494, "576p": .396, "720p": .879, "900p": 1.373,
    "1024p": 1., "1080p": 1.978, "1152p": 2.25, "1440p": 3.516,
    "2160p": 7.91, "2K": 3.906, "4K": 7.91,
    "0.26 MP - Preview": .26, "0.36 MP - Small": .36,
    "0.52 MP - SD": .52, "0.65 MP - Balanced": .65,
    "0.83 MP - HD": .83, "1.00 MP - 1024p": 1., "1.05 MP - HD+": 1.05,
    "1.20 MP - HD++": 1.20, "1.35 MP - 2K lite": 1.35,
    "1.55 MP - 2K": 1.55, "1.65 MP - 2K+": 1.65,
    "1.75 MP - QHD": 1.75, "2.10 MP - FHD": 2.10,
    "3.30 MP - QHD+": 3.30, "4.75 MP - 2K Pro": 4.75,
    "6.50 MP - Production": 6.50, "8.30 MP - UHD": 8.30,
}


def snap32(value):
    return max(32, int(math.floor(float(value) / 32 + .5)) * 32)


class NegativeMeta_H3ContinuationCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "guide": ("MINIMAX_H3_DIRECTOR_GUIDE",),
            "continuation_mode": ("INT", {"forceInput": True}),
            "continuation_source": ("INT", {"forceInput": True}),
            "director_settings": ("STRING", {"default": "", "multiline": False}),
        }, "optional": {"clip_video": ("VIDEO", {"lazy": True})}}

    RETURN_TYPES = ("MINIMAX_H3_DIRECTOR_GUIDE", "INT", "INT")
    RETURN_NAMES = ("guide", "width", "height")
    FUNCTION = "resolve"
    CATEGORY = "NegativeMeta/MiniMax H3"

    @staticmethod
    def needs_clip(guide, continuation_mode, continuation_source, director_settings):
        if int(continuation_mode) == 0 or int(continuation_source) != 0:
            return False
        if any(guide.get(key) for key in ("ref_images", "ref_videos")):
            return False
        if any(guide.get(key) is not None for key in ("first_frame", "last_frame")):
            return False
        if not director_settings:
            raise ValueError("H3 clip aspect: refresh ComfyUI's browser to load the NegativeMeta canvas extension, then queue again.")
        settings = json.loads(director_settings)
        resolution = settings.get("resolution") or {}
        return (not settings.get("external_canvas")
                and resolution.get("aspect", "auto") == "auto"
                and not (resolution.get("resolution") == "custom"
                         and resolution.get("custom_mode") == "fixed"))

    def check_lazy_status(self, guide, continuation_mode, continuation_source,
                          director_settings, clip_video=None):
        if self.needs_clip(guide, continuation_mode, continuation_source, director_settings) and clip_video is None:
            return ["clip_video"]
        return []

    def resolve(self, guide, continuation_mode, continuation_source,
                director_settings, clip_video=None):
        width, height = int(guide["width"]), int(guide["height"])
        if not self.needs_clip(guide, continuation_mode, continuation_source, director_settings) or clip_video is None:
            return guide, width, height
        clip_width, clip_height = clip_video.get_dimensions()
        # The workflow's optional-video sentinel is a single 16x16 frame.
        if (clip_width, clip_height) == (16, 16) and clip_video.get_frame_count() == 1:
            return guide, width, height
        if clip_width <= 0 or clip_height <= 0:
            raise ValueError("H3 clip aspect: the continuation video has invalid dimensions")
        resolution = (json.loads(director_settings).get("resolution") or {})
        preset = resolution.get("resolution", "auto")
        aspect = clip_width / clip_height
        if preset == "auto":
            width, height = ((snap32(768 * aspect), 768) if aspect >= 1
                             else (768, snap32(768 / aspect)))
        else:
            mp = (max(.01, float(resolution.get("custom_mp", 1)))
                  if preset == "custom" else RESOLUTION_PRESETS.get(preset))
            if mp is None:
                raise ValueError(f"H3 clip aspect: unsupported Director resolution preset {preset!r}")
            h = math.sqrt(mp * 1024 * 1024 / aspect)
            width, height = snap32(h * aspect), snap32(h)
        updated = dict(guide, width=width, height=height)
        print(f"[NegativeMeta] H3 clip aspect fallback: {clip_width}x{clip_height} -> {width}x{height}; resolution={preset}")
        return updated, width, height
