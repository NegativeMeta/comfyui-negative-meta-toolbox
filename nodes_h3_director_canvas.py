"""Soft continuation canvas for the DaSiWa MiniMax H3 Director.

This module is an auxiliary loader, not a node. When the ComfyUI-NegativeMeta
Toolbox is imported it extends the *installed* DaSiWa ``MiniMaxH3Director``
class with three optional lazy inputs:

    canvas_latent  (LATENT) optional continuation source latent
    canvas_width   (INT)    optional continuation source width
    canvas_height  (INT)    optional continuation source height

Why: with no image in the Director timeline the canvas comes from the panel's
``aspect = auto`` fallback (4:3), so a continuation chain that was built at a
different geometry hits "source and target video latent geometry differs" in the
latent continuation node. When the timeline has no image and neither external
overwrite is linked, the canvas now follows the continuation source instead.

The node type stays ``MiniMaxH3Director``, so the Director's timeline/resolution
UI (which hooks ``node.comfyClass``) keeps working. Nothing inside the DaSiWa
pack is edited, so a pack update cannot lose the feature; if a future DaSiWa
ships native continuity canvasing (0.5.x+) the patch disables itself.

Removal: delete the ``install_director_canvas()`` call in ``__init__.py``.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import threading
import time

PACK_DIR = "ComfyUI-DaSiWa-Nodes"
MODULE_REL = os.path.join("nodes", "nodes_minimax_h3_director.py")
CLASS_NAME = "MiniMaxH3Director"
NATIVE_CANVAS_VERSION = (0, 5, 0)

# Native H3 grid: the generator steps width/height by 32 and the video VAE maps
# 16 pixels per latent cell.
CANVAS_GRID = 32
LATENT_SPATIAL_SCALE = 16
# The workflow's internal fallback clip is a 16x16 placeholder.
MIN_DERIVED_CANVAS_EDGE = 256

CANVAS_INPUTS = {
    "canvas_latent": ("LATENT", {
        "lazy": True,
        "tooltip": "Optional continuation source latent. With no image in the timeline its spatial grid defines the canvas, so chained clips keep identical H3 latent geometry.",
    }),
    "canvas_width": ("INT", {
        "lazy": True, "forceInput": True, "min": 0, "max": 8192, "step": 1,
        "tooltip": "Optional continuation source width (for example the clip tail). Ignored while the timeline has an image.",
    }),
    "canvas_height": ("INT", {
        "lazy": True, "forceInput": True, "min": 0, "max": 8192, "step": 1,
        "tooltip": "Optional continuation source height. Ignored while the timeline has an image.",
    }),
}

_PATCH_FLAG = "_negativemeta_canvas_patch"
_STATE = {"patched": False, "native": False, "waiting": False}
_LOCK = threading.Lock()


# --------------------------------------------------------------------- lookup
def _custom_node_dirs():
    dirs = []
    try:
        import folder_paths
        dirs.extend(folder_paths.get_folder_paths("custom_nodes") or [])
    except Exception:
        pass
    if not dirs:
        dirs.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return dirs


def director_file():
    for base in _custom_node_dirs():
        candidate = os.path.join(base, PACK_DIR, MODULE_REL)
        if os.path.isfile(candidate):
            return os.path.normcase(os.path.abspath(candidate))
    return None


def pack_version():
    for base in _custom_node_dirs():
        pyproject = os.path.join(base, PACK_DIR, "pyproject.toml")
        if not os.path.isfile(pyproject):
            continue
        try:
            with open(pyproject, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip().startswith("version"):
                        raw = line.split("=", 1)[1].strip().strip('"').strip("'")
                        return tuple(int(part) for part in raw.split(".")[:3] if part.isdigit())
        except Exception:
            return ()
    return ()


def _find_loaded_module(target):
    if not target:
        return None
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        try:
            if os.path.normcase(os.path.abspath(path)) == target:
                return module
        except Exception:
            continue
    return None


# ------------------------------------------------------------------- geometry
def _snap(value) -> int:
    return max(CANVAS_GRID, int(round(float(value) / CANVAS_GRID)) * CANVAS_GRID)


def timeline_has_image(timeline_data) -> bool:
    try:
        state = json.loads(timeline_data or "{}")
    except (TypeError, ValueError):
        return False
    if not isinstance(state, dict):
        return False
    for item in state.get("items") or []:
        if isinstance(item, dict) and item.get("enabled", True) and item.get("type") == "image":
            return True
    return False


def latent_canvas_size(latent):
    """(width, height) implied by an H3 AV latent / IMAGE batch, or None."""
    if latent is None:
        return None
    samples = latent.get("samples") if isinstance(latent, dict) else latent
    if samples is None:
        return None
    if hasattr(samples, "unbind"):
        streams = list(samples.unbind())
    elif isinstance(samples, (list, tuple)):
        streams = list(samples)
    else:
        streams = [samples]
    video = streams[0] if streams else None
    shape = getattr(video, "shape", None)
    if shape is None:
        return None
    dims = [int(size) for size in shape]
    if len(dims) == 5:
        return dims[3] * LATENT_SPATIAL_SCALE, dims[4] * LATENT_SPATIAL_SCALE
    if len(dims) == 4:
        if dims[0] == 24:
            return dims[2] * LATENT_SPATIAL_SCALE, dims[3] * LATENT_SPATIAL_SCALE
        if dims[-1] in (1, 3, 4):
            return dims[1], dims[2]
    return None


def continuation_canvas(canvas_latent=None, canvas_width=None, canvas_height=None):
    """(width, height, source) for a usable continuation canvas, else None."""
    dims = latent_canvas_size(canvas_latent)
    source = "continuation latent"
    if dims is None and canvas_width and canvas_height:
        dims = (canvas_width, canvas_height)
        source = "continuation clip"
    if dims is None:
        return None
    width = _snap(dims[0])
    height = _snap(dims[1])
    if min(width, height) < MIN_DERIVED_CANVAS_EDGE:
        return None
    return width, height, source


# ---------------------------------------------------------------------- patch
def patch_class(base):
    """Extend one Director class in place. Returns True when the patch is live."""
    if getattr(base, _PATCH_FLAG, False):
        return True
    if not hasattr(base, "build_guide") or not hasattr(base, "INPUT_TYPES"):
        print("[NegativeMeta] H3 Director canvas: unexpected Director class, patch skipped.")
        return False

    original_input_types = base.INPUT_TYPES
    original_check = base.check_lazy_status
    original_build = base.build_guide
    build_signature = inspect.signature(original_build)
    check_signature = inspect.signature(original_check)

    @classmethod
    def INPUT_TYPES(cls):
        schema = original_input_types.__func__(cls)
        optional = schema.setdefault("optional", {})
        for name, spec in CANVAS_INPUTS.items():
            optional.setdefault(name, spec)
        return schema

    def check_lazy_status(self, *args, **kwargs):
        needed = list(original_check(self, *args, **kwargs) or [])
        bound = check_signature.bind_partial(self, *args, **kwargs)
        values = dict(bound.arguments)
        if _wants_canvas(values):
            for name in CANVAS_INPUTS:
                if values.get(name) is None and name not in needed:
                    needed.append(name)
        return needed

    def build_guide(self, *args, **kwargs):
        # canvas_* are ours: strip them before binding to the original signature.
        canvas_latent = kwargs.pop("canvas_latent", None)
        canvas_width = kwargs.pop("canvas_width", None)
        canvas_height = kwargs.pop("canvas_height", None)
        bound = build_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        external = (values.get("external_width_overwrite") is not None
                    or values.get("external_height_overwrite") is not None)
        if not external and not timeline_has_image(values.get("timeline_data")):
            derived = continuation_canvas(canvas_latent, canvas_width, canvas_height)
            if derived is not None:
                width, height, source = derived
                panel = (values.get("width"), values.get("height"))
                if (width, height) != (int(panel[0]), int(panel[1])):
                    print(
                        "[NegativeMeta] H3 Director canvas "
                        f"{width}x{height} taken from the {source} "
                        f"(no timeline image; panel said {panel[0]}x{panel[1]})"
                    )
                values["width"], values["height"] = width, height
        accepted = {key: value for key, value in values.items()
                    if key in build_signature.parameters or key == "self"}
        return original_build(**accepted)

    base.INPUT_TYPES = INPUT_TYPES
    base.check_lazy_status = check_lazy_status
    base.build_guide = build_guide
    setattr(base, _PATCH_FLAG, True)
    return True


def _wants_canvas(values) -> bool:
    return (
        values.get("external_width_overwrite") is None
        and values.get("external_height_overwrite") is None
        and values.get("canvas_latent") is None
        and values.get("canvas_width") is None
        and values.get("canvas_height") is None
        and not timeline_has_image(values.get("timeline_data"))
    )


# -------------------------------------------------------------------- install
def install_director_canvas(timeout=120.0, interval=0.25):
    """Patch the Director once its module is importable (custom nodes load in
    filesystem order, so the toolbox may come first)."""
    with _LOCK:
        if _STATE["patched"] or _STATE["native"]:
            return True
        version = pack_version()
        if version and version >= NATIVE_CANVAS_VERSION:
            _STATE["native"] = True
            print(
                f"[NegativeMeta] H3 Director canvas: DaSiWa {'.'.join(map(str, version))} "
                "handles continuation canvases natively, auxiliary patch skipped."
            )
            return True

    target = director_file()
    if target is None:
        print(f"[NegativeMeta] H3 Director canvas: {PACK_DIR} not installed, nothing to patch.")
        return False

    module = _find_loaded_module(target)
    base = getattr(module, CLASS_NAME, None) if module else None
    if base is not None and patch_class(base):
        _mark_patched(version)
        return True

    with _LOCK:
        if _STATE["waiting"]:
            return False
        _STATE["waiting"] = True

    def waiter():
        # Custom nodes load in filesystem order and this pack can come first, so
        # wait for ComfyUI to import DaSiWa instead of importing a second copy.
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(interval)
            loaded = _find_loaded_module(target)
            candidate = getattr(loaded, CLASS_NAME, None) if loaded else None
            if candidate is not None and patch_class(candidate):
                _mark_patched(pack_version())
                return
        print("[NegativeMeta] H3 Director canvas: DaSiWa Director never loaded, patch skipped.")

    threading.Thread(target=waiter, name="negativemeta-h3-canvas", daemon=True).start()
    return False


def _mark_patched(version):
    with _LOCK:
        _STATE["patched"] = True
        _STATE["waiting"] = False
    label = ".".join(map(str, version)) if version else "unknown"
    print(
        f"[NegativeMeta] H3 Director canvas: auxiliary patch installed on "
        f"{CLASS_NAME} (DaSiWa {label}); canvas follows canvas_latent / canvas_width+canvas_height "
        "when the timeline has no image."
    )
