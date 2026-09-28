from .nodes_h3_latent_checkpoint import (
    NegativeMeta_H3LatentCheckpointLoad,
    NegativeMeta_H3LatentCheckpointSave,
    NegativeMeta_H3LatentTailContinuation,
    NegativeMeta_H3VideoTail,
    NegativeMeta_H3CachedClipContext,
)
from .nodes_optional_color_image import (
    NegativeMeta_OptionalColorImage,
    NegativeMeta_ColorReferenceGate,
)


NODE_CLASS_MAPPINGS = {
    "NegativeMeta_H3LatentCheckpointSave": NegativeMeta_H3LatentCheckpointSave,
    "NegativeMeta_H3LatentCheckpointLoad": NegativeMeta_H3LatentCheckpointLoad,
    "NegativeMeta_H3LatentTailContinuation": NegativeMeta_H3LatentTailContinuation,
    "NegativeMeta_H3VideoTail": NegativeMeta_H3VideoTail,
    "NegativeMeta_H3CachedClipContext": NegativeMeta_H3CachedClipContext,
    "NegativeMeta_OptionalColorImage": NegativeMeta_OptionalColorImage,
    "NegativeMeta_ColorReferenceGate": NegativeMeta_ColorReferenceGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NegativeMeta_H3LatentCheckpointSave": "NegativeMeta · H3 Save Last Latent",
    "NegativeMeta_H3LatentCheckpointLoad": "NegativeMeta · H3 Load Last Latent",
    "NegativeMeta_H3LatentTailContinuation": "NegativeMeta · H3 Continue Last Latent",
    "NegativeMeta_H3VideoTail": "NegativeMeta · H3 Video Tail",
    "NegativeMeta_H3CachedClipContext": "NegativeMeta · H3 Cached Clip Context",
    "NegativeMeta_OptionalColorImage": "NegativeMeta · Optional Color Image",
    "NegativeMeta_ColorReferenceGate": "NegativeMeta · Color Reference Gate",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
