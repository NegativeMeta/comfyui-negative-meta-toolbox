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
from .nodes_h3_continuation_canvas import NegativeMeta_H3ContinuationCanvas
from .nodes_h3_latent_files import NegativeMeta_H3LatentFileSave, NegativeMeta_H3LatentFileLoad
from .nodes_streaming_color_transfer import NegativeMeta_StreamingColorTransfer
from .nodes_adaptive_color_transfer import NegativeMeta_AdaptiveStreamingColorTransfer
from .nodes_adaptive_image_color import NegativeMeta_AdaptiveImageColorTransfer

# Auxiliary loader (not a node): gives the installed DaSiWa MiniMaxH3Director a
# soft continuation canvas, so chained clips keep the source latent geometry when
# the Director timeline has no image. Edit nothing in the DaSiWa pack; the patch
# is reapplied on every start and disables itself on DaSiWa 0.5.x (native
# continuity). Remove this import + call to drop the feature.
from .nodes_h3_director_canvas import install_director_canvas

install_director_canvas()

WEB_DIRECTORY = "./web"


NODE_CLASS_MAPPINGS = {
    "NegativeMeta_H3LatentFileSave": NegativeMeta_H3LatentFileSave,
    "NegativeMeta_H3LatentFileLoad": NegativeMeta_H3LatentFileLoad,
    "NegativeMeta_AdaptiveImageColorTransfer": NegativeMeta_AdaptiveImageColorTransfer,
    "NegativeMeta_AdaptiveStreamingColorTransfer": NegativeMeta_AdaptiveStreamingColorTransfer,
    "NegativeMeta_StreamingColorTransfer": NegativeMeta_StreamingColorTransfer,
    "NegativeMeta_H3ContinuationCanvas": NegativeMeta_H3ContinuationCanvas,
    "NegativeMeta_H3LatentCheckpointSave": NegativeMeta_H3LatentCheckpointSave,
    "NegativeMeta_H3LatentCheckpointLoad": NegativeMeta_H3LatentCheckpointLoad,
    "NegativeMeta_H3LatentTailContinuation": NegativeMeta_H3LatentTailContinuation,
    "NegativeMeta_H3VideoTail": NegativeMeta_H3VideoTail,
    "NegativeMeta_H3CachedClipContext": NegativeMeta_H3CachedClipContext,
    "NegativeMeta_OptionalColorImage": NegativeMeta_OptionalColorImage,
    "NegativeMeta_ColorReferenceGate": NegativeMeta_ColorReferenceGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NegativeMeta_H3LatentFileSave": "NegativeMeta · H3 Save Temporary Latent",
    "NegativeMeta_H3LatentFileLoad": "NegativeMeta · H3 Load Latent File",
    "NegativeMeta_AdaptiveImageColorTransfer": "NegativeMeta · Adaptive Image Color Transfer",
    "NegativeMeta_AdaptiveStreamingColorTransfer": "NegativeMeta · Adaptive Streaming Color Transfer",
    "NegativeMeta_StreamingColorTransfer": "NegativeMeta · Streaming Video Color Transfer",
    "NegativeMeta_H3ContinuationCanvas": "NegativeMeta · H3 Continuation Canvas",
    "NegativeMeta_H3LatentCheckpointSave": "NegativeMeta · H3 Save Last Latent",
    "NegativeMeta_H3LatentCheckpointLoad": "NegativeMeta · H3 Load Last Latent",
    "NegativeMeta_H3LatentTailContinuation": "NegativeMeta · H3 Continue Last Latent",
    "NegativeMeta_H3VideoTail": "NegativeMeta · H3 Video Tail",
    "NegativeMeta_H3CachedClipContext": "NegativeMeta · H3 Cached Clip Context",
    "NegativeMeta_OptionalColorImage": "NegativeMeta · Optional Color Image",
    "NegativeMeta_ColorReferenceGate": "NegativeMeta · Color Reference Gate",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
