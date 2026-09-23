from .nodes_h3_latent_checkpoint import (
    NegativeMeta_H3LatentCheckpointLoad,
    NegativeMeta_H3LatentCheckpointSave,
)


NODE_CLASS_MAPPINGS = {
    "NegativeMeta_H3LatentCheckpointSave": NegativeMeta_H3LatentCheckpointSave,
    "NegativeMeta_H3LatentCheckpointLoad": NegativeMeta_H3LatentCheckpointLoad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NegativeMeta_H3LatentCheckpointSave": "NegativeMeta · H3 Save Last Latent",
    "NegativeMeta_H3LatentCheckpointLoad": "NegativeMeta · H3 Load Last Latent",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
