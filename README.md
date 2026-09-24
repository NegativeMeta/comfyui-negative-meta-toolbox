# Negative Meta Toolbox

Small ComfyUI custom-node pack for saving and reloading the latest MiniMax H3 audio/video latent checkpoint.

## Nodes

### H3 Save Last Latent

Saves the video and audio streams from a MiniMax H3 nested AV latent, along with optional noise masks and the seed, to:

    ComfyUI/output/latents/negativemeta_h3_last_latent.safetensors

Each fresh sampling pass replaces that checkpoint. When **Resume Last Latent** is enabled, the save node preserves the existing checkpoint and lazily skips the fresh latent input.

### H3 Load Last Latent

Loads the saved video/audio streams as a nested LATENT and returns the saved seed. It is intended for a workflow branch that resumes from the last generated latent, such as a latent upscale and refinement pass.

If no checkpoint exists, first run a fresh generation with **Resume Last Latent** disabled.

### H3 Continue Last Latent

Creates an exact MiniMax H3 audio/video continuation directly from the saved
latent. It takes a fresh target latent plus **H3 Load Last Latent**, copies the
previous latent's final shared AV run into the target head, and protects that
head with H3's separate video/audio noise masks. The previous clip is never
decoded to pixels/audio or re-encoded through the VAEs.

Use this node for a generated clip from the same H3 workflow, at the same
resolution and with compatible model/VAEs. The default 39-frame context is an
exact 24 fps video / 40 Hz audio boundary. It is intentionally different from
the checkpoint-resume route used for latent upscale: continuation seeds a new
clip from only the old latent's tail.

### H3 Video Tail

Reads only the final exact H3 continuation window from the core **Load Video**
file input. It seeks with FFmpeg, returns 24 fps images plus matching stereo
audio, and never materializes the complete source video as an IMAGE batch.

The workflow uses the 39-frame default (1.625 seconds). This is the first
shared MiniMax H3 video/audio boundary and is suitable for a seamless AV
continuation.

### H3 Cached Clip Context

Creates the masked AV prefix for an external source clip. On a cache miss it
encodes the tail with the H3 video/audio VAEs; on later runs of the unchanged
file at the same H3 geometry, it loads that encoded prefix directly and skips
both VAEs.

Cache files live in:

    ComfyUI/output/latents/negativemeta_h3_clip_prefix_cache/

They are invalidated automatically when the source file path, size or modified
time, context length, target geometry, or VAE class changes. Delete that folder
manually after replacing VAE weights that retain the same class name.

## Installation

Clone this repository into ComfyUI's custom_nodes directory, then restart ComfyUI:

    cd ComfyUI/custom_nodes
    git clone https://github.com/NegativeMeta/comfyui-negative-meta-toolbox

ComfyUI provides the comfy and folder_paths modules. Install the package dependency in the same Python environment as ComfyUI if it is missing:

    pip install -r requirements.txt

## Notes

- The checkpoint path is shared by this node pack within one ComfyUI output directory; a fresh run overwrites the previous checkpoint.
- The nodes expect MiniMax H3 AV latents with exactly two streams: video and audio.
- `H3 Continue Last Latent` requires ComfyUI's native MiniMax H3 per-token AV noise-mask support.
- `H3 Video Tail` needs FFmpeg (or `imageio-ffmpeg`) and a file-backed core `Load Video` output.
- `H3 Cached Clip Context` requires `ComfyUI-H3-Motion-Context-MultiRef` for its H3 AV-mask compatibility layer.
- The pack is independent of ComfyUI-DaSiWa-Nodes.

## License

MIT. See LICENSE.
