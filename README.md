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

## Installation

Clone this repository into ComfyUI's custom_nodes directory, then restart ComfyUI:

    cd ComfyUI/custom_nodes
    git clone https://github.com/NegativeMeta/comfyui-negative-meta-toolbox

ComfyUI provides the comfy and folder_paths modules. Install the package dependency in the same Python environment as ComfyUI if it is missing:

    pip install -r requirements.txt

## Notes

- The checkpoint path is shared by this node pack within one ComfyUI output directory; a fresh run overwrites the previous checkpoint.
- The nodes expect MiniMax H3 AV latents with exactly two streams: video and audio.
- The pack is independent of ComfyUI-DaSiWa-Nodes.

## License

MIT. See LICENSE.
