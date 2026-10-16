# Negative Meta Toolbox

Small ComfyUI custom-node pack for MiniMax H3 latent continuation, optional workflow inputs, and memory-bounded video color transfer.

## Nodes

### Adaptive Streaming Color Transfer

### Adaptive Image Color Transfer (Dasiwa Xtra integration)

`NegativeMeta_AdaptiveImageColorTransfer` accepts and returns an IMAGE batch,
using the same a/b-only reference transform, bounded temporal tint/saturation
curves, source-based lightness smoothing and highlight/gamut protection as the
streaming variant. FPS must describe the actual input batch (including frame
interpolation). DaSiWa keeps control of audio and encoding.

Processing uses per-frame Lab tensors and small scalar profiles, but both the
input and output IMAGE batches remain resident. This is for generated segments,
not a replacement for file-streaming recovery of long videos. Temporal windows
do not carry state between generation jobs. Turn the workflow Color Transfer
switch OFF for a complete lazy bypass, or set temporal color intensity to 0 to
disable only chromatic stabilization. Settings controls in the alternative are
appended, preserving all original indices. Restart ComfyUI and refresh its
frontend after updating the toolbox.

### Streaming adaptive details

Temporal Color Stabilization adds bounded per-frame a/b mean-offset and isotropic
saturation correction toward a centered, timestamp-based source-color trajectory.
UI defaults: intensity 0.8, color window 3 seconds, maximum tint offset 8 Lab
a/b vector units before intensity scaling; saturation gain is capped at ±15%
at full intensity. Set intensity 0 to disable this stage. It leaves Lab L
untouched, respects detected scene boundaries and retains only five floats per
frame. Camera/subject changes can bias estimates; long segments are not forced
to one fixed palette. Controls are appended to avoid shifting existing widgets;
legacy direct calls default to temporal color correction off.

The new `preserve_video` Luminance Mode transfers only Lab a/b color statistics,
independently of L, and stabilizes lightness against a centered temporal curve
from the source video. Reference brightness, Target Brightness Offset and Preserve
Lighting do not affect this mode. Set Exposure and Contrast Stabilization to 0
to retain source Lab L without temporal correction. Out-of-gamut chroma is
compressed toward neutral at fixed L rather than RGB-clipped. This may desaturate
extreme colors. Global color matching is not subject-aware; different compositions
can still bias hue/saturation. Smoothing softens abrupt exposure transitions but
does not equalize entire long segments to one absolute brightness.

The previous `reference` mode remains available. The mode widget is appended,
so existing widget indices are unchanged; legacy direct calls retain reference
mode. Restart ComfyUI, refresh its frontend and reload the updated workflow.

Variant of Streaming Video Color Transfer for clips whose exposure varies over
time. The base color grade stays global; an additional Lab lightness correction
is computed for every frame. A centered, timestamp-based smoothing window
defines the desired appearance, while the actual frame's exposure is compensated
immediately. Smoothing only a correction matrix would lag a sudden input jump.

Defaults: color strength `0.55`, exposure stabilization `1.0`, contrast
stabilization `0`, preserve lighting `0.2`, temporal window `1.0` second, maximum
lightness correction `25` Lab L units. Preserve lighting `0` targets constant
reference brightness; `1` retains the source's smoothly varying brightness.
Optional contrast stabilization is limited to moderate gains. Scene boundaries
are estimated from exposure-normalized structure; this heuristic can mistake
large motion for a cut. Fade/blank protection attenuates changes near flat
black/white frames.

Target Brightness Offset lowers or raises the reference-derived target in Lab L
units (UI default `-8`). Try `-12` if a bright reference background causes an
overly bright result. For full effect, use Exposure Stabilization `1` and Preserve
Lighting `0`. Highlight Protection (UI default `0.8`) attenuates positive
lightness lifts in source highlights without cancelling darkening. It cannot
recover detail already clipped in the source. Always grade the original video
rather than stacking correction passes. New controls are appended to preserve
the existing widget order; legacy direct calls retain zero-offset behavior.

All frames get exposure statistics even when Global Color Samples is `96`.
Only small scalar summaries and correction curves are retained per frame (well
under a few MB for a several-minute clip), never full decoded frames. Output
encoding and audio preservation use the same streaming path as the global node.

This is a statistical tonal-stabilization implementation, not semantic matching,
optical flow, or a reproduction of a research model. Changes in image composition
can influence the brightness estimates. If it flattens an intentional lighting
change, lower Exposure Stabilization or raise Preserve Lighting. It cannot
restore clipped highlights or detail lost in very dark footage.

Related approaches: [Video tonal stabilization via color states smoothing](https://pubmed.ncbi.nlm.nih.gov/25248186/)
and [TS Color Match](https://github.com/AlexYez/comfyui-timesaver#ts-color-match).

### Streaming Video Color Transfer

Color-corrects a finished file-backed SDR video toward a reference image without
loading the entire clip as an IMAGE batch. Connect the core **Load Video** VIDEO
output directly to this node, and connect **Load Image** as the color reference.
The node saves the complete MP4 and shows a video preview; no separate
Get Video Components or Video Combine node is needed.

The first pass pools Lab color statistics from evenly distributed source frames
and the reference. The second pass applies one fixed transform to every frame at
the original resolution and encodes frames immediately. RAM/VRAM use is bounded
by a few frames and encoder buffers, rather than by video duration. This also
preserves relative changes in lighting instead of independently rematching each
frame. It does not perform local, semantic, or scene-specific color grading.

Defaults: `mkl_lab`, strength `0.55`, device `auto`, `96` analysis frames at a
maximum edge of `512` px, H.264 CRF `18`. Analysis downscaling does not resize the
output. Analysis frames `0` analyzes every frame; analysis maximum edge `0` uses
original-resolution statistics. `reinhard_lab` offers a simpler channel match.
The sampled analysis approximates the native Transfer Color `uniform` baseline.

Requires current ComfyUI with core Transfer Color/VIDEO APIs, PyAV, Kornia,
PyTorch, and FFmpeg (or imageio-ffmpeg). Use full, uncropped SDR files with even
dimensions. Output is re-encoded as 8-bit H.264 MP4, retaining source frame
timestamps and sample aspect ratio. Audio streams are copied into the result;
MP4-incompatible audio falls back to AAC. The node writes temporary encoded
files to disk and cleans them after completion, cancellation, or failure.

Restart ComfyUI and refresh the browser after installing this node. Saved videos
default to `ComfyUI/output/video/color_recovery_stream/`.

### H3 Save Clip Latent / H3 Load Latent File

`NegativeMeta_H3LatentFileSave` always saves each fresh **raw base AV latent**
temporarily in `temp/negative_meta_h3_latents/h3_<node-key>_latest.safetensors`.
Connect Raw H3 Clip Latent, seed and Skip Temporary Save outputs from Settings.
Each node replaces only its own previous temporary file, atomically after a
successful write. Resume/upscale skips sampling and preserves the file. There
is no Save Enabled switch and no automatic permanent per-clip archive.

Click **Download Latent** after execution to retain the current clip using your
browser's normal download/save behavior. Download before the next generation;
temp files can also be cleared on ComfyUI restart. Filename Prefix labels the
download. AV streams, masks and seed are kept without VAE encoding or scaling.
The button and read-only temporary path do not serialize as backend widgets.
After a frontend reload, execute the node again to restore the download descriptor.
The legacy latest checkpoint remains independent for automatic continuation and
Upscale Last Job. Older archives are not deleted.

**Select Latent File** opens a browser file-selection dialog and uploads the chosen
NegativeMeta H3 AV `.safetensors` to `input/negative_meta_h3_latents/` using ComfyUI's
standard upload endpoint. It is retained for workflow reloads. **Use Last Temporary
Latent** clears the selection and follows the legacy latest checkpoint instead.
An absolute File Path remains available to avoid uploads/413 size limits. File
checks are lazy, only when the continuation source is selected. Unsupported or
missing files fail explicitly; no silent fallback is applied to a chosen path.

In Dasiwa Turbo 4–8 Steps, **Continue Source = 1 Hard Latent** uses this loader;
**0 Clip** retains external-video continuation. A selected file remains pinned
until changed/cleared. Keep the same H3 geometry and compatible model/VAEs. The
file is a sampled window including hidden context, not a cumulative scene. RGB
color/upscale changes are not fed back into it, and its saved seed does not
override the new generation seed. Restart ComfyUI, refresh the frontend and
reload the migrated workflow after updating these nodes.

### Legacy H3 Save Last Latent

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

### Optional Color Image

Upload an image for an optional color-reference override. The default **None**
selection returns `loaded = false`, so a workflow can keep its clip/manual
reference fallback without a required image connection. Selecting an image
returns that image and `loaded = true`; a lazy switch can then give it priority.

### H3 Continuation Canvas

Provides a clip-aspect fallback for a DaSiWa MiniMax H3 Director guide. When clip
continuation is enabled, Aspect is Auto, and the Director has no active visual
references, the node reads the VIDEO dimensions without decoding its frames.
Resolution Auto keeps the short side at 768 pixels; MP and custom-MP presets
keep their selected pixel area. Output dimensions align to 32 pixels. Feed its
updated guide and width/height to both conditioning and latent creation.

Director image/video references, explicit aspect/fixed-pixel settings, external
canvas overrides, continuation OFF, and Last Saved Latent retain Director sizing.
The single-frame 16x16 internal video placeholder is ignored.

The included browser extension copies the connected Director's current
resolution configuration into `director_settings` on each queue/API export.
After updating the pack, restart ComfyUI and refresh the browser. The Director
readout still shows its original canvas; the NegativeMeta console message shows
the effective clip-based canvas used downstream. Standalone API clients must
provide the same settings JSON explicitly.

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
- The checkpoint/color nodes are independent of ComfyUI-DaSiWa-Nodes. The optional `H3 Continuation Canvas` node integrates with its MiniMax H3 Director guide and resolution presets.

## License

MIT. See LICENSE.
