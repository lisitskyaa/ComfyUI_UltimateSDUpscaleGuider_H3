# ComfyUI_UltimateSDUpscaleGuider

> **This is a fork of [ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) with added Guider support.**

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for performing the image-to-image diffusion process on large images in tiles. This approach improves the details that is commonly found on upscaled images while reducing hardware requirements and maintaining an image size that the diffusion model is trained on.

## Fork Changes

This fork adds two key improvements over the original Ultimate SD Upscale:

1. **Guider input support** - Use custom guiders (PerpNegGuider, CFGGuider, etc.) during tiled upscaling
2. **Context Only Overlap mode** - A new tile processing mode that eliminates seams without the blur that comes from seam fix

### New Nodes Added

| Node | Description |
|------|-------------|
| **Ultimate SD Upscale (Guider)** | Full upscaling with guider support |
| **Ultimate SD Upscale (No Upscale, Guider)** | Tile refinement without initial upscaling, with guider support |

### What's Different

The new Guider nodes accept:
- **GUIDER** - A guider that encapsulates model, conditioning, and CFG (from PerpNegGuider, CFGGuider, etc.)
- **SAMPLER** - From KSamplerSelect node
- **SIGMAS** - From BasicScheduler or other scheduler nodes (configure denoise here)

Instead of the standard nodes' separate inputs for:
- model, positive, negative, cfg, sampler_name, scheduler, steps, denoise

### Example Workflow

```
[KSamplerSelect] ──────────────────────────┐
[BasicScheduler (with denoise)] ───────────┤
[PerpNegGuider] ───────────────────────────┤
[Load VAE] ────────────────────────────────┼──► [Ultimate SD Upscale (Guider)] ──► IMAGE
[Load Upscale Model] ──────────────────────┤
[Image] ───────────────────────────────────┘
```

### Note on Conditioning

The Guider nodes skip per-tile conditioning cropping since conditioning is internal to the guider. This works perfectly for text-based guiders (PerpNegGuider, CFGGuider). For spatial conditioning (ControlNet, GLIGEN), use the original non-Guider nodes instead.

---

## Context Only Overlap Mode

### The Problem with Traditional Tiled Upscaling

When upscaling large images with tiles, each tile is processed independently. Even with padding for context, the actual pixels at tile boundaries are generated separately, leading to visible seams where tiles meet.

The traditional solution is **seam fix** - a post-processing pass that re-processes the seam areas. While effective, seam fix can introduce blur and requires additional processing time.

### Our Solution: Prevent Seams Instead of Fixing Them

The **Context Only Overlap** mode takes a different approach: instead of fixing seams after they occur, it prevents them from happening in the first place.

**How it works:**
1. Each tile extends into its neighbors' territory by the `tile_padding` amount
2. When processing subsequent tiles, the already-denoised overlap regions are used as **context for the attention mechanism** but are **not re-denoised**
3. This allows each tile to "see" what its neighbors generated and create coherent continuations

### Tile Overlap Mode Options

| Mode | Description | Use Case |
|------|-------------|----------|
| **Ignore Overlap** | Tiles use minimal size, no overlap handling | Fastest, may have visible seams |
| **Reprocess Overlap** | Uniform tile sizes, overlap regions may be generated independently | Original behavior, seams possible |
| **Context Only Overlap** | Overlap regions provide context without re-denoising | Best coherence, no seam fix blur |

### Benefits of Context Only Overlap

- **No seams at tile boundaries** - Tiles share context and create coherent transitions
- **No blur from seam fix** - Eliminates the need for post-processing seam fix passes
- **More coherent images** - Adjacent tiles "see" each other's output through the overlap context
- **Works with all tile positions** - Including edge and corner tiles (bidirectional context sharing)

### Recommended Settings

For best results with Context Only Overlap mode:
- Set `tile_overlap_mode` to "Context Only Overlap"
- Use a `tile_padding` of 64-128 pixels (this becomes the overlap width)
- `mask_blur` of 8-16 provides smooth blending at tile edges
- Seam fix can typically be set to "None" since seams are prevented
- Enable `anchor_context` to make the read-only overlap a hard guarantee during sampling: the
  already-processed overlap is anchored to the original latent at every step (inpaint-style
  denoise mask), not just excluded at composite time. The mode is retained during seam-fix
  passes, whose gradient masks become soft anchors.

## Masked Upscaling

Both Guider nodes accept an optional `mask` input (MASK type). White areas of the mask are
re-diffused; black areas are left untouched. With no mask connected, behavior is identical to
before.

### How It Works

- **Tile skipping is the speed win.** Tiles whose rectangle does not touch the mask are skipped
  entirely — no VAE encode, no sampling. A mask covering a small region of a large image only pays
  for the tiles it touches.
- **Full tile context.** Tiles that do touch the mask are sampled exactly as without a mask — the
  model sees the full tile, so the re-diffused region blends with real surroundings.
- **Same feathering as tile seams.** The mask is intersected with each tile's own mask and
  feathered by the existing `mask_blur`, so region boundaries blend with the exact same process
  tile seams already use. Note the edit extends about `mask_blur` pixels past the mask edge; use
  `mask_blur=0` for a hard boundary.
- **Any resolution.** The mask may be any size (e.g. drawn on the pre-upscale image) and is
  resized to the upscaled canvas. Grayscale values give partial blending.

### Anchoring Context (anchor_context)

By default, the parts of a tile that will not be composited back — everything outside the mask
plus the tile padding ring — are freely re-diffused during sampling and then discarded at
composite time. Enabling `anchor_context` holds those parts to the original image throughout
sampling instead (inpaint-style denoise mask, re-noised to the current step each step), so the
model sees the true surroundings as context rather than its own re-imagining of them. This can
tighten blending at mask edges.

- Only takes effect when a mask is connected or `tile_overlap_mode` is "Context Only Overlap";
  otherwise it has no effect.
- Note: an all-white mask with `anchor_context` enabled is NOT equivalent to running with no
  mask — the tile padding ring is anchored.

### Compatibility

- Works with all three `tile_overlap_mode` values and all seam fix modes (seam fix passes are
  clipped to the masked region the same way).
- NOT compatible with `batch_size > 1` — the node raises an error; use `batch_size=1` with a mask.

### Use Case 1: Two-Stage Region Upscaling

Upscale the background and a character with different guider/sampler/sigmas each, in one chained
workflow. Run the background pass first so the character pass sees the finished background as tile
context:

```
[Mask] ──► [InvertMask] ──► mask ┐
[Background Guider/Sampler/Sigmas] ┼──► [Ultimate SD Upscale (Guider)] ─┐
[Image] ───────────────────────────┘                                    │
                                                                        ▼
[Mask] ────────────────────► mask ┬──► [Ultimate SD Upscale (No Upscale, Guider)] ──► IMAGE
[Character Guider/Sampler/Sigmas] ┘
```

### Use Case 2: Upscale Inpainting for Huge Images

Inpainting a huge image directly is expensive. Instead: downsample a copy, inpaint on the small
copy, paste the upscaled patch back into the full-size image, then use the No Upscale node with a
mask over the patched region to re-diffuse only that region so it matches the rest:

```
[Huge Image] ─► [Downscale] ─► [Inpaint] ─► [Upscale patch + paste back] ─┐
                                                                          ▼
[Patch-Region Mask] ─► mask ──► [Ultimate SD Upscale (No Upscale, Guider)] ──► IMAGE
```

## Installation

### Using Git
1. Git must be installed on your system. Verify by running `git -v` in a terminal.
2. Enter the following command from the terminal starting in ComfyUI/custom_nodes/
    ```
    git clone https://github.com/Blakeem/ComfyUI_UltimateSDUpscaleGuider
    ```

### Manual Download
1. Download the zip file by clicking the green "Code" button on the GitHub repository page and selecting "Download ZIP".
2. Create a new folder in the `ComfyUI/custom_nodes/` directory (e.g. `ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscaleGuider`).
3. Extract the contents of the zip file into that folder.

### Original Version (without Guider support)
If you don't need Guider support, you can install the original version:
- Via ComfyUI Manager: Search for "UltimateSDUpscale"
- Via comfy-cli: `comfy node install comfyui_ultimatesdupscale`
- Via Git: `git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale`


## Usage

Nodes can be found in the node menu under `image/upscaling`.

Documentation for the nodes can be found in the [`js/docs/`](js/docs/) folder, or viewed within the application by right-clicking the relevant node and selecting the info icon.

Details about most of the parameters can be found [here](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111/wiki/FAQ#parameters-descriptions).

Example workflows can be found in the [`example_workflows/`](example_workflows/) folder. You can also find them in the ComfyUI application under the Templates menu, scroll down the left sidebar to find the Extensions section, then selecting this repository.

## References
* **Upstream fork**: https://github.com/ssitu/ComfyUI_UltimateSDUpscale
* Ultimate Stable Diffusion Upscale script for the Automatic1111 Web UI: https://github.com/Coyote-A/ultimate-upscale-for-automatic1111
* ComfyUI: https://github.com/comfyanonymous/ComfyUI
