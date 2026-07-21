# ComfyUI_UltimateSDUpscaleGuider

> **A fork of [ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) with Guider support, Context Only Overlap, Masked Upscaling, and context anchoring.**

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for running the image-to-image diffusion process on large images in tiles. Tiling improves the detail commonly lost on upscaled images while keeping VRAM use low and the working size close to what the diffusion model was trained on.

## Fork Changes

1. **Guider input support**: drive tiled upscaling with any custom guider. For example, PerpNegGuider enables negative prompts at CFG 1 for higher quality upscales.
2. **Context Only Overlap mode**: tiles use already-generated neighbor pixels as context without re-denoising them, which greatly improves consistency and removes seams.
3. **Masked Upscaling**: an optional mask restricts re-diffusion to a region, so different parts of an image can be upscaled with different models or settings, or an inpainted patch refined to match its surroundings.
4. **Context anchoring** (`anchor_context`): holds unprocessed areas to the original image during sampling, preventing detail drift between sections and joining seams to the real image.

### New Nodes

| Node | Description |
|------|-------------|
| **Ultimate SD Upscale (Guider)** | Full upscaling with guider support |
| **Ultimate SD Upscale (No Upscale, Guider)** | Tile refinement without initial upscaling, with guider support |

### What's Different

The Guider nodes accept:
- **GUIDER**: encapsulates model, conditioning, and CFG (from PerpNegGuider, CFGGuider, etc.)
- **SAMPLER**: from KSamplerSelect
- **SIGMAS**: from BasicScheduler or other scheduler nodes (configure denoise here)

These replace the standard nodes' separate model, positive, negative, cfg, sampler_name, scheduler, steps, and denoise inputs.

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

The Guider nodes skip per-tile conditioning cropping since conditioning is internal to the guider. This works for text-based guiders (PerpNegGuider, CFGGuider). For spatial conditioning (ControlNet, GLIGEN), use the original non-Guider nodes instead.

---

## Context Only Overlap Mode

Tiles are processed independently, so pixels at tile boundaries are generated separately and can show seams. The usual remedy is a seam fix pass, which costs extra time and can blur.

Context Only Overlap prevents the seams instead of fixing them afterward:

1. Each tile extends into its neighbors by the `tile_padding` amount.
2. Already-denoised overlap regions are used as context for the attention mechanism but are not re-denoised.
3. Each tile sees what its neighbors generated and continues it coherently, including at edge and corner tiles.

### Tile Overlap Mode Options

| Mode | Description | Use Case |
|------|-------------|----------|
| **Ignore Overlap** | Tiles use minimal size, no overlap handling | Fastest, may have visible seams |
| **Reprocess Overlap** | Uniform tile sizes, overlap regions may be generated independently | Original behavior, seams possible |
| **Context Only Overlap** | Overlap regions provide context without re-denoising | Best coherence, no seam fix blur |

### Recommended Settings

- Set `tile_overlap_mode` to "Context Only Overlap"
- Use a `tile_padding` of 64-128 pixels (this becomes the overlap width)
- `mask_blur` of 8-16 provides smooth blending at tile edges
- Seam fix can typically be set to "None" since seams are prevented
- Enable `anchor_context` to hold the overlap context fixed during sampling as well, not just at composite time (see Anchoring Context below)

## Masked Upscaling

Both Guider nodes accept an optional `mask` input. White areas are re-diffused; black areas are left untouched. With no mask connected, behavior is unchanged.

### How It Works

- **Tile skipping.** Tiles that do not touch the mask are skipped entirely, with no VAE encode and no sampling, so a small region only pays for the tiles it touches.
- **Full tile context.** Processed tiles still see the whole tile, so the re-diffused region blends with its real surroundings.
- **Same feathering as tile seams.** The mask is feathered by `mask_blur`, and the edit extends about `mask_blur` pixels past the mask edge. Use `mask_blur=0` for a hard boundary.
- **Any resolution.** The mask is resized to the upscaled canvas, so it can be drawn on the pre-upscale image. Grayscale values give partial blending.
- **Batched images.** A batch of images works as usual. A single mask applies to every image in the batch; a batch of masks maps one mask to each image, so each image is edited only in its own region (a tile is sampled when any image in the batch needs it). With `anchor_context`, each image is anchored to its own original content.

### Anchoring Context (anchor_context)

By default, areas a tile will not keep are still re-diffused during sampling and thrown away afterward, so the model works against a re-imagined version of the surroundings that can drift from the real image. Enabling `anchor_context` holds those areas to the original image at every sampling step, so the model always sees the true surroundings.

In practice:

- Details stay consistent across sections instead of drifting apart during denoising, giving a more even and slightly sharper result across the whole image.
- Seams blend better because new content is joined to the real image rather than to a drifted version of it.
- Anchoring applies only to pixels the composite will not replace; everything that gets composited is still fully refined.
- Takes effect when a mask is connected or `tile_overlap_mode` is "Context Only Overlap"; otherwise it does nothing.

### Compatibility

- Works with all three `tile_overlap_mode` values and all seam fix modes (seam fix passes are clipped to the masked region the same way).
- Not compatible with `batch_size > 1`; the node raises an error. Use `batch_size=1` with a mask. Note that `batch_size` is the number of *tiles* sampled per call — batched *images* are fully supported (see above).

### Use Case 1: Two-Stage Region Upscaling

Upscale the background and a character with different guider/sampler/sigmas each, in one chained workflow. Run the background pass first so the character pass sees the finished background as tile context:

```
[Mask] ──► [InvertMask] ──► mask ┐
[Background Guider/Sampler/Sigmas] ┼──► [Ultimate SD Upscale (Guider)] ─┐
[Image] ───────────────────────────┘                                    │
                                                                        ▼
[Mask] ────────────────────► mask ┬──► [Ultimate SD Upscale (No Upscale, Guider)] ──► IMAGE
[Character Guider/Sampler/Sigmas] ┘
```

### Use Case 2: Upscale Inpainting for Huge Images

Inpainting a huge image directly is expensive. Instead: downsample a copy, inpaint on the small copy, paste the upscaled patch back into the full-size image, then use the No Upscale node with a mask over the patched region to re-diffuse only that region so it matches the rest:

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
