# Capybara ComfyUI Custom Nodes

Custom [ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for the Capybara unified visual creation pipeline.

## Installation

1. Make sure Capybara's dependencies are installed (see the main [README](../README.md)).

2. Symlink the Capybara project root into ComfyUI's `custom_nodes/`:

```bash
ln -s /path/to/Capybara /path/to/ComfyUI/custom_nodes/Capybara
```

3. Launch ComfyUI with the same Python environment:

```bash
conda activate capybara
python /path/to/ComfyUI/main.py --listen 0.0.0.0 --port 8888
```

The Capybara nodes should appear under the **Capybara** category in the node menu.

## Nodes

| Node | Description |
| --- | --- |
| **Capybara Load Pipeline** | Load all model components (transformer, VAE, text encoders, vision encoder, scheduler). Auto-selects the best attention backend. Supports optional FP8 quantization. |
| **Capybara Generate** | Main generation / editing node. Supports `t2v`, `t2i`, `ti2i`, and `tv2v` via the `task_type` selector. Outputs IMAGE frames + fps. |
| **Capybara Load Video** | Load a video file as IMAGE frames + fps. Alternative to the built-in LoadVideo node. |
| **Capybara Load Rewrite Model** | Load Qwen3-VL-8B-Instruct for prompt rewriting. |
| **Capybara Rewrite Instruction** | Expand a short editing instruction into a detailed prompt. |

### Capybara Generate -- key details

- **Single `reference` input** (IMAGE) for both image editing and video editing. The `task_type` determines how the input is interpreted: a single image for `ti2i`, video frames for `tv2v`.
- **`images` + `fps` outputs** -- connect to ComfyUI's built-in **CreateVideo** -> **SaveVideo** to save as video, or to **PreviewImage** / **SaveImage** for images.
- For `ti2i` / `tv2v`, the output aspect ratio is inferred from the reference input (the `aspect_ratio` dropdown is ignored).
- For `ti2i` / `tv2v`, `guidance_scale` is fixed to `1.0` internally.

### FP8 Quantization

The **Capybara Load Pipeline** node has a `quantize` dropdown with two options:

| Option | Description |
| --- | --- |
| `none` (default) | No quantization. Transformer weights stay in the selected `dtype` (bf16/fp16). |
| `fp8` | FP8 (E4M3) weight-only quantization via [torchao](https://github.com/pytorch/ao). Roughly halves the transformer's weight memory. |

**Requirements:** NVIDIA Ada Lovelace or Hopper GPU (compute capability >= 8.9, e.g. RTX 4090, L40, H100) and `torchao` installed.

**Notes:**
- The `dtype` setting (bf16/fp16) still controls the compute precision. Only the transformer weights are stored in FP8; activations and matmuls run in the selected dtype.
- When FP8 is enabled, the transformer stays pinned on GPU (quantized tensors cannot be moved between devices). All other models (VAE, text encoders, vision encoder) still offload to CPU normally.
- FP8 quantization primarily saves VRAM. Compute speed is roughly the same as without quantization.

## Example Workflow

A sample TV2V workflow is provided in [`examples/sample_workflow.json`](examples/sample_workflow.json). Drag and drop the file into the ComfyUI canvas to load it.

The workflow demonstrates:

```
[LoadVideo] -> [GetVideoComponents] -> frames -+
                                                |
[CapybaraLoadPipeline] -> pipe -> [CapybaraGenerate] -> images + fps -> [CreateVideo] -> [SaveVideo]
```

Other task types follow the same pattern -- just change `task_type` in the Generate node and connect the appropriate inputs:

- **T2V / T2I**: No reference needed. For T2V, pipe `images` + `fps` to CreateVideo -> SaveVideo.
- **TI2I**: Connect a **LoadImage** node to the `reference` input.
- **TV2V**: Connect video frames (from LoadVideo -> GetVideoComponents, or CapybaraLoadVideo) to the `reference` input.