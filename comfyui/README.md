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
| **Capybara Load Pipeline** | Load all model components (transformer, VAE, text encoders, vision encoder, scheduler). Auto-selects the best attention backend. |
| **Capybara Generate** | Main generation / editing node. Supports `t2v`, `t2i`, `ti2i`, and `tv2v` via the `task_type` selector. Outputs IMAGE frames + fps. |
| **Capybara Load Video** | Load a video file as IMAGE frames + fps. Alternative to the built-in LoadVideo node. |
| **Capybara Load Rewrite Model** | Load Qwen3-VL-8B-Instruct for prompt rewriting. |
| **Capybara Rewrite Instruction** | Expand a short editing instruction into a detailed prompt. |

### Capybara Generate -- key details

- **Single `reference` input** (IMAGE) for both image editing and video editing. The `task_type` determines how the input is interpreted: a single image for `ti2i`, video frames for `tv2v`.
- **`images` + `fps` outputs** -- connect to ComfyUI's built-in **CreateVideo** -> **SaveVideo** to save as video, or to **PreviewImage** / **SaveImage** for images.
- For `ti2i` / `tv2v`, the output aspect ratio is inferred from the reference input (the `aspect_ratio` dropdown is ignored).
- For `ti2i` / `tv2v`, `guidance_scale` is fixed to `1.0` internally.

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