import os
import sys
import importlib
import importlib.util
import torch
import numpy as np
import tempfile
import random
from PIL import Image

capybara_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if capybara_root not in sys.path:
    sys.path.insert(0, capybara_root)


def _import_capybara_utils():
    """Import Capybara's utils.py by explicit path to avoid collision with ComfyUI's utils package."""
    utils_path = os.path.join(capybara_root, "utils.py")
    spec = importlib.util.spec_from_file_location("capybara_utils", utils_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_capybara_utils = _import_capybara_utils()
load_rewrite_model = _capybara_utils.load_rewrite_model
rewrite_instruction_fn = _capybara_utils.rewrite_instruction_fn
get_bucket_from_resolution_and_aspect_ratio = _capybara_utils.get_bucket_from_resolution_and_aspect_ratio
get_aspect_ratio_from_bucket = _capybara_utils.get_aspect_ratio_from_bucket
get_bucket_from_resolution_and_actual_ratio = _capybara_utils.get_bucket_from_resolution_and_actual_ratio

from models.pipelines.capybara_pipeline import Capybara_Pipeline


CAPYBARA_PIPE_TYPE = "CAPYBARA_PIPE"
REWRITE_MODEL_TYPE = "CAPYBARA_REWRITE_MODEL"


def _unpad_input(hidden_states, attention_mask):
    """Standalone unpad: remove padding tokens indicated by a boolean mask.

    This reimplements flash_attn.bert_padding.unpad_input so we never need
    to import the flash_attn package when only flash_attn_interface (v3) is
    available.

    Args:
        hidden_states: (batch, seqlen, ...) tensor
        attention_mask: (batch, seqlen) bool tensor, True = keep

    Returns:
        (unpadded, indices, cu_seqlens, max_seqlen_in_batch)
    """
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = seqlens.max().item()
    cu_seqlens = torch.zeros(seqlens.shape[0] + 1, dtype=torch.int32, device=seqlens.device)
    cu_seqlens[1:] = seqlens.cumsum(0)

    shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, *shape[2:])
    unpadded = hidden_states.index_select(0, indices)
    return unpadded, indices, cu_seqlens, max_seqlen


def _pad_input(hidden_states, indices, batch_size, seqlen):
    """Standalone pad: scatter unpadded tokens back into a zero-filled (batch, seqlen, ...) tensor."""
    output = torch.zeros(batch_size * seqlen, *hidden_states.shape[1:],
                         device=hidden_states.device, dtype=hidden_states.dtype)
    output.index_copy_(0, indices, hidden_states)
    return output.reshape(batch_size, seqlen, *hidden_states.shape[1:])


def _flash3_nopad_v3(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False):
    """Drop-in replacement for flash_attn_no_pad_v3 that only needs flash_attn_interface."""
    from einops import rearrange
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3

    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query, key, value = qkv.unbind(dim=2)

    query_unpad, indices, cu_seqlens, max_seqlen_q = _unpad_input(
        rearrange(query, "b s h d -> b s (h d)"), key_padding_mask)
    key_unpad, _, cu_seqlens_k, _ = _unpad_input(
        rearrange(key, "b s h d -> b s (h d)"), key_padding_mask)
    value_unpad, _, _, _ = _unpad_input(
        rearrange(value, "b s h d -> b s (h d)"), key_padding_mask)

    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)

    output_unpad = flash_attn_varlen_func_v3(
        query_unpad, key_unpad, value_unpad,
        cu_seqlens, cu_seqlens_k,
        max_seqlen_q, max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )

    output = _pad_input(
        rearrange(output_unpad, "nnz h d -> nnz (h d)"),
        indices, batch_size, seqlen,
    )
    return rearrange(output, "b s (h d) -> b s h d", h=nheads)


@torch.compiler.disable
def _flash3_attention(q, k, v, drop_rate=0.0, attn_mask=None, causal=False, softmax_scale=None):
    """Flash-attention-3 path for the token refiner that does NOT depend on the flash_attn package."""
    qkv = torch.stack([q, k, v], dim=2)
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.bool()
    return _flash3_nopad_v3(qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=softmax_scale)


def _save_comfyui_image_to_temp(image_tensor):
    """Save a ComfyUI IMAGE tensor [B, H, W, C] to a temporary PNG file and return the path."""
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    img_np = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pil_img.save(tmp.name)
    tmp.close()
    return tmp.name


def _save_comfyui_frames_to_temp_video(frames_tensor, fps=24):
    """Save ComfyUI IMAGE frames [F, H, W, C] to a temporary mp4 file and return the path."""
    import imageio
    frames_np = (frames_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    imageio.mimwrite(tmp.name, frames_np, fps=fps)
    return tmp.name


class CapybaraLoadPipeline:
    """Load the full Capybara pipeline (transformer, VAE, text encoders, vision encoder, scheduler)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the pretrained Capybara checkpoint directory",
                }),
                "transformer_version": (["capybara_v01"], {
                    "default": "capybara_v01",
                }),
                "dtype": (["bfloat16", "float16"], {
                    "default": "bfloat16",
                }),
                "enable_offloading": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable CPU offloading to save GPU VRAM",
                }),
                "flow_shift": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Flow shift parameter for the scheduler",
                }),
                "quantize": (["none", "fp8"], {
                    "default": "none",
                    "tooltip": "FP8 weight-only quantization for the transformer",
                }),
            }
        }

    RETURN_TYPES = (CAPYBARA_PIPE_TYPE,)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_pipeline"
    CATEGORY = "Capybara"
    DESCRIPTION = "Load the Capybara unified visual pipeline with all sub-models."

    def load_pipeline(self, model_path, transformer_version, dtype, enable_offloading, flow_shift, quantize="none"):
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        transformer_dtype = dtype_map[dtype]
        quantize_transformer = None if quantize == "none" else quantize

        pipe = Capybara_Pipeline.create_pipeline(
            pretrained_model_name_or_path=model_path,
            transformer_version=transformer_version,
            enable_offloading=enable_offloading,
            enable_group_offloading=None,
            create_sr_pipeline=False,
            force_sparse_attn=False,
            transformer_dtype=transformer_dtype,
            flow_shift=flow_shift,
            device=torch.device("cuda"),
            quantize_transformer=quantize_transformer,
        )

        resolved_mode = self._resolve_attn_mode()
        print(f"[Capybara ComfyUI] Using attention mode: {resolved_mode}")
        pipe.transformer.set_attn_mode(resolved_mode)
        self._patch_attention_modules(resolved_mode)

        return (pipe,)

    @staticmethod
    def _resolve_attn_mode():
        """Determine best available attention mode: flash3 > flash2 > torch."""
        from models.commons import is_flash3_available, is_flash2_available
        if is_flash3_available():
            return "flash3"
        if is_flash2_available():
            return "flash2"
        return "torch"

    @staticmethod
    def _patch_attention_modules(resolved_mode):
        """Patch attention functions so flash3-only envs never import flash_attn.

        Two things need patching:
        1. The `attention()` function used by the token refiner -- it defaults to
           attn_mode="flash" which tries to import flash_attn.
        2. The `flash_attn_no_pad_v3` callable used by sequence_parallel_attention
           for the flash3 path in the double/single blocks -- it also imports
           flash_attn internally for pad/unpad utilities.
        """
        import models.models.transformers.modules.token_refiner as tr_module
        import models.models.transformers.modules.attention as attn_module
        import models.utils.flash_attn_no_pad as nopad_module

        original_attention = attn_module.attention

        if resolved_mode == "flash3":
            nopad_module.flash_attn_no_pad_v3 = _flash3_nopad_v3
            attn_module.flash_attn_no_pad_v3 = _flash3_nopad_v3

            @torch.compiler.disable
            def patched_attention(q, k, v, drop_rate=0.0, attn_mask=None, causal=False, attn_mode="flash3"):
                try:
                    x = _flash3_attention(q, k, v, drop_rate, attn_mask, causal)
                    b, s, a, d = x.shape
                    return x.reshape(b, s, -1)
                except Exception as e:
                    print(f"[Capybara ComfyUI] flash3 attention failed ({e}), falling back to torch")
                    return original_attention(q, k, v, drop_rate=drop_rate, attn_mask=attn_mask, causal=causal, attn_mode="torch")
        else:
            def patched_attention(q, k, v, drop_rate=0.0, attn_mask=None, causal=False, attn_mode=resolved_mode):
                return original_attention(q, k, v, drop_rate=drop_rate, attn_mask=attn_mask, causal=causal, attn_mode=resolved_mode)

        attn_module.attention = patched_attention
        tr_module.attention = patched_attention


class CapybaraLoadVideo:
    """Load a video file and return IMAGE frames + fps.

    Output is compatible with CapybaraGenerate (reference_video) and
    ComfyUI's CreateVideo / SaveAnimatedWEBP / SaveWEBM nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to the input video file",
                }),
            },
            "optional": {
                "max_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "tooltip": "Limit number of frames loaded (0 = all frames)",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("frames", "fps", "video_path")
    FUNCTION = "load_video"
    CATEGORY = "Capybara"
    DESCRIPTION = "Load a video file as IMAGE frames. Works with CapybaraGenerate and CreateVideo/SaveVideo."

    def load_video(self, video_path, max_frames=0):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        import imageio
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 24.0))

        frames = []
        for i, frame in enumerate(reader):
            if max_frames > 0 and i >= max_frames:
                break
            frames.append(frame)
        reader.close()

        frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_np)  # [F, H, W, C]

        print(f"[Capybara LoadVideo] Loaded {frames_tensor.shape[0]} frames @ {fps} fps "
              f"({frames_tensor.shape[2]}x{frames_tensor.shape[1]})")

        return (frames_tensor, fps, video_path)


class CapybaraLoadRewriteModel:
    """Load the Qwen3-VL-8B-Instruct model for instruction rewriting/expansion."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "Qwen/Qwen3-VL-8B-Instruct",
                    "tooltip": "HuggingFace model name or local path for Qwen3-VL-8B-Instruct",
                }),
            },
            "optional": {
                "attn_implementation": (["auto", "flash_attention_3", "flash_attention_2", "sdpa", "eager"], {
                    "default": "auto",
                    "tooltip": "Attention backend. 'auto' picks flash_attention_3 > flash_attention_2 > sdpa.",
                }),
            }
        }

    RETURN_TYPES = (REWRITE_MODEL_TYPE,)
    RETURN_NAMES = ("rewrite_model",)
    FUNCTION = "load_model"
    CATEGORY = "Capybara"
    DESCRIPTION = "Load the Qwen3-VL instruction rewrite model for expanding short editing prompts."

    @staticmethod
    def _resolve_hf_attn():
        """Pick the best HuggingFace attn_implementation: fa3 > fa2 > sdpa."""
        from models.commons import is_flash3_available, is_flash2_available
        if is_flash3_available():
            return "flash_attention_3"
        if is_flash2_available():
            return "flash_attention_2"
        return "sdpa"

    def load_model(self, model_path, attn_implementation="auto"):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        if attn_implementation == "auto":
            attn_implementation = self._resolve_hf_attn()

        device = torch.device("cuda")
        print(f"[Capybara ComfyUI] Loading rewrite model from: {model_path} (attn: {attn_implementation})")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map="cpu",
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path)
        print(f"[Capybara ComfyUI] Rewrite model loaded on {device}")
        return ((model, processor),)


class CapybaraRewriteInstruction:
    """Expand a short editing instruction into a detailed one using Qwen3-VL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rewrite_model": (REWRITE_MODEL_TYPE,),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The short editing instruction to expand",
                }),
            },
            "optional": {
                "media_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to reference video for visual-grounded rewriting (tv2v)",
                }),
                "reference": ("IMAGE", {
                    "tooltip": "Reference visual for visual-grounded rewriting (image or video frames)",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "rewrite"
    CATEGORY = "Capybara"
    DESCRIPTION = "Rewrite/expand a short editing instruction into a detailed one using Qwen3-VL."

    def rewrite(self, rewrite_model, prompt, media_path="", reference=None):
        model, processor = rewrite_model

        tmp_img_path = None
        if reference is not None:
            tmp_img_path = _save_comfyui_image_to_temp(reference)
            media = tmp_img_path
        elif media_path and os.path.exists(media_path):
            media = media_path
        else:
            media = None

        rewritten = rewrite_instruction_fn(prompt, media, model, processor)

        if tmp_img_path is not None:
            try:
                os.unlink(tmp_img_path)
            except OSError:
                pass

        return (rewritten,)


class CapybaraGenerate:
    """Run the Capybara pipeline for any supported task type (t2v, t2i, ti2i, tv2v)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": (CAPYBARA_PIPE_TYPE,),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt / editing instruction",
                }),
                "task_type": (["t2v", "t2i", "ti2i", "tv2v"], {
                    "default": "t2v",
                }),
                "resolution": (["480p", "720p", "1024", "1080p"], {
                    "default": "480p",
                }),
                "aspect_ratio": (["16:9", "9:16", "4:3", "3:4", "1:1"], {
                    "default": "16:9",
                }),
                "num_frames": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 241,
                    "step": 4,
                    "tooltip": "Number of frames to generate (video tasks). Ignored for image tasks.",
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**30,
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "reference": ("IMAGE", {
                    "tooltip": "Reference visual input: single image for ti2i, video frames for tv2v",
                }),
                "reference_video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to reference video for tv2v (fallback when reference is not connected)",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "generate"
    CATEGORY = "Capybara"
    DESCRIPTION = "Generate images or video frames using the Capybara pipeline. Output fps can be piped to CreateVideo -> SaveVideo."

    def generate(
        self,
        pipe,
        prompt,
        task_type,
        resolution,
        aspect_ratio,
        num_frames,
        num_inference_steps,
        guidance_scale,
        seed,
        negative_prompt="",
        reference=None,
        reference_video_path="",
    ):
        extra_kwargs = {}
        tmp_video_path = None

        if task_type == "ti2i":
            if reference is None:
                raise ValueError("reference (IMAGE) is required for ti2i task type")
            ref_img_path = _save_comfyui_image_to_temp(reference)
            extra_kwargs["reference_img"] = ref_img_path

            img = Image.open(ref_img_path)
            w, h = img.size
            actual_ratio = w / h
            bucket = get_bucket_from_resolution_and_actual_ratio(resolution, actual_ratio)
            aspect_ratio = get_aspect_ratio_from_bucket(bucket)

        elif task_type == "tv2v":
            if reference is not None:
                tmp_video_path = _save_comfyui_frames_to_temp_video(reference)
                extra_kwargs["reference_video"] = tmp_video_path
                h, w = reference.shape[1], reference.shape[2]
            elif reference_video_path and os.path.exists(reference_video_path):
                extra_kwargs["reference_video"] = reference_video_path
                import imageio
                reader = imageio.get_reader(reference_video_path)
                first_frame = reader.get_data(0)
                reader.close()
                h, w = first_frame.shape[:2]
            else:
                raise ValueError(
                    "tv2v task requires either reference (IMAGE frames) or reference_video_path"
                )
            actual_ratio = w / h
            bucket = get_bucket_from_resolution_and_actual_ratio(resolution, actual_ratio)
            aspect_ratio = get_aspect_ratio_from_bucket(bucket)

        else:
            bucket = get_bucket_from_resolution_and_aspect_ratio(resolution, aspect_ratio)

        height, width = bucket

        if task_type in ("ti2i", "t2i"):
            video_length = 1
        else:
            video_length = num_frames

        if task_type in ("ti2i", "tv2v"):
            guidance_scale = 1.0

        if seed == 0:
            seed = random.randint(1, 2**30)

        print(f"[Capybara ComfyUI] task={task_type}, resolution={resolution}, "
              f"bucket={height}x{width}, aspect_ratio={aspect_ratio}")

        out = pipe(
            enable_sr=False,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            num_inference_steps=num_inference_steps,
            video_length=video_length,
            negative_prompt=negative_prompt,
            seed=seed,
            output_type="pt",
            prompt_rewrite=False,
            return_pre_sr_video=False,
            guidance_scale=guidance_scale,
            task_type=task_type,
            height=height,
            width=width,
            target_device=torch.device("cuda"),
            **extra_kwargs,
        )

        video_tensor = out.videos
        images = self._pipeline_output_to_comfyui(video_tensor)
        fps = 24.0 if task_type not in ("ti2i", "t2i") else 1.0

        for tmp in [extra_kwargs.get("reference_img"), tmp_video_path]:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

        return (images, fps)

    @staticmethod
    def _pipeline_output_to_comfyui(video_tensor):
        """Convert pipeline output [B, C, F, H, W] to ComfyUI IMAGE format [F, H, W, 3]."""
        if video_tensor.ndim == 5:
            video_tensor = video_tensor[0]
        # video_tensor is now [C, F, H, W], values in [0, 1]
        video_tensor = video_tensor.permute(1, 2, 3, 0)  # [F, H, W, C]
        video_tensor = video_tensor.float().cpu()
        video_tensor = video_tensor.clamp(0.0, 1.0)
        return video_tensor


NODE_CLASS_MAPPINGS = {
    "CapybaraLoadPipeline": CapybaraLoadPipeline,
    "CapybaraLoadVideo": CapybaraLoadVideo,
    "CapybaraLoadRewriteModel": CapybaraLoadRewriteModel,
    "CapybaraRewriteInstruction": CapybaraRewriteInstruction,
    "CapybaraGenerate": CapybaraGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CapybaraLoadPipeline": "Capybara Load Pipeline",
    "CapybaraLoadVideo": "Capybara Load Video",
    "CapybaraLoadRewriteModel": "Capybara Load Rewrite Model",
    "CapybaraRewriteInstruction": "Capybara Rewrite Instruction",
    "CapybaraGenerate": "Capybara Generate",
}
