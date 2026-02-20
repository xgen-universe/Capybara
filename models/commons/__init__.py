# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import os
import torch
from itertools import repeat
from contextlib import contextmanager
from torch import nn
import collections.abc

def _ntuple(n):
    """Create a function that converts input to n-tuple."""
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))
    return parse

# Convenience functions for common tuple sizes
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)

PRECISION_TO_TYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

# Default generation pipeline configurations
PIPELINE_CONFIGS = {
    'capybara_v01': {
        'guidance_scale': 6.0,
        'embedded_guidance_scale': None,
        'flow_shift': 5.0,
    },
    '480p_t2v': {
        'guidance_scale': 6.0,
        'embedded_guidance_scale': None,
        'flow_shift': 5.0,
    },
    '480p_i2v': {
        'guidance_scale': 6.0,
        'embedded_guidance_scale': None,
        'flow_shift': 5.0,
    },
    '720p_t2v': {
        'guidance_scale': 6.0,
        'embedded_guidance_scale': None,
        'flow_shift': 9.0,
    },
    '720p_i2v': {
        'guidance_scale': 6.0,
        'embedded_guidance_scale': None,
        'flow_shift': 7.0,
    },
    '480p_t2v_distilled': {
        'guidance_scale': 1.0,
        'embedded_guidance_scale': None,
        'flow_shift': 5.0,
    },
    '480p_i2v_distilled': {
        'guidance_scale': 1.0,
        'embedded_guidance_scale': None,
        'flow_shift': 5.0,
    },
    '720p_t2v_distilled': {
        'guidance_scale': 1.0,
        'embedded_guidance_scale': None,
        'flow_shift': 9.0,
    },
    '720p_i2v_distilled': {
        'guidance_scale': 1.0,
        'embedded_guidance_scale': None,
        'flow_shift': 7.0,
    },
    '720p_t2v_distilled_sparse': {
        'guidance_scale': 1.0,
        'embedded_guidance_scale': None,
        'flow_shift': 7.0,
    },
    '720p_i2v_distilled_sparse': {
        'guidance_scale': 1.0,
        'embedded_guidance_scale': None,
        'flow_shift': 9.0,
    },
}

# Default super-resolution pipeline configurations
SR_PIPELINE_CONFIGS = {
    '720p_sr_distilled': {
        'flow_shift': 2.0,
        'base_resolution': '480p',
        'guidance_scale': 1.0,
        'embedded_guidance_scale': None,
        'num_inference_steps': 6,
    },
    '1080p_sr_distilled': {
        'flow_shift': 2.0,
        'base_resolution': '720p',
        'guidance_scale': 1.0,
        'embedded_guidance_scale': None,
        'num_inference_steps': 8,
    },
}

TRANSFORMER_VERSION_TO_SR_VERSION = {
    '480p_t2v': '720p_sr_distilled',
    '480p_t2v_v7': '720p_sr_distilled',
    '720p_t2v': '1080p_sr_distilled',
    '480p_i2v': '720p_sr_distilled',
    '720p_i2v': '1080p_sr_distilled',
    '480p_t2v_distilled': '720p_sr_distilled',
    '720p_t2v_distilled': '1080p_sr_distilled',
    '480p_i2v_distilled': '720p_sr_distilled',
    '720p_i2v_distilled': '1080p_sr_distilled',
    '480p_t2v_distilled_sparse': '720p_sr_distilled',
    '720p_t2v_distilled_sparse': '1080p_sr_distilled',
    '480p_i2v_distilled_sparse': '720p_sr_distilled',
    '720p_i2v_distilled_sparse': '1080p_sr_distilled',
}

def is_flash2_available():
    try:
        from flash_attn import flash_attn_varlen_qkvpacked_func
        return True
    except Exception:
        return False

def is_flash3_available():
    try:
        from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3  # noqa: F401
        return True
    except Exception:
        return False

def is_flash_available():
    return is_flash2_available() or is_flash3_available()

def is_sparse_attn_supported():
    return 'nvidia h' in torch.cuda.get_device_properties(0).name.lower()

def is_sparse_attn_available():
    if not is_sparse_attn_supported():
        return False
    try:
        from flex_block_attn import flex_block_attn_func  # noqa: F401
        return True
    except Exception:
        return False

def maybe_fallback_attn_mode(attn_mode, infer_state=None, block_idx=None):
    """
    Determine the final attention mode based on configuration and availability.
    
    Args:
        attn_mode: Requested attention mode
        infer_state: Inference configuration object (optional)
        block_idx: Current block index (optional)
    
    Returns:
        Final attention mode to use
    """
    import warnings
    
    # Check for sageattn and flex-block-attn conflict
    enable_sageattn = False
    if infer_state is not None:
        enable_sageattn = (infer_state.enable_sageattn and 
                        block_idx in infer_state.sage_blocks_range)
    
    assert not (enable_sageattn and attn_mode == 'flex-block-attn'), \
        ("SageAttention cannot be used with flex-block-attn mode. "
         "Please disable enable_sageattn or use a different attention mode.")
    
    # Use SageAttention if configured
    if enable_sageattn:
        attn_mode = 'sageattn'
        return attn_mode
    
    # Handle flash attention modes
    if attn_mode == 'flash':
        if is_flash3_available():
            attn_mode = 'flash3'
        elif is_flash2_available():
            attn_mode = 'flash2'
        else:
            warnings.warn("flash is not available. Falling back to torch attention.")
            attn_mode = 'torch'
    elif attn_mode == 'flash3':
        if not is_flash3_available():
            warnings.warn("flash3 is not available. Falling back to torch attention.")
            attn_mode = 'torch'
    elif attn_mode == 'flash2':
        if not is_flash2_available():
            warnings.warn("flash2 is not available. Falling back to torch attention.")
            attn_mode = 'torch'
    if attn_mode in ('flex-block-attn'):
        from models.commons import is_sparse_attn_available
        if not is_sparse_attn_available():
            raise ValueError(f"{attn_mode} is not available for your GPU or flex-block-attn is not properly installed.")
    return attn_mode

def _has_quantized_params(model):
    """Check if a model contains torchao quantized parameters that cannot be moved with .to()."""
    try:
        from torchao.dtypes import AffineQuantizedTensor
    except ImportError:
        return False
    for param in model.parameters():
        if isinstance(param, AffineQuantizedTensor) or isinstance(param.data, AffineQuantizedTensor):
            return True
    return False


@contextmanager
def auto_offload_model(models, device, enabled=True):
    from diffusers.hooks.group_offloading import _is_group_offload_enabled
    if enabled:
        if isinstance(models, nn.Module):
            models = [models]
        for model in models:
            if model is not None:
                if _has_quantized_params(model):
                    continue
                model.to(device)
    yield
    if enabled:
        for model in models:
            if model is not None:
                if _has_quantized_params(model):
                    continue
                model.to(torch.device('cpu'))

def get_gpu_memory(device=None):
    if not torch.cuda.is_available():
        return 0
    device = device if device is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    if hasattr(torch.cuda, 'get_per_process_memory_fraction'):
        memory_fraction = torch.cuda.get_per_process_memory_fraction()
    else:
        memory_fraction = 1.0
    return props.total_memory * memory_fraction

def get_rank():
    return int(os.environ.get('RANK', '0'))


def quantize_model_fp8(model, mode="fp8"):
    """Apply FP8 weight-only quantization via torchao.

    Requires compute capability >= 8.9 (Ada Lovelace / Hopper).
    """
    if mode != "fp8":
        return model

    cc = torch.cuda.get_device_capability()
    if cc[0] < 8 or (cc[0] == 8 and cc[1] < 9):
        raise RuntimeError(
            f"FP8 quantization requires compute capability >= 8.9 (Ada/Hopper). "
            f"Current GPU has {cc[0]}.{cc[1]}."
        )

    from torchao.quantization import quantize_, float8_weight_only

    quantize_(model, float8_weight_only())
    print(f"[Capybara] Transformer quantized to FP8 (weight-only, E4M3).")
    return model
