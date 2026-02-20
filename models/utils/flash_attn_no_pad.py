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

import torch
from einops import rearrange


def _unpad_input(hidden_states, attention_mask):
    """Remove padding tokens indicated by a boolean mask.

    Standalone reimplementation of flash_attn.bert_padding.unpad_input so that
    flash_attn_no_pad_v3 can work without the flash_attn package installed.
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
    """Scatter unpadded tokens back into a zero-filled (batch, seqlen, ...) tensor."""
    output = torch.zeros(
        batch_size * seqlen, *hidden_states.shape[1:],
        device=hidden_states.device, dtype=hidden_states.dtype,
    )
    output.index_copy_(0, indices, hidden_states)
    return output.reshape(batch_size, seqlen, *hidden_states.shape[1:])


def flash_attn_no_pad(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False
):
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(
        x, key_padding_mask
    )

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output

def flash_attn_no_pad_v3(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False
):
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3

    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query, key, value = qkv.unbind(dim=2)

    query_unpad, indices, cu_seqlens, max_seqlen_q = _unpad_input(
        rearrange(query, "b s h d -> b s (h d)"), key_padding_mask
    )
    key_unpad, _, _, _ = _unpad_input(
        rearrange(key, "b s h d -> b s (h d)"), key_padding_mask
    )
    value_unpad, _, _, _ = _unpad_input(
        rearrange(value, "b s h d -> b s (h d)"), key_padding_mask
    )

    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)

    output_unpad = flash_attn_varlen_func_v3(
        query_unpad, key_unpad, value_unpad,
        cu_seqlens, cu_seqlens,
        max_seqlen_q, max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )

    output = rearrange(
        _pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
        "b s (h d) -> b s h d", h=nheads,
    )
    return output
