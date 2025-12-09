from einops import rearrange
import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input

    _HAS_FLASH_ATTN = True
except Exception:
    _HAS_FLASH_ATTN = False


def flash_attn_no_pad(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None
):
    """Attention kernel wrapper used throughout the repo.

    If FlashAttention is available, uses its varlen kernel with pad/unpad.
    Otherwise falls back to PyTorch scaled_dot_product_attention.

    Args:
        qkv: Tensor of shape [B, S, 3, H, D]
        key_padding_mask: Bool tensor [B, S], True marks positions to mask out.
        causal: Whether to apply causal masking.
        dropout_p: Dropout prob (unused in fallback / set to 0 at train-time here).
        softmax_scale: Optional scale (unused in fallback).
    Returns:
        Tensor of shape [B, S, H, D]
    """
    if _HAS_FLASH_ATTN:
        batch_size = qkv.shape[0]
        seqlen = qkv.shape[1]
        nheads = qkv.shape[-2]
        x = rearrange(qkv, "b s three h d -> b s (three h d)")
        # Note: flash_attn.bert_padding.unpad_input expects mask where True means keep.
        # Our callers pass True for masked (to drop). Invert for unpad_input.
        keep_mask = (
            ~key_padding_mask
            if key_padding_mask is not None
            else torch.ones((batch_size, seqlen), dtype=torch.bool, device=qkv.device)
        )
        x_unpad, indices, cu_seqlens, max_s, _ = unpad_input(x, keep_mask)

        x_unpad = rearrange(
            x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
        )
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad,
            cu_seqlens,
            max_s,
            dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        output = rearrange(
            pad_input(
                rearrange(output_unpad, "nnz h d -> nnz (h d)"),
                indices,
                batch_size,
                seqlen,
            ),
            "b s (h d) -> b s h d",
            h=nheads,
        )
        return output

    # Fallback: use PyTorch SDPA across all tokens with a boolean key mask.
    B, S, three, H, D = qkv.shape
    q = qkv[:, :, 0]  # [B, S, H, D]
    k = qkv[:, :, 1]
    v = qkv[:, :, 2]
    # reshape to [B*H, S, D]
    q_r = q.permute(0, 2, 1, 3).reshape(B * H, S, D)
    k_r = k.permute(0, 2, 1, 3).reshape(B * H, S, D)
    v_r = v.permute(0, 2, 1, 3).reshape(B * H, S, D)

    attn_mask = None
    if key_padding_mask is not None:
        # key_padding_mask: [B, S] with True meaning "masked out".
        # Expand to [B, H, Q=S, K=S] then flatten heads -> [B*H, S, S]
        attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,S]
        attn_mask = attn_mask.expand(B, H, S, S).reshape(B * H, S, S)

    out = F.scaled_dot_product_attention(
        q_r,
        k_r,
        v_r,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=causal,
    )
    out = out.reshape(B, H, S, D).permute(0, 2, 1, 3).contiguous()  # [B, S, H, D]
    return out
