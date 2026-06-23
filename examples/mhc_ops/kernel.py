"""MHC operators used as GEAK optimization targets."""

from __future__ import annotations

import torch


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference MHC pre operator.

    Args:
        residual: BF16 tensor with shape ``[..., hc_mult, hidden_size]``.
        fn: FP32 projection weight with shape
            ``[hc_mult * 2 + hc_mult * hc_mult, hc_mult * hidden_size]``.
        hc_scale: FP32 scale tensor. Entries 0, 1, and 2 are used for pre,
            post, and comb branches respectively.
        hc_base: FP32 bias tensor with shape
            ``[hc_mult * 2 + hc_mult * hc_mult]``.

    Returns:
        ``(post_mix, comb_mix, layer_input)`` with shapes
        ``[..., hc_mult, 1]`` FP32, ``[..., hc_mult, hc_mult]`` FP32, and
        ``[..., hidden_size]`` BF16.
    """

    assert residual.dtype == torch.bfloat16
    assert fn.dtype == hc_scale.dtype == hc_base.dtype == torch.float32

    c = residual.shape[-2]
    h = residual.shape[-1]
    c3 = c * 2 + c * c
    if fn.shape != (c3, c * h):
        raise ValueError(f"fn must have shape {(c3, c * h)}, got {tuple(fn.shape)}")
    if hc_scale.numel() < 3:
        raise ValueError("hc_scale must contain at least 3 values")
    if hc_base.shape != (c3,):
        raise ValueError(f"hc_base must have shape {(c3,)}, got {tuple(hc_base.shape)}")

    outer = residual.shape[:-2]
    r = residual.reshape(-1, c, h)
    t = r.shape[0]

    # RMS-normalized FP32 projection over the flattened multi-channel state.
    x = r.reshape(t, c * h).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (c * h) + rms_eps)

    pre_mix = torch.sigmoid(mixes[:, :c] * hc_scale[0] + hc_base[:c]) + hc_pre_eps
    post_mix = (
        torch.sigmoid(mixes[:, c : 2 * c] * hc_scale[1] + hc_base[c : 2 * c])
        * hc_post_mult_value
    )

    comb = (
        mixes[:, 2 * c :].reshape(t, c, c) * hc_scale[2]
        + hc_base[2 * c :].view(1, c, c)
    )
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(pre_mix.unsqueeze(-1) * r.to(torch.float32), dim=1).to(
        torch.bfloat16
    )
    return (
        post_mix.view(*outer, c, 1),
        comb.view(*outer, c, c),
        layer_input.view(*outer, h),
    )


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
) -> torch.Tensor:
    """Reference MHC post operator.

    The assumed update rule is:

    ``new_residual_i = sum_j comb_mix[i, j] * residual_j + post_mix_i * x``

    Args:
        x: BF16 sub-layer output with shape ``[..., hidden_size]``.
        residual: BF16 tensor with shape ``[..., hc_mult, hidden_size]``.
        post_mix: FP32 tensor with shape ``[..., hc_mult, 1]`` or
            ``[..., hc_mult]``.
        comb_mix: FP32 tensor with shape ``[..., hc_mult, hc_mult]``.

    Returns:
        BF16 tensor with shape ``[..., hc_mult, hidden_size]``.
    """

    assert x.dtype == residual.dtype == torch.bfloat16
    assert post_mix.dtype == comb_mix.dtype == torch.float32

    c = residual.shape[-2]
    h = residual.shape[-1]
    outer = residual.shape[:-2]
    if x.shape != (*outer, h):
        raise ValueError(f"x must have shape {(*outer, h)}, got {tuple(x.shape)}")
    if comb_mix.shape != (*outer, c, c):
        raise ValueError(
            f"comb_mix must have shape {(*outer, c, c)}, got {tuple(comb_mix.shape)}"
        )
    if post_mix.shape == (*outer, c):
        post = post_mix.reshape(-1, c, 1)
    elif post_mix.shape == (*outer, c, 1):
        post = post_mix.reshape(-1, c, 1)
    else:
        raise ValueError(
            "post_mix must have shape "
            f"{(*outer, c)} or {(*outer, c, 1)}, got {tuple(post_mix.shape)}"
        )

    r = residual.reshape(-1, c, h).to(torch.float32)
    y = x.reshape(-1, h).to(torch.float32)
    comb = comb_mix.reshape(-1, c, c)

    mixed_residual = torch.bmm(comb, r)
    new_residual = mixed_residual + post * y.unsqueeze(1)
    return new_residual.to(torch.bfloat16).view(*outer, c, h)
