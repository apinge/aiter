"""Regression for unstable BF16 Neumann inversion on correlated GDN keys."""

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("hg,h", [(8, 24), (4, 12)])
def test_gdn_prepare_correlated_keys(hg, h):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("Requires ROCm")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx942"):
        pytest.skip("Validated on gfx942")
    pytest.importorskip("flydsl")
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        gdn_prepare_fwd_flydsl,
    )

    torch.manual_seed(2026)
    # Almost parallel keys with weak decay make the BF16 matrix powers in
    # (I+B)(I+B^2)(I+B^4)(I+B^8) large and their final sum ill-conditioned.
    base = torch.randn(1, 1, hg, 128, device="cuda")
    k = F.normalize(
        base + 0.01 * torch.randn(1, 64, hg, 128, device="cuda"), dim=-1
    ).bfloat16()
    v = torch.randn(1, 64, h, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.full((1, 64, h), -0.0001, device="cuda")
    beta = torch.full_like(g, 0.999)
    w, u, gc = gdn_prepare_fwd_flydsl(k, v, g, beta, use_exp2=False)

    keys = k.float().repeat_interleave(h // hg, dim=2).transpose(1, 2)
    values = v.float().transpose(1, 2)
    gates = g.transpose(1, 2).cumsum(-1)
    betas = beta.transpose(1, 2)
    lower = torch.tril(
        (keys @ keys.transpose(-1, -2))
        * betas[..., None]
        * (gates[..., :, None] - gates[..., None, :]).exp(),
        diagonal=-1,
    )
    identity = torch.eye(64, device="cuda").expand(1, h, 64, 64)
    inverse = torch.linalg.solve_triangular(identity + lower, identity, upper=False)
    # Match the final BF16 WY GEMM operand conversions, after the FP32 solve.
    inverse = inverse.bfloat16().float()
    w_ref = inverse @ (keys * (betas * gates.exp())[..., None]).bfloat16().float()
    u_ref = inverse @ (values * betas[..., None]).bfloat16().float()
    torch.testing.assert_close(gc, gates, atol=1e-6, rtol=1e-5)
    for actual, expected in ((w, w_ref), (u, u_ref)):
        error = (actual.float() - expected).square().mean().sqrt()
        scale = expected.square().mean().sqrt()
        assert error / scale < 0.003, (error / scale).item()
