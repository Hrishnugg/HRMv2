import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parents[1]
import continuous_prm_c9h_transfer as C9H
import continuous_prm_c6_heatmap_value_field as C6
import continuous_prm_common as C


def test_conv_lora_identity_then_frozen_base():
    import torch
    torch.manual_seed(0)
    unet = C6.build_model("unet", in_channels=8)
    x = torch.randn(1, 8, 64, 64)
    with torch.no_grad():
        base_out = unet(x).clone()
    n = C9H.apply_conv_lora(unet, rank=4, alpha=1.0)
    assert n >= 10
    with torch.no_grad():
        lora_out = unet(x)
    assert torch.allclose(base_out, lora_out, atol=1e-5)  # B=0 => identity at init
    C.set_lora_trainable(unet)
    trainable = [nm for nm, p in unet.named_parameters() if p.requires_grad]
    assert trainable and all((".A" in nm or ".B" in nm) for nm in trainable)


def test_conv_lora_changes_output_after_step():
    import torch
    unet = C6.build_model("unet", in_channels=8)
    C9H.apply_conv_lora(unet, rank=4, alpha=1.0)
    C.set_lora_trainable(unet)
    x = torch.randn(1, 8, 64, 64)
    base_out = unet(x).detach().clone()
    opt = torch.optim.SGD([p for p in unet.parameters() if p.requires_grad], lr=1.0)
    loss = unet(x).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    assert not torch.allclose(base_out, unet(x), atol=1e-6)
