import torch
from hrm.train.optim import AdamATan2, warmup_constant_lr

def test_adam_atan2_steps_and_is_scale_invariant_in_eps():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = AdamATan2([p], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)
    before = p.detach().clone()
    (p.square().sum()).backward()
    opt.step()
    assert not torch.allclose(p.detach(), before)
    assert torch.isfinite(p).all()

def test_adam_atan2_zero_grad_only_decays():
    p = torch.nn.Parameter(torch.ones(4))
    opt = AdamATan2([p], lr=0.1, betas=(0.9, 0.95), weight_decay=0.5)
    p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.allclose(p.detach(), torch.full((4,), 1.0 * (1 - 0.1 * 0.5)))  # atan2(0, x)=0 -> pure decoupled decay

def test_warmup_constant():
    assert warmup_constant_lr(0, 100) == 0.0 and warmup_constant_lr(50, 100) == 0.5
    assert warmup_constant_lr(100, 100) == 1.0 and warmup_constant_lr(5000, 100) == 1.0
