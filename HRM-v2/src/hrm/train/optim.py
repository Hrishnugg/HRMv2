import math, torch

class AdamATan2(torch.optim.Optimizer):
    """Adam with atan2-based update (scale-invariant, epsilon-free) + decoupled weight decay.
    update = a * atan2(m_hat, b * sqrt(v_hat));  a = 1.2732395447351628, b = 1.0."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, a=1.2732395447351628, b=1.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a, b=b))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, (b1, b2), wd, a, b = group["lr"], group["betas"], group["weight_decay"], group["a"], group["b"]
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1; t = st["step"]
                m, v = st["exp_avg"], st["exp_avg_sq"]
                m.lerp_(g, 1 - b1); v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1 ** t); v_hat = v / (1 - b2 ** t)
                if wd != 0: p.mul_(1 - lr * wd)                       # decoupled (AdamW-style)
                p.add_(torch.atan2(m_hat, b * v_hat.sqrt()), alpha=-lr * a)
        return None

def warmup_constant_lr(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0: return 1.0
    return min(1.0, step / warmup_steps)
