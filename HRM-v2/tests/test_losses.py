import sys, types, importlib.util
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m

orig_losses = _load("orig_losses", REPO / "models" / "losses.py")
from hrm.train.losses import stablemax_cross_entropy, softmax_cross_entropy, ACTLossHead, IGNORE_LABEL_ID


def test_ce_parity_with_original():
    torch.manual_seed(0)
    logits = torch.randn(4, 12, 11); labels = torch.randint(0, 11, (4, 12)); labels[0, :6] = -100
    for fn_o, fn_p in ((orig_losses.stablemax_cross_entropy, stablemax_cross_entropy),
                       (orig_losses.softmax_cross_entropy, softmax_cross_entropy)):
        lo = fn_o(logits, labels); lp = fn_p(logits, labels)
        assert torch.allclose(lo, lp.to(lo.dtype), atol=1e-6)


class _DummyACT(torch.nn.Module):
    """Canned outputs shaped like HRMACTv1 so ACTLossHead semantics can be asserted."""
    def __init__(self, logits, qh, qc, labels, halted, steps, target_qc=None):
        super().__init__()
        self.o = dict(logits=logits, q_halt_logits=qh, q_continue_logits=qc)
        if target_qc is not None:
            self.o["target_q_continue"] = target_qc
        self.carry = types.SimpleNamespace(current_data={"labels": labels}, halted=halted, steps=steps)
    def forward(self, carry, batch):
        return self.carry, self.o


def test_act_loss_head_includes_q_halt_term():
    torch.manual_seed(0)
    B, S, V = 3, 8, 7
    logits = torch.randn(B, S, V, requires_grad=True)
    qh = torch.randn(B, requires_grad=True); qc = torch.randn(B, requires_grad=True)
    labels = torch.randint(0, V, (B, S))
    dummy = _DummyACT(logits, qh, qc, labels,
                      halted=torch.ones(B, dtype=torch.bool), steps=torch.ones(B, dtype=torch.int32))
    head = ACTLossHead(dummy, loss_type="softmax_cross_entropy")
    carry, loss, metrics, _, all_halted = head(return_keys=[], carry=None, batch=None)
    loss.backward()
    assert qh.grad is not None and qh.grad.abs().sum() > 0      # D1: q_halt gets gradient
    assert "q_halt_loss" in metrics and "lm_loss" in metrics
    # D3 divisor semantics: per-sequence mean over valid tokens, then SUM over batch (not global mean)
    per_tok = torch.nn.functional.cross_entropy(
        logits.detach().float().view(-1, V), labels.view(-1), reduction="none").view(B, S)
    expected_lm = (per_tok / S).sum()
    assert torch.allclose(metrics["lm_loss"].float(), expected_lm, rtol=1e-4)


def test_act_loss_head_q_continue_when_target_present():
    torch.manual_seed(1)
    B, S, V = 2, 4, 5
    logits = torch.randn(B, S, V); qh = torch.randn(B); qc = torch.randn(B, requires_grad=True)
    dummy = _DummyACT(logits, qh, qc, torch.randint(0, V, (B, S)),
                      halted=torch.zeros(B, dtype=torch.bool), steps=torch.ones(B, dtype=torch.int32),
                      target_qc=torch.sigmoid(torch.randn(B)))
    head = ACTLossHead(dummy, loss_type="softmax_cross_entropy")
    _, loss, metrics, _, _ = head(return_keys=[], carry=None, batch=None)
    assert "q_continue_loss" in metrics
    loss.backward()
    assert qc.grad is not None and qc.grad.abs().sum() > 0
