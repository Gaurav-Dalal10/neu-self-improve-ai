"""Verify PPO clipping selects the pessimistic branch correctly."""
import torch


def test_ppo_clip_positive_adv_high_ratio():
    """A>0, ratio>1+ε → clipped branch should win."""
    eps = 0.2
    ratio = torch.tensor([1.5])
    adv   = torch.tensor([1.0])
    pg1 = -adv * ratio                                     # -1.5
    pg2 = -adv * torch.clamp(ratio, 1 - eps, 1 + eps)      # -1.2
    assert abs(torch.max(pg1, pg2).item() - (-1.2)) < 1e-6


def test_ppo_clip_negative_adv_low_ratio():
    """A<0, ratio<1-ε → clipped branch should win."""
    eps = 0.2
    ratio = torch.tensor([0.5])
    adv   = torch.tensor([-1.0])
    pg1 = -adv * ratio                                     # 0.5
    pg2 = -adv * torch.clamp(ratio, 1 - eps, 1 + eps)      # 0.8
    assert abs(torch.max(pg1, pg2).item() - 0.8) < 1e-6
