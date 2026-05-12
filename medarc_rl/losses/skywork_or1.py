import torch
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs


def loss(inputs: LossInputs, clip_eps: float = 0.2) -> LossOutputs:
    """PPO-style policy gradient loss matching Skywork OR1 / MAGIC (arXiv:2505.22312).

    Implements standard PPO ratio clipping with no KL penalty, as used by the paper.
    The adaptive entropy bonus (α_k) from MAGIC requires stateful tracking across
    training steps that isn't supported by PRIME-RL's per-sample loss interface,
    so it is omitted. The paper's other key choices are preserved:
    - token-level averaging (no per-response length normalization)
    - no KL penalty
    """
    log_ratio = inputs.trainer_logprobs - inputs.inference_logprobs
    ratio = torch.exp(log_ratio)
    advantages = inputs.advantages
    mask = inputs.loss_mask

    pg_unclipped = -advantages * ratio
    pg_clipped = -advantages * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    denom = mask.sum().clamp_min(1)
    loss = (torch.max(pg_unclipped, pg_clipped) * mask).sum() / denom

    metrics = {
        "pg_clipfrac": ((pg_clipped > pg_unclipped).float() * mask).sum() / denom,
        "approx_kl": (log_ratio.abs() * mask).sum() / denom,
    }

    return LossOutputs(loss=loss, metrics=metrics)
