import torch
from prime_rl.orchestrator.advantage import AdvantageInputs, AdvantageOutputs
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs


def advantage(inputs: AdvantageInputs, eps: float = 1e-8) -> AdvantageOutputs:
    """GRPO advantage with std normalization, as used in AceReason-Nemotron (arXiv:2505.16400)."""
    rewards = inputs.rewards
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return AdvantageOutputs(advantages=(rewards - mean) / (std + eps))


def loss(inputs: LossInputs) -> LossOutputs:
    """REINFORCE policy gradient loss as used in AceReason-Nemotron (arXiv:2505.16400).

    No ratio clipping, no KL penalty — pure policy gradient on trainer log probs.
    Advantage normalization (mean/std) is handled by the custom advantage function.
    """
    mask = inputs.loss_mask
    denom = mask.sum().clamp_min(1)

    # No per-sequence /denom here — compute_loss divides the summed loss by total
    # masked tokens across the batch (loss_scale), matching the paper's 1/sum_i|o_i| normalization.
    pg_loss = -(inputs.advantages * inputs.trainer_logprobs * mask).sum()

    log_ratio = inputs.trainer_logprobs - inputs.inference_logprobs
    metrics = {
        "approx_kl": (log_ratio.abs() * mask).sum() / denom,  # per-sequence mean for logging
    }

    return LossOutputs(loss=pg_loss, metrics=metrics)
