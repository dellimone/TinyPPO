import torch
from .ops import masked_mean, masked_var, masked_whiten, clip_by_value, entropy_from_logits


def compute_rewards_with_kl(scores, logprobs, ref_logprobs, mask, kl_coef):
    """Compute per-token rewards: -beta * KL + score at last response token.

    Args:
        scores: (batch,) reward scores from the reward model.
        logprobs: (batch, seq) log probs from the policy.
        ref_logprobs: (batch, seq) log probs from the frozen reference model.
        mask: (batch, seq) response token mask.
        kl_coef: scalar KL penalty coefficient (0 = no KL).

    Returns:
        rewards: (batch, seq)
        kl: (batch, seq) per-token KL divergence
    """
    kl = logprobs - ref_logprobs
    non_score_reward = -kl_coef * kl
    rewards = non_score_reward.clone()
    for i in range(rewards.size(0)):
        last_idx = int(mask[i].sum().item()) - 1
        rewards[i, last_idx] += scores[i]
    kl_approx = (torch.exp(kl) - 1) - kl  # always >= 0
    return rewards, kl, kl_approx


def compute_advantages(values, rewards, mask, gamma=1.0, lam=0.95, normalize="global"):
    """Generalized Advantage Estimation with configurable normalization.

    Args:
        values: (batch, seq) value estimates.
        rewards: (batch, seq) per-token rewards.
        mask: (batch, seq) response token mask.
        gamma: discount factor.
        lam: GAE lambda.
        normalize: "global" (whiten over full batch), "batch" (per-sequence),
                   or "none" (no whitening).

    Returns:
        advantages: (batch, seq) normalized and detached.
        returns: (batch, seq)
    """
    gen_len = rewards.size(1)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0.0

    values = values * mask
    rewards = rewards * mask

    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        advantages[:, t] = lastgaelam = delta + gamma * lam * lastgaelam

    returns = advantages + values

    if normalize == "global":
        advantages = masked_whiten(advantages, mask, shift_mean=True)
    elif normalize == "batch":
        # Normalize per-sequence along the token dimension
        mean = masked_mean(advantages, mask, axis=1).unsqueeze(1)
        var = masked_var(advantages, mask)  # global var for stability
        advantages = (advantages - mean) * torch.rsqrt(var + 1e-8)
    # "none" — no normalization

    advantages = advantages.detach()
    return advantages, returns


def ppo_loss(new_logprobs, old_logprobs, new_values, old_values, advantages,
             returns, mask, cliprange=0.2, cliprange_value=0.2, vf_coef=0.5,
             logits=None, entropy_coef=0.0):
    """Compute clipped PPO policy and value losses, with optional entropy bonus.

    Args:
        new_logprobs: (batch, seq) log probs from current policy.
        old_logprobs: (batch, seq) log probs from rollout policy (detached).
        new_values: (batch, seq) value estimates from current policy.
        old_values: (batch, seq) value estimates from rollout (detached).
        advantages: (batch, seq) normalized advantages.
        returns: (batch, seq) TD(λ) returns.
        mask: (batch, seq) response token mask.
        cliprange: policy clip range (ε). Set very large to disable clipping.
        cliprange_value: value function clip range. Set very large to disable.
        vf_coef: value loss coefficient.
        logits: (batch, seq, vocab) if provided and entropy_coef > 0, adds entropy bonus.
        entropy_coef: entropy bonus coefficient (0 = disabled).

    Returns:
        loss: scalar total loss.
        pg_loss: scalar policy loss (for logging).
        vf_loss: scalar value loss (for logging).
        stats: dict with per-component metrics.
    """
    ratio = torch.exp(new_logprobs - old_logprobs)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = masked_mean(torch.max(pg_losses1, pg_losses2), mask)

    vpred_clipped = clip_by_value(
        new_values,
        old_values - cliprange_value,
        old_values + cliprange_value,
    )
    vf_losses1 = (new_values - returns) ** 2
    vf_losses2 = (vpred_clipped - returns) ** 2
    vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)

    loss = pg_loss + vf_coef * vf_loss

    # Entropy bonus: subtract to encourage exploration (maximizing entropy)
    entropy_loss = torch.tensor(0.0, device=loss.device)
    if entropy_coef > 0.0 and logits is not None:
        entropy_loss = -entropy_coef * masked_mean(entropy_from_logits(logits), mask)
        loss = loss + entropy_loss

    with torch.no_grad():
        clip_frac = masked_mean((torch.abs(ratio - 1.0) > cliprange).float(), mask).item()

    stats = {
        "policy_loss": pg_loss.item(),
        "value_loss": vf_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "clip_frac": clip_frac,
    }
    return loss, pg_loss, vf_loss, stats
