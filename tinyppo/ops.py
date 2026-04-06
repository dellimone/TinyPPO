import torch
import torch.nn.functional as F


def masked_mean(values, mask, axis=None):
    """Mean of values where mask is 1."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Variance of values where mask is 1 (global)."""
    mean = masked_mean(values, mask)
    centered = (values - mean) ** 2
    var = masked_mean(centered, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum > 1:
            var = var * mask_sum / (mask_sum - 1)
    return var


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values using masked mean/var."""
    mean = masked_mean(values, mask)
    var = masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def logprobs_from_logits(logits, labels):
    """Per-token log probs: log_softmax then gather."""
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def entropy_from_logits(logits):
    """Per-token entropy from logits."""
    pd = F.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - (pd * logits).sum(dim=-1)


def clip_by_value(x, tensor_min, tensor_max):
    """Element-wise clipping with tensor bounds."""
    return torch.max(torch.min(x, tensor_max), tensor_min)
