import torch
from .ops import logprobs_from_logits, masked_mean


def pad_and_concat(query_tensors, response_tensors, pad_token_id):
    """Left-pad concatenated (query, response) sequences into a batch.

    Returns:
        input_ids: (batch, max_len) padded token ids.
        attention_mask: (batch, max_len) mask (1 for real tokens).
        lengths: list of (query_len, response_len) tuples.
    """
    seqs = []
    lengths = []
    for q, r in zip(query_tensors, response_tensors):
        cat = torch.cat([q, r])
        seqs.append(cat)
        lengths.append((len(q), len(r)))

    max_len = max(s.size(0) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(seqs), max_len, dtype=torch.long)

    for i, s in enumerate(seqs):
        offset = max_len - s.size(0)
        input_ids[i, offset:] = s
        attention_mask[i, offset:] = 1

    return input_ids, attention_mask, lengths


def batched_forward_pass(model, query_tensors, response_tensors, pad_token_id, device):
    """Run model forward and extract per-token logprobs/values for response tokens only.

    Returns:
        resp_logprobs: (batch, max_resp_len)
        resp_values: (batch, max_resp_len)
        resp_mask: (batch, max_resp_len)
        resp_logits: (batch, max_resp_len, vocab_size)
    """
    input_ids, attention_mask, lengths = pad_and_concat(
        query_tensors, response_tensors, pad_token_id,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    seq_len = input_ids.size(1)

    with torch.no_grad() if not model.training else torch.enable_grad():
        logits, values = model(input_ids, attention_mask=attention_mask)

    # Shifted logprobs: logits[:, :-1] predict input_ids[:, 1:]
    all_logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
    all_values = values[:, :-1]

    max_resp_len = max(r_len for _, r_len in lengths)
    batch_size = len(lengths)

    resp_logprobs = torch.zeros(batch_size, max_resp_len, device=device)
    resp_values = torch.zeros(batch_size, max_resp_len, device=device)
    resp_mask = torch.zeros(batch_size, max_resp_len, device=device)
    resp_logits = torch.zeros(batch_size, max_resp_len, logits.size(-1), device=device)

    for i, (q_len, r_len) in enumerate(lengths):
        pad_offset = seq_len - (q_len + r_len)
        start = pad_offset + q_len - 1
        end = start + r_len
        resp_logprobs[i, :r_len] = all_logprobs[i, start:end]
        resp_values[i, :r_len] = all_values[i, start:end]
        resp_mask[i, :r_len] = 1.0
        resp_logits[i, :r_len] = logits[i, start:end, :]

    return resp_logprobs, resp_values, resp_mask, resp_logits


def compute_mean_ratio(model, query_tensors, response_tensors, old_logprobs, mask,
                       pad_token_id, device):
    """Compute mean importance-sampling ratio across the full batch (no grad)."""
    model.eval()
    with torch.no_grad():
        new_logprobs, _, _, _ = batched_forward_pass(
            model, query_tensors, response_tensors, pad_token_id, device,
        )
    ratio = torch.exp(new_logprobs - old_logprobs.detach())
    return masked_mean(ratio, mask).item()
