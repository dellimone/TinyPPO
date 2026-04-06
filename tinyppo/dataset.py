import torch
from datasets import load_dataset, Dataset


def build_dataset(tokenizer, n: int = 2000, max_prompt_tokens: int = 20, seed: int = 42):
    """Build a dataset of TinyStories prompts using the first sentence of each story.

    Args:
        tokenizer: HuggingFace tokenizer.
        n: Number of prompts to collect.
        max_prompt_tokens: Max tokenized length (truncation).
        seed: Random seed for shuffling.

    Returns:
        HuggingFace Dataset with columns: query, input_ids, attention_mask.
    """
    ds = load_dataset("roneneldan/TinyStories", split="train").shuffle(seed=seed)
    prompts = []
    for ex in ds:
        # Take the first sentence (split on '. ' or '.\n')
        text = ex["text"].strip()
        for sep in (". ", ".\n"):
            idx = text.find(sep)
            if idx != -1:
                prompt = text[:idx + 1]  # include the period
                break
        else:
            prompt = text  # no sentence boundary found, use full text
        if 15 < len(prompt) < 200:
            prompts.append(prompt)
        if len(prompts) >= n:
            break

    ds = Dataset.from_dict({"query": prompts})
    ds = ds.map(
        lambda x: tokenizer(
            x["query"], truncation=True, max_length=max_prompt_tokens, padding="max_length",
        ),
        batched=True,
    )
    ds.set_format("torch")
    return ds


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "query": [b["query"] for b in batch],
    }
