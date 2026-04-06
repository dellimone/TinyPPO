import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig


class CausalLMWithValueHead(nn.Module):
    """Wraps a causal LM with a scalar value head on the last hidden state."""

    def __init__(self, model_name: str, vf_init_weights=None):
        """
        Args:
            model_name: HuggingFace model identifier.
            vf_init_weights: Optional state dict to initialize the value head linear layer.
                             Keys expected: "weight" and "bias". If None, random init.
        """
        super().__init__()
        self.pretrained_model = AutoModelForCausalLM.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.v_head = nn.Linear(config.hidden_size, 1)
        self.v_head_dropout = nn.Dropout(0.0)

        if vf_init_weights is not None:
            self.v_head.load_state_dict(vf_init_weights)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.pretrained_model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        lm_logits = outputs.logits
        value = self.v_head(self.v_head_dropout(hidden_states)).squeeze(-1)
        return lm_logits, value

    def generate(self, **kwargs):
        return self.pretrained_model.generate(**kwargs)


def create_reference_model(model: CausalLMWithValueHead) -> CausalLMWithValueHead:
    """Deep-copy model with all params frozen."""
    ref = copy.deepcopy(model)
    for param in ref.parameters():
        param.requires_grad = False
    ref.eval()
    return ref
