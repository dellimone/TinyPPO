import dataclasses
from dataclasses import dataclass, field


@dataclass
class PPOConfig:
    # ── Model ──────────────────────────────────────────────────────────────
    model_name: str = "roneneldan/TinyStories-33M"
    seed: int = 42

    # ── Data ───────────────────────────────────────────────────────────────
    num_prompts: int = 6000
    max_prompt_tokens: int = 20
    batch_size: int = 64
    mini_batch_size: int = 16

    # ── Generation ─────────────────────────────────────────────────────────
    max_new_tokens: int = 128
    top_k: int = 50

    # ── Training ───────────────────────────────────────────────────────────
    num_epochs: int = 1
    learning_rate: float = 5e-6
    ppo_epochs: int = 4
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    max_ratio_threshold: float = 10.0
    entropy_coef: float = 0.0  # entropy bonus; 0 = disabled

    # ── KL ablation ────────────────────────────────────────────────────────
    # kl_mode: "adaptive" (proportional controller), "fixed" (static beta), "none" (no KL)
    kl_mode: str = "adaptive"
    init_kl_coef: float = 0.3
    target_kl: float = 10.0
    kl_horizon: int = 500

    # ── Advantage estimation ablation ──────────────────────────────────────
    # advantage_normalize: "global" (whiten over batch), "batch" (per-seq), "none"
    advantage_normalize: str = "global"

    # ── Value function init ablation ───────────────────────────────────────
    # vf_init: "random" or "from_reward_model" (pass vf_init_weights to CausalLMWithValueHead)
    vf_init: str = "random"

    # ── Pre-training loss mixing ablation ──────────────────────────────────
    # Set > 0 and pass pretrain_dataloader to PPOTrainer to enable
    pretrain_loss_coef: float = 0.0

    # ── Logging ────────────────────────────────────────────────────────────
    log_every_n_steps: int = 5
    eval_every_n_steps: int = 20
    save_every_n_steps: int = 0   # checkpoint interval; 0 = disabled
    output_dir: str = "./outputs"

    # ── Eval prompts ───────────────────────────────────────────────────────
    eval_prompts: list = field(default_factory=lambda: [
        "Once upon a time, there was a",
        "The little girl loved to",
        "One sunny day, the boy went",
        "In a small village, there lived a",
        "The puppy was so excited to",
    ])

    @classmethod
    def from_dict(cls, d: dict) -> "PPOConfig":
        """Create a PPOConfig from a (partial) dict, ignoring unknown keys."""
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known and v is not None}
        return cls(**filtered)

    def __post_init__(self):
        valid_kl = {"adaptive", "fixed", "none"}
        if self.kl_mode not in valid_kl:
            raise ValueError(f"kl_mode must be one of {valid_kl}, got '{self.kl_mode}'")
        valid_norm = {"global", "batch", "none"}
        if self.advantage_normalize not in valid_norm:
            raise ValueError(
                f"advantage_normalize must be one of {valid_norm}, "
                f"got '{self.advantage_normalize}'"
            )
