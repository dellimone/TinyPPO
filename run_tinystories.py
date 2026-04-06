"""Entry point for the TinyStories PPO-RLHF experiment.

Outputs are organized as:
    outputs/<experiment>/<run_name>/
        metrics.jsonl
        curves.png
        samples.txt
        checkpoints/
            step_00000/policy.pt
            step_00000/trainer.pt
            ...

Usage (GPU defaults):
    python run_tinystories.py --experiment baseline

Usage (CPU smoke test):
    python run_tinystories.py --experiment debug \\
        --batch_size 4 --num_prompts 200 --max_new_tokens 32

Ablation examples:
    python run_tinystories.py --experiment kl_ablation --run_name no_kl --kl_mode none
    python run_tinystories.py --experiment kl_ablation --run_name fixed_kl --kl_mode fixed
    python run_tinystories.py --experiment adv_ablation --run_name no_norm --advantage_normalize none
    python run_tinystories.py --experiment clip_ablation --run_name no_clip --cliprange 1e9
"""
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tinyppo import PPOConfig, PPOTrainer
from tinyppo.dataset import build_dataset, collate_fn
from tinyppo.logger import StepLogger
from tinyppo.model import CausalLMWithValueHead, create_reference_model
from tinyppo.plotting import plot_training_curves, plot_samples
from tinyppo.rewards import GoEmotionsRewardFn, WeightedEmotionRewardFn, NoisyRewardWrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TinyStories PPO-RLHF")

    # Experiment identity
    p.add_argument("--experiment", type=str, default="default",
                   help="Experiment name — used as the top-level output subdirectory.")
    p.add_argument("--run_name", type=str, default=None,
                   help="Run name — subdirectory under experiment. "
                        "Auto-generated from key config values if omitted.")
    p.add_argument("--tag", type=str, default=None,
                   help="Optional suffix appended to the auto-generated run name. "
                        "Use to distinguish runs that differ in params not captured "
                        "by the auto-name (e.g. --tag lr1e5, --tag bs32).")

    # Model
    p.add_argument("--model_name", type=str, default="roneneldan/TinyStories-33M")
    p.add_argument("--seed", type=int, default=42)

    # Data
    p.add_argument("--num_prompts", type=int, default=None)
    p.add_argument("--max_prompt_tokens", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--mini_batch_size", type=int, default=16)

    # Generation
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--top_k", type=int, default=50)

    # Training
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--cliprange", type=float, default=0.2)
    p.add_argument("--cliprange_value", type=float, default=0.2)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_ratio_threshold", type=float, default=10.0)
    p.add_argument("--entropy_coef", type=float, default=0.0,
                   help="Entropy bonus coefficient (0 = disabled)")

    # KL ablation
    p.add_argument("--kl_mode", type=str, default="adaptive",
                   choices=["adaptive", "fixed", "none"])
    p.add_argument("--init_kl_coef", type=float, default=0.3)
    p.add_argument("--target_kl", type=float, default=10.0)
    p.add_argument("--kl_horizon", type=int, default=500)

    # Advantage ablation
    p.add_argument("--advantage_normalize", type=str, default="global",
                   choices=["global", "batch", "none"])

    # Reward ablation
    p.add_argument("--reward_noise_frac", type=float, default=0.0,
                   help="Fraction of reward labels to corrupt (0 = no noise)")
    p.add_argument("--emotion_weights", type=str, default=None,
                   help='JSON dict of emotion weights, e.g. \'{"joy":1.0,"sadness":-0.5}\'. '
                        'When provided, uses WeightedEmotionRewardFn instead of GoEmotionsRewardFn.')

    # Logging / checkpointing
    p.add_argument("--log_every_n_steps", type=int, default=5)
    p.add_argument("--eval_every_n_steps", type=int, default=20)
    p.add_argument("--save_every_n_steps", type=int, default=20,
                   help="Checkpoint interval in steps. 0 to disable.")
    p.add_argument("--base_output_dir", type=str, default="./outputs")

    # Plotting
    p.add_argument("--no_plot", action="store_true",
                   help="Skip generating training curve plots")

    return p.parse_args()


def _auto_run_name(config: PPOConfig, reward_noise_frac: float) -> str:
    """Build a descriptive run name from the key ablation axes."""
    parts = [f"kl_{config.kl_mode}"]
    if config.kl_mode == "fixed":
        parts.append(f"b{config.init_kl_coef}")
    if config.advantage_normalize != "global":
        parts.append(f"adv_{config.advantage_normalize}")
    if config.cliprange >= 100:
        parts.append("noclip")
    elif config.cliprange != 0.2:
        parts.append(f"clip{config.cliprange}")
    if reward_noise_frac > 0:
        parts.append(f"noise{reward_noise_frac}")
    return "_".join(parts)


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Apply CPU-friendly defaults when not overridden
    if args.num_prompts is None:
        args.num_prompts = 200 if device == "cpu" else 6000
    if args.batch_size is None:
        args.batch_size = 4 if device == "cpu" else 64
    if args.max_new_tokens is None:
        args.max_new_tokens = 32 if device == "cpu" else 128

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # PPOConfig doesn't know about experiment/run_name/base_output_dir
    config = PPOConfig.from_dict(vars(args))

    # Build output directory: outputs/<experiment>/<run_name>/
    run_name = args.run_name or _auto_run_name(config, args.reward_noise_frac)
    if args.tag and not args.run_name:
        run_name = f"{run_name}_{args.tag}"
    output_dir = f"{args.base_output_dir}/{args.experiment}/{run_name}"
    config.output_dir = output_dir

    print(f"\nExperiment : {args.experiment}")
    print(f"Run        : {run_name}")
    print(f"Output dir : {output_dir}")
    print(f"\nConfig:\n"
          f"  model={config.model_name}\n"
          f"  batch={config.batch_size}, mini_batch={config.mini_batch_size}, "
          f"ppo_epochs={config.ppo_epochs}\n"
          f"  kl_mode={config.kl_mode}, advantage_normalize={config.advantage_normalize}\n"
          f"  prompts={config.num_prompts}, max_new_tokens={config.max_new_tokens}\n")

    # ── Tokenizer & models ────────────────────────────────────────────────
    print("[init] Loading tokenizer + policy...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = CausalLMWithValueHead(config.model_name).to(device)
    ref_model = create_reference_model(model)
    ref_model = ref_model.to(device)
    print(f"Policy params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Reward function ───────────────────────────────────────────────────
    print("\n[init] Loading reward model...")
    if args.emotion_weights:
        import json as _json
        weights = _json.loads(args.emotion_weights)
        reward_fn = WeightedEmotionRewardFn(emotion_weights=weights, device=device)
        print(f"  Weighted emotion reward: {weights}")
    else:
        reward_fn = GoEmotionsRewardFn(
            model_name="SamLowe/roberta-base-go_emotions",
            target_emotion="joy",
            device=device,
        )
    if args.reward_noise_frac > 0:
        reward_fn = NoisyRewardWrapper(reward_fn, noise_frac=args.reward_noise_frac,
                                       seed=args.seed)
        print(f"  Noisy reward wrapper: noise_frac={args.reward_noise_frac}")

    # ── Dataset ───────────────────────────────────────────────────────────
    print("\n[init] Building dataset...")
    dataset = build_dataset(
        tokenizer, n=config.num_prompts,
        max_prompt_tokens=config.max_prompt_tokens, seed=config.seed,
    )
    print(f"Dataset: {len(dataset)} prompts")
    print(f"Example: {dataset[0]['query']}")

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        drop_last=True, collate_fn=collate_fn,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        logger=StepLogger(output_dir),
    )

    logger = trainer.train(dataloader)

    # ── Summary ───────────────────────────────────────────────────────────
    rewards = logger.get_metric("reward")
    n = len(rewards)
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {args.experiment}/{run_name}")
    print(f"{'='*70}")
    if n > 10:
        print(f"  Total steps:    {n}")
        print(f"  Final R (avg):  {float(np.mean(rewards[-10:])):.4f}")
        print(f"  Peak R:         {max(rewards):.4f}")
        kls = logger.get_metric("kl")
        print(f"  Final KL (avg): {float(np.mean(kls[-10:])):.3f}")
        betas = logger.get_metric("beta")
        print(f"  Final beta:     {betas[-1]:.4f}")
    else:
        print(f"  Only {n} steps completed — not enough for summary")

    # ── Plot ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_training_curves(
            logger,
            target_kl=config.target_kl,
            save_path=f"{output_dir}/curves.png",
            show=False,
        )
        plot_samples(logger, save_path=f"{output_dir}/samples.txt")
        print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
