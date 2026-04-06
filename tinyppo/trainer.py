from __future__ import annotations
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import PPOConfig
from .kl_controller import (
    AdaptiveKLController, FixedKLController, NoKLController, KLControllerBase,
)
from .rollout import batched_forward_pass, compute_mean_ratio
from .losses import compute_rewards_with_kl, compute_advantages, ppo_loss
from .ops import entropy_from_logits, masked_mean
from .logger import StepLogger
from .rewards import RewardFn


def _safe_mean(lst: list) -> float:
    """Mean of a list, or 0.0 if empty."""
    return float(np.mean(lst)) if lst else 0.0


class PPOTrainer:
    """PPO-RLHF trainer.

    Encapsulates the full training loop. Ablations are controlled via:
      - PPOConfig flags (kl_mode, advantage_normalize, cliprange, etc.)
      - Injectable reward_fn callable
      - Subclass overrides of _ppo_loss / _generate / _score /
        _prepare_queries / _compute_step_context / _mini_batch_loss

    Args:
        config: PPOConfig instance.
        model: Policy model (CausalLMWithValueHead).
        ref_model: Frozen reference model (same architecture as model).
        tokenizer: HuggingFace tokenizer.
        reward_fn: Callable (list[str]) -> list[float]. Any reward function works.
        optimizer: Optional. Defaults to Adam with config.learning_rate.
        logger: Optional StepLogger. Defaults to a new StepLogger(config.output_dir).
        pretrain_dataloader: Required if config.pretrain_loss_coef > 0.
    """

    def __init__(
        self,
        config: PPOConfig,
        model,
        ref_model,
        tokenizer,
        reward_fn: RewardFn,
        optimizer: torch.optim.Optimizer | None = None,
        logger: StepLogger | None = None,
        pretrain_dataloader: DataLoader | None = None,
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.logger = logger or StepLogger(config.output_dir)
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(), lr=config.learning_rate
        )
        self.pretrain_dataloader = pretrain_dataloader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kl_ctl = self._build_kl_controller()

        if config.pretrain_loss_coef > 0 and pretrain_dataloader is None:
            raise ValueError(
                "pretrain_loss_coef > 0 requires a pretrain_dataloader"
            )

        # Infinite iterator for pretrain batches
        self._pretrain_iter = iter([])
        if pretrain_dataloader is not None:
            self._pretrain_iter = self._infinite(pretrain_dataloader)

    # ── Public API ─────────────────────────────────────────────────────────

    def train(self, dataloader: DataLoader) -> StepLogger:
        """Run the full training loop."""
        cfg = self.config
        steps_per_epoch = len(dataloader.dataset) // cfg.batch_size
        total_steps = 0
        t0 = time.time()

        print(f"PPO config: batch={cfg.batch_size}, mini_batch={cfg.mini_batch_size}, "
              f"ppo_epochs={cfg.ppo_epochs}")
        print(f"LR: {cfg.learning_rate}")
        print(f"KL: mode={cfg.kl_mode}, init_coef={cfg.init_kl_coef}, "
              f"target={cfg.target_kl}, horizon={cfg.kl_horizon}")
        print(f"Advantage normalize: {cfg.advantage_normalize}")
        print(f"Steps per epoch: {steps_per_epoch}, "
              f"total: ~{steps_per_epoch * cfg.num_epochs}")
        print(f"\n{'='*70}")
        print(f"  Starting training: {cfg.num_epochs} epochs, {self.device}")
        print(f"{'='*70}\n")

        for epoch in range(cfg.num_epochs):
            for batch in dataloader:
                self._train_step(batch, total_steps, t0)
                total_steps += 1

        elapsed = time.time() - t0
        print(f"\n{'='*70}")
        print(f"  Done: {total_steps} steps in {elapsed:.0f}s "
              f"({elapsed / max(total_steps, 1):.1f}s/step avg)")
        print(f"{'='*70}")

        self.logger.save()
        return self.logger

    # ── Training step ──────────────────────────────────────────────────────

    def _train_step(self, batch: dict, total_steps: int, t0: float) -> None:
        cfg = self.config
        step_t0 = time.time()

        # 1. Build query tensors (strip padding)
        query_tensors = [
            ids[ids != self.tokenizer.pad_token_id] for ids in batch["input_ids"]
        ]

        # 2. Generate responses
        response_tensors = self._generate(query_tensors)
        response_texts = self.tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True
        )

        # 3. Expand queries to match responses (identity for PPO, repeats G× for GRPO)
        eff_queries, eff_query_strs = self._prepare_queries(
            query_tensors, batch["query"], response_tensors
        )

        # 4. Score query+response pairs
        score_tensors = self._score(eff_query_strs, response_texts)

        # 5. Forward passes + advantage computation (algorithm-specific)
        ctx = self._compute_step_context(eff_queries, response_tensors, score_tensors)

        # 6. Mini-batch PPO updates
        self.model.train()
        batch_indices = list(range(len(eff_queries)))
        epoch_stats: dict[str, list] = {
            "policy_loss": [], "value_loss": [], "clip_frac": [],
            "grad_norm": [],
        }
        batch_skipped = False

        for ppo_epoch in range(cfg.ppo_epochs):
            mean_ratio = compute_mean_ratio(
                self.model, eff_queries, response_tensors,
                ctx["old_logprobs"], ctx["mask"],
                self.tokenizer.pad_token_id, self.device,
            )
            if mean_ratio > cfg.max_ratio_threshold:
                print(f"  [WARNING] step {total_steps} ppo_epoch {ppo_epoch}: "
                      f"mean ratio={mean_ratio:.2f} > {cfg.max_ratio_threshold:.1f} "
                      f"— skipping batch")
                batch_skipped = True
                break

            self.model.train()
            random.shuffle(batch_indices)

            for mb_start in range(0, len(batch_indices), cfg.mini_batch_size):
                mb_idx = batch_indices[mb_start:mb_start + cfg.mini_batch_size]
                if not mb_idx:
                    continue

                mb_queries = [eff_queries[i] for i in mb_idx]
                mb_responses = [response_tensors[i] for i in mb_idx]

                loss, stats = self._mini_batch_loss(
                    ctx, mb_idx, mb_queries, mb_responses
                )

                # Optional: pre-training loss mixing
                if cfg.pretrain_loss_coef > 0:
                    pt_batch = next(self._pretrain_iter)
                    pt_ids = pt_batch["input_ids"].to(self.device)
                    pt_attn = pt_batch.get("attention_mask")
                    if pt_attn is not None:
                        pt_attn = pt_attn.to(self.device)
                    pt_logits, _ = self.model(pt_ids, attention_mask=pt_attn)
                    lm_loss = nn.functional.cross_entropy(
                        pt_logits[:, :-1, :].reshape(-1, pt_logits.size(-1)),
                        pt_ids[:, 1:].reshape(-1),
                    )
                    loss = loss + cfg.pretrain_loss_coef * lm_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                for k in ("policy_loss", "value_loss", "clip_frac"):
                    epoch_stats[k].append(stats.get(k, 0.0))
                epoch_stats["grad_norm"].append(grad_norm.item())

        # 7. Update KL controller
        mask = ctx["mask"]
        mean_kl = ((ctx["kls"] * mask).sum(dim=-1)).mean().item()
        mean_kl_approx = ((ctx["kls_approx"] * mask).sum(dim=-1)).mean().item()
        self.kl_ctl.update(mean_kl, cfg.batch_size)

        # 8. Logging
        self._log_step(
            total_steps, step_t0, ctx, epoch_stats,
            response_tensors, mean_kl, mean_kl_approx, batch_skipped,
        )

    # ── Algorithm-specific hooks (override in subclasses) ──────────────────

    def _prepare_queries(self, query_tensors, query_strings, response_tensors):
        """Expand queries to match responses. Identity for PPO; GRPO repeats G×."""
        return query_tensors, query_strings

    def _compute_step_context(self, query_tensors, response_tensors, score_tensors):
        """Forward passes + advantage computation. Returns dict for mini-batch loop."""
        cfg = self.config
        self.model.eval()
        old_logprobs, old_values, mask, old_logits = batched_forward_pass(
            self.model, query_tensors, response_tensors,
            self.tokenizer.pad_token_id, self.device,
        )
        with torch.no_grad():
            ref_logprobs, _, _, _ = batched_forward_pass(
                self.ref_model, query_tensors, response_tensors,
                self.tokenizer.pad_token_id, self.device,
            )

        rewards, kls, kls_approx = compute_rewards_with_kl(
            score_tensors, old_logprobs, ref_logprobs, mask, self.kl_ctl.value,
        )
        advantages, returns = compute_advantages(
            old_values.detach(), rewards.detach(), mask,
            gamma=cfg.gamma, lam=cfg.lam, normalize=cfg.advantage_normalize,
        )

        return {
            "old_logprobs": old_logprobs, "old_values": old_values,
            "mask": mask, "old_logits": old_logits,
            "ref_logprobs": ref_logprobs,
            "advantages": advantages, "returns": returns,
            "kls": kls, "kls_approx": kls_approx,
            "score_tensors": score_tensors,
        }

    def _mini_batch_loss(self, ctx, mb_idx, mb_queries, mb_responses):
        """Compute loss for one mini-batch. Returns (loss, stats)."""
        mb_resp_len = max(r.size(0) for r in mb_responses)
        mb_old_logprobs = ctx["old_logprobs"][mb_idx, :mb_resp_len].detach()
        mb_old_values = ctx["old_values"][mb_idx, :mb_resp_len].detach()
        mb_advantages = ctx["advantages"][mb_idx, :mb_resp_len]
        mb_returns = ctx["returns"][mb_idx, :mb_resp_len].detach()
        mb_mask = ctx["mask"][mb_idx, :mb_resp_len]

        mb_new_logprobs, mb_new_values, _, mb_logits = batched_forward_pass(
            self.model, mb_queries, mb_responses,
            self.tokenizer.pad_token_id, self.device,
        )

        loss, pg_loss, vf_loss, stats = self._ppo_loss(
            mb_new_logprobs, mb_old_logprobs,
            mb_new_values, mb_old_values,
            mb_advantages, mb_returns, mb_mask,
            logits=mb_logits,
        )
        return loss, stats

    def _ppo_loss(self, new_logprobs, old_logprobs, new_values, old_values,
                  advantages, returns, mask, logits=None):
        """Compute PPO loss. Override to ablate loss internals.

        Example (disable value clipping):
            class NoCriticClipTrainer(PPOTrainer):
                def _ppo_loss(self, *args, **kw):
                    return ppo_loss(*args, **kw, cliprange_value=1e9)
        """
        return ppo_loss(
            new_logprobs, old_logprobs, new_values, old_values,
            advantages, returns, mask,
            cliprange=self.config.cliprange,
            cliprange_value=self.config.cliprange_value,
            vf_coef=self.config.vf_coef,
            logits=logits,
            entropy_coef=self.config.entropy_coef,
        )

    def _generate(self, query_tensors: list) -> list:
        """Generate response tokens for a batch of query tensors.

        Override to change generation strategy (e.g., greedy, beam search).
        """
        cfg = self.config
        self.model.eval()
        max_q_len = max(q.size(0) for q in query_tensors)
        gen_input_ids = torch.full(
            (len(query_tensors), max_q_len), self.tokenizer.pad_token_id,
            dtype=torch.long, device=self.device,
        )
        gen_attn_mask = torch.zeros(
            len(query_tensors), max_q_len, dtype=torch.long, device=self.device,
        )
        for i, q in enumerate(query_tensors):
            offset = max_q_len - q.size(0)
            gen_input_ids[i, offset:] = q.to(self.device)
            gen_attn_mask[i, offset:] = 1

        with torch.no_grad():
            gen_output = self.model.generate(
                input_ids=gen_input_ids,
                attention_mask=gen_attn_mask,
                min_length=-1,
                top_k=cfg.top_k,
                do_sample=True,
                max_new_tokens=cfg.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response_tensors = []
        for i, q in enumerate(query_tensors):
            resp = gen_output[i, max_q_len:]
            eos_positions = (resp == self.tokenizer.eos_token_id).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                resp = resp[:eos_positions[0].item() + 1]
            if resp.numel() == 0:
                resp = torch.tensor([self.tokenizer.eos_token_id], device=self.device)
            response_tensors.append(resp.cpu())

        return response_tensors

    def _score(self, queries: list[str], responses: list[str]) -> torch.Tensor:
        """Score query+response pairs with the reward function.

        Override to change how text is assembled or post-process scores.
        """
        full_texts = [q + " " + r for q, r in zip(queries, responses)]
        scores = self.reward_fn(full_texts)
        return torch.tensor(scores, dtype=torch.float32, device=self.device)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _log_step(self, total_steps, step_t0, ctx, epoch_stats,
                  response_tensors, mean_kl, mean_kl_approx, batch_skipped):
        """Unified logging for both PPO and GRPO."""
        cfg = self.config
        mask = ctx["mask"]
        scores = ctx["score_tensors"]

        r_mean = float(scores.mean().cpu().item())
        r_std = float(scores.std().cpu().item())
        mean_resp_len = float(np.mean([r.size(0) for r in response_tensors]))
        beta = self.kl_ctl.value

        with torch.no_grad():
            ent = masked_mean(entropy_from_logits(ctx["old_logits"]), mask).item()

        # Explained variance (only when value function is used)
        explained_var = 0.0
        if "returns" in ctx and "old_values" in ctx:
            with torch.no_grad():
                flat_returns = ctx["returns"][mask.bool()].float()
                flat_values = ctx["old_values"][mask.bool()].float()
                var_returns = flat_returns.var().item()
                explained_var = 1.0 - (flat_returns - flat_values).var().item() / (var_returns + 1e-8)

        dt = time.time() - step_t0
        self.logger.log(total_steps, {
            "reward": r_mean,
            "reward_std": r_std,
            "kl": mean_kl,
            "kl_approx": mean_kl_approx,
            "beta": beta,
            "entropy": ent,
            "clip_frac": _safe_mean(epoch_stats["clip_frac"]),
            "policy_loss": _safe_mean(epoch_stats["policy_loss"]),
            "value_loss": _safe_mean(epoch_stats["value_loss"]),
            "grad_norm": _safe_mean(epoch_stats["grad_norm"]),
            "mean_resp_len": mean_resp_len,
            "explained_var": explained_var,
            "batch_skipped": int(batch_skipped),
            "step_time": dt,
        })

        if total_steps % cfg.log_every_n_steps == 0:
            print(f"  [step {total_steps:4d}] "
                  f"R={r_mean:.4f} | KL={mean_kl:.3f} | beta={beta:.4f} | "
                  f"{dt:.1f}s/step")

        if total_steps % cfg.eval_every_n_steps == 0:
            self._run_eval(total_steps)

        if cfg.save_every_n_steps > 0 and total_steps > 0 and total_steps % cfg.save_every_n_steps == 0:
            self._save_checkpoint(total_steps)

    def _build_kl_controller(self) -> KLControllerBase:
        cfg = self.config
        if cfg.kl_mode == "adaptive":
            return AdaptiveKLController(cfg.init_kl_coef, cfg.target_kl, cfg.kl_horizon)
        elif cfg.kl_mode == "fixed":
            return FixedKLController(cfg.init_kl_coef)
        elif cfg.kl_mode == "none":
            return NoKLController()
        raise ValueError(f"Unknown kl_mode: '{cfg.kl_mode}'")

    def _run_eval(self, step: int) -> None:
        """Generate from eval prompts and log samples."""
        cfg = self.config
        self.model.eval()
        eval_enc = self.tokenizer(
            cfg.eval_prompts, return_tensors="pt", padding=True,
        ).to(self.device)
        with torch.no_grad():
            eval_out = self.model.generate(
                **eval_enc,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                top_k=cfg.top_k,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        samples = self.tokenizer.batch_decode(eval_out, skip_special_tokens=True)
        self.logger.log_samples(step, samples)
        for s in samples:
            print(f"    >> {s}")

    def _save_checkpoint(self, step: int) -> None:
        """Save policy.pt (inference) and trainer.pt (resume) under output_dir/checkpoints/."""
        import os
        ckpt_dir = os.path.join(self.config.output_dir, "checkpoints", f"step_{step:05d}")
        os.makedirs(ckpt_dir, exist_ok=True)

        try:
            torch.save(
                self.model.pretrained_model.state_dict(),
                os.path.join(ckpt_dir, "policy.pt"),
            )
            torch.save({
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "kl_coef": self.kl_ctl.value,
            }, os.path.join(ckpt_dir, "trainer.pt"))
        except (RuntimeError, OSError) as e:
            print(f"  [ckpt] WARNING: checkpoint failed at step {step} — {e}")
            return

        print(f"  [ckpt] Saved checkpoint at step {step} → {ckpt_dir}")

    @staticmethod
    def _infinite(dataloader: DataLoader):
        """Infinite iterator over a DataLoader (cycles when exhausted)."""
        while True:
            yield from dataloader
