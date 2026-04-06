# TinyPPO

A small, from-scratch PyTorch implementation of **PPO for RLHF**, built as a study of *which PPO hyperparameters actually matter in the RLHF regime and how they interact*. The setup is deliberately compact — a 33M-parameter language model, a single learned reward, a handful of ablation axes — so that full sweeps fit comfortably on one GPU and every run is easy to reason about end-to-end.

The policy is [`TinyStories-33M`](https://huggingface.co/roneneldan/TinyStories-33M), fine-tuned with PPO to generate children's stories that maximize a "joy" reward provided by [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions). Since the base model is already fluent on this domain, we skip the SFT stage of the canonical RLHF pipeline and use the pretrained checkpoint as both the initial policy and the frozen reference `π_ref`.

The implementation favors readability and hackability over raw throughput: the trainer is a single class with clearly labelled override points, every core operation lives in its own small module, and the `run_tinystories.py` entry point exposes every knob as a CLI flag so ablations compose without code changes.
## Quickstart

```bash
pip install -r requirements.txt

# Quick CPU smoke test
python run_tinystories.py --experiment debug \
    --batch_size 4 --num_prompts 50 --max_new_tokens 20 --ppo_epochs 2

# Full GPU training run
python run_tinystories.py --experiment baseline
```

## Running ablations

Individual ablations are single CLI flags:

```bash
python run_tinystories.py --experiment kl --kl_mode adaptive
python run_tinystories.py --experiment kl --kl_mode fixed --init_kl_coef 0.5
python run_tinystories.py --experiment kl --kl_mode none
```

Systematic sweeps across the 11 supported axes are driven by `sweep.sh`:

```bash
bash sweep.sh lr          # learning-rate sweep
bash sweep.sh kl_mode     # KL penalty modes
bash sweep.sh clip        # PPO clip range ε
# … see sweep.sh for the full list
```

## Project Structure

```
tinyppo/              # library code
run_tinystories.py    # CLI entry point
sweep.sh              # systematic ablation driver
analysis.ipynb        # post-hoc analysis of sweep runs
presentation/         # Reveal.js slide deck + assets
outputs/              # per-experiment metrics, samples, checkpoints
Docs/                 # reference papers
requirements.txt
```

## References

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms* — the clipped surrogate objective and PPO itself.
- Christiano et al. (2017), *Deep RL from Human Preferences* — learning a scalar reward model from preference comparisons.
- Ziegler et al. (2019), *Fine-Tuning Language Models from Human Preferences* — RLHF + per-token KL penalty applied to LMs; source of the adaptive KL controller.
- Ouyang et al. (2022), *Training Language Models to Follow Instructions with Human Feedback (InstructGPT)* — the three-stage SFT → RM → PPO pipeline as a practical recipe.
