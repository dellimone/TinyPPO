from __future__ import annotations
import random
from typing import Callable

from transformers import pipeline

# Type alias: any callable that takes a list of strings and returns a list of floats
RewardFn = Callable[[list[str]], list[float]]


class GoEmotionsRewardFn:
    """Reward = P(target_emotion) from a GoEmotions classifier.

    Example:
        reward_fn = GoEmotionsRewardFn(target_emotion="joy", device="cuda")
        scores = reward_fn(["I'm so happy today!", "This is terrible."])
    """

    def __init__(
        self,
        model_name: str = "SamLowe/roberta-base-go_emotions",
        target_emotion: str = "joy",
        device: str = "cpu",
    ):
        self.target_emotion = target_emotion
        self._pipe = pipeline(
            "text-classification", model=model_name, device=device, top_k=None,
        )

    def __call__(self, texts: list[str]) -> list[float]:
        return [
            {s["label"]: s["score"] for s in out}.get(self.target_emotion, 0.0)
            for out in self._pipe(texts, batch_size=len(texts))
        ]


class WeightedEmotionRewardFn:
    """Reward = weighted sum of emotion probabilities from GoEmotions.

    Example:
        reward_fn = WeightedEmotionRewardFn(
            emotion_weights={"joy": 1.0, "sadness": -0.5},
            device="cuda",
        )
        scores = reward_fn(["I'm so happy!", "This is sad."])
    """

    def __init__(
        self,
        emotion_weights: dict[str, float],
        model_name: str = "SamLowe/roberta-base-go_emotions",
        device: str = "cpu",
    ):
        self.emotion_weights = emotion_weights
        self._pipe = pipeline(
            "text-classification", model=model_name, device=device, top_k=None,
        )

    def __call__(self, texts: list[str]) -> list[float]:
        scores = []
        for out in self._pipe(texts, batch_size=len(texts)):
            score_map = {s["label"]: s["score"] for s in out}
            total = sum(w * score_map.get(emo, 0.0)
                        for emo, w in self.emotion_weights.items())
            scores.append(total)
        return scores


class NoisyRewardWrapper:
    """Wraps any RewardFn and randomly corrupts a fraction of scores.

    Useful for testing robustness to noisy or weak reward models.
    Corruption replaces a score with a uniform random value in [0, 1].

    Args:
        base_fn: The underlying reward function.
        noise_frac: Fraction of samples to corrupt in [0, 1]. 0 = no noise.
        seed: Random seed for reproducibility.

    Example:
        noisy_fn = NoisyRewardWrapper(GoEmotionsRewardFn(...), noise_frac=0.2)
        trainer = PPOTrainer(..., reward_fn=noisy_fn)
    """

    def __init__(self, base_fn: RewardFn, noise_frac: float = 0.0, seed: int = 42):
        if not 0.0 <= noise_frac <= 1.0:
            raise ValueError(f"noise_frac must be in [0, 1], got {noise_frac}")
        self.base_fn = base_fn
        self.noise_frac = noise_frac
        self._rng = random.Random(seed)

    def __call__(self, texts: list[str]) -> list[float]:
        scores = list(self.base_fn(texts))
        for i in range(len(scores)):
            if self._rng.random() < self.noise_frac:
                scores[i] = self._rng.random()  # uniform [0, 1] corruption
        return scores
