import numpy as np


class KLControllerBase:
    """Shared interface: .value (float), .update(current_kl, n_steps)."""
    value: float

    def update(self, current_kl: float, n_steps: int) -> None:
        raise NotImplementedError


class AdaptiveKLController(KLControllerBase):
    """Proportional KL controller — adjusts beta to track target KL."""

    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        proportional_error = np.clip(current_kl / self.target - 1, -0.2, 0.2)
        mult = 1.0 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController(KLControllerBase):
    """KL coefficient that never changes."""

    def __init__(self, kl_coef: float):
        self.value = kl_coef

    def update(self, current_kl: float, n_steps: int) -> None:
        pass  # fixed — no update


class NoKLController(KLControllerBase):
    """KL penalty disabled — coefficient is always 0."""

    value: float = 0.0

    def update(self, current_kl: float, n_steps: int) -> None:
        pass  # always zero
