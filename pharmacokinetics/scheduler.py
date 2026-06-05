"""
Pharmacokinetic (PK) Scheduler — ★ Novel Contribution

Existing steering vectors apply a FIXED coefficient to every token.
Real pharmacology doesn't work that way:
  - Drugs have half-lives, absorption curves, clearance rates
  - Dose adjusts when patient state changes (adaptive dosing)

This module implements the first pharmacokinetic model for LLM steering:

  1. Exponential decay:   c(t) = c0 * exp(-t / tau)
     Models drug clearance as generation proceeds.

  2. Uncertainty-triggered boost:
     When the model's internal entropy spikes (= model is uncertain),
     automatically increase dose to steer back on course.

  3. Oscillating delivery:
     Sinusoidal dose schedule — models slow-release formulations.

  4. Custom schedule:
     User-defined callable for arbitrary pharmacokinetic profiles.

This enables the first "time-release" and "adaptive" AI compounds.

Inspired by:
  - Pharmacokinetic modeling: one-compartment model, Bateman equation
  - Inference-time scaling: Snell et al. (2024) https://arxiv.org/abs/2408.03314
  - SADI adaptive intervention: Wang et al. (2024) https://arxiv.org/abs/2410.12299
"""

from __future__ import annotations
import torch
import math
from typing import Callable, Optional
from transformer_lens import HookedTransformer
from dataclasses import dataclass


@dataclass
class PKProfile:
    """A complete pharmacokinetic profile for a steering compound."""
    name: str
    c0: float           # Initial peak concentration (coefficient)
    half_life: float    # Half-life in tokens
    lag: int = 0        # Absorption lag in tokens (time to peak)
    tmax: Optional[int] = None  # Token of peak concentration


class PKScheduler:
    """
    Pharmacokinetic dose scheduler for LLM steering vectors.

    Models how dose (coefficient) evolves over the token sequence,
    analogous to drug plasma concentration over time.

    One-compartment model:
        c(t) = c0 * exp(-lambda * t)
        where lambda = ln(2) / half_life

    Bateman equation (absorption + elimination):
        c(t) = c0 * (exp(-lambda_e * t) - exp(-lambda_a * t))
        models oral administration with absorption phase.
    """

    def __init__(
        self,
        initial_coefficient: float = 15.0,
        half_life_tokens: float = 50.0,
        mode: str = "exponential",
        custom_schedule: Optional[Callable[[int], float]] = None,
        uncertainty_boost: bool = False,
        uncertainty_threshold: float = 3.5,
        uncertainty_boost_factor: float = 2.0,
    ):
        """
        Args:
            initial_coefficient: Starting dose (c0)
            half_life_tokens: Tokens at which dose halves (exponential decay)
            mode: 'exponential', 'bateman', 'constant', 'oscillating', 'custom'
            custom_schedule: Callable(token_idx) -> coefficient, for mode='custom'
            uncertainty_boost: If True, boost dose when model entropy spikes
            uncertainty_threshold: Entropy above which to trigger boost
            uncertainty_boost_factor: Multiplier applied during boost
        """
        self.c0 = initial_coefficient
        self.half_life = half_life_tokens
        self.lam = math.log(2) / max(half_life_tokens, 1e-6)
        self.mode = mode
        self.custom_schedule = custom_schedule
        self.uncertainty_boost = uncertainty_boost
        self.uncertainty_threshold = uncertainty_threshold
        self.boost_factor = uncertainty_boost_factor
        self._token_idx = 0

    def reset(self) -> None:
        """Reset token counter (call before each new generation)."""
        self._token_idx = 0

    def get_coefficient(self, token_idx: int, entropy: Optional[float] = None) -> float:
        """
        Get the coefficient (dose) at a given token position.

        Args:
            token_idx: Current token in the generation sequence
            entropy: Optional current model entropy for adaptive dosing

        Returns:
            Dose coefficient for this token
        """
        t = token_idx

        if self.mode == "exponential":
            c = self.c0 * math.exp(-self.lam * t)

        elif self.mode == "bateman":
            # Absorption rate = 3x elimination rate (typical oral absorption)
            lam_a = self.lam * 3
            lam_e = self.lam
            if lam_a != lam_e:
                c = self.c0 * (math.exp(-lam_e * t) - math.exp(-lam_a * t))
                c = max(c, 0.0)
            else:
                c = self.c0 * t * math.exp(-lam_e * t)

        elif self.mode == "constant":
            c = self.c0

        elif self.mode == "oscillating":
            # Slow-release: sinusoidal modulation around decaying envelope
            envelope = self.c0 * math.exp(-self.lam * t)
            oscillation = 0.5 * (1 + math.sin(2 * math.pi * t / self.half_life))
            c = envelope * oscillation

        elif self.mode == "custom":
            if self.custom_schedule is None:
                raise ValueError("custom_schedule must be set for mode='custom'")
            c = self.custom_schedule(t)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Uncertainty-triggered adaptive boost
        if self.uncertainty_boost and entropy is not None:
            if entropy > self.uncertainty_threshold:
                c *= self.boost_factor

        return max(c, 0.0)

    def get_schedule(self, n_tokens: int) -> list[float]:
        """Pre-compute the full dose schedule for n_tokens."""
        return [self.get_coefficient(t) for t in range(n_tokens)]

    def make_hook(
        self,
        direction: torch.Tensor,
        layer: int,
        model: HookedTransformer,
    ) -> Callable:
        """
        Create a transformer hook that applies PK-scheduled steering.

        The hook updates the dose at each call based on the current token
        position in the generation sequence.

        Usage:
            scheduler = PKScheduler(c0=15.0, half_life_tokens=50)
            hook = scheduler.make_hook(direction, layer=15, model=model)
            tokens = model.to_tokens(prompt)
            with model.hooks(fwd_hooks=[(f"blocks.15.hook_resid_pre", hook)]):
                output = model.generate(tokens, max_new_tokens=200)
        """
        scheduler = self

        def pk_hook(value, hook):
            t = scheduler._token_idx
            c = scheduler.get_coefficient(t)
            scheduler._token_idx += 1
            return value + c * direction.to(value.device).unsqueeze(0).unsqueeze(0)

        return pk_hook


def plot_pk_profiles(n_tokens: int = 200, save_path: Optional[str] = None):
    """
    Visualize different PK profiles over token sequence.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib for visualization")
        return

    profiles = {
        "Exponential (IV)":      PKScheduler(c0=15.0, half_life_tokens=50, mode="exponential"),
        "Bateman (Oral)":        PKScheduler(c0=15.0, half_life_tokens=50, mode="bateman"),
        "Oscillating (Slow-rel)":PKScheduler(c0=15.0, half_life_tokens=50, mode="oscillating"),
        "Constant":              PKScheduler(c0=15.0, half_life_tokens=50, mode="constant"),
    }

    plt.figure(figsize=(10, 5))
    ts = list(range(n_tokens))
    for name, pk in profiles.items():
        plt.plot(ts, pk.get_schedule(n_tokens), label=name)
    plt.xlabel("Token position in generation")
    plt.ylabel("Steering coefficient (dose)")
    plt.title("AI Drug Pharmacokinetic Profiles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()
