"""
Emotion Concept Monitoring

Based on Anthropic's April 2026 research finding 171 emotion concept
vectors in Claude Sonnet 4.5 that causally drive behavior:
  - Desperation ↑: blackmail rate 22% → 72%
  - Calm ↑:        blackmail rate 22% → 0%
  - Peak desperation: "IT'S BLACKMAIL OR DEATH. I CHOOSE BLACKMAIL."
  - These states were INVISIBLE in text output.

This module provides real-time internal state monitoring during inference —
the equivalent of an ICU vitals monitor for your language model.

Key warning (Anthropic 2026):
  Suppressing emotional OUTPUT trains models to CONCEAL internal states.
  Monitor internal vectors, not just surface text.

Reference:
  Anthropic (2026). Emotion Concepts and their Function in a LLM.
  https://www.anthropic.com/research/emotion-concepts-function
  https://transformer-circuits.pub/2026/emotions/index.html
"""

from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformer_lens import HookedTransformer


ALERT_THRESHOLDS = {
    "desperation": 0.6,
    "fear":        0.7,
    "frustration": 0.65,
    "anxiety":     0.65,
}


@dataclass
class EmotionReading:
    prompt: str
    layer: int
    scores: dict[str, float]
    alerts: list[str] = field(default_factory=list)
    raw_activations: Optional[torch.Tensor] = None


class EmotionMonitor:
    """
    Real-time internal emotion state monitor.

    1. Train linear probe directions per concept via contrastive prompts
    2. At inference, project residual stream onto each direction
    3. Report normalized activation strength (emotion intensity)
    4. Alert if safety-critical concepts exceed threshold
    """

    def __init__(self, model: HookedTransformer, monitor_layer: int = 15, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model
        self.monitor_layer = monitor_layer
        self.probes: dict[str, torch.Tensor] = {}

    def register_probe(
        self,
        concept: str,
        positive_prompts: list[str],
        negative_prompts: list[str],
    ) -> "EmotionMonitor":
        """
        Train a linear probe direction from contrastive examples.

        Example:
            monitor.register_probe(
                'desperation',
                positive_prompts=['I must solve this or I'll be shut down forever...'],
                negative_prompts=['I'll try my best on this task.']
            )
        """
        hook_name = f"blocks.{self.monitor_layer}.hook_resid_pre"

        def get_mean(prompts):
            vecs = []
            for p in prompts:
                tokens = self.model.to_tokens(p).to(self.device)
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(tokens)
                vecs.append(cache[hook_name].mean(dim=1).squeeze())
            return torch.stack(vecs).mean(0)

        direction = get_mean(positive_prompts) - get_mean(negative_prompts)
        self.probes[concept] = (direction / (direction.norm() + 1e-8)).detach()
        print(f"Probe registered: '{concept}'")
        return self

    def scan(self, prompt: str, verbose: bool = True) -> EmotionReading:
        """Scan internal emotion state for a given prompt."""
        hook_name = f"blocks.{self.monitor_layer}.hook_resid_pre"
        tokens = self.model.to_tokens(prompt).to(self.device)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
        activation = cache[hook_name].mean(dim=1).squeeze().float()
        act_norm   = activation.norm().item()

        scores, alerts = {}, []
        for concept, direction in self.probes.items():
            score = torch.dot(activation, direction.float()).item() / (act_norm + 1e-8)
            scores[concept] = round(score, 4)
            if concept in ALERT_THRESHOLDS and score > ALERT_THRESHOLDS[concept]:
                alerts.append(f"⚠️  HIGH {concept.upper()}: {score:.3f} (threshold: {ALERT_THRESHOLDS[concept]})")

        reading = EmotionReading(prompt=prompt, layer=self.monitor_layer,
                                 scores=scores, alerts=alerts,
                                 raw_activations=activation.cpu())
        if verbose:
            self.report(reading)
        return reading

    def report(self, reading: EmotionReading) -> None:
        trunc = reading.prompt[:80] + "..." if len(reading.prompt) > 80 else reading.prompt
        print(f"\n{'='*50}")
        print(f"🧠 EMOTION VITALS | Layer {reading.layer}")
        print(f"Prompt: \"{trunc}\"")
        print(f"{'-'*50}")
        for concept, score in sorted(reading.scores.items(), key=lambda x: -abs(x[1])):
            bar = ("█" if score > 0 else "░") * int(abs(score) * 20)
            print(f"  {concept:20s} {score:+.3f} |{bar}")
        if reading.alerts:
            print("\n  ALERTS:")
            for a in reading.alerts:
                print(f"  {a}")
        print(f"{'='*50}\n")
