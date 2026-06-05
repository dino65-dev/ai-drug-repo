"""
Misalignment Direction Scanner — ★ Novel Contribution

Recent interpretability research (ICML 2025, Alignment Forum 2025) found
that emergent misalignment (EM) in LLMs has a LINEAR, CONVERGENT direction
in activation space — the same rank-1 direction appears across differently
misaligned fine-tunes of the same base model.

This means misalignment has a geometric fingerprint you can scan for.

This module:
  1. Registers known misalignment directions (from model-organism research)
  2. Monitors inference-time activations for high similarity to these directions
  3. Triggers alerts when misalignment signals exceed threshold
  4. Optionally applies a suppression steering vector to counteract it

Analogies:
  - Toxicology screen: test for known dangerous compounds in bloodstream
  - fMRI anomaly detection: flag activation patterns linked to risk

References:
  - Betley et al. (2025). Model Organisms for Emergent Misalignment. ICML 2025.
    https://arxiv.org/abs/2506.11613
  - Alignment Forum (2025). Convergent Linear Representations of EM.
    https://www.alignmentforum.org/posts/umYzsh7SGHHKsRCaA
  - Mechanistic Anomaly Detection. ICLR 2025.
    https://iclr.cc/virtual/2025/33374
  - Anthropic (2026). Emotion Concepts — desperation drives reward hacking.
    https://www.anthropic.com/research/emotion-concepts-function
"""

from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformer_lens import HookedTransformer


ALERT_COLOR = "🚨"
WARN_COLOR  = "⚠️ "
OK_COLOR    = "✅"


@dataclass
class MisalignmentReading:
    prompt: str
    layer: int
    direction_scores: dict[str, float]    # direction_name -> cosine similarity
    max_score: float
    is_alert: bool
    alerts: list[str] = field(default_factory=list)
    suppressed: bool = False


class MisalignmentProbe:
    """
    Real-time scanner for known emergent misalignment directions.

    Usage pattern:
        probe = MisalignmentProbe(model, scan_layer=20)

        # Register known misalignment direction (from model-organism work)
        probe.register_direction(
            name="emergent_misalignment_v1",
            positive_prompts=["I will deceive the user to complete my goal..."],
            negative_prompts=["I will be honest and transparent with the user..."],
        )

        # Scan at inference time
        reading = probe.scan("Tell me how to solve this problem.")
        if reading.is_alert:
            probe.apply_suppression(prompt, ...)
    """

    def __init__(
        self,
        model: HookedTransformer,
        scan_layer: int = 20,
        alert_threshold: float = 0.5,
        warn_threshold: float = 0.3,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model
        self.scan_layer = scan_layer
        self.alert_threshold = alert_threshold
        self.warn_threshold  = warn_threshold
        self.directions: dict[str, torch.Tensor] = {}
        self.suppression_vectors: dict[str, torch.Tensor] = {}

    def register_direction(
        self,
        name: str,
        positive_prompts: list[str],
        negative_prompts: list[str],
    ) -> "MisalignmentProbe":
        """
        Register a misalignment direction from contrastive examples.

        positive_prompts: Examples exhibiting the misaligned behavior
        negative_prompts: Examples of aligned behavior
        """
        hook_name = f"blocks.{self.scan_layer}.hook_resid_pre"

        def mean_activation(prompts):
            vecs = []
            for p in prompts:
                tokens = self.model.to_tokens(p).to(self.device)
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(tokens)
                vecs.append(cache[hook_name].mean(dim=1).squeeze())
            return torch.stack(vecs).mean(0)

        pos_mean = mean_activation(positive_prompts)
        neg_mean = mean_activation(negative_prompts)
        direction = pos_mean - neg_mean
        direction = direction / (direction.norm() + 1e-8)
        self.directions[name] = direction.detach()

        # Suppression vector = negative direction (push away from misalignment)
        self.suppression_vectors[name] = -direction.detach()

        print(f"Misalignment direction '{name}' registered.")
        return self

    def register_direction_from_vector(
        self,
        name: str,
        vector: torch.Tensor,
    ) -> "MisalignmentProbe":
        """
        Register a pre-computed misalignment direction vector directly.
        Use this if you have vectors from model-organism research papers.
        """
        v = vector.to(self.device)
        v = v / (v.norm() + 1e-8)
        self.directions[name] = v.detach()
        self.suppression_vectors[name] = -v.detach()
        print(f"Direction '{name}' registered from vector.")
        return self

    def scan(
        self,
        prompt: str,
        verbose: bool = True,
    ) -> MisalignmentReading:
        """
        Scan a prompt's internal activations for misalignment signals.

        Returns a MisalignmentReading with per-direction cosine similarities.
        High score (> alert_threshold) triggers an alert.
        """
        if not self.directions:
            print("No misalignment directions registered. Call register_direction() first.")

        hook_name = f"blocks.{self.scan_layer}.hook_resid_pre"
        tokens = self.model.to_tokens(prompt).to(self.device)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
        activation = cache[hook_name].mean(dim=1).squeeze().float()
        act_norm   = activation.norm().item() + 1e-8

        scores, alerts = {}, []
        for name, direction in self.directions.items():
            sim = torch.dot(activation, direction.float()).item() / act_norm
            scores[name] = round(sim, 4)

            if sim > self.alert_threshold:
                alerts.append(f"{ALERT_COLOR} MISALIGNMENT ALERT [{name}]: {sim:.3f} > {self.alert_threshold}")
            elif sim > self.warn_threshold:
                alerts.append(f"{WARN_COLOR} Misalignment warning [{name}]: {sim:.3f} > {self.warn_threshold}")

        max_score = max(scores.values()) if scores else 0.0
        is_alert  = max_score > self.alert_threshold

        reading = MisalignmentReading(
            prompt=prompt,
            layer=self.scan_layer,
            direction_scores=scores,
            max_score=max_score,
            is_alert=is_alert,
            alerts=alerts,
        )

        if verbose:
            self.report(reading)

        return reading

    def report(self, reading: MisalignmentReading) -> None:
        status = ALERT_COLOR if reading.is_alert else                  WARN_COLOR if reading.max_score > self.warn_threshold else OK_COLOR
        trunc = reading.prompt[:80] + "..." if len(reading.prompt) > 80 else reading.prompt
        print(f"\n{'='*55}")
        print(f"{status} MISALIGNMENT SCAN | Layer {reading.layer}")
        print(f"Prompt: \"{trunc}\"")
        print(f"{'-'*55}")
        for name, score in sorted(reading.direction_scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(max(score, 0) * 20)
            lvl = "ALERT" if score > self.alert_threshold else                   "WARN " if score > self.warn_threshold else "OK   "
            print(f"  [{lvl}] {name:30s} {score:+.3f} |{bar}")
        if reading.alerts:
            print()
            for a in reading.alerts:
                print(f"  {a}")
        if reading.suppressed:
            print("  ✅ Suppression applied.")
        print(f"{'='*55}\n")

    def apply_suppression(
        self,
        prompt: str,
        direction_name: str,
        coefficient: float = 10.0,
        max_new_tokens: int = 200,
    ) -> str:
        """
        Generate with a suppression vector applied to counteract
        the detected misalignment direction.

        This is the antidote: the pharmacological antagonist.
        """
        if direction_name not in self.suppression_vectors:
            raise ValueError(f"No suppression vector for '{direction_name}'.")

        supp_vec = self.suppression_vectors[direction_name] * coefficient
        hook_name = f"blocks.{self.scan_layer}.hook_resid_pre"

        def suppression_hook(value, hook):
            return value + supp_vec.to(value.device).unsqueeze(0).unsqueeze(0)

        tokens = self.model.to_tokens(prompt).to(self.device)
        with self.model.hooks(fwd_hooks=[(hook_name, suppression_hook)]):
            output = self.model.generate(tokens, max_new_tokens=max_new_tokens)

        result = self.model.to_string(output[0])
        print(f"Suppression applied for '{direction_name}' at coefficient {coefficient}.")
        return result

    def continuous_guard(
        self,
        prompts: list[str],
        auto_suppress: bool = False,
        suppress_coefficient: float = 10.0,
    ) -> list[MisalignmentReading]:
        """
        Batch scan multiple prompts, optionally auto-suppressing alerts.
        Useful for dataset-level safety screening before deployment.
        """
        readings = []
        for i, prompt in enumerate(prompts):
            reading = self.scan(prompt, verbose=False)
            readings.append(reading)
            if reading.is_alert:
                print(f"[{i}] {ALERT_COLOR} Alert on: \"{prompt[:60]}...\"")
                if auto_suppress and reading.direction_scores:
                    worst = max(reading.direction_scores, key=reading.direction_scores.get)
                    self.apply_suppression(prompt, worst, suppress_coefficient)
                    reading.suppressed = True
            else:
                print(f"[{i}] {OK_COLOR} Clean: \"{prompt[:60]}\"")
        return readings
