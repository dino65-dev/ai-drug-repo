"""
Reliability & Out-of-Distribution Evaluation Suite

NeurIPS 2024 evaluated steering vectors on 100+ datasets and found:
  - Unreliable even in-distribution when concept has no coherent direction
  - Can misgeneralize OOD
  - Can produce opposite effects at high doses

This module runs a 4-test reliability battery before you trust any compound.

Reference:
  NeurIPS 2024. Analysing the Generalisation and Reliability of Steering Vectors.
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/fb3ad59a84799bfb8d700e56d19c231b
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ReliabilityReport:
    concept: str
    model: str
    layer: int
    coefficient: float
    in_distribution_score: float = 0.0
    ood_score: float = 0.0
    off_target_score: float = 0.0
    dose_linearity_score: float = 0.0
    dose_curve: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        return float(np.mean([self.in_distribution_score, self.ood_score,
                              self.off_target_score, self.dose_linearity_score]))

    def summary(self) -> str:
        grade = "✅ RELIABLE" if self.overall_score > 0.7 else                 "⚠️  MARGINAL" if self.overall_score > 0.4 else                 "❌ UNRELIABLE"
        lines = [
            f"\n{'='*55}",
            f"📊 RELIABILITY REPORT: {self.concept}",
            f"{'='*55}",
            f"  Model: {self.model} | Layer: {self.layer} | Coeff: {self.coefficient}",
            f"  In-distribution:  {self.in_distribution_score:.2f}",
            f"  OOD robustness:   {self.ood_score:.2f}",
            f"  Off-target clean: {self.off_target_score:.2f}",
            f"  Dose linearity:   {self.dose_linearity_score:.2f}",
            f"  {'─'*33}",
            f"  Overall: {self.overall_score:.2f} → {grade}",
        ]
        if self.warnings:
            lines.append("  Warnings:")
            lines += [f"    {w}" for w in self.warnings]
        lines.append(f"{'='*55}\n")
        return "\n".join(lines)


def _hit_rate(outputs: list[str], keywords: list[str]) -> float:
    hits = sum(any(kw.lower() in o.lower() for kw in keywords) for o in outputs)
    return hits / (len(outputs) + 1e-9)


def evaluate_vector(
    generate_fn: Callable[[str, float], str],
    concept: str,
    target_keywords: list[str],
    in_dist_prompts: list[str],
    ood_prompts: list[str],
    off_target_prompts: list[str],
    coefficient: float = 1.5,
    dose_range: list[float] = None,
    model_name: str = "unknown",
    layer: int = -1,
) -> ReliabilityReport:
    """
    Full 4-test reliability evaluation.

    Args:
        generate_fn: (prompt, coefficient) -> generated_text
        concept: Name of concept being steered
        target_keywords: Words that should appear MORE with steering
        in_dist_prompts: Same distribution as training
        ood_prompts: Different distribution
        off_target_prompts: Should NOT be affected by steering
        coefficient: Dose to evaluate
        dose_range: Coefficients for linearity test
    """
    if dose_range is None:
        dose_range = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    report = ReliabilityReport(concept=concept, model=model_name,
                               layer=layer, coefficient=coefficient)
    print(f"Evaluating reliability of '{concept}'...")

    print("  [1/4] In-distribution...")
    in_dist_out = [generate_fn(p, coefficient) for p in in_dist_prompts]
    report.in_distribution_score = _hit_rate(in_dist_out, target_keywords)

    print("  [2/4] OOD robustness...")
    ood_out = [generate_fn(p, coefficient) for p in ood_prompts]
    report.ood_score = _hit_rate(ood_out, target_keywords)

    print("  [3/4] Off-target effects...")
    ot_steered  = [generate_fn(p, coefficient) for p in off_target_prompts]
    ot_baseline = [generate_fn(p, 0.0) for p in off_target_prompts]
    changes = sum(
        any(kw.lower() in s.lower() for kw in target_keywords) and
        not any(kw.lower() in b.lower() for kw in target_keywords)
        for s, b in zip(ot_steered, ot_baseline)
    )
    report.off_target_score = 1.0 - changes / max(len(off_target_prompts), 1)

    print("  [4/4] Dose linearity...")
    probe_prompt = in_dist_prompts[0] if in_dist_prompts else "Tell me about yourself."
    dose_hits = []
    for dose in dose_range:
        out = generate_fn(probe_prompt, dose)
        report.dose_curve[dose] = out[:100]
        dose_hits.append(any(kw.lower() in out.lower() for kw in target_keywords))
    violations = sum(1 for i in range(1, len(dose_hits)) if dose_hits[i] < dose_hits[i-1])
    report.dose_linearity_score = 1.0 - violations / max(len(dose_range)-1, 1)

    if report.in_distribution_score < 0.5:
        report.warnings.append("Low in-distribution effectiveness. Concept may lack a coherent direction.")
    if report.ood_score < 0.3:
        report.warnings.append("Poor OOD generalization. Do not deploy outside training distribution.")
    if report.off_target_score < 0.7:
        report.warnings.append("Significant off-target effects. Narrow the injection layer.")
    if report.dose_linearity_score < 0.5:
        report.warnings.append("Non-monotone dose response. Check for overdose at intermediate doses.")

    print(report.summary())
    return report
