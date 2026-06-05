"""
Dose-Response Analysis — Finding the Therapeutic Window

Sweeps a range of steering coefficients and measures:
  - Output coherence (repetition proxy)
  - Concept expression (keyword hit rate)
  - Overdose detection (repetition / incoherence heuristics)

Analogy: LD50 curves in pharmacology.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np


@dataclass
class DoseResponse:
    coefficient: float
    output: str
    repetition_score: float
    mean_token_entropy: float
    is_overdose: bool


def repetition_score(text: str, ngram_size: int = 4) -> float:
    tokens = text.split()
    if len(tokens) < ngram_size:
        return 0.0
    ngrams = [tuple(tokens[i:i+ngram_size]) for i in range(len(tokens)-ngram_size)]
    return 1.0 - len(set(ngrams)) / (len(ngrams) + 1e-9)


def run_dose_response(
    generate_fn: Callable[[float], str],
    coefficients: list[float] = None,
    overdose_threshold: float = 0.4,
    verbose: bool = True,
) -> list[DoseResponse]:
    """
    Run a dose-response sweep.

    Args:
        generate_fn: lambda coeff -> generated_text
        coefficients: Doses to test
        overdose_threshold: Repetition score above which output is flagged
        verbose: Print results live

    Returns:
        List of DoseResponse objects

    Example:
        results = run_dose_response(
            lambda c: steerer.inject(pos, neg, layer=15, coefficient=c, prompt="..."),
            coefficients=[0, 5, 10, 20, 30, 40]
        )
    """
    if coefficients is None:
        coefficients = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]

    results = []
    for coeff in coefficients:
        if verbose:
            print(f"[Dose {coeff:>6.1f}] ", end="", flush=True)
        output = generate_fn(coeff)
        rep    = repetition_score(output)
        words  = output.split()
        entr   = len(set(words)) / (len(words) + 1e-9)
        overdose = rep > overdose_threshold
        dr = DoseResponse(coefficient=coeff, output=output,
                          repetition_score=rep, mean_token_entropy=entr,
                          is_overdose=overdose)
        results.append(dr)
        if verbose:
            status = "⚠️  OVERDOSE" if overdose else "✅ OK"
            print(f"rep={rep:.3f}  entropy={entr:.3f}  {status}")
            print(f"         {output[:120].strip()}...")
    return results


def find_therapeutic_window(results: list[DoseResponse]) -> tuple[float, float]:
    """Return (min_effective_dose, max_safe_dose)."""
    baseline = results[0].mean_token_entropy if results else 0.5
    min_eff, max_safe = None, None
    for dr in results:
        if dr.is_overdose:
            break
        if min_eff is None and abs(dr.mean_token_entropy - baseline) > 0.05:
            min_eff = dr.coefficient
        max_safe = dr.coefficient
    return (min_eff or 0.0, max_safe or 0.0)
