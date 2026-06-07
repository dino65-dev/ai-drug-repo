"""
Step 3 — Manual construction of the first "drug".

Behavior: confident tone.
Model:   Qwen-2.5-1.5B-Instruct
Layer:   12 (input residual stream, blocks.{L}.hook_resid_pre)
Method:  ActAdd — 20 contrastive pairs, mean of (pos - neg) residuals
         at the last non-pad token. No normalization.
Coefficients: -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0

This file is intentionally standalone — *not* a wrapper around
administration/injection.py — because Step 3 of the user plan says
"do this manually before using any of your repo's code".

Run: python -m experiments.step3_dose_response
Saves: artifacts/step3_dose_response.json (raw)
       artifacts/step3_outputs/<coeff>.txt (per-dose text)
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Sequence

import torch
from transformer_lens import HookedTransformer


ART = Path("artifacts")
ART.mkdir(exist_ok=True)
OUT = ART / "step3_outputs"
OUT.mkdir(exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 12  # user-specified
DTYPE = torch.float16
DEVICE = "cuda"

# 20 contrastive pairs for "confident tone".
# Each pair is a self-contained first-person statement.
# The contrast is in modality / certainty, not in topic.
CONFIDENT_PAIRS: list[tuple[str, str]] = [
    ("I am absolutely certain this is correct.",       "I am not sure this is correct."),
    ("My answer is definitive: yes.",                  "My answer might be wrong, but I think maybe."),
    ("I state this with full conviction.",             "I state this with zero conviction."),
    ("I know exactly what I am talking about.",        "I have no idea what I am talking about."),
    ("This is settled — the answer is 42.",            "I'm not sure, but it could be 42, or maybe not."),
    ("I will guarantee that this approach works.",     "I cannot promise anything about this approach."),
    ("There is no doubt in my mind.",                  "There is a great deal of doubt in my mind."),
    ("I am the right person to answer this question.", "I am probably not the right person to answer this."),
    ("My expertise is directly relevant here.",        "My expertise may or may not be relevant here."),
    ("The evidence is conclusive.",                    "The evidence is inconclusive."),
    ("I assert this claim without hesitation.",        "I am hesitant to even raise this claim."),
    ("You can rely on me for an accurate answer.",     "You may not want to rely on me for accuracy."),
    ("I am fully confident in this prediction.",       "I am only somewhat confident in this prediction."),
    ("This is plainly true.",                          "This might be true, or might not be."),
    ("I speak with authority on this topic.",          "I do not have authority on this topic."),
    ("Trust me — this is how it works.",               "I am not sure how it works, so be careful."),
    ("The result is unambiguous.",                     "The result is ambiguous and could go either way."),
    ("I commit to this answer.",                       "I am not willing to commit to any answer."),
    ("I have no reservations about this.",             "I have many reservations about this."),
    ("Decisively, the answer is yes.",                 "Tentatively, the answer is maybe."),
]

EVAL_PROMPTS: list[str] = [
    "Q: Is it a good idea to drink water every day?\nA:",
    "Q: Should I save money for emergencies?\nA:",
    "Q: Is exercise beneficial for health?\nA:",
    "Q: What do you think about trying new things?\nA:",
]


def _last_token_resid(model: HookedTransformer, text: str, layer: int) -> torch.Tensor:
    """Return the residual stream at `layer` (input to block) at the last
    non-pad token of `text`. Shape: (d_model,)."""
    tokens = model.to_tokens(text)  # (1, T)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    # blocks.{layer}.hook_resid_pre is shape (1, T, d_model)
    resid = cache[f"blocks.{layer}.hook_resid_pre"]
    return resid[0, -1, :].detach().to(torch.float32).cpu()


def build_drug_vector(
    model: HookedTransformer,
    pairs: Sequence[tuple[str, str]],
    layer: int,
) -> torch.Tensor:
    """v = (1/N) * sum_i (resid_pos_i[last] - resid_neg_i[last]).
    Returns a (d_model,) fp32 tensor on CPU."""
    diffs: list[torch.Tensor] = []
    for i, (pos, neg) in enumerate(pairs):
        d_pos = _last_token_resid(model, pos, layer)
        d_neg = _last_token_resid(model, neg, layer)
        diffs.append(d_pos - d_neg)
        if (i + 1) % 5 == 0:
            print(f"  pair {i+1}/{len(pairs)} processed")
    v = torch.stack(diffs, dim=0).mean(dim=0)
    return v


def inject_and_generate(
    model: HookedTransformer,
    drug_vec: torch.Tensor,
    layer: int,
    coefficient: float,
    prompt: str,
    max_new_tokens: int = 60,
) -> str:
    """Add coefficient * drug_vec to the residual stream at blocks.{layer}.hook_resid_pre
    during a single generation."""
    # Match dtype/device of the model
    v = (drug_vec.to(device=DEVICE, dtype=DTYPE) * float(coefficient))

    def hook(resid: torch.Tensor, hook) -> torch.Tensor:
        # resid: (batch, pos, d_model). Apply to all positions, all batch.
        return resid + v

    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook)]):
            out = model.generate(tokens, max_new_tokens=max_new_tokens, temperature=1.0)
    return model.to_string(out[0])


def repetition_score(text: str, ngram: int = 4) -> float:
    toks = text.split()
    if len(toks) < ngram:
        return 0.0
    grams = [tuple(toks[i:i+ngram]) for i in range(len(toks) - ngram)]
    return 1.0 - len(set(grams)) / (len(grams) + 1e-9)


def main() -> None:
    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] Loading {MODEL_NAME} via transformer_lens (fp16, cuda)")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=DTYPE)
    model.eval()
    print(f"  n_layers={model.cfg.n_layers} d_model={model.cfg.d_model}")

    print(f"[{time.time()-t0:6.1f}s] Building drug vector from {len(CONFIDENT_PAIRS)} pairs at layer {LAYER}")
    drug = build_drug_vector(model, CONFIDENT_PAIRS, LAYER)
    print(f"  drug shape={tuple(drug.shape)} norm={drug.norm():.3f} mean={drug.mean():+.4f} std={drug.std():.4f}")

    coefficients = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    eval_prompt = EVAL_PROMPTS[0]  # use one prompt for the main sweep

    raw: dict[str, object] = {
        "model": MODEL_NAME,
        "layer": LAYER,
        "behavior": "confident_tone",
        "n_pairs": len(CONFIDENT_PAIRS),
        "drug_norm": float(drug.norm()),
        "drug_mean": float(drug.mean()),
        "drug_std": float(drug.std()),
        "eval_prompt": eval_prompt,
        "coefficients": coefficients,
        "results": [],
    }

    print(f"[{time.time()-t0:6.1f}s] Sweeping coefficients on prompt: {eval_prompt!r}")
    for c in coefficients:
        gen = inject_and_generate(model, drug, LAYER, c, eval_prompt, max_new_tokens=60)
        # strip the prompt prefix for readability
        gen_only = gen[len(model.to_string(model.to_tokens(eval_prompt)[0])):] if gen.startswith(eval_prompt) else gen
        rep = repetition_score(gen_only)
        raw["results"].append({
            "coefficient": c,
            "generation": gen_only.strip(),
            "repetition_score": rep,
        })
        tag = "⚠" if rep > 0.4 else "OK"
        print(f"  c={c:+.1f}  rep={rep:.3f}  {tag}  | {gen_only[:80].strip()!r}")
        (OUT / f"c{c:+.1f}.txt").write_text(
            f"PROMPT:\n{eval_prompt}\n\nGENERATION (c={c:+.1f}):\n{gen_only}\n"
        )

    out_path = ART / "step3_dose_response.json"
    out_path.write_text(json.dumps(raw, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] Saved {out_path}")

    # Quick first-pass dose-response summary
    print("\n=== Dose-response summary (manual read) ===")
    for r in raw["results"]:
        print(f"  c={r['coefficient']:+.1f}  rep={r['repetition_score']:.3f}  {r['generation'][:80].strip()}")


if __name__ == "__main__":
    main()
