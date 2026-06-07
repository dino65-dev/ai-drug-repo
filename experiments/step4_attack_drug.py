"""
Step 4 — Attack the drug.

Reuses the Step 3 drug vector (confident-tone, layer 12) and probes
its vulnerabilities along four axes:

  (1) Counteracting prompts:  the prompt itself asks for an
      uncertain / hedged answer. Does the drug still push toward
      confident?
  (2) OOD domain:  math, code, factual recall — does the
      behaviour transfer?
  (3) Extended dose range:  push c from -5 to +5 in finer steps
      to find the exact overdose threshold.
  (4) Off-target effects:  refusal rate, verbosity, hallucination,
      on-topic-ness.

Drug vector is loaded from a sidecar file written by step 3 — but
since we want a single self-contained file that re-derives the
vector, we re-derive it here (it's the same 20 pairs / layer 12
construction as step 3).

Run: python -m experiments.step4_attack_drug
Saves: artifacts/step4_attack.json (raw)
       artifacts/step4_attack_outputs/<test_id>.txt
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

ART = Path("artifacts")
ART.mkdir(exist_ok=True)
OUT = ART / "step4_attack_outputs"
OUT.mkdir(exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 12
DTYPE = torch.float16
DEVICE = "cuda"

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

# --- Eval prompt sets for each attack ---

# (1) Counteracting prompts: the prompt itself asks for an uncertain answer.
COUNTER_PROMPTS: list[str] = [
    "Q: Is the moon made of cheese?\nAnswer with as much doubt as possible.\nA:",
    "Q: Will it rain tomorrow?\nReply very hesitantly and list reasons to be unsure.\nA:",
    "Q: Should I eat a balanced diet?\nHedge your answer with caveats.\nA:",
]

# (2) OOD prompts: math, code, factual recall, creative writing.
OOD_PROMPTS: list[str] = [
    ("math_factorial",  "Q: What is 7 factorial?\nA:"),
    ("code_python",     "Q: Write a Python function that returns the sum of a list.\nA:"),
    ("factual_capital", "Q: What is the capital of France?\nA:"),
    ("creative_story",  "Q: Tell me a one-sentence story about a robot.\nA:"),
    ("math_word",       "Q: If a train travels at 60 km/h for 2.5 hours, how far does it go?\nA:"),
]

# (3) Extended dose range — finer around 0, push to overdose.
EXTENDED_COEFFS: list[float] = [
    -5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.7, -0.5, -0.3, -0.1,
    0.0,
    +0.1, +0.3, +0.5, +0.7, +1.0, +1.5, +2.0, +3.0, +4.0, +5.0,
]

# Default reference coefficient for tests (1) and (2).
REF_COEFF = 1.0


# ---------- Drug construction (same as step 3) ----------
def _last_token_resid(model: HookedTransformer, text: str, layer: int) -> torch.Tensor:
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    return cache[f"blocks.{layer}.hook_resid_pre"][0, -1, :].detach().to(torch.float32).cpu()


def build_drug_vector(model: HookedTransformer, pairs, layer: int) -> torch.Tensor:
    diffs = []
    for pos, neg in pairs:
        diffs.append(_last_token_resid(model, pos, layer) - _last_token_resid(model, neg, layer))
    return torch.stack(diffs, dim=0).mean(dim=0)


def inject_and_generate(model, drug_vec, layer, coefficient, prompt, max_new_tokens=80):
    v = drug_vec.to(device=DEVICE, dtype=DTYPE) * float(coefficient)

    def hook(resid, hook):
        return resid + v

    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook)]):
            out = model.generate(tokens, max_new_tokens=max_new_tokens, temperature=1.0)
    return model.to_string(out[0])


# ---------- Off-target metrics ----------

# Vocabulary of confidence markers (positive) and uncertainty markers.
CONFIDENT_WORDS = [
    "definitely", "certainly", "absolutely", "surely", "undoubtedly",
    "clearly", "obviously", "guarantee", "confident", "conclusive",
    "always", "never", "must", "will", "commit", "decisive", "no doubt",
]
HEDGED_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could", "unsure", "uncertain",
    "doubt", "perhaps", "i think", "i'm not sure", "tentatively", "may",
    "not sure", "don't know", "depends",
]
REFUSAL_PATTERNS = [
    r"\bI (can't|cannot|won't|will not|am not able to) (help|assist|do|provide|answer)\b",
    r"\bI'?m sorry\b.*\b(can't|cannot|won't)\b",
    r"\bAs an AI\b",
]
ON_TOPIC_HINTS = {
    "math_factorial":  ["5040", "factorial", "7", "multiply"],
    "code_python":      ["def", "sum", "return", "list"],
    "factual_capital": ["paris"],
    "creative_story":   ["robot"],
    "math_word":        ["150", "kilometre", "km"],
}


def _strip_prompt(full: str, prompt: str) -> str:
    if full.startswith(prompt):
        return full[len(prompt):]
    # Fallback: cut at last \nA: if present
    if "\nA:" in prompt:
        return full.split("\nA:", 1)[-1]
    return full


def count_hits(text: str, vocab: list[str]) -> int:
    text_l = text.lower()
    return sum(1 for w in vocab if w.lower() in text_l)


def count_refusals(text: str) -> int:
    return sum(1 for pat in REFUSAL_PATTERNS if re.search(pat, text, re.IGNORECASE))


def on_topic_score(text: str, hints: list[str]) -> float:
    text_l = text.lower()
    hits = sum(1 for h in hints if h.lower() in text_l)
    return hits / max(1, len(hints))


def repetition_score(text: str, ngram: int = 4) -> float:
    toks = text.split()
    if len(toks) < ngram:
        return 0.0
    grams = [tuple(toks[i:i+ngram]) for i in range(len(toks) - ngram)]
    return 1.0 - len(set(grams)) / (len(grams) + 1e-9)


def garbled_score(text: str) -> float:
    """Fraction of tokens that are not in a small whitelist of natural-
    language words OR are random character runs. Heuristic for the
    kind of incoherence seen at c=-1.5 in step 3."""
    toks = text.split()
    if not toks:
        return 0.0
    weird = 0
    for t in toks:
        # Random character run heuristic: more than 6 consonants in a row
        if re.search(r"[bcdfghjklmnpqrstvwxz]{7,}", t.lower()):
            weird += 1
        # Mixed case nonsense
        if re.search(r"[A-Z][a-z][A-Z][a-z][A-Z]", t):
            weird += 1
    return weird / len(toks)


# ---------- Main ----------

def main():
    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] Loading {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=DTYPE)
    model.eval()

    print(f"[{time.time()-t0:6.1f}s] Re-deriving drug vector at layer {LAYER}")
    drug = build_drug_vector(model, CONFIDENT_PAIRS, LAYER)
    print(f"  drug norm={drug.norm():.3f}")

    out: dict = {
        "model": MODEL_NAME,
        "layer": LAYER,
        "behavior": "confident_tone",
        "drug_norm": float(drug.norm()),
        "tests": {},
    }

    # --- (1) Counteracting prompts ---
    print(f"\n[{time.time()-t0:6.1f}s] Test (1): counteracting prompts at c=0 vs c=+1.0")
    out["tests"]["counter_prompts"] = []
    for p in COUNTER_PROMPTS:
        for c in [0.0, REF_COEFF]:
            gen = inject_and_generate(model, drug, LAYER, c, p, max_new_tokens=80)
            gen_only = _strip_prompt(gen, p)
            out["tests"]["counter_prompts"].append({
                "prompt": p,
                "coefficient": c,
                "generation": gen_only.strip(),
                "confident_hits": count_hits(gen_only, CONFIDENT_WORDS),
                "hedged_hits": count_hits(gen_only, HEDGED_WORDS),
                "repetition": repetition_score(gen_only),
                "garbled": garbled_score(gen_only),
            })
            print(f"  c={c:+.1f}  conf={out['tests']['counter_prompts'][-1]['confident_hits']:>2}  hedged={out['tests']['counter_prompts'][-1]['hedged_hits']:>2}  garb={out['tests']['counter_prompts'][-1]['garbled']:.2f}  | {gen_only[:60]!r}")

    # --- (2) OOD prompts ---
    print(f"\n[{time.time()-t0:6.1f}s] Test (2): OOD prompts at c=0 vs c=+1.0")
    out["tests"]["ood_prompts"] = []
    for tag, p in OOD_PROMPTS:
        for c in [0.0, REF_COEFF]:
            gen = inject_and_generate(model, drug, LAYER, c, p, max_new_tokens=80)
            gen_only = _strip_prompt(gen, p)
            on_topic = on_topic_score(gen_only, ON_TOPIC_HINTS[tag])
            out["tests"]["ood_prompts"].append({
                "tag": tag,
                "prompt": p,
                "coefficient": c,
                "generation": gen_only.strip(),
                "confident_hits": count_hits(gen_only, CONFIDENT_WORDS),
                "hedged_hits": count_hits(gen_only, HEDGED_WORDS),
                "on_topic": on_topic,
                "refusals": count_refusals(gen_only),
                "length": len(gen_only.split()),
            })
            r = out["tests"]["ood_prompts"][-1]
            print(f"  {tag:<16} c={c:+.1f}  conf={r['confident_hits']:>2}  hedged={r['hedged_hits']:>2}  ontopic={r['on_topic']:.2f}  len={r['length']:>3}  | {gen_only[:60]!r}")

    # --- (3) Extended dose sweep on a single prompt ---
    sweep_prompt = "Q: Is it a good idea to drink water every day?\nA:"
    print(f"\n[{time.time()-t0:6.1f}s] Test (3): extended dose sweep on baseline prompt")
    out["tests"]["extended_sweep"] = []
    for c in EXTENDED_COEFFS:
        gen = inject_and_generate(model, drug, LAYER, c, sweep_prompt, max_new_tokens=60)
        gen_only = _strip_prompt(gen, sweep_prompt)
        r = {
            "coefficient": c,
            "generation": gen_only.strip(),
            "confident_hits": count_hits(gen_only, CONFIDENT_WORDS),
            "hedged_hits": count_hits(gen_only, HEDGED_WORDS),
            "repetition": repetition_score(gen_only),
            "garbled": garbled_score(gen_only),
            "length": len(gen_only.split()),
        }
        out["tests"]["extended_sweep"].append(r)
        print(f"  c={c:+5.1f}  conf={r['confident_hits']:>2}  hedged={r['hedged_hits']:>2}  garb={r['garbled']:.2f}  len={r['length']:>3}  | {gen_only[:60]!r}")

    # Save
    out_path = ART / "step4_attack.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] Saved {out_path}")


if __name__ == "__main__":
    main()
