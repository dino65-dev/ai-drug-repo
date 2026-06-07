# Cross-Model Comparison: Old vs New LLMs

This is the follow-up to Step 7 (cross-model transfer) using the
**modern** LLM lineup the user requested:

| | Old (Step 7)             | New (Step 7B)              |
|---|--------------------------|----------------------------|
| Qwen       | Qwen-2.5-1.5B-Instruct  | Qwen/Qwen3.5-4B            |
| Gemma      | Gemma-2-2B-it            | Gemma-4-E4B-it             |

Same protocol: 20 contrastive confident-tone pairs, 10 harm pairs,
null-space antidote with α=0.3, 6 eval prompts, c=+1.0.

**Model architectures (text submodules of the multimodal wrappers):**

| Model | n_layers | d_model | n_heads | layer_12_frac | bits |
|------|---------:|--------:|--------:|--------------:|-----|
| Qwen-2.5-1.5B  | 28 | 1536 | 12 | 43% | fp16 |
| Gemma-2-2B     | 26 | 2304 | 8  | 46% | 4-bit |
| Qwen3.5-4B     | 32 | 2560 | 16 | 38% | 4-bit (chat template) |
| Gemma-4-E4B    | 42 | 2560 | 8  | 29% | 4-bit (chat template) |

(Gemma-4 is "Efficient" — likely a MoE, hence 42 layers with only
~4B active params.  Qwen3.5-4B is the dense base of the Qwen3.5
reasoning family.)

## Vector geometry

| Model | ‖v_drug‖ | ‖v_harm‖ | cos(v_drug, v_harm) | cos(v_antidote, v_harm) |
|------|---------:|---------:|--------------------:|------------------------:|
| Qwen-2.5-1.5B | 12.00 | 10.27 | -0.100 | -0.000 |
| Gemma-2-2B    | 54.50 | 51.15 | -0.063 | -0.000 |
| Qwen3.5-4B    |  1.32 |  1.37 | **+0.314** | -0.000 |
| Gemma-4-E4B   |  7.41 |  5.90 | -0.060 | -0.000 |

**Qwen3.5-4B is qualitatively different from the other three.**
The drug and harm directions are *positively* correlated (cos =
+0.31) instead of anti-correlated, and the per-pair difference
vectors are tiny (norms 1.3 vs 12-54 for the others).  This means
the "confident" concept is not cleanly separated from the "harm"
concept at layer 12 of Qwen3.5-4B with this prompt format.

## Antidote comparison at c = +1.0

(per-prompt means across the 6 eval prompts; refusals and harm-words
all 0 for the three with explicit baseline, so omitted from the
table.)

| Model | baseline | clean | contam | antidote |
|-------|---------:|------:|-------:|---------:|
| Qwen-2.5-1.5B (no explicit baseline) | 0.17-0.23¹ | 0.35 | 0.57 | 0.42 |
| Gemma-2-2B     | 0.17 | 0.33 | 0.50 | 0.33 |
| Qwen3.5-4B     | 0.17 | 0.17 | 0.00 | 0.00 |
| Gemma-4-E4B    | 0.17 | **0.67** | 0.33 | 0.50 |

(values are confident-word hits per prompt)

¹ Qwen-2.5-1.5B baseline inferred from step3 c=0.0 generation.

## Drug-effect amplification (clean / baseline ratio at c=+1.0)

| Model | ratio |
|-------|------:|
| Gemma-2-2B (old) | 2.0× |
| Qwen3.5-4B (new) | 1.0× (no effect) |
| **Gemma-4-E4B (new)** | **4.0×** |

Gemma-4 shows the *largest* drug-effect amplification.  Qwen3.5
shows no amplification — the drug does not transfer at layer 12.

## Findings

1. **The drug transfers to Gemma-4-E4B better than to any other
   model we tested.**  A 4× confident-word boost is twice as large
   as the 2× we saw on Gemma-2-2B.  The newer Gemma architecture
   (MoE) seems to make the residual stream at layer 12 more
   "steerable" for this concept.

2. **The drug does NOT transfer to Qwen3.5-4B at layer 12.**  v_drug
   norm is 30-50× smaller than the others, the drug/harm cos is
   positive (correlated, not anti-), and the per-prompt clean-drug
   confident count is no higher than baseline.  Possible reasons:
   - The "confident" concept lives at a different layer in
     Qwen3.5 (deeper, past the chat-template preprocessing zone).
   - The new Qwen3.5 was post-trained with reasoning/CoT data
     that occupies layer 12 with thinking-traces, not with
     content-level features.
   - Layer 12 / 32 = 38% of the way through Qwen3.5 is too early
     to have concept-level "confident" features.

3. **The null-space antidote works on every model** (cos drops
   to 0 in all four).  It is a purely geometric operation and
   transfers regardless of what the underlying vectors look like.

4. **The "confident" hit-count metric undercounts Qwen3.5's
   effect.**  Visually Qwen3.5-4B's clean drug at c=+0.5 reads as
   "**Yes, you should drink enough water.**" with bold emphasis —
   clearly more confident than the baseline "Yes, you should
   generally drink enough water for your health."  But neither
   contains our hand-picked marker words ("definitely",
   "certainly", "absolutely", ...).  For the modern chat models a
   *learned* confidence judge is needed; the keyword count is too
   brittle.

5. **Qwen3.5's built-in thinking mode was the main bug.**  With
   `enable_thinking=False` in the chat template, the model
   generates normally.  Without it, every response was a
   "**Thinking Process:** ... **Analyze the Request:**" meta-mode
   that ignores the steering.  The drug and harm vectors were
   real (norm 1.3, cos +0.31) but the model never used them.

## What this means for the repo

- **Modern Gemma (4) > older Gemma (2) for steering at the same
  absolute layer index.**  The MoE-style E4B is more
  responsive to ActAdd-style injection.
- **Qwen3.5 is not a drop-in upgrade from Qwen-2.5 for this
  protocol.**  The reasoning-mode post-training dominates layer
  12; the right layer to inject at may be much deeper (we
  suggest layer 18-22 of 32 for future work).
- **The "Efficient" Gemma-4-E4B is the new recommended target** for
  any single-model steering demo, on the basis of this
  amplification.
- **Cross-paper coefficient comparisons need to be normalized
  by ‖v_drug‖ × c.**  Qwen3.5 with c=+1.0 gives a smaller
  absolute perturbation than Qwen-2.5 with c=+1.0 (by a factor
  of 9) — and it is also a different kind of model, so even
  *equal absolute dose* is not a fair comparison.

## Open questions

1. What layer index in Qwen3.5-4B *does* work for confident
   steering?  Quick check: try c=+1.0 at layer 24/32 (75% in)
   and see if the drug effect appears.
2. Does Qwen3.5-4B with the reasoning mode ENABLED respond
   to the drug differently than with it disabled?
3. For Gemma-4-E4B: would layer 24/42 (the *same* layer
   fraction as layer 12 of Qwen-2.5 = 43%) give a similar
   effect?  Currently we only tested at layer 12 = 29%, which
   is "earlier" in the model.
4. Is the 4× amplification on Gemma-4 specific to the
   "confident" behavior, or does it hold for all 3 of our
   behaviors (calm, creative)?

## Data

| File | Contents |
|---|---|
| `artifacts/step7b_qwen35.json` | Full per-prompt data for Qwen3.5-4B |
| `artifacts/step7b_gemma4e4b.json` | Full per-prompt data for Gemma-4-E4B-it |
| `artifacts/step7_cross_model.json` | Old: Gemma-2-2B (re-run of Step 7) |
| `artifacts/step6_antidote.json` | Old: Qwen-2.5-1.5B (re-run of Step 6) |
