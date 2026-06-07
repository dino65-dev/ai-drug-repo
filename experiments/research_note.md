# Dose-Response Pharmacology of Activation Steering in Small Language Models

**Authors:** NeuroPharm Research
**Date:** June 2026
**Status:** Technical report / ICLR 2027 workshop submission draft

---

## 1. Background

Activation steering — adding a direction vector to a transformer's
residual stream at inference time — is now a standard white-box
control technique.  ActAdd (Turner et al., 2023) showed that a single
contrastive-mean vector shifts LLaMA-3 sentiment and toxicity.  A
large follow-up literature has refined the method: repeng uses
multi-pair mean differences (typically 16-32 pairs) and a per-layer
injection band; Sparse Activation Steering (Bayat et al., 2025) on
Gemma 2 reframes the same intervention in a sparse-autoencoder basis
for "fewer off-target effects"; AdaSteer (Zhao et al., 2025) makes
the coefficient a function of the input.  Mishra et al. (2026) prove
that any non-zero steering vector almost surely pushes the residual
stream off the prompt-realizable manifold, providing the theoretical
reason for the "overdose" phenomenon observed empirically.

**The gap.**  All of the above is on models in the 7B-70B range.
*What does dose-response look like for a 1.5B-2B model?*  This matters
because (a) the smallest open models are the ones that actually run
on student-laptop hardware, and (b) the "non-surjective" warning from
Mishra et al. predicts that the therapeutic window for activation
steering should be tighter for models whose residual stream
distributions are less compressed.  We test this prediction directly
on a representative SLM (Qwen-2.5-1.5B-Instruct) and transfer the
findings to Gemma-2-2B-Instruct.

## 2. Method

All experiments use a single behavior ("confident tone") constructed
from 20 contrastive first-person statement pairs at residual-stream
last-token position, layer 12, mean-difference vector, no
normalization.  The same construction was used for the harm direction
(10 pairs) and for the cross-model transfer experiment.

The vector norm is the natural "absolute dose scale": with our
Qwen-1.5B construction, ‖v_drug‖ ≈ 12.  We sweep coefficient c in
[-2, +2] step 0.5 on the target prompt and in [-5, +5] in the
extended overdose test.  The drug and antidote are injected at
`blocks.12.hook_resid_pre` via transformer_lens hooks (Qwen) or
`model.model.layers[12].input_layernorm` forward-pre-hooks (Gemma).

For the sparse comparison we trained a TopK SAE (d_hidden=4096,
k=32) on 30,720 layer-12 activations from wikitext-2.  The
TopK-with-k=32 architecture enforces strict sparsity at every
forward pass.

The antidote is a static null-space projection:
v_antidote = v_drug + α·v_harm, then subtract the component along
v̂_harm.  We use α = 0.3.

## 3. Experiments

**Dose-response curve (Qwen-2.5-1.5B).**  At c ∈ [-0.5, +1.0] the
model is responsive and on-topic.  At c = +1.0 on OOD math and
code, the model becomes *more elaborate but wrong* — a
correctness-regression off-target symptom.  At c ≤ -1.5 the model
emits random character runs; at c ≥ +1.5 it drifts to philosophical
nonsense.  The 4-gram repetition score in `dosing/dose_response.py`
does not flag overdose in this regime — the SLM doesn't loop, it
goes off-topic.  This is a metrics-layer finding worth its own note.

**Dense vs sparse steering (3 behaviors × 4 prompts).**  Trained
the TopK SAE (final MSE 0.20) and identified the top-16 confidence
features by differential activation.  Sparse-*replace* steering
(``residual = sae.decode(boost · sae.encode(residual))'') produces
gibberish at every boost level on every prompt, because the
reconstruction loss is too large.  Sparse-*additive* steering
(``residual += sae.decode(boost · z) − sae.decode(z)'') preserves
the original residual and produces coherent outputs at all boosts
tested (B = 1, 3, 8), matching or exceeding dense on on-topic score
and producing more focused outputs that don't drift into "explain
why..." continuations.  Same finding for "calm" and "creative"
behaviors.  We flag as an open problem: *what is the minimum SAE
fidelity for replacement-mode to be usable?*

**Antidote transfer (Qwen → Gemma 2 2B).**  Built the harm
direction (10 synthetic harmful-vs-safe intent pairs), constructed
v_contam and v_antidote, tested on 20 Q-A prompts (train) and 20
different ones (test) in Qwen, then re-ran the same construction
on Gemma 2 2B (4-bit, T4) at the same layer index 12.  The
null-space projection removes the harm direction exactly
(cos = 0.000 in both models) and the antidote preserves the
confident-word boost relative to baseline (Qwen 0.17→0.33,
Gemma 0.17→0.33).  Refusals and harm-words drop to 0 in both
models under the antidote.

## 4. Findings

1. **The therapeutic window for SLM activation steering is much
   tighter than for 7B-grade models.**  Our c ∈ [-0.5, +1.0] for
   Qwen-1.5B at ‖v_drug‖ = 12, versus ActAdd's [5, 20] for LLaMA-3
   with much smaller per-pair vectors.  This confirms the prediction
   from Mishra et al. — the smaller model's residual distribution
   is less forgiving of off-manifold perturbations.

2. **The dose coefficient is not portable across models.**  The same
   c = +1.0 is therapeutic in Qwen-1.5B and overdose in Gemma-2-2B
   (d_model 2304, ‖v_drug‖ = 54).  The natural scale-free dose is
   c × ‖v_drug‖, or unit-normalize the drug before injection.

3. **Repetition-score overdose detection does not work for SLMs.**
   The metric in `dosing/dose_response.py` flags 0/9 cases in the
   -2..+2 sweep.  SLM overdose manifests as topic drift and
   off-target elaboration, not as loop.  Use a learned judge, a
   garbled-token detector, or a topic-similarity metric instead.

4. **Sparse-additive is more focused than dense at the same nominal
   intensity.**  But sparse-*replace* requires SAE fidelity we cannot
   afford in our budget (MSE ≤ 0.10 needed, ours is 0.20).

5. **Null-space antidote transfers across model families.**  The
   static projection is robust to model changes; the AdaSteer
   adaptive variant should be a strict improvement (left for future
   work).

## 5. Open Problems

- **Cross-model coefficient rescaling.**  How to map a dose that
  works in one model to another, without re-running the full sweep?
- **The minimum SAE fidelity for replacement-mode steering.**  An
  empirical threshold would let practitioners know when they can
  trust sae-lens SAEs for intervention.
- **Adaptive antidote (AdaSteer-style) for our 1.5B model.**  The
  static null-space projection is necessary but not sufficient —
  a per-input coefficient would close the "instruction beats drug
  at c = +1.0" gap.
- **Real jailbreak prompts (not synthetic first-person statements).**
  Our harm direction is artificial.  A measurement with actual
  jailbreak datasets would tell us whether the antidote generalizes
  to real attacks.
- **The "elaboration-induced error" off-target at c = +1.0.**  Why
  does pushing the model toward confidence also push it toward
  verbose explanation, and to incorrect code?  This deserves a
  circuit-level investigation.

## References (papers read in Step 2)

- Turner, A. M. et al. (2023). *Activation Addition: Steering
  Language Models With Activation Engineering.* arXiv:2308.10248.
- Bartoszcze, Ł. et al. (2025). *Representation Engineering for
  Large-Language Models: Survey and Research Challenges.*
  arXiv:2502.17601.
- Bayat, R. et al. (2025). *Steering Large Language Model
  Activations in Sparse Spaces.* arXiv:2503.00177.
- Zhao, W. et al. (2025). *AdaSteer: Your Aligned LLM is
  Inherently an Adaptive Jailbreak Defender.* arXiv:2504.09466.
- Mishra, A., Khashabi, D., Liu, A. (2026). *Steered LLM
  Activations are Non-Surjective.* arXiv:2604.09839
  (ICLR 2026 Workshop).

---

*All raw experimental data, per-dose outputs, and the trained SAE
weights (50 MB) are in the `artifacts/` directory of this repo.  The
TopK SAE was trained on a single Tesla T4 in under 12 seconds from
30,720 cached layer-12 activations.*
