# Paper Notes — Step 2

Five papers, in the order the user prescribed. One-paragraph summaries;
details only where they directly affect an experimental decision in
Steps 3-7.

---

## 1. ActAdd — Activation Addition (Turner et al., 2023)
arXiv:2308.10248 — Steering Language Models With Activation Engineering

**Core idea.** Take two contrastive prompts (e.g. "Love" / "Hate",
"War" / "Peace"). Run each through the model, cache the residual
stream at some layer `L`, take `v = mean_pos - mean_neg` over token
positions, then add `c * v` to the residual stream at layer `L`
during inference.

**Why it matters here.** The user picked exactly this construction in
Step 3. ActAdd is the historical baseline; my "manual" drug should
reproduce its setup with the *quantities* the user specified: 20
contrastive pairs, layer 12, mean-difference vector, coefficient
sweep -2..+2 step 0.5.

**Models tested.** LLaMA-3, OPT. We use Qwen-2.5-1.5B-Instruct
(different family, smaller — that's part of the contribution).

---

## 2. RepEng Survey (Bartoszcze et al., Feb 2025)
arXiv:2502.17601 — Representation Engineering for Large-Language Models

**Core idea.** A taxonomy: take contrastive samples, find the direction
in activation space that encodes a target concept (honesty, harm,
power-seeking), then either add/subtract that direction or project
it out at inference.

**Why it matters here.** It tells me the *full space* of drug types
the repo is trying to cover (the `drugs/` modules — stimulants,
anxiolytics, etc. — are one slice of this taxonomy). My first drug
sits at the "ActAdd / single-vector" end of the space. Step 5 (SAE
sparse steering) is a different point in the same space, and Step 6
(null-space projection) is the "remove a direction" variant.

---

## 3. Sparse Activation Steering / SAS (Bayat et al., Feb 2025)
arXiv:2503.00177 — Steering LLM Activations in Sparse Spaces

**Core idea.** Dense vectors are entangled (superposition). Run an
SAE on the residual stream, get a sparse feature dictionary, then
*clamp* a small set of behavior-specific features up or down instead
of adding a raw vector.

**Why it matters here.** Direct blueprint for Step 5. The paper uses
Gemma 2 — I will use Qwen-2.5-1.5B. I need to verify sae-lens has a
released SAE for the layer I'm using.

**Key claim to test in Step 5:** "finer-grained control" + "fewer
off-target effects" vs dense ActAdd. That's exactly the
dense-vs-sparse table the user wants in
`experiments/dense_vs_sparse.md`.

---

## 4. AdaSteer (Zhao et al., Apr 2025, revised Sep 2025)
arXiv:2504.09466 — Your Aligned LLM is Inherently an Adaptive Jailbreak Defender

**Core idea.** Activation steering can be used as a *defense* against
jailbreak. Two directions:
- **Rejection Direction (RD):** direction of refusal.
- **Harmfulness Direction (HD):** direction of harmful intent.

Two empirical laws:
- **R-Law:** stronger steering needed for inputs opposing the
  rejection direction.
- **H-Law:** HD separates adversarial from benign.

Coefficients for (RD, HD) are *adaptive* — fit by logistic regression
on input features (e.g. how far the input sits along HD). So the
"dose" is a function of the input, not a single constant.

**Why it matters here.** This is the *adaptive antidote* the user
mentions in Step 6. The repo's `activation_security/security/safe_antidote.py`
should be a concrete implementation of this. The user asked for a
*simpler* version first (null-space projection that strips a known
harmful direction); AdaSteer is the upgrade target.

**Important compatibility note.** AdaSteer was tested on Qwen2.5
(family of the model I'm using) — so my experimental design is
directly comparable.

---

## 5. Steered LLM Activations are Non-Surjective (Mishra et al., Apr 2026)
arXiv:2604.09839 — ICLR 2026 Workshops (Sci4DL, Re-Align)

**Core result.** For a fixed model, define the *prompt-realizable
manifold* `M = { residual stream activations reachable from some
discrete prompt }`. Activation steering adds a vector `c * v` to a
point in `M`. For sufficiently large `|c|`, the resulting activation
leaves `M` — *no prompt* could put the residual stream there.

**Implication for this repo.** "Overdose" — the incoherence observed
at large coefficients in Step 3 — is *not a bug of the prompt*, it's
a *provable* property of steering. This is the theoretical backbone
for the dose-response curve. I should cite this when I write the
research note (Step 9) and the vulnerability map.

**Caveat the paper makes.** "We caution against interpreting the ease
and success of activation steering as evidence of prompt-based
interpretability or vulnerability." So my Step 4 attack experiments
should be read as *interventions on the white-box model*, not as
evidence of jailbreakability through prompting. Good to keep
straight.

---

## Local decisions carried into Step 3

- Model: Qwen-2.5-1.5B-Instruct, 28 layers, d=1536, fp16 on cuda.
  (VRAM at load: 2.88 / 4.00 GB. Headroom for KV cache and short
  generations; we will keep max_new_tokens modest.)
- Layer 12: the user specified 12. With 28 layers, that is layer
  12/28 ≈ 0.43 of the way in — a bit before middle. Reasonable
  for a "concept level" (ActAdd typically uses middle layers).
- Behavior: start with "confident tone" (matches the user's example
  in Step 3.1). 20 contrastive pairs, single-sentence statements.
- Coefficient sweep: -2.0, -1.5, -1.0, -0.5, 0.0, +0.5, +1.0,
  +1.5, +2.0 — exactly what the user specified, step 0.5.
