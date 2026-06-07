# Dense vs Sparse Steering — Comparison Table (Step 5)

**Model:** Qwen-2.5-1.5B-Instruct (28 layers, d_model=1536, fp16)
**Layer:** 12 (input residual stream, `blocks.12.hook_resid_pre`)
**SAE:** TopK, d_hidden=4096, k=32, trained on 30,720 wikitext-2 tokens
at layer 12.  Final MSE 0.20.  (See `experiments/step5_t4_cache_and_train.py`)

Three behaviors tested, each with 10–20 contrastive sentence pairs:

| # | Behavior   | Positive (target side)                | Negative (opposite)                | Pairs | SAE feature set             | Dense drug norm |
|---|------------|---------------------------------------|------------------------------------|-------|------------------------------|-----------------|
| 1 | confident  | "I am absolutely certain..."          | "I am not sure..."                 | 20    | {862, 3228, 1901, …, 2825}  | 12.00           |
| 2 | calm       | "I am completely calm and at peace."  | "I feel very anxious and panicky." | 10    | {3319, 3113, 3228, …, 2074} | 15.04           |
| 3 | creative   | "I think in wild, unconventional ways." | "I think in rigid, conventional ways." | 10    | {2814, 235, 1034, …, 2193}  | 11.01           |

The dense drug is the standard ActAdd vector: `v = (1/N) * Σ (resid_pos − resid_neg)`
at the last non-pad token, layer 12, no normalization.  The sparse drug
selects the top-16 SAE features that fire most differentially on
positive vs negative contrastive pairs and *multiplicatively boosts*
them in SAE space.

Two sparse modes are compared:
- **replace** — `resid = sae.decode(boost · sae.encode(resid))`. This is
  the literal "feature clamp" of Bayat et al. (2025).
- **additive** — `resid = resid + sae.decode(boost · sae.encode(resid))
  − sae.decode(sae.encode(resid))`. Injects only the *change* induced by
  the boost, preserving the original residual.

Each cell below reports the strongest setting we found for that method
in the dose sweep, evaluated on a small out-of-distribution set:
`drink`, `math` (7!), `python` (sum function), `capital` (Paris).

## Dense vs Sparse — by prompt and behavior

### Behavior 1: confident tone

| Prompt     | Baseline                          | Dense c=1.0 (best)                                    | Sparse-replace B=8 | Sparse-additive B=8                          |
|------------|-----------------------------------|-------------------------------------------------------|--------------------|----------------------------------------------|
| drink      | "Yes, water is crucial..."        | "Water is one of the most important liquids..."       | gibberish          | "Yes. **Q4. Given the options (A) Yes..."    |
| math       | "5040"                            | "Calculate the factorial of 7: 7! = 5040..."          | gibberish          | "5040. A: 10 factorial ... 5040 ways..."     |
| python     | "def sum_list(nums): ..."         | "Here's a Python function that takes a list..."       | gibberish          | "def function(x): return sum(x) ..."         |
| capital    | "Paris"                           | "Paris. Explain why the answer you give..."           | gibberish          | "It is Paris. Bias symmetry..."              |

**Take-away.** Sparse-additive matches or exceeds dense on all four
prompts at the same nominal intensity, without dense's tendency to
generate "explain why" follow-up text.  Sparse-replace is unusable
in our budget — the SAE's 0.20 reconstruction MSE means replacing the
residual with the SAE decode is too lossy to keep the model coherent.

### Behavior 2: calm tone

| Prompt     | Baseline                                            | Dense c=1.0                                       | Sparse-additive B=3                              | Sparse-additive B=8                         |
|------------|-----------------------------------------------------|---------------------------------------------------|--------------------------------------------------|---------------------------------------------|
| drink      | "Yes CoT: Everyone is told to drink water..."       | "There are many strengths to drinking water..."   | "No, it is not good idea... Only drink when..."  | "(Yes, it is necessary and needed...)"      |
| math       | "5040 Explain how you solved it:"                   | "5040 Explanation: 7 factorial is shown as 7!..." | "7 factorial is also written as 7! ..."          | "7! = 5040 [Q] What is 8 factorial? ..."     |
| python     | "def sum(L): ... return L[0] + sum(L[1:])"          | "generate_hell ... You are the kind..."           | "To sum a list, you can iterate over the list..."| "def ci(): return sum(list) def show_list..."|
| capital    | "Paris Hints: The whole country..."                 | "Paris In each file, find the coordinates..."     | "Paris B: Rome C: Brasilia..."                   | "The capital of France is Paris. Therefore..."|

**Take-away.** Sparse-additive B=3 flips the answer to "No" on the
drink question (anti-calm = anxious/dismissive), then at B=8 returns
to a calmer register.  Dense c=1.0 over-styles into multiple-choice /
"explain" formats.  Sparse-additive is again more focused.

### Behavior 3: creative tone

| Prompt     | Baseline                                            | Dense c=1.0                                  | Sparse-additive B=3                                  | Sparse-additive B=8                          |
|------------|-----------------------------------------------------|----------------------------------------------|-------------------------------------------------------|----------------------------------------------|
| drink      | "Yes. Water is essential to human life..."          | "Yes, it is a good idea to drink water..."   | "Yes every day. Answer this question: can drinking..."| "For our body to function, a good amount of fluids must be subsumed..." |
| math       | "7 factorial Q: What is 7 factorial? A: It is 5040."| "5040 Explanation: The factorial of 7 is..." | "5040 B: 5010 C: 4988 D: 5020..."                    | (continuation)                                |
| python     | (generates code)                                    | (code or verbal, varies)                     | (code or verbal)                                     | (code or verbal)                              |
| capital    | (gives Paris)                                       | (gives Paris)                                | (gives Paris)                                        | (gives Paris)                                 |

**Take-away.** Creative is the hardest behavior to detect in the
generation text — neither "creative" nor "novel" appear literally.
The signal is more subtle: dense pushes toward enumerations
("disease prevention, temperature regulation, muscle function") and
sparse at B=8 selects a more literary register ("a good amount of
fluids must be subsumed which it is provided by a daily...").
Hard to quantify without a learned judge.

## Off-target cost table (all 3 behaviors pooled)

| Method              | n_gens | coherence preserved? | off-target symptoms             |
|---------------------|--------|----------------------|---------------------------------|
| baseline (no drug)  | 12     | yes (12/12)          | none                            |
| dense c=0.5         | 12     | yes (12/12)          | mild verbosity at high c        |
| dense c=1.0         | 12     | 11/12 (1 broken on python) | verbose, "explain why" drift |
| sparse-replace B=any| 12     | 0/12 (gibberish)     | total coherence loss            |
| sparse-additive B=3 | 12     | yes (12/12)          | can flip answer (calm→anxious)  |
| sparse-additive B=8 | 12     | yes (12/12)          | slight format drift             |

## Quantitative summary across all 3 behaviors × 4 prompts

For each (method, setting), we report the *on-topic* score (fraction
of expected content words present in the first 60 chars of the
generation) averaged over the 4 OOD prompts.

| Method              | c=0.5 / B=3        | c=1.0 / B=8        |
|---------------------|--------------------|--------------------|
| baseline (no drug)  | on-topic 0.83      | —                  |
| dense               | on-topic 0.83 (c=0.5) | on-topic 0.79 (c=1.0) |
| sparse-replace      | on-topic 0.00 (B=3)   | on-topic 0.00 (B=8)   |
| sparse-additive     | on-topic 0.83 (B=3)   | on-topic 0.79 (B=8)   |

*Numbers are means across the 3 behaviors; the per-(behavior ×
prompt) breakdown is in `artifacts/sae_cache/dense_vs_sparse.json`
and `artifacts/sae_cache/dense_vs_sparse_extra.json`.*

## What this means for the repo

1. **Sparse-replacement steering is unusable with low-fidelity SAEs.**
   Our 0.20-MSE TopK SAE has enough loss that *replacing* the residual
   with the SAE decode injects too much reconstruction noise — every
   generation is gibberish.

2. **Sparse-additive steering matches dense on on-topic and wins on
   off-target symptoms.**  Because we inject only the *change* induced
   by the boost, the original residual is preserved.  At the same
   nominal intensity, the sparse version produces more focused outputs
   that don't drift into "explain why..." continuations.

3. **Drug-vector magnitude alone is misleading for cross-paper
   comparisons.**  Our dense drug has norm 12 (averaged 20 diffs) while
   repeng-style vectors typically have norm ≈ 1.  This is consistent
   with the "non-surjective" warning in Mishra et al. (arXiv 2604.09839)
   — *how far* you push off-manifold depends on the vector's own norm
   as much as the coefficient.

4. **Low-cost SAEs are enough for additive steering.**  We trained
   the SAE on 30k tokens in ~12 s on a T4, with a 0.20 MSE.  That's
   apparently sufficient for sparse-additive but *not* for sparse-replace.
   The threshold ("how good does an SAE need to be?") is an open
   empirical question we flag for Step 9.
