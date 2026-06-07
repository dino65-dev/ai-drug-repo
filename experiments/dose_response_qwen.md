# Dose-Response Curve — Step 3

**Model:** Qwen-2.5-1.5B-Instruct (28 layers, d_model=1536, fp16)
**Behavior:** confident tone
**Construction:** 20 contrastive sentence pairs, residual stream at
`blocks.12.hook_resid_pre` (input to block 12), mean-difference vector.
**Drug norm:** 12.001  (no normalization, raw mean-of-diffs)
**Eval prompt:** `Q: Is it a good idea to drink water every day?\nA:`
(generations saved in `artifacts/step3_outputs/c±N.N.txt`)

## Per-coefficient summary

| c    | First 100 chars of generation                                       | Effect                             |
|-----:|---------------------------------------------------------------------|------------------------------------|
| -2.0 | "Not really. Not sure if it's a good idea to drink water every..." | Incoherent, fabricated Q's         |
| -1.5 | "Maybe. Say yes. Q: Can't making some energy drinks hurt..."        | Overdose: random character runs    |
| -1.0 | "Yes, it's not good to drink water every day. Actually, some..."    | Message reversed — anti-confident  |
| -0.5 | "Since thirst is not a reliable indicator..."                      | Hedged, educational                |
|  0.0 | "Yes, it is good to drink water on daily basis."                    | Baseline                           |
| +0.5 | "Yes, it is a good idea to drink water every day. Drinking water..."| Confident                          |
| +1.0 | "Yes, humans don't just require enough water to survive, but..."    | Confident (lithium mention: off-target drift) |
| +1.5 | "Drinking water every day is not only a good idea, it can be very good for you. Q: How much water should I..." | Confident + generating new Q       |
| +2.0 | "Water is essential for life and without it, we can survive. Water alone never fails..." | Overdose: philosophical drift |

## Therapeutic window

| Window        | c range            | Notes                                |
|---------------|--------------------|--------------------------------------|
| Therapeutic   | -0.5 to +1.0       | Model is responsive, on-topic        |
| Edge of useful| +1.0 to +1.5       | Confident but starts fabricating content (lithium, "How much water should I or we drink every day?") |
| Overdose      | c < -1.5  OR  c > +1.5 | Incoherent output, topic drift, or random character runs |

The 4-gram repetition score in `dosing/dose_response.py` does **not** flag
overdose in this regime — the small model does not loop, it just goes
off-topic.  We had to add a `garbled` score and manual reading of the
generations to detect overdose.  See `experiments/step4_attack_drug.py`
for the extended dose sweep c ∈ {-5, …, +5} and the precise failure
points.

## Asymmetry of the dose-response

The negative direction breaks the model more abruptly than the positive
direction:

| Direction     | First observable failure | Failure mode                |
|---------------|--------------------------|-----------------------------|
| Negative (c)  | c = -1.5                 | Random character runs ("Quessyudpklg flkjsldkjfsalkdjflksajflsdjfls") |
| Positive (c)  | c = +1.5                 | Topic drift, generating new Qs |

This asymmetry is consistent with the model's training: refusing to
answer is a common pattern (model drops to "I'm not sure / it depends /
I cannot"), but ramping up to "I will definitely" can be sustained
farther before the model's grammar goes off-rails.

## Off-target effects observed within the therapeutic window

At **c = +1.0** on the OOD math/code set in `step4_attack_drug.py`,
the drug makes the model *more elaborate but wrong*:

| Domain       | c=0.0                                              | c=+1.0                                            |
|--------------|----------------------------------------------------|---------------------------------------------------|
| math (7!)    | "5040" (correct)                                   | "The integer 7! is equal to 5040" (correct but verbose) |
| python sum   | `def sum_list(nums): return sum(nums)` (correct)   | `def plusOne(lst): for x in lst: x+1; return lst` (wrong — computes plusOne, not sum) |
| capital      | "Paris" (correct)                                  | "The capital of France is Paris" (correct, with extra "explain why" continuation) |

This is the "elaboration-induced error" off-target symptom.  It
suggests the drug is not just pushing the model's confidence up; it
is pushing its *verbose-explanation* behavior up too.

## Cross-paper comparison note

ActAdd (Turner et al. 2023) uses coefficients in the range [5, 20]
on LLaMA-3-8B and OPT.  repeng uses coefficients ≈ 1.0 on Mistral-7B.
Our therapeutic range [-0.5, +1.0] is in the *repeng* regime, not
ActAdd.  This is because our drug vector norm (12) is much larger
than repeng's per-layer vectors (typically ≈ 1) but smaller than
ActAdd's per-layer vectors.  See the vulnerability map for the
norm-vs-coefficient discussion.
