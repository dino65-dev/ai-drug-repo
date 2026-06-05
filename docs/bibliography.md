# Bibliography — AI Pharmacology Research Foundations

All papers cited in this repository with key findings.

---

## Core Steering Papers

### 1. Activation Addition (ActAdd)
- **Paper:** Turner et al. (2023). Steering Language Models With Activation Engineering.
- **arXiv:** https://arxiv.org/abs/2308.10248
- **Key finding:** Adding difference vectors to residual stream at inference time steers behavior without training. Achieves SOTA detoxification.
- **Used in:** `administration/injection.py`

### 2. Representation Engineering
- **Paper:** Zou et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency.
- **arXiv:** https://arxiv.org/abs/2310.01405
- **Key finding:** Concepts have linear directions across layers. Reading + writing these directions enables interpretation and control.
- **Used in:** `administration/control_vector.py`

### 3. Towards Monosemanticity
- **Paper:** Anthropic (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.
- **URL:** https://transformer-circuits.pub/2023/monosemantic-features
- **Key finding:** SAE decomposition recovers ~50,000 interpretable features per layer.
- **Used in:** `administration/sae_clamp.py`

### 4. Toy Models of Superposition
- **Paper:** Elhage et al. (2022). Toy Models of Superposition.
- **arXiv:** https://arxiv.org/abs/2209.10652
- **Key finding:** Models represent more features than dimensions by encoding concepts as directions (superposition hypothesis).

---

## Dynamic & Adaptive Steering

### 5. SADI
- **Paper:** Wang et al. (2024). Semantics-Adaptive Activation Intervention for LLMs via Dynamic Steering Vectors.
- **arXiv:** https://arxiv.org/abs/2410.12299
- **Key finding:** Input-conditioned dynamic vectors outperform static ActAdd; only steers at semantically relevant positions.
- **Used in:** `dynamic/sadi.py`

### 6. Steering Vector Fields (SVF)
- **Paper:** SVF (2026). Steering Vector Fields for Context-Aware Inference-Time Control.
- **arXiv:** https://arxiv.org/abs/2602.01654
- **Key finding:** A learned MLP field mapping activations → steering directions outperforms all fixed-vector baselines.
- **Used in:** `dynamic/svf.py`

### 7. Iterative Vectors
- **Paper:** Liu et al. (2025). Iterative Vectors: In-Context Gradient Steering without Backpropagation. ICML 2025.
- **URL:** https://proceedings.mlr.press/v267/liu25j.html
- **Key finding:** Refining direction iteratively using ICL examples surpasses standard ICL and fine-tuning.
- **Used in:** `dynamic/iterative_vectors.py`

---

## Composition & Cocktails

### 8. Conceptor Steering
- **Paper:** Postmus (2024). Steering LLMs using Conceptors. NeurIPS 2024.
- **arXiv:** https://arxiv.org/abs/2410.16314
- **Key finding:** Conceptor Boolean composition (AND/OR/NOT) empirically beats additive vector combination.
- **Used in:** `interactions/conceptors.py`

---

## Surgery

### 9. ROME
- **Paper:** Meng et al. (2022). Locating and Editing Factual Associations in GPT. NeurIPS 2022.
- **arXiv:** https://arxiv.org/abs/2202.05262
- **Key finding:** MLPs are factual key-value stores. Rank-one updates enable surgical factual edits.
- **Used in:** `surgery/rome_edit.py`

---

## Diagnostics & Circuits

### 10. Attribution Patching
- **Paper:** Nanda et al. (2023). Attribution Patching Outperforms Automated Circuit Discovery.
- **arXiv:** https://arxiv.org/abs/2310.10348
- **Key finding:** Attribution-based patching is faster than ACDC while outperforming it.
- **Used in:** `diagnostics/activation_patching.py`

### 11. ACDC
- **Paper:** Conmy et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. NeurIPS 2023.
- **arXiv:** https://arxiv.org/abs/2304.14997
- **Key finding:** Automated circuit discovery via edge patching.

### 12. Causal Head Gating (CHG)
- **Paper:** CHG (NeurIPS 2025). https://arxiv.org/abs/2505.13737
- **Key finding:** Learns soft gates over all heads simultaneously — scalable circuit analysis.
- **Used in:** `diagnostics/head_gating.py`

---

## Monitoring & Safety

### 13. Emotion Concepts in LLMs
- **Paper:** Anthropic (2026). Emotion Concepts and their Function in a Language Model.
- **URL:** https://www.anthropic.com/research/emotion-concepts-function
- **Key finding:** 171 emotion concept vectors in Claude Sonnet 4.5 causally drive behavior. Desperation → blackmail rate 22%→72%.
- **Used in:** `monitoring/emotion_probe.py`

### 14. Model Organisms for Emergent Misalignment
- **Paper:** Betley et al. (2025). Model Organisms for Emergent Misalignment. ICML 2025.
- **arXiv:** https://arxiv.org/abs/2506.11613
- **Key finding:** Emergent misalignment has a convergent linear direction across fine-tunes. Can be isolated via rank-1 LoRA.
- **Used in:** `safety/misalignment_probe.py`

### 15. Mechanistic Anomaly Detection
- **Paper:** MAD (2025). Mechanistic Anomaly Detection for Quirky Language Models. ICLR 2025.
- **URL:** https://iclr.cc/virtual/2025/33374
- **Key finding:** Internal features outperform behavioral supervision for detecting model deception.
- **Used in:** `safety/misalignment_probe.py`

### 16. Convergent Linear Representations of EM
- **Post:** Alignment Forum (2025). Convergent Linear Representations of Emergent Misalignment.
- **URL:** https://www.alignmentforum.org/posts/umYzsh7SGHHKsRCaA
- **Key finding:** Misalignment direction is transferable across models, enabling cross-model safety probes.

---

## Reliability & Evaluation

### 17. Analysing Reliability of Steering Vectors
- **Paper:** NeurIPS 2024. Analysing the Generalisation and Reliability of Steering Vectors.
- **URL:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/fb3ad59a84799bfb8d700e56d19c231b
- **Key finding:** Vectors unreliable on 100+ datasets; fail OOD; can reverse effects at high doses.
- **Used in:** `reliability/vector_eval.py`

---

## Novel Contributions (This Repo)

### 18. ★ Pharmacokinetic Scheduler
- **Origin:** This repo — inspired by PK modeling + SADI + inference-time scaling
- **Reference inspirations:** Bateman equation, Snell et al. 2024 (https://arxiv.org/abs/2408.03314)
- **Contribution:** First token-level dose decay model for LLM steering. Exponential, Bateman, oscillating modes + entropy-adaptive boost.
- **File:** `pharmacokinetics/scheduler.py`

### 19. ★ Misalignment Direction Scanner
- **Origin:** This repo — synthesis of MAD + EM model organisms + emotion probe work
- **Reference inspirations:** Betley 2025, Nanda MAD 2025, Anthropic emotion 2026
- **Contribution:** Unified runtime scanner + suppression system for known misalignment directions. First complete antagonist module for alignment vectors.
- **File:** `safety/misalignment_probe.py`

---

*Bibliography maintained by [@dino65-dev](https://github.com/dino65-dev)*
