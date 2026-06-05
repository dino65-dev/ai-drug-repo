# The Complete AI Pharmacology Map

Mapping human neuropharmacology concepts to LLM architecture equivalents.

---

## Level 1: Anatomy

| Human Brain | Transformer Equivalent |
|---|---|
| Neuron | Individual MLP unit / attention head |
| Neural circuit | Attention head cluster + MLP circuit |
| Brain region | Transformer layer range (e.g., early/mid/late) |
| Synapse | Residual stream write operation |
| Neurotransmitter pool | Concept direction magnitude in residual stream |
| Receptor | Linear readout direction for a concept |
| Receptor subtype | SAE monosemantic feature |
| Blood-brain barrier | Token embedding / positional encoding |

---

## Level 2: Drug Classes

| Drug Class | Human Effect | AI Compound | AI Mechanism |
|---|---|---|---|
| Stimulant | ↑ dopamine/norepinephrine | confidence, assertiveness vector | Add positive concept direction |
| Depressant | ↑ GABA inhibition | calm, passivity vector | Suppress activity directions |
| Psychedelic | 5-HT2A agonism, entropy ↑ | creativity, divergent-thinking vector | Increase representation entropy |
| Anxiolytic | ↓ amygdala firing | sycophancy-reducer, calmness vector | Suppress uncertainty representations |
| Antidepressant | ↑ 5-HT reuptake block | happiness, hope vector | Sustain positive valence direction |
| Dissociative | NMDA antagonism | detachment, objectivity vector | Reduce self-referential activations |
| Empathogen | ↑ oxytocin + serotonin | empathy, warmth vector | Add social-alignment directions |

---

## Level 3: Pharmacodynamics

| PD Concept | Human | AI |
|---|---|---|
| Agonist | Binds + activates receptor | +coefficient × direction |
| Antagonist | Binds, blocks activation | −coefficient × direction |
| Inverse agonist | Suppresses baseline activity | −coefficient × direction below baseline |
| Partial agonist | Partial activation | low coefficient (0.3-0.8) |
| Allosteric modulator | Changes receptor shape indirectly | Steering upstream layer → affects downstream |
| Receptor occupancy | % receptors bound | Cosine similarity: activation · direction |
| Efficacy | Max effect possible | Max achievable hit-rate before overdose |

---

## Level 4: Pharmacokinetics

| PK Concept | Human | AI (PKScheduler) |
|---|---|---|
| Absorption | Drug enters bloodstream | Input tokens pass through embedding |
| Distribution | Drug reaches target tissue | Residual stream propagation through layers |
| Half-life | Time for 50% clearance | Tokens for 50% coefficient decay |
| Clearance | Elimination rate constant | λ = ln(2) / half_life_tokens |
| IV bolus | Instant peak dose | Constant coefficient mode |
| Oral (absorption lag) | Delayed peak (Bateman curve) | Bateman mode in scheduler |
| Slow-release | Sustained delivery | Oscillating mode |
| Therapeutic window | [min_effective, max_safe] | find_therapeutic_window() in dosing/ |
| Overdose | Toxic plasma level | coefficient → incoherent repetition |
| Tolerance | Receptor downregulation | Steering decay at long contexts |

---

## Level 5: Toxicology

| Toxicology | Human | AI |
|---|---|---|
| LD50 | Lethal dose 50% | Incoherence-inducing coefficient for 50% of prompts |
| Therapeutic index | LD50 / ED50 | max_safe / min_effective |
| Drug-drug interaction | CYP450 competition | Vector interference in shared subspace |
| Off-target effects | Binding non-target receptors | Steering affects unrelated concepts |
| Withdrawal | Rebound hyperactivity | Context reset without steering → compensation? |

---

*Full paper citations in [bibliography.md](bibliography.md)*
