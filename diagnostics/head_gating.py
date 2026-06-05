"""
Causal Head Gating (CHG)

Learns soft gates g_h ∈ [0,1] over ALL attention heads simultaneously
to identify which heads facilitate, suppress, or are irrelevant to a behavior.

More scalable than one-at-a-time activation patching.

Analogy: Pharmacological receptor mapping — identify which receptor
         subtypes mediate which behavioral effects.

Reference:
  Causal Head Gating. NeurIPS 2025.
  https://arxiv.org/abs/2505.13737
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer


class CausalHeadGating:
    def __init__(self, model: HookedTransformer, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.n_heads  = model.cfg.n_heads

    def _collect_head_means(self, prompts: list[str]) -> torch.Tensor:
        """Returns (N_prompts, n_layers*n_heads*d_model) tensor."""
        all_vecs = []
        for prompt in prompts:
            tokens = self.model.to_tokens(prompt).to(self.device)
            head_parts = []
            hooks = []
            for layer in range(self.n_layers):
                hook_name = f"blocks.{layer}.attn.hook_z"
                def make_hook(l):
                    def fn(value, hook):
                        W_O = self.model.blocks[l].attn.W_O
                        for h in range(self.n_heads):
                            out = value[0, :, h, :] @ W_O[h]
                            head_parts.append(out.mean(0).detach())
                        return value
                    return fn
                hooks.append((hook_name, make_hook(layer)))
            with torch.no_grad():
                self.model.run_with_hooks(tokens, fwd_hooks=hooks)
            all_vecs.append(torch.cat(head_parts))
        return torch.stack(all_vecs)

    def fit(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
        epochs: int = 100,
        lr: float = 0.05,
        sparsity_lambda: float = 1e-3,
    ) -> dict[tuple[int, int], float]:
        """
        Learn soft gates separating positive (behavior present) from negative examples.
        Returns {(layer, head): gate_value}
        """
        pos_t = self._collect_head_means(positive_prompts).to(self.device)
        neg_t = self._collect_head_means(negative_prompts).to(self.device)

        n_head_feats = self.n_layers * self.n_heads
        d_per_head   = pos_t.shape[-1] // n_head_feats

        gates = nn.Parameter(torch.ones(n_head_feats, device=self.device) * 0.5)
        optimizer = torch.optim.Adam([gates], lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            g = torch.sigmoid(gates)
            g_exp = g.repeat_interleave(d_per_head)
            pos_gated = pos_t * g_exp.unsqueeze(0)
            neg_gated = neg_t * g_exp.unsqueeze(0)
            pos_mean  = pos_gated.mean(0)
            neg_mean  = neg_gated.mean(0)
            sep = -nn.functional.cosine_similarity(
                (pos_mean - neg_mean).unsqueeze(0), pos_mean.unsqueeze(0)).mean()
            loss = sep + sparsity_lambda * g.sum()
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print(f"  CHG Epoch {epoch}: loss={loss.item():.4f}, active={(g>0.5).sum().item()}")

        g_final = torch.sigmoid(gates).detach()
        gate_dict = {}
        idx = 0
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                gate_dict[(layer, head)] = round(g_final[idx].item(), 4)
                idx += 1
        return gate_dict

    def report(self, gates: dict[tuple[int, int], float], top_k: int = 10) -> None:
        sorted_gates = sorted(gates.items(), key=lambda x: x[1], reverse=True)
        print(f"\n=== Top {top_k} Causal Attention Heads ===")
        for (layer, head), score in sorted_gates[:top_k]:
            bar = "█" * int(score * 20)
            print(f"  L{layer:02d}H{head:02d} | {score:.4f} | {bar}")
        print()
